"""Sandboxed tools and multi-layer memory tools for the Local LLM Agent Harness."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

_WORK_LOG_LOCK = threading.Lock()

import httpx
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS  # type: ignore

MAX_OUTPUT_CHARS = 6000

ALLOWED_MEMORY_FILES = (
    "rule.md",
    "profile.md",
    "memory.md",
    "plan.md",
    "plan_master.md",
    "work_log.md",
)

DANGEROUS_COMMAND_PATTERNS = [
    r"\brm\s+-rf\s+[/\\]",
    r"\bdel\s+/[sqf]+",
    r"\brmdir\s+/[sq]+",
    r"\bformat\s+[a-zA-Z]:",
    r"\bmkfs\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\breg\s+(delete|add)\b",
    r"\bdiskpart\b",
    r"\bbcdedit\b",
    r"\bpowershell\b.*-enc",
    r":\(\)\s*\{\s*:\|:&\s*\};:",
]


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return f"{text[:half]}\n\n... [省略: {omitted} 文字] ...\n\n{text[-half:]}"


def normalize_memory_filename(target: str) -> str:
    """Normalize memory file target name to one of ALLOWED_MEMORY_FILES."""
    cleaned = target.strip().lower()
    if not cleaned.endswith(".md"):
        cleaned = f"{cleaned}.md"
    if cleaned == "master_plan.md":
        cleaned = "plan_master.md"
    if cleaned not in ALLOWED_MEMORY_FILES:
        raise ValueError(
            f"無効なメモリファイル名 '{target}' です。使用可能: {', '.join(ALLOWED_MEMORY_FILES)}"
        )
    return cleaned


class WorkspaceTools:
    """Provides sandboxed file, command, Python, web, and memory tools."""

    def __init__(
        self,
        workspace_root: Path,
        memory_dir: Path,
        session_dir: Optional[Path] = None,
        plan_path: Optional[Path] = None,
        work_log_path: Optional[Path] = None,
        shared_workspace_dir: Optional[Path] = None,
        ultrathink_dir: Optional[Path] = None,
        on_memory_update: Optional[Callable[[str, str], Awaitable[None]]] = None,
        on_files_update: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.base_workspace_root: Path = self.workspace_root
        self.active_isolated_task_id: Optional[str] = None
        self.active_isolated_task_dir: Optional[Path] = None
        self._isolated_baseline_digests: dict[str, str] = {}
        self.memory_dir = memory_dir.resolve()
        self.session_dir = session_dir.resolve() if session_dir else None
        self.plan_path = plan_path.resolve() if plan_path else None
        self.work_log_path = work_log_path.resolve() if work_log_path else None
        self.shared_workspace_dir = shared_workspace_dir.resolve() if shared_workspace_dir else None
        self.ultrathink_dir = ultrathink_dir.resolve() if ultrathink_dir else None
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        if self.plan_path:
            self.plan_path.parent.mkdir(parents=True, exist_ok=True)
        if self.work_log_path:
            self.work_log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.shared_workspace_dir:
            self.shared_workspace_dir.mkdir(parents=True, exist_ok=True)
        if self.ultrathink_dir:
            self.ultrathink_dir.mkdir(parents=True, exist_ok=True)
        self.on_memory_update = on_memory_update
        self.on_files_update = on_files_update

    def resolve_safe_path(self, rel_path: str) -> Path:
        """Resolve a relative path strictly inside the active session sandbox."""
        if not rel_path or rel_path.strip() in ("", "."):
            return self.workspace_root
        cleaned = rel_path.strip().replace("\\", "/").lstrip("/")
        if cleaned.startswith("./"):
            cleaned = cleaned[2:]
        if not cleaned or cleaned == ".":
            return self.workspace_root
        if re.match(r"^[a-zA-Z]:", cleaned):
            raise ValueError(f"アクセス拒否: ドライブ絶対パス '{rel_path}' にはアクセスできません。")
        cleaned_lower = cleaned.lower()
        if cleaned_lower == "trash_bin" or cleaned_lower.startswith("trash_bin/") or "/trash_bin/" in f"/{cleaned_lower}":
            raise ValueError("アクセス拒否: trash_bin は隔離済みゴミ箱のためエージェントからは参照できません。")

        if self.plan_path and cleaned in ("plan.md", "plan_master.md", "master_plan.md"):
            return self.plan_path
        if self.work_log_path and cleaned == "work_log.md":
            return self.work_log_path

        if self.ultrathink_dir:
            ut_cleaned = cleaned
            while ut_cleaned.startswith("ultrathink/"):
                ut_cleaned = ut_cleaned[len("ultrathink/"):]
            while ut_cleaned.startswith("10_active_workspace/workspace/"):
                ut_cleaned = "10_active_workspace/" + ut_cleaned[len("10_active_workspace/workspace/"):]
            while ut_cleaned.startswith("workspace/"):
                ut_cleaned = ut_cleaned[len("workspace/"):]
            if ut_cleaned == "workspace":
                ut_cleaned = "10_active_workspace"
            while ut_cleaned.startswith("10_active_workspace/10_active_workspace/"):
                ut_cleaned = ut_cleaned[len("10_active_workspace/"):]

            if self.active_isolated_task_dir:
                is_hyp_doc = bool(re.match(r"^hyp_[^/]+\.md$", Path(ut_cleaned).name, re.IGNORECASE))
                if ut_cleaned in ("NORTH_STAR.md", "GLOBAL_INDEX.md", "20_knowledge_base", "90_archive") or ut_cleaned.startswith(
                    ("20_knowledge_base/", "90_archive/")
                ):
                    candidate = (self.ultrathink_dir / ut_cleaned).resolve()
                    if candidate != self.ultrathink_dir and self.ultrathink_dir not in candidate.parents:
                        raise ValueError(f"アクセス拒否: ultrathink外のパス '{rel_path}' にはアクセスできません。")
                    return candidate
                if is_hyp_doc:
                    hyp_rel = ut_cleaned if ut_cleaned.startswith("10_active_workspace/") else f"10_active_workspace/{ut_cleaned}"
                    candidate = (self.ultrathink_dir / hyp_rel).resolve()
                    if candidate != self.ultrathink_dir and self.ultrathink_dir not in candidate.parents:
                        raise ValueError(f"アクセス拒否: ultrathink外のパス '{rel_path}' にはアクセスできません。")
                    return candidate
                if ut_cleaned == "10_active_workspace":
                    return self.workspace_root
                iso_sub = ut_cleaned[len("10_active_workspace/"):] if ut_cleaned.startswith("10_active_workspace/") else ut_cleaned
                candidate = (self.workspace_root / iso_sub).resolve()
                if candidate != self.workspace_root and self.workspace_root not in candidate.parents:
                    raise ValueError(f"アクセス拒否: 隔離workspace外のパス '{rel_path}' にはアクセスできません。")
                return candidate

            if ut_cleaned in (
                "NORTH_STAR.md",
                "GLOBAL_INDEX.md",
                "10_active_workspace",
                "20_knowledge_base",
                "90_archive",
            ) or ut_cleaned.startswith(
                ("10_active_workspace/", "20_knowledge_base/", "90_archive/")
            ):
                candidate = (self.ultrathink_dir / ut_cleaned).resolve()
                if candidate != self.ultrathink_dir and self.ultrathink_dir not in candidate.parents:
                    raise ValueError(f"アクセス拒否: ultrathink外のパス '{rel_path}' にはアクセスできません。")
                return candidate
            candidate = (self.ultrathink_dir / "10_active_workspace" / ut_cleaned).resolve()
            act_root = (self.ultrathink_dir / "10_active_workspace").resolve()
            if candidate != act_root and act_root not in candidate.parents:
                raise ValueError(f"アクセス拒否: ultrathink外のパス '{rel_path}' にはアクセスできません。")
            return candidate

        if (
            self.shared_workspace_dir
            and self.workspace_root == self.shared_workspace_dir
            and (
                cleaned == "sub_agents"
                or cleaned.startswith("sub_agents/")
                or cleaned == "longthink/sub_agents"
                or cleaned.startswith("longthink/sub_agents/")
            )
        ):
            sub_agents_root = (self.shared_workspace_dir.parent / "sub_agents").resolve()
            norm_cleaned = cleaned[len("longthink/"):] if cleaned.startswith("longthink/") else cleaned
            sub_rel = norm_cleaned[len("sub_agents"):].lstrip("/")
            candidate = (sub_agents_root / sub_rel).resolve() if sub_rel else sub_agents_root
            if candidate != sub_agents_root and sub_agents_root not in candidate.parents:
                raise ValueError(f"アクセス拒否: sub_agents外のパス '{rel_path}' にはアクセスできません。")
            return candidate

        if self.shared_workspace_dir and (cleaned == "shared_workspace" or cleaned.startswith("shared_workspace/")):
            sub_rel = cleaned[len("shared_workspace"):].lstrip("/")
            candidate = (self.shared_workspace_dir / sub_rel).resolve() if sub_rel else self.shared_workspace_dir
            if candidate != self.shared_workspace_dir and self.shared_workspace_dir not in candidate.parents:
                raise ValueError(f"アクセス拒否: shared_workspace外のパス '{rel_path}' にはアクセスできません。")
            return candidate

        if cleaned.startswith("workspace/") and self.workspace_root.name == "workspace":
            cleaned = cleaned[len("workspace/"):]
        elif cleaned == "workspace" and self.workspace_root.name == "workspace":
            return self.workspace_root

        candidate = (self.workspace_root / cleaned).resolve()
        if candidate != self.workspace_root and self.workspace_root not in candidate.parents:
            raise ValueError(f"アクセス拒否: workspace外のパス '{rel_path}' にはアクセスできません。")
        return candidate

    def _display_rel_path(self, target: Path) -> str:
        resolved = target.resolve()
        if self.ultrathink_dir:
            try:
                return resolved.relative_to(self.ultrathink_dir).as_posix()
            except ValueError:
                pass
        if self.session_dir:
            try:
                return resolved.relative_to(self.session_dir).as_posix()
            except ValueError:
                pass
        try:
            return resolved.relative_to(self.workspace_root).as_posix()
        except ValueError:
            return resolved.name

    def _append_work_log(self, entry_line: str) -> None:
        if not self.work_log_path:
            return
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            with _WORK_LOG_LOCK:
                existing = (
                    self.work_log_path.read_text(encoding="utf-8", errors="replace")
                    if self.work_log_path.exists()
                    else ""
                )
                sep = "" if (not existing or existing.endswith("\n")) else "\n"
                self.work_log_path.write_text(f"{existing}{sep}- [{ts}] {entry_line}\n", encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def _file_sha256(path: Path) -> str:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return ""

    @staticmethod
    def _is_temp_isolated_artifact(rel_path: Path) -> bool:
        if any(part in {".git", "venv", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache"} for part in rel_path.parts):
            return True
        name_lower = rel_path.name.lower()
        stem_lower = rel_path.stem.lower()
        suffix_lower = rel_path.suffix.lower()
        if suffix_lower in {".tmp", ".bak", ".log", ".pyc", ".pyo", ".swp"}:
            return True
        if stem_lower.startswith(("tmp", "temp_", "debug_", "scratch_")) or stem_lower.endswith(("_tmp", "_temp", "_debug")):
            return True
        if name_lower.startswith("._"):
            return True
        return False

    @staticmethod
    def _is_final_promotable_artifact(rel_path: Path) -> bool:
        if WorkspaceTools._is_temp_isolated_artifact(rel_path):
            return False
        suffix_lower = rel_path.suffix.lower()
        return suffix_lower in {".py", ".md", ".json", ".csv", ".txt", ".html", ".tsv", ".yaml", ".yml"}

    def enter_isolated_workspace(self, step_id: str) -> Path:
        """Create/activate tool_isolated_ws/task_<step_id>/workspace/ and snapshot current workspace files."""
        sdir = self.session_dir or self.base_workspace_root.parent
        clean_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(step_id or "0")).strip("_") or "0"
        if clean_id.startswith("task_"):
            dir_name = clean_id
            norm_id = clean_id[5:] or "0"
        else:
            dir_name = f"task_{clean_id}"
            norm_id = clean_id

        iso_task_dir = (sdir / "tool_isolated_ws" / dir_name).resolve()
        iso_ws = (iso_task_dir / "workspace").resolve()
        if self.active_isolated_task_dir == iso_task_dir and iso_ws.exists():
            self.workspace_root = iso_ws
            return iso_ws

        iso_ws.mkdir(parents=True, exist_ok=True)
        baseline_digests: dict[str, str] = {}
        skip_names = {"plan.md", "plan_master.md", "master_plan.md", "work_log.md", "north_star.md", "global_index.md"}
        if self.base_workspace_root.exists():
            for src in sorted(self.base_workspace_root.rglob("*")):
                if not src.is_file():
                    continue
                if any(part in self.IGNORED_DIR_NAMES for part in src.parts):
                    continue
                if src.name.lower() in skip_names or (src.name.lower().startswith("hyp_") and src.suffix.lower() == ".md"):
                    continue
                try:
                    if src.stat().st_size > 1024 * 1024:
                        continue
                    rel_path = src.relative_to(self.base_workspace_root)
                except (OSError, ValueError):
                    continue
                dst = iso_ws / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))
                baseline_digests[rel_path.as_posix()] = self._file_sha256(src)

        self._isolated_baseline_digests = baseline_digests
        self.active_isolated_task_id = norm_id
        self.active_isolated_task_dir = iso_task_dir
        self.workspace_root = iso_ws
        return iso_ws

    def complete_isolated_workspace(
        self,
        step_id: Optional[str] = None,
        mode: str = "normal",
    ) -> Optional[dict[str, Any]]:
        """Promote verified final artifacts to canonical dirs and move all temporary files to trash_bin/task_<id>/."""
        sdir = self.session_dir or self.base_workspace_root.parent
        target_task_dir: Optional[Path] = None
        norm_id = self.active_isolated_task_id or "0"
        if step_id is not None and str(step_id).strip():
            clean_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(step_id)).strip("_") or "0"
            dir_name = clean_id if clean_id.startswith("task_") else f"task_{clean_id}"
            norm_id = dir_name[5:]
            cand_dir = (sdir / "tool_isolated_ws" / dir_name).resolve()
            if cand_dir.exists():
                target_task_dir = cand_dir
        if target_task_dir is None:
            target_task_dir = self.active_isolated_task_dir

        baseline_digests = dict(self._isolated_baseline_digests)
        self._isolated_baseline_digests.clear()
        self.workspace_root = self.base_workspace_root
        self.active_isolated_task_id = None
        self.active_isolated_task_dir = None

        if target_task_dir is None or not target_task_dir.exists():
            return None

        iso_ws = target_task_dir / "workspace"
        promoted_files: list[str] = []
        promoted_dest = "shared_workspace/final_artifacts/"

        if iso_ws.exists():
            all_ws_files = [p for p in sorted(iso_ws.rglob("*")) if p.is_file()]
            for src in all_ws_files:
                try:
                    rel_path = src.relative_to(iso_ws)
                except ValueError:
                    continue
                rel_posix = rel_path.as_posix()
                # Only pure temporary/debug artifacts stay in iso_ws to be moved into trash_bin/
                if self._is_temp_isolated_artifact(rel_path):
                    continue

                cur_digest = self._file_sha256(src)
                was_baseline_unchanged = (rel_posix in baseline_digests and baseline_digests[rel_posix] == cur_digest)
                if was_baseline_unchanged:
                    try:
                        src.unlink()
                    except OSError:
                        pass
                    continue

                if mode == "ultrathink" and self.ultrathink_dir:
                    final_art_dir = (sdir / "shared_workspace" / "final_artifacts").resolve()
                    kb_dir = (self.ultrathink_dir / "20_knowledge_base").resolve()
                    promoted_dest = "10_active_workspace/ & shared_workspace/final_artifacts/"
                    dest_targets = [self.base_workspace_root / rel_path, final_art_dir / rel_path]
                    if rel_path.suffix.lower() != ".py" and self._is_final_promotable_artifact(rel_path):
                        promoted_dest = "10_active_workspace/, 20_knowledge_base/ & shared_workspace/final_artifacts/"
                        dest_targets.append(kb_dir / rel_path)
                elif mode == "longthink" and self.shared_workspace_dir:
                    final_art_dir = (self.shared_workspace_dir / "final_artifacts").resolve()
                    promoted_dest = "longthink/shared_workspace/final_artifacts/ & workspace/"
                    dest_targets = [final_art_dir / rel_path, self.base_workspace_root / rel_path]
                else:
                    final_art_dir = (sdir / "shared_workspace" / "final_artifacts").resolve()
                    promoted_dest = "shared_workspace/final_artifacts/ & workspace/"
                    dest_targets = [final_art_dir / rel_path, self.base_workspace_root / rel_path]

                seen_dsts: set[Path] = set()
                for dst in dest_targets:
                    res_dst = dst.resolve()
                    if res_dst in seen_dsts or res_dst == src.resolve():
                        continue
                    seen_dsts.add(res_dst)
                    res_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src), str(res_dst))
                promoted_files.append(rel_posix)
                try:
                    src.unlink()
                except OSError:
                    pass

        # Move ONLY remaining temporary/debug/cache files in tool_isolated_ws/task_<id>/ into trash_bin/task_<id>/
        trash_task_dir = (sdir / "trash_bin" / target_task_dir.name).resolve()
        trashed_files: list[str] = []
        for rem in sorted(target_task_dir.rglob("*")):
            if not rem.is_file():
                continue
            try:
                rel_rem = rem.relative_to(target_task_dir)
            except ValueError:
                rel_rem = Path(rem.name)
            dst_rem = trash_task_dir / rel_rem
            if dst_rem.exists():
                dst_rem = trash_task_dir / f"{dst_rem.stem}_{norm_id}{dst_rem.suffix}"
            dst_rem.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(rem), str(dst_rem))
            trashed_files.append(rel_rem.as_posix())

        try:
            shutil.rmtree(str(target_task_dir), ignore_errors=True)
        except Exception:
            pass

        trash_rel = f"trash_bin/{target_task_dir.name}"
        self._append_work_log(
            f"🗑️ 仮ファイルを {trash_rel} へ隔離完了 (昇格: {len(promoted_files)}件, ゴミ箱移動: {len(trashed_files)}件)"
        )
        return {
            "task_id": norm_id,
            "promoted_files": promoted_files,
            "promoted_dest": promoted_dest,
            "trash_path": trash_rel,
            "trashed_files": trashed_files,
            "trashed_count": len(trashed_files),
            "message": "🗑️ 仮ファイルを trash_bin へ隔離完了",
        }

    async def web_search(self, query: str, max_results: int = 5) -> str:
        """Search the web using DuckDuckGo."""
        if not query or not query.strip():
            return "エラー: 検索クエリが空です。"
        max_results = max(1, min(int(max_results), 10))

        def _search() -> list[dict[str, Any]]:
            with DDGS() as ddgs:
                return list(ddgs.text(query.strip(), max_results=max_results))

        try:
            results = await asyncio.to_thread(_search)
            if not results:
                return f"検索結果なし: '{query}'"
            lines = [f"検索結果 ('{query}'):"]
            for i, item in enumerate(results, 1):
                title = item.get("title", "No title")
                href = item.get("href", item.get("link", ""))
                body = item.get("body", item.get("snippet", ""))
                lines.append(f"{i}. {title}\n   URL: {href}\n   概要: {body}")
            return _truncate("\n".join(lines))
        except Exception as exc:
            return f"web_search エラー: {exc}"

    async def fetch_url(self, url: str) -> str:
        """Fetch and extract readable text from a URL."""
        if not url or not url.strip().startswith(("http://", "https://")):
            return "エラー: URLは http:// または https:// で始まる必要があります。"
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            }
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.get(url.strip(), headers=headers)
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")
                if "text/html" in content_type or "<html" in resp.text[:500].lower():
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                        tag.decompose()
                    text = soup.get_text(separator="\n")
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    extracted = "\n".join(lines)
                else:
                    extracted = resp.text
                return _truncate(f"URL: {url}\nステータス: {resp.status_code}\n\n{extracted}")
        except Exception as exc:
            return f"fetch_url エラー ({url}): {exc}"

    def _sync_sibling_plan_files(self, content: str) -> None:
        """Keep session plan_master.md and plan.md synchronized when session_dir is set."""
        if not self.session_dir:
            return
        try:
            for name in ("plan_master.md", "plan.md"):
                p = self.session_dir / name
                if not self.plan_path or p.resolve() != self.plan_path:
                    p.write_text(content, encoding="utf-8")
        except Exception:
            pass

    async def write_file(self, path: str, content: str) -> str:
        """Write content to a file inside workspace/."""
        try:
            target = self.resolve_safe_path(path)
            if target == self.workspace_root or (self.ultrathink_dir and target == self.ultrathink_dir):
                return "エラー: ファイル名を指定してください。"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            rel = self._display_rel_path(target)
            self._append_work_log(f"write_file: {rel} ({len(content)} 文字)")
            if self.plan_path and target.resolve() == self.plan_path:
                self._sync_sibling_plan_files(content)
                if self.on_memory_update:
                    await self.on_memory_update("plan.md", content)
            if self.on_files_update:
                await self.on_files_update()
            return f"ファイル書き込み完了: {rel} ({len(content)} 文字)"
        except Exception as exc:
            return f"write_file エラー: {exc}"

    async def read_file(self, path: str) -> str:
        """Read a file inside workspace/."""
        try:
            target = self.resolve_safe_path(path)
            if not target.exists():
                return f"エラー: ファイルが見つかりません: {path}"
            if target.is_dir():
                return f"エラー: '{path}' はディレクトリです。list_files を使用してください。"
            text = target.read_text(encoding="utf-8", errors="replace")
            return _truncate(text)
        except Exception as exc:
            return f"read_file エラー: {exc}"

    IGNORED_DIR_NAMES = {
        ".git",
        "venv",
        ".venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "trash_bin",
        "tool_isolated_ws",
    }

    async def list_files(self, path: str = ".") -> str:
        """List files and directories inside workspace/."""
        try:
            target = self.resolve_safe_path(path)
            if not target.exists():
                return f"エラー: パスが見つかりません: {path}"
            if not target.is_dir():
                return f"エラー: '{path}' はディレクトリではありません。"
            entries: list[str] = []
            for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if item.name in self.IGNORED_DIR_NAMES:
                    continue
                rel = self._display_rel_path(item)
                if item.is_dir():
                    entries.append(f"[DIR]  {rel}/")
                else:
                    size = item.stat().st_size
                    entries.append(f"[FILE] {rel} ({size} bytes)")
            if not entries:
                return (
                    "ディレクトリは空です（正常な初期状態です。まだファイルは作成されていないため、"
                    "再確認は不要です。次は write_file で必要なスクリプトやファイルを新規作成してください）。"
                )
            if len(entries) > 50:
                omitted = len(entries) - 50
                return "\n".join(entries[:50]) + f"\n... (他 {omitted} 件省略)"
            return "\n".join(entries)
        except Exception as exc:
            return f"list_files エラー: {exc}"

    async def run_python(self, code: str, timeout: int = 60) -> str:
        """Execute Python code in workspace/ using the active Python interpreter with UTF-8 mode."""
        if not code or not code.strip():
            return "エラー: 実行するPythonコードが空です。"
        timeout = max(1, min(int(timeout), 300))

        def _exec() -> str:
            tmp_path: Optional[Path] = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".py",
                    dir=self.workspace_root,
                    delete=False,
                    encoding="utf-8",
                ) as tmp:
                    tmp.write(code)
                    tmp_path = Path(tmp.name)

                utf8_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
                proc = subprocess.run(
                    [sys.executable, "-X", "utf8", str(tmp_path)],
                    cwd=str(self.workspace_root),
                    env=utf8_env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    encoding="utf-8",
                    errors="replace",
                )
                out = proc.stdout.strip()
                err = proc.stderr.strip()
                parts = [f"終了コード: {proc.returncode}"]
                if out:
                    parts.append(f"--- STDOUT ---\n{out}")
                if err:
                    parts.append(f"--- STDERR ---\n{err}")
                if not out and not err:
                    parts.append("(出力なし)")
                return _truncate("\n".join(parts))
            except subprocess.TimeoutExpired:
                return f"エラー: Python実行がタイムアウトしました ({timeout}秒)。"
            finally:
                if tmp_path and tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

        result = await asyncio.to_thread(_exec)
        first_line = result.splitlines()[0] if result else ""
        preview_body = " ".join(
            line.strip()
            for line in result.splitlines()[1:]
            if line.strip() and not line.strip().startswith("---")
        )
        if len(preview_body) > 180:
            preview_body = preview_body[:177] + "..."
        summary = f"{first_line} | {preview_body}".strip(" |")
        self._append_work_log(f"run_python ({self._display_rel_path(self.workspace_root)}) -> {summary}")
        if self.on_files_update:
            await self.on_files_update()
        return result

    async def run_command(self, command: str, timeout: int = 60) -> str:
        """Execute a shell command inside workspace/ with safety checks."""
        if not command or not command.strip():
            return "エラー: コマンドが空です。"
        for pattern in DANGEROUS_COMMAND_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return f"エラー: 安全上の理由により、このコマンドはブロックされました: {command}"

        timeout = max(1, min(int(timeout), 300))

        def _exec() -> str:
            try:
                utf8_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
                proc = subprocess.run(
                    command,
                    shell=True,
                    cwd=str(self.workspace_root),
                    env=utf8_env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    encoding="utf-8",
                    errors="replace",
                )
                out = proc.stdout.strip()
                err = proc.stderr.strip()
                parts = [f"終了コード: {proc.returncode}"]
                if out:
                    parts.append(f"--- STDOUT ---\n{out}")
                if err:
                    parts.append(f"--- STDERR ---\n{err}")
                if not out and not err:
                    parts.append("(出力なし)")
                return _truncate("\n".join(parts))
            except subprocess.TimeoutExpired:
                return f"エラー: コマンド実行がタイムアウトしました ({timeout}秒)。"
            except Exception as exc:
                return f"run_command エラー: {exc}"

        result = await asyncio.to_thread(_exec)
        self._append_work_log(f"run_command: {command.strip()[:80]}")
        if self.on_files_update:
            await self.on_files_update()
        return result

    async def update_memory(self, target: str, content: str, mode: str = "overwrite") -> str:
        """Update persistent Markdown memory (session-isolated for plan_master.md/plan.md/work_log.md)."""
        try:
            filename = normalize_memory_filename(target)
            is_plan_target = filename in ("plan.md", "plan_master.md")
            if is_plan_target and self.plan_path:
                path = self.plan_path
                scope_label = self._display_rel_path(path)
            elif filename == "work_log.md" and self.work_log_path:
                path = self.work_log_path
                scope_label = self._display_rel_path(path)
            else:
                path = self.memory_dir / ("plan.md" if filename == "plan_master.md" else filename)
                scope_label = f"memory/{filename}"

            path.parent.mkdir(parents=True, exist_ok=True)
            if mode == "append" and path.exists():
                existing = path.read_text(encoding="utf-8", errors="replace").rstrip()
                new_content = f"{existing}\n\n{content.strip()}\n"
            else:
                new_content = content
            path.write_text(new_content, encoding="utf-8")
            if is_plan_target and self.plan_path:
                self._sync_sibling_plan_files(new_content)
            if self.on_memory_update:
                await self.on_memory_update("plan.md" if is_plan_target else filename, new_content)
            if self.on_files_update:
                await self.on_files_update()
            return f"永続メモリ {scope_label} を更新しました ({mode})。"
        except Exception as exc:
            return f"update_memory エラー: {exc}"

    async def update_plan(self, content: str) -> str:
        """Convenience wrapper to update active plan_master.md / plan.md."""
        return await self.update_memory("plan_master.md", content, mode="overwrite")

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch tool execution by name."""
        handlers: dict[str, Callable[..., Awaitable[str]]] = {
            "web_search": self.web_search,
            "fetch_url": self.fetch_url,
            "write_file": self.write_file,
            "read_file": self.read_file,
            "list_files": self.list_files,
            "list_dir": self.list_files,
            "run_python": self.run_python,
            "run_command": self.run_command,
            "update_memory": self.update_memory,
            "update_plan": self.update_plan,
        }
        handler = handlers.get(name)
        if not handler:
            return f"エラー: 未知のツール '{name}' です。"
        try:
            return await handler(**arguments)
        except TypeError as exc:
            return f"ツール '{name}' の引数エラー: {exc} (受信引数: {json.dumps(arguments, ensure_ascii=False)})"
        except Exception as exc:
            return f"ツール '{name}' 実行エラー: {exc}"


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "update_memory",
            "description": (
                "多層永続メモリ (memory/ 内の Markdown ファイル) を更新する。"
                "target には rule.md, profile.md, memory.md, plan.md, work_log.md のいずれかを指定する。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "enum": ["rule.md", "profile.md", "memory.md", "plan.md", "plan_master.md", "work_log.md"],
                        "description": "更新対象のメモリファイル名",
                    },
                    "content": {
                        "type": "string",
                        "description": "書き込むMarkdownテキスト",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["overwrite", "append"],
                        "description": "overwrite (上書き) または append (追記)",
                        "default": "overwrite",
                    },
                },
                "required": ["target", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_plan",
            "description": "進行中タスクの計画チェックリスト (plan_master.md / plan.md) をMarkdownチェックリスト (- [ ] / - [x]) で更新する。",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "plan.md に書き込む完全なMarkdownテキスト",
                    }
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "DuckDuckGoを使用してインターネット検索を行い、タイトル・URL・概要を取得する。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "検索クエリ"},
                    "max_results": {
                        "type": "integer",
                        "description": "取得する最大件数 (1-10, デフォルト5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "指定したURLのWebページ本文テキストを取得する。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "取得対象のURL (http/https)"}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "workspace/ フォルダ内にファイルを作成または上書き保存する。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "workspace/ からの相対パス"},
                    "content": {"type": "string", "description": "書き込む内容"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "workspace/ フォルダ内のファイル内容を読み取る。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "workspace/ からの相対パス"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "workspace/ フォルダ内のファイル・ディレクトリ一覧を取得する。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "workspace/ からの相対ディレクトリパス (デフォルト '.')",
                        "default": ".",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "workspace/ をカレントディレクトリとしてPythonコードを実行し、標準出力・標準エラーを返す。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "実行するPythonコード"},
                    "timeout": {
                        "type": "integer",
                        "description": "タイムアウト秒数 (デフォルト60、最大300)",
                        "default": 60,
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "workspace/ をカレントディレクトリとしてシェルコマンドを実行する。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "実行するコマンド"},
                    "timeout": {
                        "type": "integer",
                        "description": "タイムアウト秒数 (デフォルト60、最大300)",
                        "default": 60,
                    },
                },
                "required": ["command"],
            },
        },
    },
]

LONGTHINK_EXTRA_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "update_master_plan",
            "description": "/longthink 指揮者専用: shared_workspace/master_plan.md を全体計画チェックリスト (- [ ] / - [x]) で更新する。",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "master_plan.md に書き込む完全なMarkdownテキスト",
                    }
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_parallel_workers",
            "description": (
                "/longthink マスター (Orchestrator) 専用: 課題の性質や進行状況に応じて今何体のサブエージェント "
                "(1〜4体) を立ち上げるべきかを動的に決定し、各サブエージェントに個別の隔離フォルダ "
                "(sub_agents/<sub_agent_id>/plan.md および workspace/) を自動生成して asyncio.gather で並列実行する。"
                "行き詰まった既存の agent_id を指定して方針変更指示を与える再実行にも使用できる。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "description": "立ち上げるサブエージェントごとのタスク配列 (1〜4件を動的に指定)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "agent_id": {
                                    "type": "string",
                                    "description": "任意のサブエージェントID (例: researcher_01, coder_01, worker_01)。既存ID指定時は方針変更・追加指示として同フォルダを再利用。省略時は自動採番。",
                                    "default": "",
                                },
                                "agent_role": {
                                    "type": "string",
                                    "enum": ["worker", "critic"],
                                    "description": "worker (実作業・コード・調査担当) または critic (検算・レビュー・検証担当)",
                                    "default": "worker",
                                },
                                "task_title": {
                                    "type": "string",
                                    "description": "サブエージェントに任せる独立サブタスクの短いタイトル",
                                },
                                "instruction": {
                                    "type": "string",
                                    "description": "サブエージェントへの具体的な作業・検証・方針変更指示",
                                },
                            },
                            "required": ["task_title", "instruction"],
                        },
                    }
                },
                "required": ["tasks"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_sub_agent",
            "description": (
                "/longthink マスター (Orchestrator) 専用: 1体のサブエージェント (worker / critic) を動的に生成 "
                "または既存の sub_agents/<agent_id>/ に方針変更指示を与えて実行し、専用 workspace/ と plan.md で作業させて "
                "handoff_board.md へ結果報告を書かせる。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_role": {
                        "type": "string",
                        "enum": ["worker", "critic"],
                        "description": "worker (実作業・コード・調査担当) または critic (検算・レビュー・検証担当)",
                    },
                    "task_title": {
                        "type": "string",
                        "description": "サブエージェントに任せるタスクの短いタイトル",
                    },
                    "instruction": {
                        "type": "string",
                        "description": "サブエージェントへの具体的な作業・検証・方針変更指示",
                    },
                    "agent_id": {
                        "type": "string",
                        "description": "任意のID (例: worker_01, critic_01)。既存ID指定時は同じ隔離フォルダに対して方針変更・追加指示として実行。省略時は自動採番。",
                        "default": "",
                    },
                },
                "required": ["agent_role", "task_title", "instruction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_final_artifact",
            "description": "/longthink 指揮者専用: 合格した最終成果物を shared_workspace/final_artifacts/ に保存する。",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "保存するファイル名 (例: report.md, solution.py)",
                    },
                    "content": {
                        "type": "string",
                        "description": "成果物の内容",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
]

ULTRATHINK_EXTRA_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "upsert_hypothesis",
            "description": (
                "/ultrathink ハイギア専用: 10_active_workspace/hyp_XXXX.md をYAMLフロントマター付きで作成・更新し、"
                "GLOBAL_INDEX.md を同期する。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hyp_id": {
                        "type": "string",
                        "description": "仮説ID (例: HYP_0001)。省略時は新規採番。",
                        "default": "",
                    },
                    "title": {
                        "type": "string",
                        "description": "サブタスク・仮説の簡潔なタイトル",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["UNVERIFIED", "VERIFIED", "REJECTED"],
                        "description": "検証ステータス (デフォルト UNVERIFIED)",
                        "default": "UNVERIFIED",
                    },
                    "content": {
                        "type": "string",
                        "description": "仮説の内容・根拠・検証手順・反証チェック結果 (Markdown本文)",
                    },
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_hypothesis_status",
            "description": (
                "/ultrathink ローギア専用: 思考なしで即答し、対象仮説のステータス (UNVERIFIED/VERIFIED/REJECTED) と "
                "GLOBAL_INDEX.md を高速更新する。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hyp_id": {
                        "type": "string",
                        "description": "対象仮説ID (例: HYP_0001)",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["UNVERIFIED", "VERIFIED", "REJECTED"],
                        "description": "確定ステータス",
                    },
                    "summary": {
                        "type": "string",
                        "description": "確定結果または棄却理由の1〜2文要約",
                    },
                },
                "required": ["hyp_id", "status", "summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_global_index",
            "description": "/ultrathink 専用: GLOBAL_INDEX.md の目次テーブル全体を更新する。",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "GLOBAL_INDEX.md に書き込むMarkdownテキスト",
                    }
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crystallize_knowledge",
            "description": (
                "/ultrathink GC/知識統合専用: 検証済みの確定事実を 20_knowledge_base/fact_XXXX.md へ結晶化保存し、"
                "20_knowledge_base/INDEX_KB.md を更新する。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact_id": {
                        "type": "string",
                        "description": "確定知識ID (例: FACT_0001)。省略時は自動採番。",
                        "default": "",
                    },
                    "title": {
                        "type": "string",
                        "description": "確定知識のタイトル",
                    },
                    "content": {
                        "type": "string",
                        "description": "統合された確定事実・根拠数値・結論 (Markdown)",
                    },
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_ultrathink",
            "description": (
                "/ultrathink 専用: NORTH_STAR.md の全ゴール・絶対制約が VERIFIED 事実によって完全に達成されたときのみ呼び出し、"
                "最終回答を提出する。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "final_report": {
                        "type": "string",
                        "description": "ユーザー向けの包括的な最終報告書 (Markdown)",
                    }
                },
                "required": ["final_report"],
            },
        },
    },
]
