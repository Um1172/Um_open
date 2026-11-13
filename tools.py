"""Sandboxed tools and multi-layer memory tools for the Local LLM Agent Harness."""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

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
        on_memory_update: Optional[Callable[[str, str], Awaitable[None]]] = None,
        on_files_update: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.memory_dir = memory_dir.resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.on_memory_update = on_memory_update
        self.on_files_update = on_files_update

    def resolve_safe_path(self, rel_path: str) -> Path:
        """Resolve a relative path strictly inside workspace_root."""
        if not rel_path or rel_path.strip() == "":
            return self.workspace_root
        cleaned = rel_path.strip().lstrip("/\\")
        candidate = (self.workspace_root / cleaned).resolve()
        if candidate != self.workspace_root and self.workspace_root not in candidate.parents:
            raise ValueError(f"アクセス拒否: workspace外のパス '{rel_path}' にはアクセスできません。")
        return candidate

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

    async def write_file(self, path: str, content: str) -> str:
        """Write content to a file inside workspace/."""
        try:
            target = self.resolve_safe_path(path)
            if target == self.workspace_root:
                return "エラー: ファイル名を指定してください。"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            rel = target.relative_to(self.workspace_root).as_posix()
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
                rel = item.relative_to(self.workspace_root).as_posix()
                if item.is_dir():
                    entries.append(f"[DIR]  {rel}/")
                else:
                    size = item.stat().st_size
                    entries.append(f"[FILE] {rel} ({size} bytes)")
            if not entries:
                return f"ディレクトリは空です: {path}"
            return "\n".join(entries)
        except Exception as exc:
            return f"list_files エラー: {exc}"

    async def run_python(self, code: str, timeout: int = 30) -> str:
        """Execute Python code in workspace/ using the active Python interpreter."""
        if not code or not code.strip():
            return "エラー: 実行するPythonコードが空です。"
        timeout = max(1, min(int(timeout), 120))

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

                proc = subprocess.run(
                    [sys.executable, str(tmp_path)],
                    cwd=str(self.workspace_root),
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
        if self.on_files_update:
            await self.on_files_update()
        return result

    async def run_command(self, command: str, timeout: int = 30) -> str:
        """Execute a shell command inside workspace/ with safety checks."""
        if not command or not command.strip():
            return "エラー: コマンドが空です。"
        for pattern in DANGEROUS_COMMAND_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return f"エラー: 安全上の理由により、このコマンドはブロックされました: {command}"

        timeout = max(1, min(int(timeout), 120))

        def _exec() -> str:
            try:
                proc = subprocess.run(
                    command,
                    shell=True,
                    cwd=str(self.workspace_root),
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
        if self.on_files_update:
            await self.on_files_update()
        return result

    async def update_memory(self, target: str, content: str, mode: str = "overwrite") -> str:
        """Update one of the persistent memory Markdown files in memory/."""
        try:
            filename = normalize_memory_filename(target)
            path = self.memory_dir / filename
            if mode == "append" and path.exists():
                existing = path.read_text(encoding="utf-8").rstrip()
                new_content = f"{existing}\n\n{content.strip()}\n"
            else:
                new_content = content
            path.write_text(new_content, encoding="utf-8")
            if self.on_memory_update:
                await self.on_memory_update(filename, new_content)
            return f"永続メモリ memory/{filename} を更新しました ({mode})。"
        except Exception as exc:
            return f"update_memory エラー: {exc}"

    async def update_plan(self, content: str) -> str:
        """Convenience wrapper to update memory/plan.md."""
        return await self.update_memory("plan.md", content, mode="overwrite")

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch tool execution by name."""
        handlers: dict[str, Callable[..., Awaitable[str]]] = {
            "web_search": self.web_search,
            "fetch_url": self.fetch_url,
            "write_file": self.write_file,
            "read_file": self.read_file,
            "list_files": self.list_files,
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
                        "enum": ["rule.md", "profile.md", "memory.md", "plan.md", "work_log.md"],
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
            "description": "進行中タスクの計画チェックリスト (memory/plan.md) をMarkdownチェックリスト (- [ ] / - [x]) で更新する。",
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
                        "description": "タイムアウト秒数 (デフォルト30)",
                        "default": 30,
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
                        "description": "タイムアウト秒数 (デフォルト30)",
                        "default": 30,
                    },
                },
                "required": ["command"],
            },
        },
    },
]
