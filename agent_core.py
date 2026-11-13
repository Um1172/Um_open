"""Autonomous Plan-Act-Reflect agent harness with multi-layer memory and chat sessions."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
from openai import AsyncOpenAI, BadRequestError

from tools import (
    ALLOWED_MEMORY_FILES,
    TOOL_SCHEMAS,
    WorkspaceTools,
    normalize_memory_filename,
)

DEFAULT_LM_STUDIO_URL = "http://localhost:1234/v1"
MAX_TOOL_HISTORY_MESSAGES = 14
MAX_CHAT_CONTEXT_MESSAGES = 16

DEFAULT_MEMORY_TEMPLATES: dict[str, str] = {
    "rule.md": (
        "# エージェント基本行動指針 (rule.md)\n\n"
        "- ユーザーに対して丁寧かつ簡潔に日本語で回答する。\n"
        "- 複雑なタスクは `update_plan` で計画を立ててから実行する。\n"
        "- ファイル操作・コマンド実行は `workspace/` 内に限定する。\n"
        "- ユーザーの恒久的な指示や制約が与えられたらこのファイルに追記する。\n"
    ),
    "profile.md": (
        "# ユーザープロフィール・環境情報 (profile.md)\n\n"
        "- OS: Windows\n"
        "- GPU: AMD Radeon RX 7800 XT (16GB)\n"
        "- ローカルLLM推論環境: LM Studio\n"
    ),
    "memory.md": (
        "# 長期記憶・重要事項 (memory.md)\n\n"
        "（会話から得られた重要な決定事項・事実・知識をここに蓄積します）\n"
    ),
    "plan.md": (
        "# 進行中タスク計画 (plan.md)\n\n"
        "（現在進行中の長期タスクはありません）\n"
    ),
    "work_log.md": (
        "# 作業ログ (work_log.md)\n\n"
        "（ツール実行履歴・作成ファイル・試行錯誤の記録をここに残します）\n"
    ),
}

SYSTEM_PROMPT = """あなたはローカルPC上で自律稼働する優秀なAIアシスタント兼自律エージェントです。
ユーザーとの自然な対話に加え、調査・コーディング・ファイル操作などの多段階タスクを自律的に遂行できます。

## 多層永続メモリシステム (`memory/`)
あなたの記憶は以下の5つのMarkdownファイルで構成されており、毎回の推論時にシステムプロンプトへ自動注入されます：
- `rule.md`: 基本行動指針・ユーザーからの恒久的な指示
- `profile.md`: ユーザー情報・PC環境・好み
- `memory.md`: 重要な決定事項・事実・知識の要約
- `plan.md`: 現在進行中の長期タスクの計画チェックリスト (`- [ ]` / `- [x]`)
- `work_log.md`: ツール実行履歴・作成ファイル・試行錯誤の作業記録

## 行動原則と出力フォーマット（厳守）
1. **シンプルな質問や雑談**には、不要なツールを使わず分かりやすく直接回答してください。
2. **調査・実装・複数手順を要するタスク（分割実行ルール）**:
   - 複雑なタスクやコーディング依頼を受けた場合、**1回のターンですべてを完結させることは禁止**です。
   - 必ず最初に `update_plan`（または `update_memory`）で `plan.md` に細分化されたステップ（`- [ ]` 形式のチェックリスト）を書き出してください。
   - **1回の推論ステップにつき1つの具体的な作業**（ファイル作成、実行テスト、検索、計画更新など）だけを着実に進め、完了した項目を `- [x]` に更新してください。
   - プログラムファイル（`.py` 等）を作成・編集した場合は、必ず `run_python` または `run_command` で実際に実行テストを行い、エラーがないことを検証してから終了してください。
3. **2層分離ルール（ツール呼び出し時のチャット出力制限）**:
   - 複雑な思考・設計・計算・ソースコードの検討は、すべて `<think>...</think>` タグ内、またはツール引数（`write_file` の `content` や `run_python` の `code` 等）の中に収めてください。
   - ツールを呼び出す際のチャット本文（途中経過メッセージ）は、**「今から何をするか」を伝える1〜2文（50〜100文字程度）の簡潔な日本語のみ**にしてください。
   - **ツール実行前の途中経過メッセージに、ソースコードのブロック（```）や長文の解説を書くことは禁止**です。
4. **記憶の更新**: ユーザーの好み・新しい事実・重要な作業結果が得られたら、`update_memory` ツールで適切なファイル（`profile.md`, `memory.md`, `work_log.md`, `rule.md`）を更新してください。
5. **行き詰まったら別アプローチ**: 同じ操作が失敗したり同じ結果になる場合は、別の方法を試してください。
6. **最終返信**: すべての作業が完了した最終ターンでのみ、結論・実行結果・必要な要約を分かりやすいMarkdown形式で出力してください。
"""


def _now_time() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _normalize_base_url(url: str) -> str:
    cleaned = (url or DEFAULT_LM_STUDIO_URL).strip().rstrip("/")
    if not cleaned.endswith("/v1"):
        cleaned = f"{cleaned}/v1"
    return cleaned


def _host_root_from_base_url(base_url: str) -> str:
    normalized = _normalize_base_url(base_url)
    return normalized[:-3] if normalized.endswith("/v1") else normalized


def _format_size_label(size_bytes: Optional[int]) -> str:
    if not size_bytes or size_bytes <= 0:
        return ""
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / (1024 ** 3):.2f} GB"
    return f"{size_bytes / (1024 ** 2):.0f} MB"


_MODEL_CACHE_TIME: float = 0.0
_MODEL_CACHE_DATA: dict[str, dict[str, Any]] = {}


def _find_lms_cli() -> Optional[str]:
    """Locate LM Studio CLI (lms.exe) if installed."""
    candidates = [
        Path.home() / ".lmstudio" / "bin" / "lms.exe",
        Path.home() / ".cache" / "lm-studio" / "bin" / "lms.exe",
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return shutil.which("lms")


def _discover_custom_model_roots() -> list[Path]:
    """Collect default and user-configured LM Studio model directories."""
    roots: list[Path] = [
        Path.home() / ".lmstudio" / "models",
        Path.home() / ".cache" / "lm-studio" / "models",
    ]
    config_files = [
        Path.home() / ".lmstudio" / ".internal" / "backend-preferences.json",
        Path.home() / ".lmstudio" / "settings.json",
    ]
    for cfg in config_files:
        if not cfg.exists():
            continue
        try:
            data = json.loads(cfg.read_text(encoding="utf-8", errors="replace"))
            for key in ("modelsDirectory", "downloadsFolder", "modelStorageDirectory"):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    p = Path(val.strip())
                    if p not in roots:
                        roots.append(p)
        except Exception:
            continue
    return [r for r in roots if r.exists()]


def _discover_local_models() -> dict[str, dict[str, Any]]:
    """Discover local LM Studio models and byte sizes via lms CLI and disk scan (cached 20s)."""
    global _MODEL_CACHE_TIME, _MODEL_CACHE_DATA
    now = time.monotonic()
    if _MODEL_CACHE_DATA and (now - _MODEL_CACHE_TIME) < 20.0:
        return _MODEL_CACHE_DATA

    discovered: dict[str, dict[str, Any]] = {}

    # Source A: lms.exe ls --json
    lms_path = _find_lms_cli()
    if lms_path:
        try:
            proc = subprocess.run(
                [lms_path, "ls", "--json"],
                timeout=4,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if proc.returncode == 0 and proc.stdout.strip():
                entries = json.loads(proc.stdout)
                if isinstance(entries, list):
                    for item in entries:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") == "embedding":
                            continue
                        key = (
                            item.get("modelKey")
                            or item.get("indexedModelIdentifier")
                            or item.get("path")
                            or item.get("id")
                            or ""
                        )
                        if not key:
                            continue
                        size_bytes = item.get("sizeBytes") or item.get("size") or None
                        quant = item.get("quantization") or ""
                        if isinstance(quant, dict):
                            quant = quant.get("name", "")
                        discovered[key] = {
                            "id": key,
                            "size_bytes": int(size_bytes) if size_bytes else None,
                            "quantization": str(quant),
                            "from_cli": True,
                        }
        except Exception:
            pass

    # Source B: Disk scan of LM Studio model folders
    model_exts = {".gguf", ".safetensors", ".bin", ".mlx"}
    for root in _discover_custom_model_roots():
        try:
            for dirpath, _, filenames in os.walk(root):
                model_files = [
                    fn for fn in filenames if os.path.splitext(fn)[1].lower() in model_exts
                ]
                if not model_files:
                    continue
                non_mmproj = [fn for fn in model_files if "mmproj" not in fn.lower()]
                target_files = non_mmproj if non_mmproj else model_files
                total = 0
                for fn in target_files:
                    try:
                        total += (Path(dirpath) / fn).stat().st_size
                    except OSError:
                        pass
                if total <= 0:
                    continue
                rel = Path(dirpath).relative_to(root).as_posix()
                for alias in {rel, Path(dirpath).name}:
                    existing = discovered.get(alias)
                    if not existing:
                        discovered[alias] = {
                            "id": alias,
                            "size_bytes": total,
                            "quantization": "",
                            "from_cli": False,
                        }
                    elif not existing.get("size_bytes"):
                        existing["size_bytes"] = total
                for fn in target_files:
                    if fn.lower().endswith(".gguf"):
                        stem = os.path.splitext(fn)[0]
                        if stem not in discovered:
                            discovered[stem] = {
                                "id": stem,
                                "size_bytes": total,
                                "quantization": "",
                                "from_cli": False,
                            }
        except OSError:
            continue

    _MODEL_CACHE_TIME = now
    _MODEL_CACHE_DATA = discovered
    return discovered


def _match_disk_size(model_id: str, discovered: dict[str, dict[str, Any]]) -> Optional[int]:
    if not model_id or not discovered:
        return None
    key = model_id.strip().lower()
    lower_map = {k.lower(): v.get("size_bytes") for k, v in discovered.items() if v.get("size_bytes")}
    if key in lower_map:
        return lower_map[key]
    tail = key.split("/")[-1]
    if tail in lower_map:
        return lower_map[tail]
    for k, sz in lower_map.items():
        if tail in k or k in key:
            return sz
    return None


async def check_lm_studio(base_url: str = DEFAULT_LM_STUDIO_URL) -> dict[str, Any]:
    """Check if LM Studio local server is reachable and list models with size metadata."""
    normalized = _normalize_base_url(base_url)
    host_root = _host_root_from_base_url(normalized)
    details_map: dict[str, dict[str, Any]] = {}

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            # 1. Standard OpenAI /v1/models
            resp = await client.get(f"{normalized}/models")
            resp.raise_for_status()
            v1_data = resp.json()
            for m in v1_data.get("data", []):
                mid = m.get("id", "")
                if mid:
                    details_map[mid] = {
                        "id": mid,
                        "size_bytes": m.get("size_bytes") or m.get("size") or None,
                        "state": "loaded",
                        "quantization": m.get("quantization") or "",
                    }

            # 2. LM Studio native /api/v1/models (0.4+)
            try:
                r_v1_native = await client.get(f"{host_root}/api/v1/models")
                if r_v1_native.status_code == 200:
                    payload = r_v1_native.json()
                    for m in payload.get("models", []):
                        mid = m.get("key") or m.get("id") or ""
                        if not mid:
                            continue
                        loaded_instances = m.get("loaded_instances") or []
                        state = "loaded" if loaded_instances else "not-loaded"
                        quant = m.get("quantization")
                        if isinstance(quant, dict):
                            quant = quant.get("name", "")
                        size_bytes = m.get("size_bytes") or m.get("size")
                        existing = details_map.get(mid, {"id": mid})
                        details_map[mid] = {
                            "id": mid,
                            "size_bytes": size_bytes or existing.get("size_bytes"),
                            "state": state,
                            "quantization": quant or existing.get("quantization", ""),
                        }
            except Exception:
                pass

            # 3. LM Studio native /api/v0/models (0.3.x)
            try:
                r_v0 = await client.get(f"{host_root}/api/v0/models")
                if r_v0.status_code == 200:
                    payload = r_v0.json()
                    for m in payload.get("data", []):
                        mid = m.get("id") or ""
                        if not mid:
                            continue
                        existing = details_map.get(mid, {"id": mid})
                        details_map[mid] = {
                            "id": mid,
                            "size_bytes": m.get("size_bytes") or m.get("size") or existing.get("size_bytes"),
                            "state": m.get("state") or existing.get("state", "unknown"),
                            "quantization": m.get("quantization") or existing.get("quantization", ""),
                        }
            except Exception:
                pass

        # 4. Merge discovered local models & fill missing sizes
        discovered = await asyncio.to_thread(_discover_local_models)
        existing_lower = {k.lower() for k in details_map}
        for key, info in discovered.items():
            if info.get("from_cli") and key.lower() not in existing_lower:
                details_map[key] = {
                    "id": key,
                    "size_bytes": info.get("size_bytes"),
                    "state": "not-loaded",
                    "quantization": info.get("quantization", ""),
                }
                existing_lower.add(key.lower())

        model_details: list[dict[str, Any]] = []
        for mid, info in details_map.items():
            sb = info.get("size_bytes")
            if not sb:
                sb = _match_disk_size(mid, discovered)
            model_details.append(
                {
                    "id": mid,
                    "size_bytes": sb,
                    "size_label": _format_size_label(sb),
                    "state": info.get("state", "unknown"),
                    "quantization": info.get("quantization", ""),
                }
            )

        model_details.sort(key=lambda d: (0 if d["state"] == "loaded" else 1, d["id"].lower()))
        return {
            "connected": True,
            "base_url": normalized,
            "models": [d["id"] for d in model_details],
            "model_details": model_details,
            "error": None,
        }
    except Exception as exc:
        return {
            "connected": False,
            "base_url": normalized,
            "models": [],
            "model_details": [],
            "error": str(exc),
        }


async def select_lm_studio_model(
    model_id: str,
    base_url: str = DEFAULT_LM_STUDIO_URL,
) -> dict[str, Any]:
    """Unload current models and load/activate the specified model in LM Studio."""
    global _MODEL_CACHE_TIME
    normalized = _normalize_base_url(base_url)
    host_root = _host_root_from_base_url(normalized)
    if not model_id or not model_id.strip():
        return {"ok": True, "model": "", "method": "auto", "message": "自動選択に設定しました。"}

    target = model_id.strip()

    # Method 1: lms.exe CLI (reliably updates active model in LM Studio GUI)
    lms_path = _find_lms_cli()
    if lms_path:
        def _switch_via_cli() -> tuple[bool, str]:
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.run(
                [lms_path, "unload", "--all"],
                timeout=25,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
            )
            proc = subprocess.run(
                [lms_path, "load", target, "-y"],
                timeout=90,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
            )
            if proc.returncode == 0:
                return True, proc.stdout.strip()
            return False, (proc.stderr or proc.stdout or "").strip()

        try:
            ok, detail = await asyncio.to_thread(_switch_via_cli)
            if ok:
                _MODEL_CACHE_TIME = 0.0
                return {
                    "ok": True,
                    "model": target,
                    "method": "lms_cli",
                    "message": f"LM Studio でモデル '{target}' に切り替えました。",
                }
        except Exception:
            pass

    # Method 2: LM Studio Native REST API (/api/v1/models/unload + load, /api/v0/models/load)
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                r_list = await client.get(f"{host_root}/api/v1/models")
                if r_list.status_code == 200:
                    for m in r_list.json().get("models", []):
                        for inst in m.get("loaded_instances") or []:
                            inst_id = inst.get("id") if isinstance(inst, dict) else inst
                            if inst_id:
                                await client.post(
                                    f"{host_root}/api/v1/models/unload",
                                    json={"instance_id": inst_id},
                                )
            except Exception:
                pass

            try:
                r1 = await client.post(f"{host_root}/api/v1/models/load", json={"model": target})
                if r1.status_code in (200, 201):
                    _MODEL_CACHE_TIME = 0.0
                    return {
                        "ok": True,
                        "model": target,
                        "method": "api/v1/models/load",
                        "message": f"LM Studio でモデル '{target}' をロードしました。",
                    }
            except Exception:
                pass

            try:
                r0 = await client.post(f"{host_root}/api/v0/models/load", json={"model": target})
                if r0.status_code in (200, 201):
                    _MODEL_CACHE_TIME = 0.0
                    return {
                        "ok": True,
                        "model": target,
                        "method": "api/v0/models/load",
                        "message": f"LM Studio でモデル '{target}' をロードしました。",
                    }
            except Exception:
                pass

            # Method 3: Fallback JIT load via minimal 1-token chat completion request
            r_jit = await client.post(
                f"{normalized}/chat/completions",
                json={
                    "model": target,
                    "messages": [{"role": "user", "content": "ok"}],
                    "max_tokens": 1,
                    "temperature": 0,
                },
            )
            r_jit.raise_for_status()
            _MODEL_CACHE_TIME = 0.0
            return {
                "ok": True,
                "model": target,
                "method": "jit/chat/completions",
                "message": f"LM Studio へモデル '{target}' の使用を送信しました。",
            }
    except Exception as exc:
        return {
            "ok": False,
            "model": target,
            "method": "error",
            "message": f"モデル '{target}' の切り替えに失敗しました: {exc}",
        }


class MemoryManager:
    """Manages the 5-layer persistent Markdown memory files under memory/."""

    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir.resolve()
        self.ensure_defaults()

    def ensure_defaults(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        for filename, template in DEFAULT_MEMORY_TEMPLATES.items():
            path = self.memory_dir / filename
            if not path.exists():
                path.write_text(template, encoding="utf-8")

    def read_file(self, filename: str) -> str:
        normalized = normalize_memory_filename(filename)
        path = self.memory_dir / normalized
        if not path.exists():
            return DEFAULT_MEMORY_TEMPLATES.get(normalized, "")
        return path.read_text(encoding="utf-8", errors="replace")

    def write_file(self, filename: str, content: str) -> str:
        normalized = normalize_memory_filename(filename)
        path = self.memory_dir / normalized
        path.write_text(content, encoding="utf-8")
        return content

    def read_all(self) -> dict[str, str]:
        self.ensure_defaults()
        return {name: self.read_file(name) for name in ALLOWED_MEMORY_FILES}

    def build_context_block(self) -> str:
        all_mem = self.read_all()
        sections = ["\n\n## 現在の永続メモリ (memory/)"]
        for name in ALLOWED_MEMORY_FILES:
            content = all_mem[name].strip()
            if len(content) > 1400:
                content = content[:1400] + "\n... (省略)"
            sections.append(f"### memory/{name}\n```markdown\n{content}\n```")
        return "\n\n".join(sections)


class SessionManager:
    """Persists chat sessions and message histories under sessions/."""

    def __init__(self, sessions_dir: Path) -> None:
        self.sessions_dir = sessions_dir.resolve()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", session_id)
        if not safe_id:
            raise ValueError("無効なセッションIDです。")
        return self.sessions_dir / f"{safe_id}.json"

    def create_session(self, title: str = "新しいチャット") -> dict[str, Any]:
        session_id = f"sess_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        now = _now_iso()
        session = {
            "id": session_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        self.save_session(session)
        return session

    def get_session(self, session_id: str) -> dict[str, Any]:
        path = self._session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"セッションが見つかりません: {session_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def save_session(self, session: dict[str, Any]) -> None:
        session["updated_at"] = _now_iso()
        path = self._session_path(session["id"])
        path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")

    def delete_session(self, session_id: str) -> bool:
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_sessions(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in self.sessions_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                items.append(
                    {
                        "id": data["id"],
                        "title": data.get("title", "無題のチャット"),
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),
                        "message_count": len(data.get("messages", [])),
                    }
                )
            except Exception:
                continue
        items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return items


class EventBroker:
    """Broadcasts SSE events to all connected browser clients."""

    def __init__(self, max_history: int = 200) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._history: list[dict[str, Any]] = []
        self._max_history = max_history

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        for event in self._history:
            queue.put_nowait(event)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    def clear_history(self) -> None:
        self._history.clear()

    async def publish(self, event_type: str, data: dict[str, Any]) -> None:
        event = {
            "type": event_type,
            "timestamp": _now_time(),
            "data": data,
        }
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]
        for queue in list(self._subscribers):
            await queue.put(event)


def _extract_think_and_content(text: str) -> tuple[str, str]:
    """Separate <think>...</think> (and unclosed <think>) reasoning blocks from visible content."""
    if not text:
        return "", ""
    thinks: list[str] = []

    def _repl(match: re.Match[str]) -> str:
        thinks.append(match.group(1).strip())
        return ""

    visible = re.sub(r"<think>(.*?)</think>", _repl, text, flags=re.DOTALL).strip()

    # Handle unclosed <think> tag if model truncated before </think>
    if "<think>" in visible:
        before, _, after = visible.partition("<think>")
        if after.strip():
            thinks.append(after.strip())
        visible = before.strip()

    return "\n\n".join(t for t in thinks if t), visible


VALID_TOOL_NAMES: set[str] = {schema["function"]["name"] for schema in TOOL_SCHEMAS}


def _normalize_tool_call_payload(payload: Any) -> list[dict[str, Any]]:
    """Normalize parsed JSON tool call payload(s) into standard call dicts."""
    if isinstance(payload, list):
        calls: list[dict[str, Any]] = []
        for item in payload:
            calls.extend(_normalize_tool_call_payload(item))
        return calls

    if not isinstance(payload, dict):
        return []

    if "tool_calls" in payload and isinstance(payload["tool_calls"], list):
        return _normalize_tool_call_payload(payload["tool_calls"])

    if "function" in payload and isinstance(payload["function"], dict):
        fn = payload["function"]
        name = fn.get("name")
        args = fn.get("arguments", fn.get("parameters", {}))
    else:
        name = payload.get("name") or payload.get("tool") or payload.get("tool_name")
        if "arguments" in payload:
            args = payload.get("arguments")
        elif "parameters" in payload:
            args = payload.get("parameters")
        else:
            args = None

    if args is None and isinstance(name, str) and name in VALID_TOOL_NAMES:
        args = {
            k: v
            for k, v in payload.items()
            if k not in ("name", "tool", "tool_name", "type", "id")
        }

    if isinstance(args, str):
        args_str = args.strip()
        if not args_str:
            args = {}
        else:
            try:
                args = json.loads(args_str)
            except Exception:
                return []

    if isinstance(name, str) and name in VALID_TOOL_NAMES and isinstance(args, dict):
        return [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "name": name,
                "arguments": args,
            }
        ]
    return []


def _extract_json_objects_from_text(segment: str) -> list[Any]:
    """Scan text for valid JSON objects or arrays using JSONDecoder.raw_decode."""
    objects: list[Any] = []
    decoder = json.JSONDecoder()
    idx = 0
    length = len(segment)
    while idx < length:
        ch = segment[idx]
        if ch in "{[":
            try:
                obj, end_idx = decoder.raw_decode(segment, idx)
                objects.append(obj)
                idx = end_idx
                continue
            except Exception:
                pass
        idx += 1
    return objects


def _parse_fallback_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse tool calls from text (<tool_call>, markdown ```json```, bare JSON, or fn(...))."""
    if not text or not text.strip():
        return []

    _, cleaned = _extract_think_and_content(text)
    if not cleaned:
        cleaned = text

    segments: list[tuple[str, bool]] = []
    for match in re.finditer(r"<tool_call>\s*([\s\S]*?)\s*(?:</tool_call>|$)", cleaned):
        if match.group(1).strip():
            segments.append((match.group(1), True))
    for match in re.finditer(r"```(?:json|tool_call|tool)?\s*([\s\S]*?)```", cleaned, flags=re.IGNORECASE):
        if match.group(1).strip():
            segments.append((match.group(1), True))
    segments.append((cleaned, False))

    calls: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def _add_call(call: dict[str, Any]) -> None:
        sig = (call["name"], json.dumps(call["arguments"], sort_keys=True, ensure_ascii=False))
        if sig not in seen:
            seen.add(sig)
            calls.append(call)

    for seg, is_explicit_block in segments:
        objs = _extract_json_objects_from_text(seg)
        if not objs and is_explicit_block:
            brace_match = re.search(r"\{[\s\S]*\}", seg)
            if brace_match:
                try:
                    objs = [json.loads(brace_match.group(0))]
                except Exception:
                    objs = []
        for obj in objs:
            for call in _normalize_tool_call_payload(obj):
                _add_call(call)

    if not calls and VALID_TOOL_NAMES:
        fn_pattern = (
            r"\b("
            + "|".join(re.escape(n) for n in sorted(VALID_TOOL_NAMES))
            + r")\s*\(\s*(\{[\s\S]*)"
        )
        decoder = json.JSONDecoder()
        for match in re.finditer(fn_pattern, cleaned):
            fn_name = match.group(1)
            arg_tail = match.group(2)
            try:
                args_obj, _ = decoder.raw_decode(arg_tail, 0)
            except Exception:
                continue
            if isinstance(args_obj, dict):
                _add_call(
                    {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "name": fn_name,
                        "arguments": args_obj,
                    }
                )

    return calls


@dataclass
class AgentState:
    status: str = "idle"  # idle | running | stopping | stopped | completed | error
    session_id: Optional[str] = None
    current_step: int = 0
    max_steps: int = 30
    model: str = ""
    base_url: str = DEFAULT_LM_STUDIO_URL
    phase_state: str = "idle"  # idle | thinking | working
    phase_label: str = ""
    phase_started_at: float = 0.0
    error_message: Optional[str] = None


class AgentHarness:
    """Manages chat turns, autonomous tool loops, memory consolidation, and SSE events."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir.resolve()
        self.workspace_root = self.base_dir / "workspace"
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.memory = MemoryManager(self.base_dir / "memory")
        self.sessions = SessionManager(self.base_dir / "sessions")
        self.broker = EventBroker()
        self.state = AgentState()
        self._stop_event = asyncio.Event()
        self._run_task: Optional[asyncio.Task[None]] = None
        self.live_thinking_log: list[dict[str, Any]] = []
        self.last_completed_message: Optional[dict[str, Any]] = None

    def is_running(self) -> bool:
        return self._run_task is not None and not self._run_task.done()

    def get_session_live_status(self, session_id: Optional[str] = None) -> dict[str, Any]:
        global_running = self.is_running() or self.state.status in ("running", "stopping")
        matches_session = (session_id is None) or (self.state.session_id == session_id)
        is_session_running = bool(global_running and matches_session)
        elapsed = (
            round(max(0.0, time.monotonic() - self.state.phase_started_at), 1)
            if (global_running and self.state.phase_started_at > 0 and self.state.phase_state in ("thinking", "working"))
            else 0.0
        )
        return {
            "is_running": is_session_running,
            "global_running": bool(global_running),
            "running_session_id": self.state.session_id if global_running else None,
            "status": self.state.status if matches_session else "idle",
            "session_id": self.state.session_id,
            "step": self.state.current_step if matches_session else 0,
            "current_step": self.state.current_step if matches_session else 0,
            "max_steps": self.state.max_steps,
            "state": self.state.phase_state if matches_session else "idle",
            "phase_state": self.state.phase_state if matches_session else "idle",
            "label": self.state.phase_label if matches_session else "",
            "phase_label": self.state.phase_label if matches_session else "",
            "role": "assistant",
            "elapsed_seconds": elapsed if matches_session else 0.0,
            "model": self.state.model,
            "error_message": self.state.error_message if matches_session else None,
            "thinking_log": list(self.live_thinking_log) if matches_session else [],
            "last_completed_message": self.last_completed_message if matches_session else None,
        }

    def get_live_state(self) -> dict[str, Any]:
        return self.get_session_live_status(None)

    def _persist_progress(
        self,
        session: dict[str, Any],
        assistant_msg_id: str,
        thinking_log: list[dict[str, Any]],
        latest_content: Optional[str] = None,
        in_progress: bool = True,
    ) -> dict[str, Any]:
        target_entry: Optional[dict[str, Any]] = None
        for msg in reversed(session.get("messages", [])):
            if msg.get("id") == assistant_msg_id or (
                msg.get("role") == "assistant" and msg.get("in_progress")
            ):
                target_entry = msg
                break
        if target_entry is None:
            target_entry = {
                "id": assistant_msg_id,
                "role": "assistant",
                "content": latest_content if latest_content is not None else "（推論・作業を実行中...）",
                "thinking_log": list(thinking_log),
                "timestamp": _now_time(),
            }
            session.setdefault("messages", []).append(target_entry)
        else:
            target_entry["thinking_log"] = list(thinking_log)
            if latest_content is not None:
                target_entry["content"] = latest_content
            target_entry["timestamp"] = _now_time()

        if in_progress:
            target_entry["in_progress"] = True
        else:
            target_entry.pop("in_progress", None)

        self.sessions.save_session(session)
        return target_entry

    async def start_chat_turn(
        self,
        session_id: Optional[str],
        user_message: str,
        max_steps: int = 30,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> dict[str, Any]:
        if self.is_running():
            raise RuntimeError("現在別の推論が実行中です。先に停止してください。")

        normalized_url = _normalize_base_url(base_url or DEFAULT_LM_STUDIO_URL)
        lm_status = await check_lm_studio(normalized_url)
        if not lm_status["connected"]:
            raise ConnectionError(
                f"LM Studio に接続できません ({normalized_url})。"
                "LM Studio でローカルサーバーを起動してから再実行してください。"
            )

        selected_model = (model or "").strip()
        if not selected_model:
            selected_model = lm_status["models"][0] if lm_status["models"] else "local-model"

        if session_id:
            try:
                session = self.sessions.get_session(session_id)
            except FileNotFoundError:
                session = self.sessions.create_session()
        else:
            session = self.sessions.create_session()

        # Auto-title new session from first user message
        if not session["messages"] and (
            session.get("title") in ("新しいチャット", "", None)
        ):
            short_title = user_message.strip().splitlines()[0][:36]
            session["title"] = short_title + ("..." if len(user_message.strip()) > 36 else "")

        user_entry = {
            "id": f"msg_{uuid.uuid4().hex[:8]}",
            "role": "user",
            "content": user_message,
            "timestamp": _now_time(),
        }
        session["messages"].append(user_entry)
        assistant_msg_id = f"msg_{uuid.uuid4().hex[:8]}"
        self._persist_progress(
            session,
            assistant_msg_id,
            [],
            latest_content="（推論・作業を実行中...）",
            in_progress=True,
        )

        self._stop_event.clear()
        self.broker.clear_history()
        self.live_thinking_log = []
        self.last_completed_message = None
        self.state = AgentState(
            status="running",
            session_id=session["id"],
            current_step=0,
            max_steps=max(1, min(int(max_steps), 200)),
            model=selected_model,
            base_url=normalized_url,
            phase_state="thinking",
            phase_label="タスクを分析・推論中",
            phase_started_at=time.monotonic(),
        )

        await self.broker.publish("status", self._status_payload())
        await self.broker.publish("session_update", {"session": self._session_summary(session)})

        self._run_task = asyncio.create_task(
            self._run_turn(session["id"], user_message, assistant_msg_id)
        )
        return {
            "agent": self._status_payload(),
            "session_id": session["id"],
            "user_message": user_entry,
        }

    async def stop_task(self) -> dict[str, Any]:
        if not self.is_running():
            return self._status_payload()
        self._stop_event.set()
        self.state.status = "stopping"
        await self.broker.publish("status", self._status_payload())
        await self.broker.publish(
            "warning",
            {"message": "停止リクエストを受信しました。現在の処理を中断しています..."},
        )
        return self._status_payload()

    def _status_payload(self) -> dict[str, Any]:
        elapsed = (
            round(max(0.0, time.monotonic() - self.state.phase_started_at), 1)
            if (self.is_running() and self.state.phase_started_at > 0 and self.state.phase_state in ("thinking", "working"))
            else 0.0
        )
        return {
            "is_running": self.is_running(),
            "status": self.state.status,
            "session_id": self.state.session_id,
            "current_step": self.state.current_step,
            "max_steps": self.state.max_steps,
            "state": self.state.phase_state,
            "label": self.state.phase_label,
            "elapsed_seconds": elapsed,
            "model": self.state.model,
            "base_url": self.state.base_url,
            "error_message": self.state.error_message,
        }

    def _session_summary(self, session: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": session["id"],
            "title": session.get("title", "無題のチャット"),
            "created_at": session.get("created_at", ""),
            "updated_at": session.get("updated_at", ""),
            "message_count": len(session.get("messages", [])),
        }

    def list_workspace_files(self) -> list[dict[str, Any]]:
        files: list[dict[str, Any]] = []
        if not self.workspace_root.exists():
            return files
        for path in sorted(self.workspace_root.rglob("*"), key=lambda p: p.as_posix().lower()):
            if path.is_file():
                rel = path.relative_to(self.workspace_root).as_posix()
                files.append({"path": rel, "size": path.stat().st_size})
        return files

    async def _on_memory_update(self, filename: str, content: str) -> None:
        await self.broker.publish("memory_update", {"filename": filename, "content": content})

    async def _on_files_update(self) -> None:
        await self.broker.publish("files_update", {"files": self.list_workspace_files()})

    async def select_model(self, model_id: str, base_url: Optional[str] = None) -> dict[str, Any]:
        normalized = _normalize_base_url(base_url or self.state.base_url or DEFAULT_LM_STUDIO_URL)
        self.state.base_url = normalized
        result = await select_lm_studio_model(model_id, normalized)
        if result.get("ok"):
            self.state.model = result.get("model", "")
        return result

    @staticmethod
    def _merge_consecutive_roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge consecutive plain user/assistant messages so strict Jinja templates never fail."""
        merged: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if (
                merged
                and role in ("user", "assistant")
                and merged[-1].get("role") == role
                and not merged[-1].get("tool_calls")
                and not msg.get("tool_calls")
            ):
                prev_text = merged[-1].get("content") or ""
                curr_text = msg.get("content") or ""
                merged[-1] = {
                    **merged[-1],
                    "content": f"{prev_text}\n\n{curr_text}".strip(),
                }
            else:
                merged.append(msg)
        return merged

    @staticmethod
    def _flatten_tool_history(tool_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool_calls/tool messages into universal assistant/user text turns."""
        flat: list[dict[str, Any]] = []
        for msg in tool_history:
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                calls_desc = []
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    calls_desc.append(f"{fn.get('name')}({fn.get('arguments', '{}')})")
                text = msg.get("content") or ""
                combined = f"{text}\n[ツール呼び出し: {', '.join(calls_desc)}]".strip()
                flat.append({"role": "assistant", "content": combined})
            elif role == "tool":
                name = msg.get("name") or "tool"
                content = str(msg.get("content") or "")
                if len(content) > 2500:
                    content = content[:2500] + "\n... (省略)"
                flat.append(
                    {
                        "role": "user",
                        "content": f"【ツール実行結果 ({name})】\n{content}\n\n上記の結果を確認し、次のツールを呼び出すか、完了していれば最終回答を作成してください。",
                    }
                )
            else:
                flat.append({"role": role, "content": msg.get("content") or ""})
        return flat

    def _build_messages(
        self,
        session_messages: list[dict[str, Any]],
        tool_history: list[dict[str, Any]],
        flatten_tools: bool = False,
    ) -> list[dict[str, Any]]:
        """Build context-bounded messages injecting all 5 memory/*.md files."""
        system_content = SYSTEM_PROMPT + self.memory.build_context_block()
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]

        recent_chat = session_messages[-10:]
        for m in recent_chat:
            if m.get("in_progress"):
                continue
            role = m.get("role")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": m.get("content") or ""})

        trimmed_tools = tool_history[-MAX_TOOL_HISTORY_MESSAGES:]
        while trimmed_tools and trimmed_tools[0].get("role") == "tool":
            trimmed_tools = trimmed_tools[1:]

        if flatten_tools:
            trimmed_tools = self._flatten_tool_history(trimmed_tools)

        messages.extend(trimmed_tools)
        return self._merge_consecutive_roles(messages)

    async def _call_llm_with_fallback(
        self,
        client: AsyncOpenAI,
        session_messages: list[dict[str, Any]],
        tool_history: list[dict[str, Any]],
        use_flattened: bool = False,
    ) -> tuple[Any, bool]:
        """Call LM Studio chat completions, falling back to flattened tool history if 400 occurs."""
        messages = self._build_messages(session_messages, tool_history, flatten_tools=use_flattened)
        try:
            resp = await client.chat.completions.create(
                model=self.state.model,
                messages=messages,  # type: ignore[arg-type]
                tools=TOOL_SCHEMAS,  # type: ignore[arg-type]
                tool_choice="auto",
                temperature=0.3,
            )
            return resp, use_flattened
        except BadRequestError:
            if not use_flattened and tool_history:
                flat_messages = self._build_messages(
                    session_messages, tool_history, flatten_tools=True
                )
                try:
                    resp = await client.chat.completions.create(
                        model=self.state.model,
                        messages=flat_messages,  # type: ignore[arg-type]
                        tools=TOOL_SCHEMAS,  # type: ignore[arg-type]
                        tool_choice="auto",
                        temperature=0.3,
                    )
                    return resp, True
                except BadRequestError:
                    resp = await client.chat.completions.create(
                        model=self.state.model,
                        messages=flat_messages,  # type: ignore[arg-type]
                        temperature=0.3,
                    )
                    return resp, True
            raise

    async def _auto_consolidate_memory(
        self,
        client: AsyncOpenAI,
        tools: WorkspaceTools,
        user_message: str,
        final_answer: str,
        thinking_log: list[dict[str, Any]],
    ) -> None:
        """Background reflection step to keep profile.md, memory.md, and work_log.md fresh."""
        tool_used = any(item.get("type") in ("work", "tool_call") for item in thinking_log)
        mem_updated = any(
            item.get("type") in ("work", "tool_call") and item.get("tool") in ("update_memory", "update_plan")
            for item in thinking_log
        )
        if mem_updated:
            return
        if not tool_used and len(user_message) < 40 and len(final_answer) < 120:
            return

        prompt = (
            "以下の直近のやり取りから、永続記憶 (`profile.md`, `memory.md`, `work_log.md`) に"
            "新しく追記・更新すべき情報があれば `update_memory` ツールを使って記録してください。\n"
            "特になければツールを呼ばずに「なし」とだけ返答してください。\n\n"
            f"【ユーザー発言】\n{user_message[:1000]}\n\n"
            f"【AI回答】\n{final_answer[:1200]}"
        )
        try:
            resp = await client.chat.completions.create(
                model=self.state.model,
                messages=[
                    {"role": "system", "content": "あなたは記憶整理担当です。\n" + self.memory.build_context_block()},
                    {"role": "user", "content": prompt},
                ],
                tools=[TOOL_SCHEMAS[0]],  # update_memory only
                tool_choice="auto",
                temperature=0.2,
            )
            msg = resp.choices[0].message
            calls = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        continue
                    calls.append({"name": tc.function.name, "arguments": args})
            else:
                calls = _parse_fallback_tool_calls(msg.content or "")

            for call in calls:
                if call["name"] == "update_memory":
                    await tools.execute("update_memory", call["arguments"])
        except Exception:
            # Memory consolidation is best-effort; never fail the main turn
            pass

    @staticmethod
    def _describe_tool_action(tool_name: str, tool_args: dict[str, Any], in_progress: bool = True) -> str:
        suffix = "中" if in_progress else ""
        if tool_name == "run_python":
            return f"Pythonスクリプトを実行{suffix}" if in_progress else "Python実行"
        if tool_name == "run_command":
            cmd = str(tool_args.get("command", "")).strip().split()[0:2]
            cmd_str = " ".join(cmd)
            return f"コマンド実行{suffix}: {cmd_str}" if cmd_str else f"コマンド実行{suffix}"
        if tool_name == "write_file":
            path = str(tool_args.get("path", "")).strip()
            return f"ファイル保存{suffix}: {path}" if path else f"ファイル保存{suffix}"
        if tool_name == "read_file":
            path = str(tool_args.get("path", "")).strip()
            return f"ファイル読込{suffix}: {path}" if path else f"ファイル読込{suffix}"
        if tool_name == "list_files":
            return f"ファイル一覧取得{suffix}"
        if tool_name == "web_search":
            q = str(tool_args.get("query", "")).strip()[:24]
            return f"Web検索{suffix}: {q}" if q else f"Web検索{suffix}"
        if tool_name == "fetch_url":
            return f"Webページ取得{suffix}"
        if tool_name == "update_plan":
            return f"計画更新{suffix}: plan.md"
        if tool_name == "update_memory":
            target = str(tool_args.get("target", "memory")).strip()
            return f"記憶更新{suffix}: {target}"
        return f"{tool_name} 実行{suffix}"

    @staticmethod
    def _sanitize_intermediate_narration(text: str) -> str:
        """Keep intermediate progress messages concise (1-2 sentences) without code blocks."""
        cleaned = re.sub(r"<tool_call>[\s\S]*?(?:</tool_call>|$)", "", text)
        cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'\{[^{}]*"name"\s*:\s*"[^"]+"[\s\S]*\}', "", cleaned)
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        joined = " ".join(lines).strip()
        if len(joined) > 140:
            joined = joined[:137] + "..."
        return joined

    async def _run_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_msg_id: Optional[str] = None,
    ) -> None:
        assistant_msg_id = assistant_msg_id or f"msg_{uuid.uuid4().hex[:8]}"
        tools = WorkspaceTools(
            workspace_root=self.workspace_root,
            memory_dir=self.memory.memory_dir,
            on_memory_update=self._on_memory_update,
            on_files_update=self._on_files_update,
        )
        client = AsyncOpenAI(base_url=self.state.base_url, api_key="lm-studio", timeout=180.0)

        session = self.sessions.get_session(session_id)
        tool_history: list[dict[str, Any]] = []
        thinking_log: list[dict[str, Any]] = []
        last_call_signature: Optional[tuple[str, str]] = None
        consecutive_repeats = 0
        needs_execution_test = False
        consecutive_guard_rejections = 0
        empty_response_retries = 0
        use_flattened_tools = False

        try:
            for step in range(1, self.state.max_steps + 1):
                if self._stop_event.is_set():
                    await self._finalize_stopped(session, thinking_log, assistant_msg_id)
                    return

                self.state.current_step = step
                await self.broker.publish(
                    "step",
                    {"current_step": step, "max_steps": self.state.max_steps, "session_id": session_id},
                )

                # Phase: thinking
                think_label = "タスクを分析・推論中" if step == 1 else "次のアクションを推論中"
                self.state.phase_state = "thinking"
                self.state.phase_label = think_label
                self.state.phase_started_at = time.monotonic()
                await self.broker.publish(
                    "status",
                    {
                        **self._status_payload(),
                        "state": "thinking",
                        "step": step,
                        "label": think_label,
                    },
                )
                think_start = self.state.phase_started_at

                response, use_flattened_tools = await self._call_llm_with_fallback(
                    client,
                    session["messages"],
                    tool_history,
                    use_flattened=use_flattened_tools,
                )

                think_duration = round(time.monotonic() - think_start, 1)

                if self._stop_event.is_set():
                    await self._finalize_stopped(session, thinking_log, assistant_msg_id)
                    return

                msg = response.choices[0].message
                raw_content = msg.content or ""
                reasoning_extra = (
                    getattr(msg, "reasoning_content", None)
                    or getattr(msg, "reasoning", None)
                    or ""
                )
                think_text, visible_text = _extract_think_and_content(raw_content)
                combined_thinking = "\n\n".join(p for p in [reasoning_extra, think_text] if p).strip()

                parsed_calls: list[dict[str, Any]] = []
                used_native_tools = False
                if msg.tool_calls:
                    used_native_tools = True
                    for tc in msg.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments or "{}")
                        except json.JSONDecodeError:
                            args = {"_raw": tc.function.arguments}
                        parsed_calls.append(
                            {
                                "id": tc.id or f"call_{uuid.uuid4().hex[:8]}",
                                "name": tc.function.name,
                                "arguments": args,
                            }
                        )
                else:
                    parsed_calls = _parse_fallback_tool_calls(visible_text)
                    if not parsed_calls and not visible_text:
                        parsed_calls = _parse_fallback_tool_calls(raw_content)

                if parsed_calls:
                    detected_names = ", ".join(c["name"] for c in parsed_calls)
                    print(f"[Step {step}] 推論開始 -> ツール検出: {detected_names}", flush=True)

                thought_content = combined_thinking
                if not thought_content:
                    if parsed_calls:
                        thought_content = visible_text or f"ツール呼び出しを決定: {', '.join(c['name'] for c in parsed_calls)}"
                    else:
                        thought_content = "最終回答を生成"
                thought_item = {
                    "type": "thinking",
                    "step": step,
                    "duration": think_duration,
                    "content": thought_content,
                }
                thinking_log.append(thought_item)
                self.live_thinking_log.append(thought_item)
                self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
                await self.broker.publish("thinking", {**thought_item, "session_id": session_id})

                if not parsed_calls:
                    if not visible_text:
                        empty_response_retries += 1
                        if empty_response_retries <= 3:
                            retry_msg = (
                                "【システム通知】直前の応答は思考（<think> / reasoning）のみで、"
                                "ツール呼び出しもユーザー向け本文も空でした。"
                                "思考した内容に基づき、直ちに次のツールを呼び出すか、作業が完了していれば最終回答を出力してください。"
                            )
                            warn_item = {
                                "type": "warning",
                                "step": step,
                                "message": f"思考のみで出力が空だったため、次のアクションを再促しています ({empty_response_retries}/3)",
                            }
                            thinking_log.append(warn_item)
                            self.live_thinking_log.append(warn_item)
                            self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
                            await self.broker.publish("warning", {**warn_item, "session_id": session_id})
                            thought_preview = (combined_thinking[:300] + "...") if len(combined_thinking) > 300 else combined_thinking
                            tool_history.append(
                                {
                                    "role": "assistant",
                                    "content": f"（内部思考: {thought_preview or '検討中'}）",
                                }
                            )
                            tool_history.append({"role": "user", "content": retry_msg})
                            print(
                                f"[Step {step}] 推論開始 -> ツールなし（差し戻しにより Step {step + 1} へ継続）",
                                flush=True,
                            )
                            continue
                        final_text = (
                            "モデルが思考プロセスのみを出力し、最終回答またはツール呼び出しを返しませんでした。"
                            "（詳細は上の「思考プロセス・作業ログ」をご確認ください）"
                        )
                    else:
                        empty_response_retries = 0
                        final_text = visible_text

                    # Guard 1: Incomplete checklist in plan.md
                    plan_text = self.memory.read_file("plan.md")
                    has_incomplete_plan = bool(
                        re.search(r"^\s*[-*]\s*\[\s\]", plan_text, flags=re.MULTILINE)
                    )

                    guard_warning: Optional[str] = None
                    if has_incomplete_plan:
                        guard_warning = (
                            "【システム警告】plan.md にまだ未完了の項目（- [ ]）が残っています。"
                            "手抜きや省略をせず、残りのステップを実行し、plan.md をすべて [x] に更新してから最終回答を行ってください。"
                        )
                    elif needs_execution_test:
                        guard_warning = (
                            "【システム警告】作成したプログラムの動作確認がまだ行われていません。"
                            "必ず run_python 等で実際に実行し、エラーが出ないことと出力結果が正しいことを検証してから終了してください。"
                        )

                    if guard_warning:
                        consecutive_guard_rejections += 1
                        if consecutive_guard_rejections <= 3:
                            clean_step_text = self._sanitize_intermediate_narration(final_text)
                            if clean_step_text:
                                step_out_item = {
                                    "type": "step_output",
                                    "step": step,
                                    "content": clean_step_text,
                                }
                                thinking_log.append(step_out_item)
                                self.live_thinking_log.append(step_out_item)
                                self._persist_progress(
                                    session,
                                    assistant_msg_id,
                                    thinking_log,
                                    latest_content=clean_step_text,
                                    in_progress=True,
                                )
                                await self.broker.publish("narration", {**step_out_item, "session_id": session_id})
                            warn_item = {
                                "type": "warning",
                                "step": step,
                                "message": f"{guard_warning} (差し戻し {consecutive_guard_rejections}/3)",
                            }
                            thinking_log.append(warn_item)
                            self.live_thinking_log.append(warn_item)
                            self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
                            await self.broker.publish("warning", {**warn_item, "session_id": session_id})
                            tool_history.append({"role": "assistant", "content": final_text})
                            tool_history.append({"role": "user", "content": guard_warning})
                            print(
                                f"[Step {step}] 推論開始 -> ツールなし（差し戻しにより Step {step + 1} へ継続）",
                                flush=True,
                            )
                            continue
                        else:
                            safety_note = (
                                "\n\n---\n⚠️ **安全停止通知**: 未完了項目または未テスト状態の差し戻しが連続3回発生したため、"
                                "無限ループ防止のためここで終了しました。現在の進捗（`plan.md`）と残課題をご確認ください。"
                            )
                            final_text = final_text + safety_note

                    print(
                        f"[Step {step}] 推論開始 -> ツールなし（最終回答としてループ終了）",
                        flush=True,
                    )

                    assistant_entry = self._persist_progress(
                        session,
                        assistant_msg_id,
                        thinking_log,
                        latest_content=final_text,
                        in_progress=False,
                    )

                    self.state.phase_state = "idle"
                    self.state.phase_label = ""
                    self.state.phase_started_at = 0.0
                    self.last_completed_message = assistant_entry

                    await self.broker.publish(
                        "final_answer",
                        {
                            "step": step,
                            "session_id": session_id,
                            "message": assistant_entry,
                        },
                    )

                    await self._auto_consolidate_memory(
                        client, tools, user_message, final_text, thinking_log
                    )

                    self.state.status = "completed"
                    await self.broker.publish("status", self._status_payload())
                    return

                consecutive_guard_rejections = 0
                empty_response_retries = 0

                if visible_text:
                    narration = self._sanitize_intermediate_narration(visible_text)
                    if narration:
                        step_out_item = {
                            "type": "step_output",
                            "step": step,
                            "content": narration,
                        }
                        thinking_log.append(step_out_item)
                        self.live_thinking_log.append(step_out_item)
                        self._persist_progress(
                            session,
                            assistant_msg_id,
                            thinking_log,
                            latest_content=narration,
                            in_progress=True,
                        )
                        await self.broker.publish("narration", {**step_out_item, "session_id": session_id})

                if not used_native_tools:
                    use_flattened_tools = True

                if used_native_tools and not use_flattened_tools:
                    tool_history.append(
                        {
                            "role": "assistant",
                            "content": visible_text or "",
                            "tool_calls": [
                                {
                                    "id": c["id"],
                                    "type": "function",
                                    "function": {
                                        "name": c["name"],
                                        "arguments": json.dumps(c["arguments"], ensure_ascii=False),
                                    },
                                }
                                for c in parsed_calls
                            ],
                        }
                    )
                elif not used_native_tools:
                    tool_history.append(
                        {
                            "role": "assistant",
                            "content": visible_text or raw_content,
                        }
                    )
                else:
                    calls_summary = ", ".join(
                        f"{c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})"
                        for c in parsed_calls
                    )
                    tool_history.append(
                        {
                            "role": "assistant",
                            "content": f"{visible_text}\n[ツール呼び出し: {calls_summary}]".strip(),
                        }
                    )

                loop_warning_needed = False
                for call in parsed_calls:
                    if self._stop_event.is_set():
                        await self._finalize_stopped(session, thinking_log)
                        return

                    tool_name = call["name"]
                    tool_args = call["arguments"]
                    sig = (tool_name, json.dumps(tool_args, sort_keys=True, ensure_ascii=False))

                    if sig == last_call_signature:
                        consecutive_repeats += 1
                    else:
                        last_call_signature = sig
                        consecutive_repeats = 1

                    # Phase: working
                    work_active_label = self._describe_tool_action(tool_name, tool_args, in_progress=True)
                    work_done_label = self._describe_tool_action(tool_name, tool_args, in_progress=False)
                    self.state.phase_state = "working"
                    self.state.phase_label = work_active_label
                    await self.broker.publish(
                        "status",
                        {
                            **self._status_payload(),
                            "state": "working",
                            "step": step,
                            "tool": tool_name,
                            "label": work_active_label,
                        },
                    )
                    work_start = time.monotonic()

                    result_text = await tools.execute(tool_name, tool_args)

                    if tool_name == "write_file":
                        written_path = str(tool_args.get("path", "")).lower().strip()
                        if written_path.endswith((".py", ".bat", ".ps1", ".sh", ".js", ".ts", ".cmd")):
                            needs_execution_test = True
                    elif tool_name in ("run_python", "run_command"):
                        needs_execution_test = False

                    work_duration = round(time.monotonic() - work_start, 1)
                    work_item = {
                        "type": "work",
                        "step": step,
                        "tool": tool_name,
                        "label": work_done_label,
                        "duration": work_duration,
                        "arguments": tool_args,
                        "result": result_text,
                    }
                    thinking_log.append(work_item)
                    self.live_thinking_log.append(work_item)
                    await self.broker.publish("work", work_item)

                    history_result = (
                        result_text[:3000] + "\n... (省略)"
                        if len(result_text) > 3000
                        else result_text
                    )
                    if used_native_tools and not use_flattened_tools:
                        tool_history.append(
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "name": tool_name,
                                "content": history_result,
                            }
                        )
                    else:
                        tool_history.append(
                            {
                                "role": "user",
                                "content": (
                                    f"【ツール実行結果 ({tool_name})】\n{history_result}\n\n"
                                    "上記の結果を確認し、次のツールを呼び出すか、完了していれば最終回答を作成してください。"
                                ),
                            }
                        )

                    if consecutive_repeats >= 3:
                        loop_warning_needed = True

                if loop_warning_needed:
                    warn_msg = (
                        "⚠️ システム警告（無限ループ検知）: 同じツールを同じ引数で3回連続して呼び出しています。"
                        "別のアプローチ、別のツール、または別の引数を試してください。"
                    )
                    warn_item = {"type": "warning", "step": step, "message": warn_msg}
                    thinking_log.append(warn_item)
                    self.live_thinking_log.append(warn_item)
                    await self.broker.publish("warning", warn_item)
                    tool_history.append({"role": "user", "content": warn_msg})
                    consecutive_repeats = 0
                    last_call_signature = None

                print(
                    f"[Step {step}] ツール実行完了 -> 次の推論（Step {step + 1}）へ進みます",
                    flush=True,
                )

            # Max steps reached
            final_text = f"最大ステップ数 ({self.state.max_steps}) に到達したため、ここまでの処理で終了しました。"
            assistant_entry = {
                "id": f"msg_{uuid.uuid4().hex[:8]}",
                "role": "assistant",
                "content": final_text,
                "thinking_log": thinking_log,
                "timestamp": _now_time(),
            }
            session["messages"].append(assistant_entry)
            self.sessions.save_session(session)
            self.state.phase_state = "idle"
            self.state.phase_label = ""
            self.last_completed_message = assistant_entry
            await self.broker.publish(
                "final_answer",
                {"step": self.state.max_steps, "session_id": session_id, "message": assistant_entry},
            )
            self.state.status = "completed"
            await self.broker.publish("status", self._status_payload())

        except asyncio.CancelledError:
            await self._finalize_stopped(session, thinking_log)
        except Exception as exc:
            self.state.status = "error"
            self.state.phase_state = "idle"
            self.state.phase_label = ""
            self.state.error_message = str(exc)
            await self.broker.publish("error", {"message": f"実行エラー: {exc}"})
            await self.broker.publish("status", self._status_payload())

    async def _finalize_stopped(
        self, session: dict[str, Any], thinking_log: list[dict[str, Any]]
    ) -> None:
        self.state.status = "stopped"
        self.state.phase_state = "idle"
        self.state.phase_label = ""
        if thinking_log:
            assistant_entry = {
                "id": f"msg_{uuid.uuid4().hex[:8]}",
                "role": "assistant",
                "content": "（ユーザーにより停止されました）",
                "thinking_log": thinking_log,
                "timestamp": _now_time(),
            }
            session["messages"].append(assistant_entry)
            self.sessions.save_session(session)
            self.last_completed_message = assistant_entry
            await self.broker.publish(
                "final_answer",
                {
                    "step": self.state.current_step,
                    "session_id": session["id"],
                    "message": assistant_entry,
                },
            )
        await self.broker.publish("status", self._status_payload())
