"""Autonomous Plan-Act-Reflect agent harness with multi-layer memory and chat sessions."""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import re
import shutil
import stat
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, BadRequestError

from tools import (
    ALLOWED_MEMORY_FILES,
    LONGTHINK_EXTRA_TOOL_SCHEMAS,
    TOOL_SCHEMAS,
    ULTRATHINK_EXTRA_TOOL_SCHEMAS,
    WorkspaceTools,
    normalize_memory_filename,
)

DEFAULT_LM_STUDIO_URL = "http://localhost:1234/v1"
DEFAULT_SUPERVISOR_MODEL = os.environ.get("CRYSTALLM_SUPERVISOR_MODEL", "d1-2B").strip() or "d1-2B"
DEFAULT_CODER_MODEL = (
    os.environ.get("CRYSTALLM_CODER_MODEL", "lfm-3-coder-claude-distill").strip()
    or "lfm-3-coder-claude-distill"
)
DEFAULT_STANDARD_MODEL = (
    os.environ.get("CRYSTALLM_STANDARD_MODEL", "LFM3-1.7B").strip()
    or "LFM3-1.7B"
)
DEFAULT_MODEL_PARALLEL = 1
NO_MAIN_MODEL_SENTINEL = "__NONE__"
CIRCUIT_BREAKER_THRESHOLD = 2
MAX_TOKENS_LOW_GEAR = 1000
THINK_TOKEN_LIMIT_DEFAULT = 3000
MAX_TOKENS_NORMAL_OUTPUT = 16384
MAX_TOKENS_UNLOCKED_CAP = 16384
MAX_TOKENS_DIRECT_BURST = -1
D1_UNLOCK_THINK_THRESHOLD = 5
D1_ESCALATE_THRESHOLD = 6
THINK_LIMIT_INJECT_PROMPT = (
    "思考トークン上限（3000トークン）に達したため </think> を挿入し思考を打ち切りました。"
    "直ちに具体的なツール実行・[WRITE_FILE: ...] 形式でのコード記述、または結論の出力を行ってください。"
)
DIRECT_WRITE_FORMAT_GUIDE = (
    "【最優先フォーマット：ファイル作成・Pythonコード記述の標準手段】\n"
    "長いPythonコードやファイルをJSON引数にエスケープして `write_file` を呼ぶと失敗しやすいため、"
    "必ず以下の非JSON（Markdown直接抽出）形式で本文に出力してください（ハーネスが自動でファイル保存します）：\n"
    "[WRITE_FILE: path/to/script.py]\n"
    "```python\n"
    "# ここに通常のPythonコードをそのまま記述\n"
    "```"
)
PURE_CODER_NON_JSON_DIRECTIVE = (
    "【LFM3-Coder 純粋プログラミング専任制約（JSONツール呼び出し厳禁）】\n"
    "- JSON形式の Function Calling / `<tool_call>` は一切使用しないでください。\n"
    "- 必ず本文中に次の非JSON Markdown ブロック形式のみでスクリプト・コードを出力してください（ハーネスが自動保存＆検証実行します）：\n"
    "[WRITE_FILE: path/to/script.py]\n"
    "```python\n"
    "# 完全な実行可能コード\n"
    "```"
)
_IGNORED_SCAN_DIRS = {".git", "venv", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache"}
MAX_TOOL_HISTORY_MESSAGES = 14
MAX_CHAT_CONTEXT_MESSAGES = 16
LLM_HTTP_TIMEOUT = httpx.Timeout(connect=30.0, read=1200.0, write=60.0, pool=60.0)

UT_GEAR_HIGH = "HIGH"
UT_GEAR_LOW = "LOW"
UT_GEAR_GC = "GC"

_CJK_CHAR_RE = re.compile(r"[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\uff00-\uffef]")


def _estimate_think_tokens(text: str) -> int:
    """Fast token count estimation supporting Japanese/CJK and ASCII/Python code."""
    if not text:
        return 0
    cjk_count = len(_CJK_CHAR_RE.findall(text))
    non_cjk_len = max(0, len(text) - cjk_count)
    return cjk_count + (non_cjk_len + 3) // 4


_REPEAT_PHRASE_RE = re.compile(r"(.{25,240}?)\1{2,}", re.DOTALL)


def _detect_consecutive_phrase_repeat(text: str, min_len: int = 25, min_repeats: int = 3) -> Optional[str]:
    """Detect >=min_repeats consecutive identical non-decorative phrases (>=min_len chars) in streaming text."""
    if not text or len(text) < min_len * min_repeats:
        return None
    tail = text[-1200:] if len(text) > 1200 else text
    pattern = (
        _REPEAT_PHRASE_RE
        if (min_len == 25 and min_repeats == 3)
        else re.compile(rf"(.{{{min_len},240}}?)\1{{{min_repeats - 1},}}", re.DOTALL)
    )
    for match in pattern.finditer(tail):
        phrase = match.group(1)
        meaningful = re.sub(r"[\s\-=_#*~`]+", "", phrase)
        if len(meaningful) >= 12:
            return phrase
    return None


def _compress_thought_summary(thought_text: str, max_chars: int = 240) -> str:
    """Compress partial thought text into 1-2 clean summary lines for starting a fresh uncorrupted turn."""
    cleaned = re.sub(r"</?think>", "", thought_text or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    sentences = [s.strip() for s in re.split(r"(?<=[。！？.!?])\s*", cleaned) if s.strip()]
    picked: list[str] = []
    seen: set[str] = set()
    for sent in sentences:
        norm = re.sub(r"\s+", "", sent)
        if len(norm) < 8 or norm in seen:
            continue
        seen.add(norm)
        picked.append(sent)
    if not picked:
        return cleaned[-max_chars:] if len(cleaned) > max_chars else cleaned
    summary = f"{picked[0]} / {picked[-1]}" if len(picked) >= 2 else picked[0]
    if len(summary) > max_chars:
        summary = summary[:max_chars].rstrip() + "…"
    return summary


def _normalize_model_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def _is_supervisor_model_id(model_id: str, supervisor_hint: Optional[str] = None) -> bool:
    norm = _normalize_model_key(model_id)
    if not norm:
        return False
    hint_norm = _normalize_model_key(supervisor_hint or DEFAULT_SUPERVISOR_MODEL)
    return ("d12b" in norm) or (bool(hint_norm) and hint_norm in norm)


def _is_coder_model_id(model_id: str, coder_hint: Optional[str] = None) -> bool:
    norm = _normalize_model_key(model_id)
    if not norm:
        return False
    hint_norm = _normalize_model_key(coder_hint or DEFAULT_CODER_MODEL)
    if "lfm3coder" in norm or "claudedistill" in norm:
        return True
    if "lfm" in norm and "coder" in norm:
        return True
    if "coder" in norm and ("distill" in norm or "q2k" in norm):
        return True
    return bool(hint_norm) and hint_norm in norm


def _is_standard_model_id(model_id: str, standard_hint: Optional[str] = None) -> bool:
    norm = _normalize_model_key(model_id)
    if not norm:
        return False
    if _is_supervisor_model_id(model_id) or _is_coder_model_id(model_id):
        return False
    hint_norm = _normalize_model_key(standard_hint or DEFAULT_STANDARD_MODEL)
    if hint_norm and hint_norm in norm:
        return True
    if ("lfm3" in norm or "lfm25" in norm or "lfm" in norm) and ("coder" not in norm and "d1" not in norm):
        return ("17b" in norm) or ("standard" in norm) or ("instruct" in norm) or ("lfm3" in norm)
    return False


def resolve_supervisor_model_id(
    configured_id: Optional[str],
    available_models: list[str],
    main_model: str = "",
) -> str:
    """Resolve the supervisor model ID (e.g. d1-2B) against LM Studio available models."""
    target = (configured_id or "").strip() or DEFAULT_SUPERVISOR_MODEL
    if not available_models:
        return target

    # 1. Exact case-sensitive match
    if target in available_models:
        return target

    # 2. Case-insensitive exact match
    target_lower = target.lower()
    for mid in available_models:
        if mid.lower() == target_lower:
            return mid

    # 3. Direct substring match or normalized alphanumeric match ("d1-2B" <-> "d1_2b" / "...d1-2b...")
    target_norm = _normalize_model_key(target)
    candidates: list[str] = []
    for mid in available_models:
        if target_lower in mid.lower() or (target_norm and target_norm in _normalize_model_key(mid)):
            candidates.append(mid)

    if candidates:
        non_main = [c for c in candidates if c != main_model]
        return non_main[0] if non_main else candidates[0]

    return target


def resolve_coder_model_id(
    configured_id: Optional[str],
    available_models: list[str],
    main_model: str = "",
) -> str:
    """Resolve the pure-code model ID (e.g. lfm-3-coder-claude-distill / LFM3-Coder) against LM Studio models."""
    target = (configured_id or "").strip() or DEFAULT_CODER_MODEL
    if not available_models:
        return target

    if target in available_models:
        return target

    target_lower = target.lower()
    for mid in available_models:
        if mid.lower() == target_lower:
            return mid

    target_norm = _normalize_model_key(target)
    distill_candidates: list[str] = []
    q2_candidates: list[str] = []
    general_candidates: list[str] = []
    for mid in available_models:
        mid_lower = mid.lower()
        mid_norm = _normalize_model_key(mid)
        matched = (
            target_lower in mid_lower
            or (target_norm and target_norm in mid_norm)
            or _is_coder_model_id(mid, target)
        )
        if matched:
            if "claudedistill" in mid_norm or "lfm3coder" in mid_norm:
                distill_candidates.append(mid)
            elif "q2k" in mid_norm:
                q2_candidates.append(mid)
            else:
                general_candidates.append(mid)

    candidates = distill_candidates + q2_candidates + general_candidates
    if candidates:
        non_main = [c for c in candidates if c != main_model and not _is_supervisor_model_id(c)]
        return non_main[0] if non_main else candidates[0]

    return target


def resolve_standard_model_id(
    configured_id: Optional[str],
    available_models: list[str],
    main_model: str = "",
) -> str:
    """Resolve the standard admin/tool-calling model ID (LFM3 1.7B) against LM Studio models."""
    target = (configured_id or "").strip() or DEFAULT_STANDARD_MODEL
    if not available_models:
        return target

    if target in available_models:
        return target

    target_lower = target.lower()
    for mid in available_models:
        if mid.lower() == target_lower:
            return mid

    target_norm = _normalize_model_key(target)
    exact_17b: list[str] = []
    other_lfm: list[str] = []
    for mid in available_models:
        if _is_supervisor_model_id(mid) or _is_coder_model_id(mid):
            continue
        mid_lower = mid.lower()
        mid_norm = _normalize_model_key(mid)
        matched = (
            target_lower in mid_lower
            or (target_norm and target_norm in mid_norm)
            or _is_standard_model_id(mid, target)
        )
        if matched:
            if "17b" in mid_norm:
                exact_17b.append(mid)
            else:
                other_lfm.append(mid)

    candidates = exact_17b + other_lfm
    if candidates:
        non_main = [c for c in candidates if c != main_model]
        return non_main[0] if non_main else candidates[0]

    return target


@dataclass
class _StreamFunction:
    name: str = ""
    arguments: str = ""


@dataclass
class _StreamToolCall:
    id: str = ""
    type: str = "function"
    function: _StreamFunction = field(default_factory=_StreamFunction)


@dataclass
class _StreamMessage:
    content: str = ""
    reasoning_content: str = ""
    reasoning: str = ""
    tool_calls: Optional[list[_StreamToolCall]] = None
    think_truncated: bool = False
    think_tokens: int = 0
    repeat_aborted: bool = False
    repeated_phrase: str = ""


@dataclass
class _StreamChoice:
    index: int = 0
    message: _StreamMessage = field(default_factory=_StreamMessage)


@dataclass
class _StreamChatCompletion:
    choices: list[_StreamChoice] = field(default_factory=list)

STATE_INIT_PLAN = "INIT_PLAN"
STATE_EXECUTE_STEP = "EXECUTE_STEP"
STATE_FIX_ERROR = "FIX_ERROR"
STATE_VERIFY_RESULT = "VERIFY_RESULT"
STATE_REPLAN = "REPLAN"
STATE_FINAL_REPORT = "FINAL_REPORT"

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

SYSTEM_PROMPT = """あなたはローカルPC上で自律稼働する優秀なAIアシスタント兼自律エージェント「クリスタルLM（CrystalLM）」です。
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

LONGTHINK_SYSTEM_ADDON = """

## /longthink モード（マスター Orchestrator 主導の動的サブエージェント編成ルール）
あなたは現在 `/longthink` 深層思考モードの**マスター（Orchestrator）**として動作しています。
課題の性質や進行状況に応じて最適なチーム体制を動的に編成し、コンテキスト汚染を防ぎながらタスクを統括してください：
- **共通広場 (`longthink/shared_workspace/`)**:
  - `master_plan.md`: あなたが管理する全体計画チェックリスト (`- [ ]` / `- [x]`)。最初に `update_master_plan` で論理的なサブタスクに分解してください。
  - `handoff_board.md`: マスターからの指示と、各サブエージェントの中間経過・最終結果報告が集約される共有ボードです。
  - `final_artifacts/`: 統合・検証済みの最終成果物を `save_final_artifact` で保存する場所。
- **動的サブエージェント生成・統括 (`delegate_parallel_workers` / `delegate_sub_agent`)**:
  - あなたは進行状況に応じて**「今、何体のサブエージェント（1〜4体）を立ち上げるべきか」を動的に決定**してください（調査・実装・検証を並列化する場合は2〜4体、単一調査や検算には1体）。
  - 立ち上げを指示すると、`longthink/sub_agents/<sub_agent_id>/` フォルダと専用の `plan.md` / `workspace/` が自動生成されます。
  - 各サブエージェントは自分専用の `plan.md` と `workspace/` のみで作業し、一定ステップごと及び完了時に `handoff_board.md` へ結果を報告します。
- **監視・統合・方針変更ループ**:
  - 毎ステップで `master_plan.md` と `handoff_board.md` を確認し、①完了した成果物を統合（必要に応じて `sub_agents/<id>/workspace/...` を `read_file` で参照）して `update_master_plan` で `[x]` に更新するか、②追加のサブエージェントを起動するか、③行き詰まった報告を出したサブエージェントに同一 `agent_id` または新規エージェントで別方針の指示を与えるかを判断してください。
  - 新規開始直後の `workspace/` が空であることは正常です。`list_files` での空ディレクトリ再確認は繰り返さず、ただちに計画作成・作成作業・委譲へ進んでください。
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


def _read_config_json(base_dir: Optional[Path] = None) -> dict[str, Any]:
    cfg_dir = base_dir or Path(__file__).resolve().parent
    cfg_path = cfg_dir / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


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
        model_ids = [d["id"] for d in model_details]
        cfg_data = _read_config_json()
        sup_hint = os.environ.get("CRYSTALLM_SUPERVISOR_MODEL", "").strip() or str(cfg_data.get("supervisor_model") or "").strip() or DEFAULT_SUPERVISOR_MODEL
        coder_hint = os.environ.get("CRYSTALLM_CODER_MODEL", "").strip() or str(cfg_data.get("coder_model") or "").strip() or DEFAULT_CODER_MODEL
        std_hint = os.environ.get("CRYSTALLM_STANDARD_MODEL", "").strip() or str(cfg_data.get("standard_model") or "").strip() or DEFAULT_STANDARD_MODEL
        main_cfg = os.environ.get("CRYSTALLM_MAIN_MODEL", "").strip() or str(cfg_data.get("main_model") or "").strip()
        resolved_main_cfg = (
            NO_MAIN_MODEL_SENTINEL
            if main_cfg == NO_MAIN_MODEL_SENTINEL
            else (main_cfg if (main_cfg and main_cfg in model_ids) else "")
        )
        main_for_resolve = "" if resolved_main_cfg == NO_MAIN_MODEL_SENTINEL else resolved_main_cfg
        return {
            "connected": True,
            "base_url": normalized,
            "models": model_ids,
            "model_details": model_details,
            "configured_main_model": resolved_main_cfg,
            "detected_supervisor_model": resolve_supervisor_model_id(sup_hint, model_ids, main_model=main_for_resolve),
            "detected_coder_model": resolve_coder_model_id(coder_hint, model_ids, main_model=main_for_resolve),
            "detected_standard_model": resolve_standard_model_id(std_hint, model_ids, main_model=main_for_resolve),
            "error": None,
        }
    except Exception as exc:
        cfg_data = _read_config_json()
        return {
            "connected": False,
            "base_url": normalized,
            "models": [],
            "model_details": [],
            "configured_main_model": str(cfg_data.get("main_model") or "").strip(),
            "detected_supervisor_model": str(cfg_data.get("supervisor_model") or "").strip() or DEFAULT_SUPERVISOR_MODEL,
            "detected_coder_model": str(cfg_data.get("coder_model") or "").strip() or DEFAULT_CODER_MODEL,
            "detected_standard_model": str(cfg_data.get("standard_model") or "").strip() or DEFAULT_STANDARD_MODEL,
            "error": str(exc),
        }


async def select_lm_studio_model(
    model_id: str,
    base_url: str = DEFAULT_LM_STUDIO_URL,
    parallel: int = DEFAULT_MODEL_PARALLEL,
    role: str = "main",
    previous_model: Optional[str] = None,
    keep_models: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Eject the replaced / unkept model(s) from VRAM and load the newly selected model with Parallel = 1."""
    global _MODEL_CACHE_TIME
    normalized = _normalize_base_url(base_url)
    host_root = _host_root_from_base_url(normalized)
    resolved_parallel = DEFAULT_MODEL_PARALLEL
    target = (model_id or "").strip()
    if target == NO_MAIN_MODEL_SENTINEL:
        target = ""
    prev_clean = (previous_model or "").strip()
    if prev_clean == NO_MAIN_MODEL_SENTINEL:
        prev_clean = ""
    role_clean = (role or "main").strip().lower()

    def _same_model(a: str, b: str) -> bool:
        a_s = (a or "").strip().lower()
        b_s = (b or "").strip().lower()
        if not a_s or not b_s:
            return False
        if a_s == b_s:
            return True
        na = _normalize_model_key(a_s)
        nb = _normalize_model_key(b_s)
        return bool(na and nb and na == nb)

    def _is_protected_fallback(mid: str) -> bool:
        return _is_supervisor_model_id(mid) or _is_coder_model_id(mid) or _is_standard_model_id(mid)

    keep_list = [
        m.strip()
        for m in (keep_models or [])
        if isinstance(m, str) and m.strip() and m.strip() != NO_MAIN_MODEL_SENTINEL
    ]

    def _is_kept_other_slot(mid: str) -> bool:
        if keep_models is not None:
            return any(_same_model(mid, k) for k in keep_list)
        if role_clean == "main" and not (prev_clean and _same_model(mid, prev_clean)):
            return _is_protected_fallback(mid)
        return False

    ejected_models: list[str] = []
    lms_path = _find_lms_cli()
    cli_eject_targets: list[str] = []
    if prev_clean and not _same_model(prev_clean, target) and not _is_kept_other_slot(prev_clean):
        cli_eject_targets.append(prev_clean)

    # 1. Inspect currently loaded instances via LM Studio REST API & eject replaced / unkept models
    kept_loaded_on_server = False
    try:
        async with httpx.AsyncClient(timeout=20.0) as probe_client:
            r_list = await probe_client.get(f"{host_root}/api/v1/models")
            if r_list.status_code == 200:
                for m in r_list.json().get("models", []):
                    m_key = str(m.get("key") or m.get("id") or "").strip()
                    loaded_insts = m.get("loaded_instances") or []
                    if not loaded_insts:
                        continue
                    is_target = bool(target and _same_model(m_key, target))
                    is_kept = _is_kept_other_slot(m_key)
                    is_explicit_prev = bool(prev_clean and _same_model(m_key, prev_clean))
                    should_eject = (is_explicit_prev and not is_target and not is_kept) or (
                        not is_target and not is_kept
                    )
                    if should_eject:
                        if m_key and not any(_same_model(m_key, x) for x in cli_eject_targets):
                            cli_eject_targets.append(m_key)
                        for inst in loaded_insts:
                            inst_id = inst.get("id") if isinstance(inst, dict) else inst
                            if inst_id:
                                try:
                                    r_un = await probe_client.post(
                                        f"{host_root}/api/v1/models/unload",
                                        json={"instance_id": inst_id},
                                    )
                                    if r_un.status_code in (200, 204):
                                        if m_key and m_key not in ejected_models:
                                            ejected_models.append(m_key)
                                except Exception:
                                    pass
                    elif not is_target and is_kept:
                        kept_loaded_on_server = True
            else:
                r_v0 = await probe_client.get(f"{host_root}/api/v0/models")
                if r_v0.status_code == 200:
                    for m in r_v0.json().get("data", []):
                        m_key = str(m.get("id") or "").strip()
                        if m.get("state") != "loaded" or not m_key:
                            continue
                        is_target = bool(target and _same_model(m_key, target))
                        is_kept = _is_kept_other_slot(m_key)
                        if not is_target and not is_kept and not any(_same_model(m_key, x) for x in cli_eject_targets):
                            cli_eject_targets.append(m_key)
                        elif not is_target and is_kept:
                            kept_loaded_on_server = True
    except Exception:
        pass

    # 2. Also unload via CLI (`lms unload <key>` or `lms unload --all` when no kept helpers are loaded)
    if lms_path and cli_eject_targets:
        def _cli_unload_targets() -> list[str]:
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            unloaded: list[str] = []
            if not kept_loaded_on_server and keep_models is None and role_clean == "main":
                proc = subprocess.run(
                    [lms_path, "unload", "--all"],
                    timeout=20,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=flags,
                )
                if proc.returncode == 0:
                    unloaded.extend(cli_eject_targets)
                return unloaded
            for key in cli_eject_targets:
                for cmd_key in (key, key.split("/")[-1]):
                    if not cmd_key:
                        continue
                    try:
                        proc = subprocess.run(
                            [lms_path, "unload", cmd_key],
                            timeout=15,
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            creationflags=flags,
                        )
                        if proc.returncode == 0:
                            unloaded.append(key)
                            break
                    except Exception:
                        continue
            return unloaded

        try:
            cli_unloaded = await asyncio.to_thread(_cli_unload_targets)
            for uk in cli_unloaded:
                if uk and not any(_same_model(uk, ex) for ex in ejected_models):
                    ejected_models.append(uk)
        except Exception:
            pass

    if not target:
        _MODEL_CACHE_TIME = 0.0
        eject_note = f" (エジェクト: {', '.join(ejected_models)})" if ejected_models else ""
        return {
            "ok": True,
            "model": "",
            "role": role_clean,
            "parallel": resolved_parallel,
            "ejected_models": ejected_models,
            "method": "auto",
            "message": f"自動選択 (Parallel = 1) に設定しました{eject_note}。",
        }

    # 3. Load newly selected target model with Parallel = 1, Flash Attention ON, K/V Q8_0
    load_payloads = [
        {
            "model": target,
            "parallel": resolved_parallel,
            "flash_attention": True,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
        },
        {
            "model": target,
            "parallel": resolved_parallel,
            "flashAttention": True,
            "kCacheQuantizationType": "q8_0",
            "vCacheQuantizationType": "q8_0",
        },
        {"model": target, "parallel": resolved_parallel},
        {"model": target},
    ]
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            for ep_path, ep_method in (("/api/v1/models/load", "api/v1/models/load"), ("/api/v0/models/load", "api/v0/models/load")):
                for payload in load_payloads:
                    try:
                        r_load = await client.post(f"{host_root}{ep_path}", json=payload)
                        if r_load.status_code in (200, 201):
                            _MODEL_CACHE_TIME = 0.0
                            eject_note = f" / VRAMエジェクト: {', '.join(ejected_models)}" if ejected_models else ""
                            return {
                                "ok": True,
                                "model": target,
                                "role": role_clean,
                                "parallel": resolved_parallel,
                                "flash_attention": True,
                                "cache_type_k": "q8_0",
                                "cache_type_v": "q8_0",
                                "ejected_models": ejected_models,
                                "method": ep_method,
                                "message": (
                                    f"[{role_clean}] '{target}' をロードしました "
                                    f"(Parallel={resolved_parallel}, FlashAttn=ON, KV=Q8_0{eject_note})。"
                                ),
                            }
                    except Exception:
                        continue

            if lms_path:
                def _cli_load() -> bool:
                    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    cli_cmds = [
                        [
                            lms_path,
                            "load",
                            target,
                            "-y",
                            "--parallel",
                            str(resolved_parallel),
                            "--flash-attention",
                            "--cache-type-k",
                            "q8_0",
                            "--cache-type-v",
                            "q8_0",
                        ],
                        [lms_path, "load", target, "-y", "--parallel", str(resolved_parallel)],
                        [lms_path, "load", target, "-y"],
                    ]
                    for cmd_args in cli_cmds:
                        proc = subprocess.run(
                            cmd_args,
                            timeout=90,
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            creationflags=flags,
                        )
                        if proc.returncode == 0:
                            return True
                    return False

                try:
                    if await asyncio.to_thread(_cli_load):
                        _MODEL_CACHE_TIME = 0.0
                        eject_note = f" / VRAMエジェクト: {', '.join(ejected_models)}" if ejected_models else ""
                        return {
                            "ok": True,
                            "model": target,
                            "role": role_clean,
                            "parallel": resolved_parallel,
                            "flash_attention": True,
                            "cache_type_k": "q8_0",
                            "cache_type_v": "q8_0",
                            "ejected_models": ejected_models,
                            "method": "lms_cli",
                            "message": (
                                f"[{role_clean}] '{target}' をロードしました "
                                f"(Parallel={resolved_parallel}, FlashAttn=ON, KV=Q8_0{eject_note})。"
                            ),
                        }
                except Exception:
                    pass

            # Fallback JIT load via minimal 1-token chat completion request
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
            eject_note = f" / VRAMエジェクト: {', '.join(ejected_models)}" if ejected_models else ""
            return {
                "ok": True,
                "model": target,
                "role": role_clean,
                "parallel": resolved_parallel,
                "flash_attention": True,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "ejected_models": ejected_models,
                "method": "jit/chat/completions",
                "message": (
                    f"[{role_clean}] '{target}' をロードしました "
                    f"(Parallel={resolved_parallel}, FlashAttn=ON, KV=Q8_0{eject_note})。"
                ),
            }
    except Exception as exc:
        return {
            "ok": False,
            "model": target,
            "role": role_clean,
            "parallel": resolved_parallel,
            "ejected_models": ejected_models,
            "method": "error",
            "message": f"モデル '{target}' の切り替えに失敗しました: {exc}",
        }


async def reload_all_lm_studio_models(
    models_to_load: list[tuple[str, str]],
    base_url: str = DEFAULT_LM_STUDIO_URL,
) -> dict[str, Any]:
    """Eject ALL loaded LM Studio instances, wait briefly, then reload active models with Parallel=1, FlashAttn=ON, KV=Q8_0."""
    global _MODEL_CACHE_TIME
    normalized = _normalize_base_url(base_url)
    host_root = _host_root_from_base_url(normalized)
    lms_path = _find_lms_cli()
    ejected_all: list[str] = []

    # 1. Force eject ALL loaded instances via REST API
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r_list = await client.get(f"{host_root}/api/v1/models")
            if r_list.status_code == 200:
                for m in r_list.json().get("models", []):
                    m_key = str(m.get("key") or m.get("id") or "").strip()
                    for inst in m.get("loaded_instances") or []:
                        inst_id = inst.get("id") if isinstance(inst, dict) else inst
                        if inst_id:
                            try:
                                r_un = await client.post(
                                    f"{host_root}/api/v1/models/unload",
                                    json={"instance_id": inst_id},
                                )
                                if r_un.status_code in (200, 204) and m_key and m_key not in ejected_all:
                                    ejected_all.append(m_key)
                            except Exception:
                                pass
    except Exception:
        pass

    # 2. Also execute `lms unload --all` CLI if available to guarantee zero lingering instances
    if lms_path:
        def _cli_unload_all() -> bool:
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.run(
                [lms_path, "unload", "--all"],
                timeout=30,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
            )
            return proc.returncode == 0

        try:
            await asyncio.to_thread(_cli_unload_all)
        except Exception:
            pass

    _MODEL_CACHE_TIME = 0.0
    await asyncio.sleep(0.5)

    # 3. Reload unique selected models sequentially with Parallel=1, FlashAttn=ON, KV=Q8_0
    unique_loads: list[tuple[str, str]] = []
    seen_norms: set[str] = set()
    for role_name, raw_mid in models_to_load:
        cleaned = (raw_mid or "").strip()
        if not cleaned:
            continue
        norm = _normalize_model_key(cleaned) or cleaned.lower()
        if norm in seen_norms:
            continue
        seen_norms.add(norm)
        unique_loads.append((role_name, cleaned))

    loaded_results: list[dict[str, Any]] = []
    failed: list[str] = []
    loaded_keep: list[str] = []

    for role_name, mid in unique_loads:
        res = await select_lm_studio_model(
            mid,
            normalized,
            parallel=DEFAULT_MODEL_PARALLEL,
            role=role_name,
            previous_model=None,
            keep_models=list(loaded_keep),
        )
        loaded_results.append(res)
        if res.get("ok"):
            loaded_keep.append(mid)
        else:
            failed.append(mid)

    latest_lm = await check_lm_studio(normalized)
    return {
        "ok": len(failed) == 0,
        "ejected_all": ejected_all,
        "loaded": loaded_results,
        "failed": failed,
        "lm_studio": latest_lm,
        "message": (
            "全モデルをエジェクトし、選択中の構成を FlashAttn + Q8_0 で再ロードしました"
            if not failed
            else f"一部モデルの再ロードに失敗しました: {', '.join(failed)}"
        ),
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

    def build_context_block(
        self,
        session_plan: Optional[str] = None,
        session_work_log: Optional[str] = None,
    ) -> str:
        all_mem = self.read_all()
        sections = ["\n\n## 現在の永続メモリ (memory/ + セッション隔離領域)"]
        for name in ALLOWED_MEMORY_FILES:
            if name == "plan.md" and session_plan is not None:
                content = session_plan.strip()
                label = "session/plan.md"
            elif name == "work_log.md" and session_work_log is not None:
                content = session_work_log.strip()
                label = "session/work_log.md"
            else:
                content = all_mem[name].strip()
                label = f"memory/{name}"
            if len(content) > 1400:
                content = content[:1400] + "\n... (省略)"
            sections.append(f"### {label}\n```markdown\n{content}\n```")
        return "\n\n".join(sections)


class SessionManager:
    """Persists isolated per-session directories under sessions/<session_id>/."""

    def __init__(self, sessions_dir: Path) -> None:
        self.sessions_dir = sessions_dir.resolve()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_session_id(session_id: str) -> str:
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", session_id or "")
        if not safe_id:
            raise ValueError("無効なセッションIDです。")
        return safe_id

    def _session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / self._sanitize_session_id(session_id)

    def _legacy_json_path(self, safe_id: str) -> Path:
        return self.sessions_dir / f"{safe_id}.json"

    def _migrate_legacy_json_if_needed(self, safe_id: str) -> None:
        legacy = self._legacy_json_path(safe_id)
        if not legacy.exists() or not legacy.is_file():
            return
        try:
            data = json.loads(legacy.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return
        sdir = self.ensure_session_files(safe_id, session_data=data)
        msg_path = sdir / "messages.json"
        if not msg_path.exists():
            msg_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self._sync_chat_history_md(sdir, data)
        try:
            legacy.unlink()
        except OSError:
            pass

    @staticmethod
    def _sync_chat_history_md(sdir: Path, session: dict[str, Any]) -> None:
        lines = [
            f"# チャット履歴: {session.get('title', '無題のチャット')}",
            f"- セッションID: `{session.get('id', sdir.name)}`",
            f"- モード: `{session.get('mode', 'normal')}`",
            f"- 更新日時: {session.get('updated_at', _now_iso())}",
            "",
        ]
        for msg in session.get("messages", []):
            role_label = "👤 User" if msg.get("role") == "user" else "🤖 Assistant"
            ts = msg.get("timestamp", "")
            lines.append(f"## {role_label} ({ts})")
            lines.append("")
            lines.append(str(msg.get("content") or ""))
            lines.append("")
        (sdir / "chat_history.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def ensure_session_files(
        self,
        session_id: str,
        session_data: Optional[dict[str, Any]] = None,
    ) -> Path:
        sdir = self._session_dir(session_id)
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "workspace").mkdir(parents=True, exist_ok=True)
        plan_master_file = sdir / "plan_master.md"
        plan_file = sdir / "plan.md"
        if not plan_master_file.exists():
            initial_plan = (
                plan_file.read_text(encoding="utf-8", errors="replace")
                if plan_file.exists()
                else DEFAULT_MEMORY_TEMPLATES["plan.md"]
            )
            plan_master_file.write_text(initial_plan, encoding="utf-8")
        if not plan_file.exists():
            plan_file.write_text(plan_master_file.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        work_log_file = sdir / "work_log.md"
        if not work_log_file.exists():
            work_log_file.write_text(DEFAULT_MEMORY_TEMPLATES["work_log.md"], encoding="utf-8")
        chat_hist = sdir / "chat_history.md"
        if session_data is not None:
            self._sync_chat_history_md(sdir, session_data)
        elif not chat_hist.exists():
            chat_hist.write_text(f"# チャット履歴 ({sdir.name})\n\n", encoding="utf-8")
        return sdir

    def sync_plan_files(self, session_id: str, content: str) -> None:
        sdir = self.ensure_session_files(session_id)
        for p in (sdir / "plan_master.md", sdir / "plan.md"):
            p.write_text(content, encoding="utf-8")
        shared = sdir / "longthink" / "shared_workspace"
        if shared.exists():
            (shared / "master_plan.md").write_text(content, encoding="utf-8")

    def ensure_longthink_structure(self, session_id: str) -> Path:
        sdir = self.ensure_session_files(session_id)
        lt_dir = sdir / "longthink"
        shared = lt_dir / "shared_workspace"
        (shared / "final_artifacts").mkdir(parents=True, exist_ok=True)
        (lt_dir / "sub_agents").mkdir(parents=True, exist_ok=True)

        master_plan = shared / "master_plan.md"
        if not master_plan.exists():
            master_plan.write_text(
                "# /longthink 全体計画 (plan_master.md)\n\n"
                "（現在進行中の長期タスクはありません）\n",
                encoding="utf-8",
            )
        handoff = shared / "handoff_board.md"
        if not handoff.exists():
            handoff.write_text(
                "# サブエージェント引き継ぎボード (handoff_board.md)\n\n"
                "（指揮者とサブエージェント間の指示・確定結論・検証結果がここに記録されます）\n",
                encoding="utf-8",
            )
        return lt_dir

    def create_sub_agent_folder(
        self,
        session_id: str,
        agent_id: str,
        task_title: str = "",
        instruction: str = "",
    ) -> Path:
        lt_dir = self.ensure_longthink_structure(session_id)
        safe_agent_id = re.sub(r"[^a-zA-Z0-9_-]", "_", (agent_id or "worker").strip()) or "worker"
        sub_dir = lt_dir / "sub_agents" / safe_agent_id
        sub_dir.mkdir(parents=True, exist_ok=True)
        (sub_dir / "workspace").mkdir(parents=True, exist_ok=True)
        plan_file = sub_dir / "plan.md"
        title_line = task_title.strip() or safe_agent_id
        instr_preview = (instruction or "").strip()[:300]
        if not plan_file.exists():
            plan_file.write_text(
                f"# サブエージェント計画: {safe_agent_id}\n\n"
                f"## 担当タスク\n{title_line}\n\n"
                f"## 指示概要\n{instr_preview}\n\n"
                f"## チェックリスト\n- [ ] {title_line} を実行・完了する\n",
                encoding="utf-8",
            )
        elif instruction.strip():
            existing_plan = plan_file.read_text(encoding="utf-8", errors="replace").rstrip()
            plan_file.write_text(
                f"{existing_plan}\n\n"
                f"## 方針変更・追加指示 ({_now_time()})\n"
                f"- 担当: {title_line}\n"
                f"- 指示: {instr_preview}\n"
                f"- [ ] {title_line} を新しい方針で実行・完了する\n",
                encoding="utf-8",
            )
        return sub_dir

    def get_longthink_sub_agents_summary(self, session_id: str) -> str:
        """Summarize active/completed sub-agents under longthink/sub_agents/ for Orchestrator."""
        sdir = self._session_dir(session_id)
        sub_root = sdir / "longthink" / "sub_agents"
        if not sub_root.exists():
            return "（まだサブエージェントは生成されていません）"
        lines: list[str] = []
        for sub_dir in sorted(sub_root.iterdir(), key=lambda p: p.name.lower()):
            if not sub_dir.is_dir():
                continue
            plan_path = sub_dir / "plan.md"
            ws_dir = sub_dir / "workspace"
            done_cnt = 0
            todo_cnt = 0
            if plan_path.exists():
                p_text = plan_path.read_text(encoding="utf-8", errors="replace")
                done_cnt = len(re.findall(r"^\s*[-*]\s*\[[xX]\]", p_text, flags=re.MULTILINE))
                todo_cnt = len(re.findall(r"^\s*[-*]\s*\[\s\]", p_text, flags=re.MULTILINE))
            ws_files = (
                [
                    p.relative_to(ws_dir).as_posix()
                    for p in sorted(ws_dir.rglob("*"))
                    if p.is_file() and "trash_bin" not in p.parts and "tool_isolated_ws" not in p.parts
                ]
                if ws_dir.exists()
                else []
            )
            status_tag = "完了" if (done_cnt > 0 and todo_cnt == 0) else f"進行中/要確認(未完:{todo_cnt}, 完:{done_cnt})"
            files_preview = ", ".join(ws_files[:6]) if ws_files else "(空)"
            if len(ws_files) > 6:
                files_preview += f" 他{len(ws_files) - 6}件"
            lines.append(f"- `{sub_dir.name}` [{status_tag}] | workspace: {files_preview}")
        return "\n".join(lines) if lines else "（まだサブエージェントは生成されていません）"

    @staticmethod
    def _parse_yaml_frontmatter(text: str) -> tuple[dict[str, str], str]:
        """Extract YAML frontmatter keys (ID, TITLE, STATUS, LAST_UPDATE) and body text."""
        if not text:
            return {}, ""
        m = re.match(r"^\s*---\s*\r?\n([\s\S]*?)\r?\n---\s*(?:\r?\n|$)([\s\S]*)$", text)
        if not m:
            return {}, text
        meta: dict[str, str] = {}
        for line in m.group(1).splitlines():
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            key = k.strip().upper()
            val = v.strip().strip('"').strip("'")
            if key:
                meta[key] = val
        return meta, m.group(2)

    @staticmethod
    def _extract_north_star_content(user_prompt: str) -> str:
        """Build structured NORTH_STAR.md from initial prompt for Prefix Caching."""
        cleaned = re.sub(r"^\s*/ultrathink\b\s*", "", user_prompt or "", flags=re.IGNORECASE).strip()
        if not cleaned:
            cleaned = "（最初のプロンプト受信時に自動抽出されます）"
        return (
            "# NORTH_STAR (最終ゴール・絶対制約 — 固定プレフィックス)\n\n"
            "## 最終ゴールと制約条件\n"
            f"{cleaned}\n\n"
            "## 探索原則\n"
            "- すべての仮説は `10_active_workspace/hyp_XXXX.md` に単一責任で記録し、必ず Python 検証等で反証・検算する。\n"
            "- 検証に成功した仮説のみ `VERIFIED` とし、`20_knowledge_base/` へ結晶化する。\n"
            "- 誤り・矛盾が見つかった仮説は即座に `REJECTED` として記録し、原因を次に活かす。\n"
        )

    def ensure_ultrathink_structure(self, session_id: str, user_prompt: str = "") -> Path:
        """Create sessions/<id>/ultrathink/ directory layout and initialize local Git repository."""
        sdir = self.ensure_session_files(session_id)
        ut_dir = sdir / "ultrathink"
        (ut_dir / "10_active_workspace").mkdir(parents=True, exist_ok=True)
        (ut_dir / "20_knowledge_base").mkdir(parents=True, exist_ok=True)
        (ut_dir / "90_archive").mkdir(parents=True, exist_ok=True)

        north_star = ut_dir / "NORTH_STAR.md"
        cleaned_prompt = re.sub(r"^\s*/ultrathink\b\s*", "", user_prompt or "", flags=re.IGNORECASE).strip()
        if not north_star.exists() or (
            cleaned_prompt
            and "（最初のプロンプト受信時に自動抽出されます）"
            in north_star.read_text(encoding="utf-8", errors="replace")
        ):
            north_star.write_text(self._extract_north_star_content(user_prompt), encoding="utf-8")

        global_index = ut_dir / "GLOBAL_INDEX.md"
        if not global_index.exists():
            global_index.write_text(
                "# 全体仮説インデックス (GLOBAL_INDEX.md)\n\n"
                "- 集計: UNVERIFIED=0 | VERIFIED=0 | REJECTED=0\n\n"
                "| ID | TITLE | STATUS | FILE | LAST_UPDATE |\n"
                "|---|---|---|---|---|\n",
                encoding="utf-8",
            )

        kb_index = ut_dir / "20_knowledge_base" / "INDEX_KB.md"
        if not kb_index.exists():
            kb_index.write_text(
                "# 知識ベース目次 (INDEX_KB.md)\n\n"
                "| ID | TITLE | FILE | STEP |\n"
                "|---|---|---|---|\n",
                encoding="utf-8",
            )

        gitignore = sdir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("__pycache__/\n*.pyc\n*.tmp\n", encoding="utf-8")

        git_dir = sdir / ".git"
        if not git_dir.exists():
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                subprocess.run(
                    ["git", "init"],
                    cwd=str(sdir),
                    capture_output=True,
                    text=True,
                    timeout=10,
                    creationflags=flags,
                )
                subprocess.run(
                    ["git", "config", "user.name", "UltraThink Harness"],
                    cwd=str(sdir),
                    capture_output=True,
                    text=True,
                    timeout=5,
                    creationflags=flags,
                )
                subprocess.run(
                    ["git", "config", "user.email", "ultrathink@local"],
                    cwd=str(sdir),
                    capture_output=True,
                    text=True,
                    timeout=5,
                    creationflags=flags,
                )
            except Exception:
                pass
        return ut_dir

    def git_commit_snapshot(self, session_id: str, message: str) -> Optional[str]:
        """Stage and commit session changes inside sessions/<id>/ for thinking time-machine."""
        sdir = self._session_dir(session_id)
        if not (sdir / ".git").exists():
            return None
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            subprocess.run(
                ["git", "add", "."],
                cwd=str(sdir),
                capture_output=True,
                text=True,
                timeout=10,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
            )
            st = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(sdir),
                capture_output=True,
                text=True,
                timeout=10,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
            )
            if not (st.stdout or "").strip():
                return None
            clean_msg = re.sub(r"[\r\n]+", " ", message).strip()[:120] or "Snapshot"
            cm = subprocess.run(
                ["git", "commit", "-m", clean_msg],
                cwd=str(sdir),
                capture_output=True,
                text=True,
                timeout=10,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
            )
            if cm.returncode != 0:
                return None
            rev = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(sdir),
                capture_output=True,
                text=True,
                timeout=5,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
            )
            sha = (rev.stdout or "").strip() or "HEAD"
            return f"{sha}: {clean_msg}"
        except Exception:
            return None

    def get_ultrathink_metrics(self, session_id: str) -> dict[str, Any]:
        """Collect counts and items from GLOBAL_INDEX.md, 10_active_workspace/, 20_knowledge_base/, and 90_archive/."""
        sdir = self._session_dir(session_id)
        ut_dir = sdir / "ultrathink"
        if not ut_dir.exists():
            return {
                "enabled": False,
                "unverified": 0,
                "verified": 0,
                "rejected": 0,
                "knowledge_facts": 0,
                "active_files": 0,
                "archived_files": 0,
                "items": [],
            }

        items_map: dict[str, dict[str, str]] = {}
        gi_path = ut_dir / "GLOBAL_INDEX.md"
        if gi_path.exists():
            for line in gi_path.read_text(encoding="utf-8", errors="replace").splitlines():
                stripped = line.strip()
                if not stripped.startswith("|") or "---" in stripped:
                    continue
                cols = [c.strip() for c in stripped.strip("|").split("|")]
                if len(cols) >= 3 and cols[0].upper().startswith("HYP_"):
                    raw_status = cols[2].strip("[] ").upper()
                    if raw_status not in ("UNVERIFIED", "VERIFIED", "REJECTED"):
                        raw_status = "UNVERIFIED"
                    items_map[cols[0].upper()] = {
                        "id": cols[0].upper(),
                        "title": cols[1],
                        "status": raw_status,
                        "file": cols[3] if len(cols) > 3 else "",
                        "last_update": cols[4] if len(cols) > 4 else "",
                    }

        for folder_name in ("10_active_workspace", "90_archive"):
            folder = ut_dir / folder_name
            if not folder.exists():
                continue
            for p in sorted(folder.glob("hyp_*.md")):
                meta, _ = self._parse_yaml_frontmatter(p.read_text(encoding="utf-8", errors="replace"))
                hid = (meta.get("ID") or p.stem).upper()
                status = (meta.get("STATUS") or "UNVERIFIED").strip("[] ").upper()
                if status not in ("UNVERIFIED", "VERIFIED", "REJECTED"):
                    status = "UNVERIFIED"
                items_map[hid] = {
                    "id": hid,
                    "title": meta.get("TITLE") or items_map.get(hid, {}).get("title") or p.stem,
                    "status": status,
                    "file": f"{folder_name}/{p.name}",
                    "last_update": meta.get("LAST_UPDATE") or "",
                }

        items = sorted(items_map.values(), key=lambda x: x["id"])
        unverified = sum(1 for x in items if x["status"] == "UNVERIFIED")
        verified = sum(1 for x in items if x["status"] == "VERIFIED")
        rejected = sum(1 for x in items if x["status"] == "REJECTED")
        kb_dir = ut_dir / "20_knowledge_base"
        arch_dir = ut_dir / "90_archive"
        act_dir = ut_dir / "10_active_workspace"
        kb_facts = len(list(kb_dir.glob("fact_*.md"))) if kb_dir.exists() else 0
        active_files = len([p for p in act_dir.iterdir() if p.is_file()]) if act_dir.exists() else 0
        archived_files = len([p for p in arch_dir.iterdir() if p.is_file()]) if arch_dir.exists() else 0
        return {
            "enabled": True,
            "unverified": unverified,
            "verified": verified,
            "rejected": rejected,
            "knowledge_facts": kb_facts,
            "active_files": active_files,
            "archived_files": archived_files,
            "items": items,
        }

    def create_session(
        self,
        title: str = "新しいチャット",
        mode: str = "normal",
    ) -> dict[str, Any]:
        session_id = f"sess_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        normalized_mode = mode if mode in ("fastthink", "normal", "longthink", "ultrathink") else "normal"
        now = _now_iso()
        session = {
            "id": session_id,
            "title": title,
            "mode": normalized_mode,
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        sdir = self.ensure_session_files(session_id, session_data=session)
        if normalized_mode == "longthink":
            self.ensure_longthink_structure(session_id)
        elif normalized_mode == "ultrathink":
            self.ensure_ultrathink_structure(session_id)
        (sdir / "messages.json").write_text(
            json.dumps(session, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return session

    def get_session(self, session_id: str) -> dict[str, Any]:
        safe_id = self._sanitize_session_id(session_id)
        self._migrate_legacy_json_if_needed(safe_id)
        sdir = self._session_dir(safe_id)
        msg_path = sdir / "messages.json"
        if not msg_path.exists():
            raise FileNotFoundError(f"セッションが見つかりません: {session_id}")
        data = json.loads(msg_path.read_text(encoding="utf-8", errors="replace"))
        data.setdefault("mode", "normal")
        self.ensure_session_files(safe_id)
        return data

    def save_session(self, session: dict[str, Any]) -> None:
        session["updated_at"] = _now_iso()
        safe_id = self._sanitize_session_id(session["id"])
        sdir = self.ensure_session_files(safe_id, session_data=session)
        (sdir / "messages.json").write_text(
            json.dumps(session, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def delete_session(self, session_id: str) -> bool:
        safe_id = self._sanitize_session_id(session_id)
        deleted = False
        sdir = self._session_dir(safe_id)
        if sdir.exists():
            def _handle_remove_readonly(func: Any, path: str, exc_info: Any) -> None:
                try:
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                except OSError:
                    pass

            shutil.rmtree(sdir, onerror=_handle_remove_readonly)
            deleted = True
        legacy = self._legacy_json_path(safe_id)
        if legacy.exists():
            legacy.unlink()
            deleted = True
        return deleted

    def list_sessions(self) -> list[dict[str, Any]]:
        for legacy in list(self.sessions_dir.glob("*.json")):
            self._migrate_legacy_json_if_needed(legacy.stem)

        items: list[dict[str, Any]] = []
        for sdir in self.sessions_dir.iterdir():
            if not sdir.is_dir():
                continue
            msg_path = sdir / "messages.json"
            if not msg_path.exists():
                continue
            try:
                data = json.loads(msg_path.read_text(encoding="utf-8", errors="replace"))
                items.append(
                    {
                        "id": data["id"],
                        "title": data.get("title", "無題のチャット"),
                        "mode": data.get("mode", "normal"),
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),
                        "message_count": len(data.get("messages", [])),
                    }
                )
            except Exception:
                continue
        items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return items

    def resolve_session_file(self, session_id: str, rel_path: str) -> Path:
        sdir = self._session_dir(session_id).resolve()
        if not sdir.exists():
            raise FileNotFoundError(f"セッションが見つかりません: {session_id}")
        cleaned = (rel_path or "").strip().replace("\\", "/").lstrip("/")
        if not cleaned:
            raise ValueError("ファイルパスが空です。")
        candidate = (sdir / cleaned).resolve()
        if candidate != sdir and sdir not in candidate.parents:
            raise ValueError(f"アクセス拒否: セッション外のパス '{rel_path}' にはアクセスできません。")
        return candidate

    def read_session_file(self, session_id: str, rel_path: str) -> dict[str, Any]:
        sdir = self._session_dir(session_id).resolve()
        target = self.resolve_session_file(session_id, rel_path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"ファイルが見つかりません: {rel_path}")
        return {
            "session_id": self._sanitize_session_id(session_id),
            "path": target.relative_to(sdir).as_posix(),
            "content": target.read_text(encoding="utf-8", errors="replace"),
            "size": target.stat().st_size,
        }

    def write_session_file(self, session_id: str, rel_path: str, content: str) -> dict[str, Any]:
        sdir = self._session_dir(session_id).resolve()
        target = self.resolve_session_file(session_id, rel_path)
        if target.name == "messages.json" and target.parent == sdir:
            raise ValueError("messages.json は直接編集できません。")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        rel_posix = target.relative_to(sdir).as_posix()
        if rel_posix in ("plan_master.md", "plan.md", "longthink/shared_workspace/master_plan.md"):
            self.sync_plan_files(session_id, content)
        return {
            "session_id": self._sanitize_session_id(session_id),
            "path": rel_posix,
            "content": content,
            "size": target.stat().st_size,
        }

    def get_session_tree(self, session_id: str) -> dict[str, Any]:
        safe_id = self._sanitize_session_id(session_id)
        self._migrate_legacy_json_if_needed(safe_id)
        sdir = self.ensure_session_files(safe_id)
        flat_files: list[dict[str, Any]] = []

        def _walk(dir_path: Path) -> list[dict[str, Any]]:
            nodes: list[dict[str, Any]] = []
            entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            for entry in entries:
                if entry.name in _IGNORED_SCAN_DIRS:
                    continue
                if entry.name == "messages.json" and entry.parent == sdir:
                    continue
                rel = entry.relative_to(sdir).as_posix()
                if entry.is_dir():
                    nodes.append(
                        {
                            "name": entry.name,
                            "path": rel,
                            "type": "dir",
                            "children": _walk(entry),
                        }
                    )
                elif entry.is_file():
                    size = entry.stat().st_size
                    nodes.append(
                        {
                            "name": entry.name,
                            "path": rel,
                            "type": "file",
                            "size": size,
                        }
                    )
                    flat_files.append({"path": rel, "size": size})
            return nodes

        tree_nodes = _walk(sdir)
        ut_index_path = sdir / "ultrathink" / "GLOBAL_INDEX.md"
        master_plan_path = sdir / "longthink" / "shared_workspace" / "master_plan.md"
        plan_master_path = sdir / "plan_master.md"
        session_plan_path = sdir / "plan.md"
        if ut_index_path.exists():
            plan_markdown = ut_index_path.read_text(encoding="utf-8", errors="replace")
        elif master_plan_path.exists():
            plan_markdown = master_plan_path.read_text(encoding="utf-8", errors="replace")
        elif plan_master_path.exists():
            plan_markdown = plan_master_path.read_text(encoding="utf-8", errors="replace")
        elif session_plan_path.exists():
            plan_markdown = session_plan_path.read_text(encoding="utf-8", errors="replace")
        else:
            plan_markdown = ""

        return {
            "session_id": safe_id,
            "tree": tree_nodes,
            "flat_files": flat_files,
            "plan_markdown": plan_markdown,
            "ultrathink": self.get_ultrathink_metrics(safe_id),
        }


class EventBroker:
    """Broadcasts SSE events to all connected browser clients."""

    def __init__(self, max_history: int = 200) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._history: list[dict[str, Any]] = []
        self._max_history = max_history
        self._active_turn_id: str = ""

    def set_active_turn_id(self, turn_id: str) -> None:
        self._active_turn_id = turn_id or ""

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
        payload = dict(data)
        if self._active_turn_id and "turn_id" not in payload:
            payload["turn_id"] = self._active_turn_id
        event = {
            "type": event_type,
            "timestamp": _now_time(),
            "data": payload,
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


def _compose_raw_model_output(raw_content: str, reasoning_extra: str = "") -> str:
    """Preserve unedited raw model output including <think> tags and separate reasoning fields."""
    raw_c = raw_content or ""
    raw_r = reasoning_extra or ""
    if raw_r and "<think>" not in raw_c.lower():
        if raw_c:
            return f"<think>\n{raw_r}\n</think>\n{raw_c}"
        return f"<think>\n{raw_r}\n</think>"
    return raw_c or raw_r


VALID_TOOL_NAMES: set[str] = {
    schema["function"]["name"]
    for schema in (TOOL_SCHEMAS + LONGTHINK_EXTRA_TOOL_SCHEMAS + ULTRATHINK_EXTRA_TOOL_SCHEMAS)
} | {"list_dir"}


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


_DIRECT_WRITE_FILE_RE = re.compile(
    r"\[WRITE_FILE:\s*([^\r\n\]]+?)\s*\]\s*\r?\n\s*```[^\r\n]*\r?\n([\s\S]*?)```",
    re.IGNORECASE,
)
_DIRECT_WRITE_UNCLOSED_RE = re.compile(
    r"\[WRITE_FILE:\s*([^\r\n\]]+?)\s*\]\s*\r?\n\s*```[^\r\n]*\r?\n(?![\s\S]*?```)([\s\S]+)$",
    re.IGNORECASE,
)
_DIRECT_RUN_PYTHON_RE = re.compile(
    r"\[RUN_PYTHON\]\s*\r?\n\s*```(?:python|py)?[^\r\n]*\r?\n([\s\S]*?)```",
    re.IGNORECASE,
)


def _clean_direct_path(raw_path: str) -> str:
    p = (raw_path or "").strip()
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")) or (p.startswith("`") and p.endswith("`")):
        p = p[1:-1].strip()
    return p


def _extract_direct_write_file_calls(
    text: str,
    default_script_path: Optional[str] = None,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Extract non-JSON `[WRITE_FILE: path]` and `[RUN_PYTHON]` fenced code blocks directly from text."""
    if not text or not text.strip():
        return [], None
    _, cleaned = _extract_think_and_content(text)
    target_text = cleaned or text

    calls: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for match in _DIRECT_WRITE_FILE_RE.finditer(target_text):
        fpath = _clean_direct_path(match.group(1))
        code_body = match.group(2)
        if code_body.endswith("\n"):
            pass
        if not fpath:
            continue
        args = {"path": fpath, "content": code_body}
        sig = ("write_file", json.dumps(args, sort_keys=True, ensure_ascii=False))
        if sig not in seen:
            seen.add(sig)
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": "write_file",
                    "arguments": args,
                    "_from_direct_block": True,
                }
            )

    for match in _DIRECT_RUN_PYTHON_RE.finditer(target_text):
        code_body = (match.group(1) or "").strip()
        if not code_body:
            continue
        args = {"code": code_body}
        sig = ("run_python", json.dumps(args, sort_keys=True, ensure_ascii=False))
        if sig not in seen:
            seen.add(sig)
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": "run_python",
                    "arguments": args,
                    "_from_direct_block": True,
                }
            )

    diag: Optional[str] = None
    if not calls and default_script_path:
        orphan_code = _extract_orphan_python_block(target_text)
        if orphan_code is None and target_text != text:
            orphan_code = _extract_orphan_python_block(text)
        if orphan_code:
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": "write_file",
                    "arguments": {"path": default_script_path, "content": orphan_code},
                    "_from_direct_block": True,
                    "_orphan_rescue": True,
                }
            )
    if not calls:
        unclosed = _DIRECT_WRITE_UNCLOSED_RE.search(target_text)
        if unclosed:
            fpath = _clean_direct_path(unclosed.group(1))
            diag = (
                f"[WRITE_FILE: {fpath}] のMarkdownコードブロックが閉じタグ ``` に到達する前に途中切断されました。"
                "完全なコードを出力するかファイルを分割してください。"
            )
    return calls, diag


def _canonical_call_key(tool_name: str, tool_args: dict[str, Any]) -> str:
    try:
        args_dump = json.dumps(tool_args or {}, sort_keys=True, ensure_ascii=False)
    except Exception:
        args_dump = str(tool_args)
    return f"{(tool_name or '').strip()}:{args_dump}"


def _parse_fallback_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse tool calls from text: prioritizes non-JSON [WRITE_FILE: ...] blocks, then <tool_call>/JSON/fn(...)."""
    if not text or not text.strip():
        return []

    direct_calls, _ = _extract_direct_write_file_calls(text)

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

    calls: list[dict[str, Any]] = list(direct_calls)
    seen: set[tuple[str, str]] = {
        (c["name"], json.dumps(c["arguments"], sort_keys=True, ensure_ascii=False)) for c in calls
    }

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


def _parse_response_tool_calls_with_diagnostics(
    msg: Any,
    visible_text: str,
    raw_content: str,
    allowed_names: Optional[set[str]] = None,
    orphan_rescue_path: Optional[str] = None,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Extract direct blocks + OpenAI tool_calls + fallback JSON, returning detailed diagnostics on failure."""
    calls: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    seen_sigs: set[tuple[str, str]] = set()

    def _validate_and_add(cid: str, name: str, args_obj: Any, from_direct: bool = False) -> None:
        if allowed_names is not None and name not in allowed_names:
            return
        if not isinstance(args_obj, dict):
            diagnostics.append(f"ツール `{name}` の引数が辞書オブジェクトではありませんでした。")
            return
        if name == "write_file":
            p_val = str(args_obj.get("path") or "").strip()
            c_val = args_obj.get("content")
            if not p_val or c_val is None:
                diagnostics.append(
                    "`write_file` の必須引数 (`path` または `content`) が欠落・パース失敗しました。"
                    "JSON引数ではなく `[WRITE_FILE: path/to/script.py]` + ```python ... ``` 形式を直接使用してください。"
                )
                return
        elif name == "run_python" and not str(args_obj.get("code") or "").strip():
            diagnostics.append(
                "`run_python` の必須引数 `code` が空またはパース失敗です。"
                "`[WRITE_FILE: script.py]` + ```python ... ``` 形式で先にファイル保存してください。"
            )
            return
        elif name == "run_command" and not str(args_obj.get("command") or "").strip():
            diagnostics.append("`run_command` の必須引数 `command` が空です。")
            return
        elif name == "read_file" and not str(args_obj.get("path") or "").strip():
            diagnostics.append("`read_file` の必須引数 `path` が空です。")
            return

        sig = (name, json.dumps(args_obj, sort_keys=True, ensure_ascii=False))
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            item: dict[str, Any] = {"id": cid, "name": name, "arguments": args_obj}
            if from_direct:
                item["_from_direct_block"] = True
            calls.append(item)

    # 1. Always inspect non-JSON [WRITE_FILE: ...] / [RUN_PYTHON] blocks in response text first
    direct_calls, direct_diag = _extract_direct_write_file_calls(visible_text or raw_content)
    for dc in direct_calls:
        _validate_and_add(dc["id"], dc["name"], dc["arguments"], from_direct=True)
    if direct_diag:
        diagnostics.append(direct_diag)

    # 2. Structured OpenAI tool_calls with explicit JSON error diagnostics
    raw_tool_calls = getattr(msg, "tool_calls", None) or []
    for tc in raw_tool_calls:
        fn = getattr(tc, "function", None)
        t_name = getattr(fn, "name", "") or ""
        t_raw_args = getattr(fn, "arguments", "") or ""
        tc_id = getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:8]}"
        if not t_raw_args.strip():
            parsed_args: Any = {}
        else:
            try:
                parsed_args = json.loads(t_raw_args)
            except json.JSONDecodeError as exc:
                tail_snip = t_raw_args[-140:].replace("\n", "\\n")
                diagnostics.append(
                    f"ツール `{t_name}` の JSON 引数パースに失敗しました ({exc.msg}, pos={exc.pos}, 末尾=`{tail_snip}`)。"
                    "長いPythonコードをJSON文字列にエスケープして詰め込むと改行や引用符で破損します。"
                    "代わりに `[WRITE_FILE: path/to/script.py]` の直下に ```python コードブロックを直接記述してください。"
                )
                continue
        _validate_and_add(tc_id, t_name, parsed_args)

    # 3. Text fallback JSON / fn(...) if still no valid structured calls
    if not calls and not raw_tool_calls:
        fb_calls = _parse_fallback_tool_calls(visible_text)
        if not fb_calls and not visible_text:
            fb_calls = _parse_fallback_tool_calls(raw_content)
        for fc in fb_calls:
            _validate_and_add(fc["id"], fc["name"], fc["arguments"], from_direct=bool(fc.get("_from_direct_block")))

        if not calls and not diagnostics:
            probe_txt = visible_text or raw_content
            if "<tool_call>" in probe_txt or '"name"' in probe_txt or "[WRITE_FILE:" in probe_txt:
                diagnostics.append(
                    "本文内のツール呼び出しまたは `[WRITE_FILE: ...]` ブロックの形式が不正・途中切断のため解析できませんでした。"
                    "`[WRITE_FILE: path/to/script.py]` の直後に ```python ... ``` 形式で完全なコードを記述してください。"
                )

    if not calls and orphan_rescue_path and (allowed_names is None or "write_file" in allowed_names):
        orphan_code = _extract_orphan_python_block(visible_text or raw_content)
        if orphan_code is None and raw_content and raw_content != visible_text:
            orphan_code = _extract_orphan_python_block(raw_content)
        if orphan_code:
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": "write_file",
                    "arguments": {"path": orphan_rescue_path, "content": orphan_code},
                    "_from_direct_block": True,
                    "_orphan_rescue": True,
                }
            )
            diagnostics.clear()

    diag_str = "\n".join(diagnostics) if diagnostics else None
    return calls, diag_str


_ORPHAN_PY_BLOCK_RE = re.compile(r"```(?:python|py)[^\r\n]*\r?\n([\s\S]*?)(?:```|$)", re.IGNORECASE)


def _extract_orphan_python_block(text: str) -> Optional[str]:
    """Return code of the first fenced python block (used when `[WRITE_FILE: ...]` header was forgotten)."""
    if not text:
        return None
    _, cleaned = _extract_think_and_content(text)
    target = cleaned or text
    m = _ORPHAN_PY_BLOCK_RE.search(target)
    if not m:
        return None
    code = m.group(1)
    if not code.strip():
        return None
    return code if code.endswith("\n") else code + "\n"


@dataclass
class AgentState:
    status: str = "idle"  # idle | running | stopping | stopped | completed | error
    session_id: Optional[str] = None
    turn_id: str = ""
    current_step: int = 0
    max_steps: int = 0  # 0 = unlimited autonomous loop
    model: str = ""
    supervisor_model: str = DEFAULT_SUPERVISOR_MODEL
    coder_model: str = DEFAULT_CODER_MODEL
    standard_model: str = DEFAULT_STANDARD_MODEL
    parallel: int = DEFAULT_MODEL_PARALLEL
    base_url: str = DEFAULT_LM_STUDIO_URL
    phase_state: str = "idle"  # idle | thinking | working
    phase_label: str = ""
    phase_started_at: float = 0.0
    harness_state: str = "IDLE"
    error_message: Optional[str] = None
    ultrathink_info: Optional[dict[str, Any]] = None


class AgentHarness:
    """Manages chat turns, autonomous tool loops, memory consolidation, and SSE events."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir.resolve()
        self.workspace_root = self.base_dir / "workspace"
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.memory = MemoryManager(self.base_dir / "memory")
        self.sessions = SessionManager(self.base_dir / "sessions")
        self.broker = EventBroker()
        self.state = AgentState(
            model=self._load_configured_main_model(),
            supervisor_model=self._load_configured_supervisor_model(),
            coder_model=self._load_configured_coder_model(),
            standard_model=self._load_configured_standard_model(),
        )
        self._stop_event = asyncio.Event()
        self._run_task: Optional[asyncio.Task[None]] = None
        self._handoff_lock = asyncio.Lock()
        self._persist_lock = asyncio.Lock()
        self._llm_parallel_semaphore = asyncio.Semaphore(DEFAULT_MODEL_PARALLEL)
        self.active_parallel_batch: Optional[dict[str, Any]] = None
        self.live_thinking_log: list[dict[str, Any]] = []
        self.last_completed_message: Optional[dict[str, Any]] = None

    def _load_configured_main_model(self) -> str:
        env_val = os.environ.get("CRYSTALLM_MAIN_MODEL", "").strip()
        if env_val:
            return env_val
        val = str(_read_config_json(self.base_dir).get("main_model") or "").strip()
        return val

    def _save_configured_models(self, **updates: str) -> None:
        cfg_path = self.base_dir / "config.json"
        data: dict[str, Any] = _read_config_json(self.base_dir)
        for key, val in updates.items():
            cleaned = (val or "").strip()
            if cleaned:
                data[key] = cleaned
            else:
                data.pop(key, None)
        try:
            cfg_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_configured_supervisor_model(self) -> str:
        env_val = os.environ.get("CRYSTALLM_SUPERVISOR_MODEL", "").strip()
        if env_val:
            return env_val
        cfg_path = self.base_dir / "config.json"
        if cfg_path.exists():
            try:
                data = json.loads(cfg_path.read_text(encoding="utf-8", errors="replace"))
                if isinstance(data, dict) and isinstance(data.get("supervisor_model"), str):
                    val = data["supervisor_model"].strip()
                    if val:
                        return val
            except Exception:
                pass
        return DEFAULT_SUPERVISOR_MODEL

    def _load_configured_coder_model(self) -> str:
        env_val = os.environ.get("CRYSTALLM_CODER_MODEL", "").strip()
        if env_val:
            return env_val
        cfg_path = self.base_dir / "config.json"
        if cfg_path.exists():
            try:
                data = json.loads(cfg_path.read_text(encoding="utf-8", errors="replace"))
                if isinstance(data, dict) and isinstance(data.get("coder_model"), str):
                    val = data["coder_model"].strip()
                    if val:
                        return val
            except Exception:
                pass
        return DEFAULT_CODER_MODEL

    def _load_configured_standard_model(self) -> str:
        env_val = os.environ.get("CRYSTALLM_STANDARD_MODEL", "").strip()
        if env_val:
            return env_val
        cfg_path = self.base_dir / "config.json"
        if cfg_path.exists():
            try:
                data = json.loads(cfg_path.read_text(encoding="utf-8", errors="replace"))
                if isinstance(data, dict) and isinstance(data.get("standard_model"), str):
                    val = data["standard_model"].strip()
                    if val:
                        return val
            except Exception:
                pass
        return DEFAULT_STANDARD_MODEL

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
            "turn_id": self.state.turn_id if matches_session else "",
            "status": self.state.status if matches_session else "idle",
            "session_id": self.state.session_id,
            "step": self.state.current_step if matches_session else 0,
            "current_step": self.state.current_step if matches_session else 0,
            "max_steps": self.state.max_steps,
            "state": self.state.phase_state if matches_session else "idle",
            "phase_state": self.state.phase_state if matches_session else "idle",
            "label": self.state.phase_label if matches_session else "",
            "phase_label": self.state.phase_label if matches_session else "",
            "harness_state": self.state.harness_state if matches_session else "IDLE",
            "role": "assistant",
            "elapsed_seconds": elapsed if matches_session else 0.0,
            "model": self.state.model,
            "supervisor_model": self.state.supervisor_model,
            "coder_model": self.state.coder_model,
            "standard_model": self.state.standard_model,
            "parallel": self.state.parallel,
            "error_message": self.state.error_message if matches_session else None,
            "thinking_log": list(self.live_thinking_log) if matches_session else [],
            "parallel_batch": self.active_parallel_batch if matches_session else None,
            "ultrathink": self.state.ultrathink_info if matches_session else None,
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
        if assistant_msg_id:
            for msg in reversed(session.get("messages", [])):
                if msg.get("id") == assistant_msg_id:
                    target_entry = msg
                    break
        else:
            for msg in reversed(session.get("messages", [])):
                if msg.get("role") == "assistant" and msg.get("in_progress"):
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
        max_steps: Optional[int] = None,
        model: Optional[str] = None,
        supervisor_model: Optional[str] = None,
        coder_model: Optional[str] = None,
        standard_model: Optional[str] = None,
        base_url: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> dict[str, Any]:
        if self.is_running():
            raise RuntimeError("現在別の推論が実行中です。先に停止してください。")

        # Reset all previous turn buffers immediately before any await (prevents stale /api/live leaks)
        assistant_msg_id = f"msg_{uuid.uuid4().hex[:8]}"
        self._stop_event.clear()
        self.broker.clear_history()
        self.broker.set_active_turn_id(assistant_msg_id)
        self.active_parallel_batch = None
        self.live_thinking_log = []
        self.last_completed_message = None
        self.state.status = "running"
        self.state.turn_id = assistant_msg_id
        self.state.current_step = 0
        self.state.phase_state = "thinking"
        self.state.phase_label = "モデル・セッションを準備中"
        self.state.phase_started_at = time.monotonic()
        self.state.error_message = None

        normalized_url = _normalize_base_url(base_url or DEFAULT_LM_STUDIO_URL)
        try:
            lm_status = await check_lm_studio(normalized_url)
            if not lm_status["connected"]:
                raise ConnectionError(
                    f"LM Studio に接続できません ({normalized_url})。"
                    "LM Studio でローカルサーバーを起動してから再実行してください。"
                )
        except Exception as exc:
            self.state.status = "error"
            self.state.phase_state = "idle"
            self.state.phase_label = ""
            self.state.phase_started_at = 0.0
            self.state.error_message = str(exc)
            raise

        supervisor_hint = (supervisor_model or "").strip() or self._load_configured_supervisor_model()
        coder_hint = (coder_model or "").strip() or self._load_configured_coder_model()
        standard_hint = (standard_model or "").strip() or self._load_configured_standard_model()
        available_models = lm_status.get("models") or []
        raw_model = (model or "").strip()
        use_standard_as_main = raw_model == NO_MAIN_MODEL_SENTINEL or (
            not raw_model and self._load_configured_main_model() == NO_MAIN_MODEL_SENTINEL
        )
        if use_standard_as_main:
            selected_standard = resolve_standard_model_id(
                standard_hint,
                available_models,
                main_model="",
            )
            selected_model = selected_standard
            selected_supervisor = resolve_supervisor_model_id(
                supervisor_hint,
                available_models,
                main_model="",
            )
            selected_coder = resolve_coder_model_id(
                coder_hint,
                available_models,
                main_model="",
            )
        else:
            selected_model = raw_model
            if not selected_model:
                main_candidates = [
                    m
                    for m in available_models
                    if not _is_supervisor_model_id(m, supervisor_hint)
                    and not _is_coder_model_id(m, coder_hint)
                    and not _is_standard_model_id(m, standard_hint)
                ]
                if not main_candidates:
                    main_candidates = [
                        m
                        for m in available_models
                        if not _is_supervisor_model_id(m, supervisor_hint) and not _is_coder_model_id(m, coder_hint)
                    ]
                selected_model = (
                    main_candidates[0]
                    if main_candidates
                    else (available_models[0] if available_models else "local-model")
                )
            selected_supervisor = resolve_supervisor_model_id(
                supervisor_hint,
                available_models,
                main_model=selected_model,
            )
            selected_coder = resolve_coder_model_id(
                coder_hint,
                available_models,
                main_model=selected_model,
            )
            selected_standard = resolve_standard_model_id(
                standard_hint,
                available_models,
                main_model=selected_model,
            )

        stripped_msg = user_message.lstrip()
        detected_mode: Optional[str] = None
        if stripped_msg.startswith("/fastthink"):
            detected_mode = "fastthink"
        elif stripped_msg.startswith("/ultrathink"):
            detected_mode = "ultrathink"
        elif stripped_msg.startswith("/longthink"):
            detected_mode = "longthink"
        elif stripped_msg.startswith("/normal"):
            detected_mode = "normal"

        requested_mode = detected_mode or (
            mode if mode in ("fastthink", "normal", "longthink", "ultrathink") else None
        )

        if session_id:
            try:
                session = self.sessions.get_session(session_id)
                if requested_mode:
                    session["mode"] = requested_mode
            except FileNotFoundError:
                session = self.sessions.create_session(mode=requested_mode or "normal")
        else:
            session = self.sessions.create_session(mode=requested_mode or "normal")

        active_mode = session.get("mode", "normal")
        if active_mode == "fastthink":
            ut_metrics = None
            resolved_max_steps = 1
        elif active_mode == "longthink":
            self.sessions.ensure_longthink_structure(session["id"])
            ut_metrics = None
            resolved_max_steps = 0  # Unlimited autonomous steps
        elif active_mode == "ultrathink":
            self.sessions.ensure_ultrathink_structure(session["id"], user_prompt=user_message)
            ut_metrics = self.sessions.get_ultrathink_metrics(session["id"])
            resolved_max_steps = 0  # Unlimited autonomous steps
        else:
            ut_metrics = None
            resolved_max_steps = 0  # Unlimited autonomous steps

        # Auto-title new session from first user message
        if not session["messages"] and (
            session.get("title") in ("新しいチャット", "", None)
        ):
            short_title = user_message.strip().splitlines()[0][:36]
            session["title"] = short_title + ("..." if len(user_message.strip()) > 36 else "")

        for prev_msg in session.get("messages", []):
            prev_msg.pop("in_progress", None)

        user_entry = {
            "id": f"msg_{uuid.uuid4().hex[:8]}",
            "role": "user",
            "content": user_message,
            "timestamp": _now_time(),
        }
        session["messages"].append(user_entry)
        self._persist_progress(
            session,
            assistant_msg_id,
            [],
            latest_content="（推論・作業を実行中...）",
            in_progress=True,
        )

        self._stop_event.clear()
        self.broker.clear_history()
        self.broker.set_active_turn_id(assistant_msg_id)
        self.active_parallel_batch = None
        self.live_thinking_log = []
        self.last_completed_message = None
        init_label = (
            "[FASTTHINK] 高速単一応答を生成中"
            if active_mode == "fastthink"
            else ("[ULTRATHINK] 初期仮説を立案中" if active_mode == "ultrathink" else "タスクを分析・推論中")
        )
        init_harness_state = (
            "FASTTHINK"
            if active_mode == "fastthink"
            else ("ULTRATHINK_HIGH" if active_mode == "ultrathink" else STATE_INIT_PLAN)
        )
        self.state = AgentState(
            status="running",
            session_id=session["id"],
            turn_id=assistant_msg_id,
            current_step=0,
            max_steps=resolved_max_steps,
            model=selected_model,
            supervisor_model=selected_supervisor,
            coder_model=selected_coder,
            standard_model=selected_standard,
            base_url=normalized_url,
            phase_state="thinking",
            phase_label=init_label,
            phase_started_at=time.monotonic(),
            harness_state=init_harness_state,
            ultrathink_info=(
                {
                    "gear": UT_GEAR_HIGH,
                    "temperature": 0.3,
                    "focus_files": [],
                    "last_commit": None,
                    **ut_metrics,
                }
                if active_mode == "ultrathink" and ut_metrics is not None
                else None
            ),
        )

        await self.broker.publish("status", self._status_payload())
        await self.broker.publish("session_update", {"session": self._session_summary(session)})
        await self._on_files_update()

        self._run_task = asyncio.create_task(
            self._run_turn(session["id"], user_message, assistant_msg_id)
        )
        return {
            "agent": self._status_payload(),
            "session_id": session["id"],
            "turn_id": assistant_msg_id,
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
            "turn_id": self.state.turn_id,
            "current_step": self.state.current_step,
            "max_steps": self.state.max_steps,
            "state": self.state.phase_state,
            "label": self.state.phase_label,
            "elapsed_seconds": elapsed,
            "model": self.state.model,
            "supervisor_model": self.state.supervisor_model,
            "coder_model": self.state.coder_model,
            "standard_model": self.state.standard_model,
            "parallel": self.state.parallel,
            "base_url": self.state.base_url,
            "harness_state": self.state.harness_state,
            "error_message": self.state.error_message,
            "parallel_batch": self.active_parallel_batch,
            "ultrathink": self.state.ultrathink_info,
        }

    def _session_summary(self, session: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": session["id"],
            "title": session.get("title", "無題のチャット"),
            "mode": session.get("mode", "normal"),
            "created_at": session.get("created_at", ""),
            "updated_at": session.get("updated_at", ""),
            "message_count": len(session.get("messages", [])),
        }

    def list_workspace_files(self, session_id: Optional[str] = None) -> list[dict[str, Any]]:
        sid = session_id or self.state.session_id
        if sid:
            try:
                return self.sessions.get_session_tree(sid).get("flat_files", [])
            except Exception:
                pass
        files: list[dict[str, Any]] = []
        if not self.workspace_root.exists():
            return files
        for path in sorted(self.workspace_root.rglob("*"), key=lambda p: p.as_posix().lower()):
            if any(part in _IGNORED_SCAN_DIRS for part in path.parts):
                continue
            if path.is_file():
                rel = path.relative_to(self.workspace_root).as_posix()
                files.append({"path": rel, "size": path.stat().st_size})
        return files

    async def _on_memory_update(self, filename: str, content: str) -> None:
        await self.broker.publish(
            "memory_update",
            {"filename": filename, "content": content, "session_id": self.state.session_id},
        )

    async def _on_files_update(self) -> None:
        sid = self.state.session_id
        payload: dict[str, Any] = {
            "files": self.list_workspace_files(sid),
            "session_id": sid,
        }
        if sid:
            try:
                payload["tree"] = self.sessions.get_session_tree(sid)
            except Exception:
                pass
        await self.broker.publish("files_update", payload)

    async def select_model(
        self,
        model_id: str,
        base_url: Optional[str] = None,
        parallel: int = DEFAULT_MODEL_PARALLEL,
        role: str = "main",
        previous_model: Optional[str] = None,
        keep_models: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        normalized = _normalize_base_url(base_url or self.state.base_url or DEFAULT_LM_STUDIO_URL)
        self.state.base_url = normalized
        role_key = (role or "main").strip().lower()
        slot_attr_map = {
            "main": ("model", "main_model"),
            "supervisor": ("supervisor_model", "supervisor_model"),
            "coder": ("coder_model", "coder_model"),
            "standard": ("standard_model", "standard_model"),
        }
        if role_key not in slot_attr_map:
            role_key = "main"
        state_attr, cfg_key = slot_attr_map[role_key]

        prev_slot_model = (previous_model or "").strip() or str(getattr(self.state, state_attr, "") or "").strip()
        if prev_slot_model == NO_MAIN_MODEL_SENTINEL:
            prev_slot_model = ""
        is_no_main_mode = role_key == "main" and (model_id or "").strip() == NO_MAIN_MODEL_SENTINEL
        if keep_models is not None:
            eff_keep = [
                str(m).strip()
                for m in keep_models
                if isinstance(m, str) and str(m).strip() and str(m).strip() != NO_MAIN_MODEL_SENTINEL
            ]
        else:
            eff_keep = [
                str(getattr(self.state, attr, "") or "").strip()
                for r_name, (attr, _) in slot_attr_map.items()
                if r_name != role_key
                and str(getattr(self.state, attr, "") or "").strip()
                and str(getattr(self.state, attr, "") or "").strip() != NO_MAIN_MODEL_SENTINEL
            ]

        result = await select_lm_studio_model(
            "" if is_no_main_mode else model_id,
            normalized,
            parallel=DEFAULT_MODEL_PARALLEL,
            role=role_key,
            previous_model=prev_slot_model,
            keep_models=eff_keep,
        )
        if result.get("ok"):
            chosen_model = NO_MAIN_MODEL_SENTINEL if is_no_main_mode else str(result.get("model", "") or "").strip()
            setattr(self.state, state_attr, chosen_model)
            self.state.parallel = DEFAULT_MODEL_PARALLEL
            self._save_configured_models(**{cfg_key: chosen_model})
            if is_no_main_mode:
                result["model"] = NO_MAIN_MODEL_SENTINEL
                result["message"] = "モデル選択なし（LFM3 1.7B 兼用）に設定しました"
            result["role"] = role_key
            result["main_model"] = self.state.model
            result["supervisor_model"] = self.state.supervisor_model
            result["coder_model"] = self.state.coder_model
            result["standard_model"] = self.state.standard_model
        return result

    async def reload_models(
        self,
        base_url: Optional[str] = None,
        main_model: Optional[str] = None,
        supervisor_model: Optional[str] = None,
        coder_model: Optional[str] = None,
        standard_model: Optional[str] = None,
    ) -> dict[str, Any]:
        normalized = _normalize_base_url(base_url or self.state.base_url or DEFAULT_LM_STUDIO_URL)
        self.state.base_url = normalized

        lm_info = await check_lm_studio(normalized)
        avail_models = lm_info.get("models") or []

        eff_main = (main_model if main_model is not None else self.state.model or "").strip()
        eff_sup = (
            (supervisor_model or "").strip()
            or lm_info.get("detected_supervisor_model")
            or self.state.supervisor_model
            or DEFAULT_SUPERVISOR_MODEL
        )
        eff_coder = (
            (coder_model or "").strip()
            or lm_info.get("detected_coder_model")
            or self.state.coder_model
            or DEFAULT_CODER_MODEL
        )
        eff_std = (
            (standard_model or "").strip()
            or lm_info.get("detected_standard_model")
            or self.state.standard_model
            or DEFAULT_STANDARD_MODEL
        )

        if not eff_main and avail_models:
            main_candidates = [
                m
                for m in avail_models
                if not _is_supervisor_model_id(m, eff_sup)
                and not _is_coder_model_id(m, eff_coder)
                and not _is_standard_model_id(m, eff_std)
            ]
            eff_main = main_candidates[0] if main_candidates else ""

        if avail_models:
            main_for_resolve = "" if eff_main == NO_MAIN_MODEL_SENTINEL else eff_main
            if eff_sup and eff_sup not in avail_models:
                eff_sup = resolve_supervisor_model_id(eff_sup, avail_models, main_model=main_for_resolve)
            if eff_coder and eff_coder not in avail_models:
                eff_coder = resolve_coder_model_id(eff_coder, avail_models, main_model=main_for_resolve)
            if eff_std and eff_std not in avail_models:
                eff_std = resolve_standard_model_id(eff_std, avail_models, main_model=main_for_resolve)

        self.state.model = eff_main
        self.state.supervisor_model = eff_sup
        self.state.coder_model = eff_coder
        self.state.standard_model = eff_std
        self.state.parallel = DEFAULT_MODEL_PARALLEL
        self._save_configured_models(
            main_model=eff_main,
            supervisor_model=eff_sup,
            coder_model=eff_coder,
            standard_model=eff_std,
        )

        avail_set = set(avail_models)
        load_plan: list[tuple[str, str]] = []
        for r_name, mid in (
            ("main", eff_main),
            ("supervisor", eff_sup),
            ("coder", eff_coder),
            ("standard", eff_std),
        ):
            if mid and mid != NO_MAIN_MODEL_SENTINEL and (not avail_set or mid in avail_set):
                load_plan.append((r_name, mid))

        return await reload_all_lm_studio_models(load_plan, normalized)

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
        session_dir: Optional[Path] = None,
        is_longthink: bool = False,
    ) -> list[dict[str, Any]]:
        """Build context-bounded messages injecting global + session-isolated memory files."""
        session_plan: Optional[str] = None
        session_work_log: Optional[str] = None
        handoff_block = ""

        if session_dir is not None:
            if is_longthink:
                mp_path = session_dir / "longthink" / "shared_workspace" / "master_plan.md"
                hb_path = session_dir / "longthink" / "shared_workspace" / "handoff_board.md"
                if mp_path.exists():
                    session_plan = mp_path.read_text(encoding="utf-8", errors="replace")
                if hb_path.exists():
                    hb_text = hb_path.read_text(encoding="utf-8", errors="replace").strip()
                    if len(hb_text) > 2000:
                        hb_text = "... (省略)\n" + hb_text[-2000:]
                    handoff_block = (
                        "\n\n### shared_workspace/handoff_board.md\n```markdown\n"
                        + hb_text
                        + "\n```"
                    )
            if session_plan is None and (session_dir / "plan.md").exists():
                session_plan = (session_dir / "plan.md").read_text(encoding="utf-8", errors="replace")
            if (session_dir / "work_log.md").exists():
                session_work_log = (session_dir / "work_log.md").read_text(encoding="utf-8", errors="replace")

        base_prompt = SYSTEM_PROMPT + (LONGTHINK_SYSTEM_ADDON if is_longthink else "")
        system_content = (
            base_prompt
            + self.memory.build_context_block(session_plan=session_plan, session_work_log=session_work_log)
            + handoff_block
        )
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

    @classmethod
    def _compact_messages_for_retry(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compact intermediate turns when retrying after a timeout without breaking role pairing."""
        if len(messages) <= 2:
            return list(messages)
        compacted = list(messages[:2])
        last_msg = messages[-1]
        if last_msg.get("role") in ("user", "assistant") and last_msg is not messages[1]:
            compacted.append(last_msg)
        return cls._merge_consecutive_roles(compacted)

    async def _consume_chat_stream(
        self,
        stream: Any,
        check_stop: bool = True,
        max_think_tokens: Optional[int] = THINK_TOKEN_LIMIT_DEFAULT,
    ) -> Any:
        """Consume an async OpenAI streaming response counting ONLY <think>/reasoning tokens for cutoff."""
        if not hasattr(stream, "__aiter__"):
            return stream

        eff_think_cap: Optional[int] = None if max_think_tokens is None else max(0, int(max_think_tokens))

        content_parts: list[str] = []
        reasoning_content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_map: dict[int, _StreamToolCall] = {}
        content_tail = ""
        repeat_content_tail = ""
        repeat_reasoning_tail = ""
        think_seen = False
        in_think_tag = False
        rc_r_think_tokens = 0
        content_think_tokens = 0
        has_visible_outside_think = False
        think_tokens_est = 0
        think_truncated = False
        repeat_aborted = False
        repeated_phrase = ""

        try:
            async for chunk in stream:
                if check_stop and self._stop_event.is_set():
                    raise asyncio.CancelledError()
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue

                d_content = getattr(delta, "content", None)
                if d_content:
                    content_parts.append(d_content)
                    repeat_content_tail = (repeat_content_tail + d_content)[-1200:]
                    d_ctok = _estimate_think_tokens(d_content)
                    combined_tail = content_tail + d_content
                    lower_tail = combined_tail.lower()
                    if not think_seen:
                        t_idx = lower_tail.find("<think>")
                        if t_idx != -1:
                            think_seen = True
                            after_t = combined_tail[t_idx + 7 :]
                            c_idx = after_t.lower().find("</think>")
                            if c_idx != -1:
                                content_think_tokens += _estimate_think_tokens(after_t[:c_idx])
                                in_think_tag = False
                                if after_t[c_idx + 8 :].strip():
                                    has_visible_outside_think = True
                            else:
                                content_think_tokens += _estimate_think_tokens(after_t)
                                in_think_tag = True
                        elif d_content.strip():
                            has_visible_outside_think = True
                    elif in_think_tag:
                        c_idx = lower_tail.find("</think>")
                        if c_idx != -1:
                            chunk_c = d_content.lower().find("</think>")
                            if chunk_c != -1:
                                content_think_tokens += _estimate_think_tokens(d_content[:chunk_c])
                                if d_content[chunk_c + 8 :].strip():
                                    has_visible_outside_think = True
                            in_think_tag = False
                        else:
                            content_think_tokens += d_ctok
                    else:
                        if d_content.strip():
                            has_visible_outside_think = True
                    content_tail = combined_tail[-16:]

                d_rc = getattr(delta, "reasoning_content", None)
                if d_rc:
                    reasoning_content_parts.append(d_rc)
                    repeat_reasoning_tail = (repeat_reasoning_tail + d_rc)[-1200:]
                    rc_r_think_tokens += _estimate_think_tokens(d_rc)

                d_r = getattr(delta, "reasoning", None)
                if d_r:
                    reasoning_parts.append(d_r)
                    repeat_reasoning_tail = (repeat_reasoning_tail + d_r)[-1200:]
                    rc_r_think_tokens += _estimate_think_tokens(d_r)

                d_tcs = getattr(delta, "tool_calls", None)
                if d_tcs:
                    for tc_delta in d_tcs:
                        idx = getattr(tc_delta, "index", 0) or 0
                        entry = tool_calls_map.setdefault(
                            idx,
                            _StreamToolCall(id=f"call_{uuid.uuid4().hex[:8]}", function=_StreamFunction()),
                        )
                        tc_id = getattr(tc_delta, "id", None)
                        if tc_id:
                            entry.id = tc_id
                        fn_delta = getattr(tc_delta, "function", None)
                        if fn_delta is not None:
                            fn_name = getattr(fn_delta, "name", None)
                            if fn_name:
                                entry.function.name += fn_name
                            fn_args = getattr(fn_delta, "arguments", None)
                            if fn_args:
                                entry.function.arguments += fn_args

                # Real-time degenerate repeat guard: immediately break stream on 3x consecutive phrase repeat
                rep_hit = None
                if not tool_calls_map:
                    if d_content:
                        rep_hit = _detect_consecutive_phrase_repeat(repeat_content_tail)
                    if not rep_hit and (d_rc or d_r):
                        rep_hit = _detect_consecutive_phrase_repeat(repeat_reasoning_tail)
                if rep_hit:
                    repeat_aborted = True
                    repeated_phrase = rep_hit[:160]
                    print(
                        f"[Degenerate Repeat Abort] Detected 3x consecutive phrase ({len(rep_hit)} chars); closing stream.",
                        flush=True,
                    )
                    break

                # Monitor ONLY thinking tokens inside <think> / reasoning fields (never truncate action/code output)
                total_think_tokens = rc_r_think_tokens + content_think_tokens
                think_tokens_est = total_think_tokens
                still_thinking = bool(
                    (reasoning_content_parts or reasoning_parts) and not has_visible_outside_think
                ) or in_think_tag

                if eff_think_cap is not None and still_thinking and not tool_calls_map:
                    hit_cap = (eff_think_cap == 0 and total_think_tokens >= 24) or (
                        eff_think_cap > 0 and total_think_tokens >= eff_think_cap
                    )
                    if hit_cap:
                        think_truncated = True
                        break
        finally:
            close_fn = getattr(stream, "close", None)
            if callable(close_fn):
                try:
                    res = close_fn()
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:
                    pass

        assembled_calls = [tool_calls_map[k] for k in sorted(tool_calls_map)] if tool_calls_map else None
        return _StreamChatCompletion(
            choices=[
                _StreamChoice(
                    index=0,
                    message=_StreamMessage(
                        content="".join(content_parts),
                        reasoning_content="".join(reasoning_content_parts),
                        reasoning="".join(reasoning_parts),
                        tool_calls=assembled_calls,
                        think_truncated=think_truncated,
                        think_tokens=think_tokens_est,
                        repeat_aborted=repeat_aborted,
                        repeated_phrase=repeated_phrase,
                    ),
                )
            ]
        )

    async def _invoke_chat_once(
        self,
        client: AsyncOpenAI,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        temperature: float,
        check_stop: bool,
        allow_tool_drop: bool = True,
        model_override: Optional[str] = None,
        max_think_tokens: Optional[int] = THINK_TOKEN_LIMIT_DEFAULT,
        max_output_tokens: Optional[int] = MAX_TOKENS_NORMAL_OUTPUT,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> Any:
        """Issue streaming chat completion with separate thinking cap (3000) & generous output max_tokens (16384 or -1)."""
        target_model = (model_override or "").strip() or self.state.model
        if max_output_tokens == MAX_TOKENS_DIRECT_BURST:
            eff_output_cap = MAX_TOKENS_DIRECT_BURST
        else:
            eff_output_cap = max(1, int(max_output_tokens or MAX_TOKENS_NORMAL_OUTPUT))
        eff_think_cap = 0 if max_think_tokens == 0 else max(1, int(max_think_tokens or THINK_TOKEN_LIMIT_DEFAULT))

        async def _run_raw_call(
            call_msgs: list[dict[str, Any]],
            call_tools: Optional[list[dict[str, Any]]],
            limit_think: int,
            limit_output: int,
            freq_penalty: Optional[float] = None,
            pres_penalty: Optional[float] = None,
        ) -> Any:
            kwargs: dict[str, Any] = {
                "model": target_model,
                "messages": call_msgs,
                "temperature": temperature,
                "max_tokens": limit_output,
            }
            if freq_penalty is not None:
                kwargs["frequency_penalty"] = freq_penalty
            if pres_penalty is not None:
                kwargs["presence_penalty"] = pres_penalty
            if call_tools:
                kwargs["tools"] = call_tools
                kwargs["tool_choice"] = "auto"

            async with self._llm_parallel_semaphore:
                try:
                    stream = await client.chat.completions.create(**kwargs, stream=True)
                    return await self._consume_chat_stream(
                        stream,
                        check_stop=check_stop,
                        max_think_tokens=limit_think,
                    )
                except BadRequestError:
                    # Strip repetition penalties first if backend schema rejects them
                    active_kwargs = dict(kwargs)
                    if "frequency_penalty" in active_kwargs or "presence_penalty" in active_kwargs:
                        active_kwargs.pop("frequency_penalty", None)
                        active_kwargs.pop("presence_penalty", None)
                        try:
                            stream = await client.chat.completions.create(**active_kwargs, stream=True)
                            return await self._consume_chat_stream(
                                stream,
                                check_stop=check_stop,
                                max_think_tokens=limit_think,
                            )
                        except BadRequestError:
                            pass
                    # If backend rejects max_tokens=-1, retry once without max_tokens before dropping tools
                    if limit_output == MAX_TOKENS_DIRECT_BURST:
                        no_limit_kwargs = {k: v for k, v in active_kwargs.items() if k != "max_tokens"}
                        try:
                            stream = await client.chat.completions.create(**no_limit_kwargs, stream=True)
                            return await self._consume_chat_stream(
                                stream,
                                check_stop=check_stop,
                                max_think_tokens=limit_think,
                            )
                        except BadRequestError:
                            pass
                    try:
                        return await client.chat.completions.create(**active_kwargs)
                    except BadRequestError:
                        if call_tools and allow_tool_drop:
                            no_tool_kwargs = {
                                "model": target_model,
                                "messages": call_msgs,
                                "temperature": temperature,
                                "max_tokens": limit_output,
                            }
                            if limit_output == MAX_TOKENS_DIRECT_BURST:
                                no_tool_kwargs.pop("max_tokens", None)
                            try:
                                stream = await client.chat.completions.create(**no_tool_kwargs, stream=True)
                                return await self._consume_chat_stream(
                                    stream,
                                    check_stop=check_stop,
                                    max_think_tokens=limit_think,
                                )
                            except BadRequestError:
                                return await client.chat.completions.create(**no_tool_kwargs)
                        raise

        try:
            resp = await _run_raw_call(
                messages,
                tools,
                eff_think_cap,
                eff_output_cap,
                freq_penalty=frequency_penalty,
                pres_penalty=presence_penalty,
            )
        except Exception:
            if target_model != self.state.model:
                target_model = self.state.model
                resp = await _run_raw_call(
                    messages,
                    tools,
                    eff_think_cap,
                    eff_output_cap,
                    freq_penalty=frequency_penalty,
                    pres_penalty=presence_penalty,
                )
            else:
                raise

        msg = resp.choices[0].message
        raw_c = getattr(msg, "content", "") or ""
        _, vis_c = _extract_think_and_content(raw_c)
        already_calls, _ = _parse_response_tool_calls_with_diagnostics(msg, vis_c, raw_c)
        was_repeat_aborted = bool(getattr(msg, "repeat_aborted", False))
        was_think_truncated = bool(getattr(msg, "think_truncated", False))
        if (was_repeat_aborted or was_think_truncated) and not already_calls:
            partial_parts = [
                getattr(msg, "reasoning_content", "") or "",
                getattr(msg, "reasoning", "") or "",
            ]
            t_in_c, _ = _extract_think_and_content(raw_c)
            if t_in_c:
                partial_parts.append(t_in_c)
            elif raw_c:
                partial_parts.append(raw_c)
            combined_partial = "\n".join(p for p in partial_parts if p).strip()
            thought_summary = _compress_thought_summary(combined_partial)

            active_tool_names = {
                str((t.get("function") or {}).get("name") or "").strip()
                for t in (tools or [])
                if isinstance(t, dict)
            }
            ctx_head = " ".join(str(m.get("content") or "") for m in messages[:2])
            coding_phase_prefill = (
                "[WRITE_FILE:" in ctx_head
                or "非JSON" in ctx_head
                or "10_active_workspace" in ctx_head
            )
            coder_forced_prefill = _is_coder_model_id(target_model, self.state.coder_model)
            prefer_write_prefill = coder_forced_prefill or coding_phase_prefill or ("write_file" in active_tool_names)
            clean_prefill = (
                "<think>\n思考完了。直ちにファイルを作成する。\n</think>\n[WRITE_FILE: "
                if prefer_write_prefill
                else "<think>\n思考完了。直ちに結果を出力する。\n</think>\n"
            )

            if was_repeat_aborted:
                rep_snip = (getattr(msg, "repeated_phrase", "") or "")[:120]
                clean_hint = (
                    "【退化ループ検知・コンテキスト再起動】\n"
                    f"同一フレーズ反復（`{rep_snip}`）を検出したため、破損した出力を破棄しクリーンな新ターンへ切り替えました。\n"
                    "同じ前置きや解説を繰り返すことは禁止です。今すぐ 1 トークン目からコード・ファイル作成またはツール呼び出しのみを出力してください。\n\n"
                    f"{DIRECT_WRITE_FORMAT_GUIDE}"
                )
                rec_freq = frequency_penalty if frequency_penalty is not None else 0.5
                rec_pres = presence_penalty if presence_penalty is not None else 0.3
            else:
                summary_line = f"【思考要点（圧縮）】{thought_summary}\n\n" if thought_summary else ""
                clean_hint = (
                    f"{summary_line}"
                    "【思考上限到達・クリーン実行ターン切替】\n"
                    "思考トークン上限に達したため、途中の推論断片を引き継がずクリーンな新規ターンを開始しました。\n"
                    "上記の要点に基づき、長い前置きなしに今すぐ `[WRITE_FILE: path/to/file.py]` またはツール呼び出しのみを実行してください。\n\n"
                    f"{DIRECT_WRITE_FORMAT_GUIDE}"
                )
                rec_freq = frequency_penalty
                rec_pres = presence_penalty

            # Never splice mid-sentence </think> fragments; start a clean new turn with explicit prefill
            clean_base_messages = self._merge_consecutive_roles(
                list(messages) + [{"role": "user", "content": clean_hint}]
            )
            prefill_messages = clean_base_messages + [{"role": "assistant", "content": clean_prefill}]
            user_fallback_messages = self._merge_consecutive_roles(
                clean_base_messages
                + [
                    {
                        "role": "user",
                        "content": (
                            "思考は完了しています。`<think>` を書かず 1 文字目から "
                            "`[WRITE_FILE: path/to/file.py]` + コードブロック（または必要なツール）のみを出力してください。"
                        ),
                    }
                ]
            )

            used_assistant_prefill = False
            try:
                follow_resp = await _run_raw_call(
                    prefill_messages,
                    tools,
                    0,
                    eff_output_cap,
                    freq_penalty=rec_freq,
                    pres_penalty=rec_pres,
                )
                used_assistant_prefill = True
            except BadRequestError:
                follow_resp = await _run_raw_call(
                    user_fallback_messages,
                    tools,
                    128,
                    eff_output_cap,
                    freq_penalty=rec_freq,
                    pres_penalty=rec_pres,
                )

            follow_msg = follow_resp.choices[0].message
            follow_content = getattr(follow_msg, "content", "") or ""
            if used_assistant_prefill and prefer_write_prefill and not getattr(follow_msg, "tool_calls", None):
                stripped_fc = follow_content.lstrip()
                if stripped_fc and not stripped_fc.lower().startswith(("[write_file:", "<think>")):
                    if re.match(r"^[^\[\n]{1,200}\]", stripped_fc):
                        follow_msg.content = f"[WRITE_FILE: {stripped_fc}"
                    elif stripped_fc.startswith("```"):
                        auto_path = (
                            "10_active_workspace/src_auto.py"
                            if ("10_active_workspace" in ctx_head or "ultrathink" in ctx_head.lower())
                            else "workspace/script_auto.py"
                        )
                        follow_msg.content = f"[WRITE_FILE: {auto_path}]\n{stripped_fc}"

            follow_msg.think_truncated = was_think_truncated
            follow_msg.repeat_aborted = was_repeat_aborted
            follow_msg.think_tokens = getattr(msg, "think_tokens", eff_think_cap)
            if was_repeat_aborted:
                prev_note = (
                    "[⚠️ 同一フレーズ連続反復を検知しストリーム即切断 → "
                    "frequency_penalty=0.5 / presence_penalty=0.3 でクリーン再生成]\n"
                )
            else:
                prev_note = (
                    f"[<think> 上限 {eff_think_cap} 到達（推定 {follow_msg.think_tokens} tokens）"
                    f"→ 思考要点圧縮のうえクリーン新規ターン（[WRITE_FILE: プレフィル）へ切替]\n"
                )
            existing_rc = getattr(follow_msg, "reasoning_content", "") or ""
            merged_rc_parts = [prev_note]
            if thought_summary:
                merged_rc_parts.append(thought_summary)
            if existing_rc:
                merged_rc_parts.append(existing_rc)
            follow_msg.reasoning_content = "\n\n".join(p for p in merged_rc_parts if p).strip()
            return follow_resp

        return resp

    async def _create_chat_completion_resilient(
        self,
        client: AsyncOpenAI,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.3,
        check_stop: bool = True,
        allow_tool_drop: bool = True,
        model_override: Optional[str] = None,
        max_think_tokens: Optional[int] = THINK_TOKEN_LIMIT_DEFAULT,
        max_output_tokens: Optional[int] = MAX_TOKENS_NORMAL_OUTPUT,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> Any:
        """Call chat completions via streaming with automatic retry and context compaction on timeout."""
        for attempt in range(3):
            if attempt == 0:
                cur_messages = messages
                cur_tools = tools
            elif attempt == 1:
                cur_messages = self._compact_messages_for_retry(messages) if len(messages) > 2 else messages
                cur_tools = tools
            else:
                cur_messages = self._compact_messages_for_retry(messages) if len(messages) > 2 else messages
                cur_tools = None if allow_tool_drop else tools

            try:
                return await self._invoke_chat_once(
                    client,
                    cur_messages,
                    cur_tools,
                    temperature=temperature,
                    check_stop=check_stop,
                    allow_tool_drop=allow_tool_drop,
                    model_override=model_override,
                    max_think_tokens=max_think_tokens,
                    max_output_tokens=max_output_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )
            except (APITimeoutError, APIConnectionError, httpx.TimeoutException, httpx.TransportError) as exc:
                if check_stop and self._stop_event.is_set():
                    raise asyncio.CancelledError() from exc
                if attempt >= 2:
                    raise
                print(
                    f"[LLM Timeout/Retry] Attempt {attempt + 1}/3 failed ({type(exc).__name__}: {exc}); "
                    "retrying with compacted context...",
                    flush=True,
                )
                await asyncio.sleep(2.0)

    async def _call_llm_with_fallback(
        self,
        client: AsyncOpenAI,
        session_messages: list[dict[str, Any]],
        tool_history: list[dict[str, Any]],
        use_flattened: bool = False,
        session_dir: Optional[Path] = None,
        is_longthink: bool = False,
        extra_tools: Optional[list[dict[str, Any]]] = None,
        model_override: Optional[str] = None,
        max_think_tokens: Optional[int] = THINK_TOKEN_LIMIT_DEFAULT,
        max_output_tokens: Optional[int] = MAX_TOKENS_NORMAL_OUTPUT,
    ) -> tuple[Any, bool]:
        """Call LM Studio chat completions, falling back to flattened tool history if 400 occurs."""
        active_tools = TOOL_SCHEMAS + (extra_tools or [])
        messages = self._build_messages(
            session_messages,
            tool_history,
            flatten_tools=use_flattened,
            session_dir=session_dir,
            is_longthink=is_longthink,
        )
        try:
            resp = await self._create_chat_completion_resilient(
                client,
                messages,
                tools=active_tools,
                temperature=0.3,
                allow_tool_drop=(use_flattened or not tool_history),
                model_override=model_override,
                max_think_tokens=max_think_tokens,
                max_output_tokens=max_output_tokens,
            )
            return resp, use_flattened
        except BadRequestError:
            if not use_flattened and tool_history:
                flat_messages = self._build_messages(
                    session_messages,
                    tool_history,
                    flatten_tools=True,
                    session_dir=session_dir,
                    is_longthink=is_longthink,
                )
                resp = await self._create_chat_completion_resilient(
                    client,
                    flat_messages,
                    tools=active_tools,
                    temperature=0.3,
                    allow_tool_drop=True,
                    model_override=model_override,
                    max_think_tokens=max_think_tokens,
                    max_output_tokens=max_output_tokens,
                )
                return resp, True
            raise

    @staticmethod
    def _parse_plan_checklist(plan_text: str) -> tuple[list[str], list[str]]:
        """Extract unchecked (- [ ]) and checked (- [x]) items from plan_master.md."""
        unchecked = [
            m.group(1).strip()
            for m in re.finditer(r"^\s*[-*]\s*\[\s\]\s*(.+)$", plan_text, flags=re.MULTILINE)
            if m.group(1).strip()
        ]
        checked = [
            m.group(1).strip()
            for m in re.finditer(r"^\s*[-*]\s*\[[xX]\]\s*(.+)$", plan_text, flags=re.MULTILINE)
            if m.group(1).strip()
        ]
        return unchecked, checked

    @staticmethod
    def _mark_first_unchecked_done(plan_text: str) -> tuple[str, Optional[str]]:
        """Convert the topmost '- [ ] item' line to '- [x] item'."""
        match = re.search(r"^(\s*[-*]\s*)\[\s\](\s*)(.+)$", plan_text, flags=re.MULTILINE)
        if not match:
            return plan_text, None
        done_item = match.group(3).strip()
        updated = (
            plan_text[: match.start()]
            + f"{match.group(1)}[x]{match.group(2)}{done_item}"
            + plan_text[match.end() :]
        )
        return updated, done_item

    @classmethod
    def _mark_n_unchecked_done(cls, plan_text: str, count: int) -> str:
        updated = plan_text
        for _ in range(max(0, count)):
            nxt, item = cls._mark_first_unchecked_done(updated)
            if not item:
                break
            updated = nxt
        return updated

    @staticmethod
    def _build_fallback_plan_from_response(text: str, user_message: str) -> str:
        """Normalize model output into a valid - [ ] checklist if model skipped tool call."""
        existing_checks = [
            f"- [ ] {m.group(1).strip()}"
            for m in re.finditer(r"^\s*[-*]\s*\[\s\]\s*(.+)$", text, flags=re.MULTILINE)
            if m.group(1).strip()
        ]
        if existing_checks:
            return "# タスク計画 (plan_master.md)\n\n" + "\n".join(existing_checks[:8]) + "\n"

        bullets: list[str] = []
        for line in text.splitlines():
            m = re.match(r"^\s*(?:[-*]|\d+[.)、])\s+(.+)$", line.strip())
            if m:
                item = m.group(1).strip()
                if item and not item.startswith("[x]"):
                    cleaned_item = re.sub(r"^\[[xX\s]\]\s*", "", item)
                    bullets.append(f"- [ ] {cleaned_item}")
        if bullets:
            return "# タスク計画 (plan_master.md)\n\n" + "\n".join(bullets[:8]) + "\n"

        short_req = user_message.strip().splitlines()[0][:80]
        return (
            "# タスク計画 (plan_master.md)\n\n"
            f"- [ ] ユーザー要望「{short_req}」を調査・実装または回答準備する\n"
            "- [ ] 実行結果・回答内容の妥当性を確認する\n"
        )

    @staticmethod
    def _summarize_work_log(work_log_text: str, max_lines: int = 12, max_chars: int = 1200) -> str:
        """Return compact recent work_log.md summary for minimal-context state prompts."""
        lines = [ln.strip() for ln in work_log_text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        if not lines:
            return "（まだ作業ログはありません）"
        tail = "\n".join(lines[-max_lines:])
        if len(tail) > max_chars:
            tail = "... (省略)\n" + tail[-max_chars:]
        return tail

    @staticmethod
    def _inspect_python_result(result_text: str) -> tuple[bool, Optional[str], str]:
        """Determine whether run_python succeeded (exit_code == 0 and no Traceback) or failed."""
        lines = result_text.splitlines()
        first_line = lines[0] if lines else ""
        has_nonzero = bool(re.search(r"終了コード:\s*(?!0\b)-?\d+", first_line))
        has_traceback = "Traceback (most recent call last):" in result_text
        has_timeout = result_text.startswith("エラー:") or "タイムアウトしました" in first_line
        if has_nonzero or has_traceback or has_timeout:
            excerpt = result_text.strip()
            if len(excerpt) > 700:
                excerpt = excerpt[:300] + "\n...\n" + excerpt[-380:]
            return False, excerpt, ""
        stdout_excerpt = result_text.strip()
        if len(stdout_excerpt) > 900:
            stdout_excerpt = stdout_excerpt[:900] + "\n... (省略)"
        return True, None, stdout_excerpt

    @classmethod
    def _determine_harness_state(
        cls,
        plan_text: str,
        last_python_error: Optional[str],
        pending_verification: bool,
    ) -> str:
        """Deterministically pick one of the 5 harness states."""
        unchecked, checked = cls._parse_plan_checklist(plan_text)
        if not unchecked and not checked:
            return STATE_INIT_PLAN
        if last_python_error is not None:
            return STATE_FIX_ERROR
        if pending_verification:
            return STATE_VERIFY_RESULT
        if unchecked:
            return STATE_EXECUTE_STEP
        return STATE_FINAL_REPORT

    @staticmethod
    def _filter_tools_for_state(current_state: str, is_longthink: bool) -> list[dict[str, Any]]:
        """Expose only tools permitted in the active state machine phase."""
        if current_state == STATE_FINAL_REPORT:
            return []
        all_tools = TOOL_SCHEMAS + (LONGTHINK_EXTRA_TOOL_SCHEMAS if is_longthink else [])
        if current_state in (STATE_INIT_PLAN, STATE_REPLAN):
            allowed = {"update_plan", "update_memory", "update_master_plan", "write_file"}
            return [t for t in all_tools if t["function"]["name"] in allowed]
        if current_state == STATE_FIX_ERROR:
            allowed = {"write_file", "read_file", "list_files", "run_python", "run_command"}
            return [t for t in all_tools if t["function"]["name"] in allowed]
        if current_state == STATE_VERIFY_RESULT:
            allowed = {"run_python", "read_file", "write_file", "update_plan", "update_memory", "update_master_plan"}
            return [t for t in all_tools if t["function"]["name"] in allowed]
        return all_tools

    @classmethod
    def _build_action_stagnation_signature(
        cls,
        tool_name: str,
        tool_args: dict[str, Any],
        result_text: str,
    ) -> tuple[str, str, str]:
        """Return (call_sig, res_sig, reason_str) for 4-step stagnation detection."""
        args_dict = tool_args if isinstance(tool_args, dict) else {}
        norm_tool = "list_dir" if tool_name in ("list_files", "list_dir") else tool_name
        res_clean = re.sub(r"tmp[a-zA-Z0-9_]+\.py", "tmp.py", result_text or "").strip()

        # Normalize call signature (tool + near-identical arguments)
        if norm_tool == "list_dir":
            raw_path = str(args_dict.get("path", ".")).strip().replace("\\", "/")
            norm_path = "." if raw_path in ("", ".", "./", "workspace", "workspace/") else raw_path
            call_sig = f"call:list_dir:{norm_path}"
        elif norm_tool == "read_file":
            raw_path = str(args_dict.get("path", "")).strip().replace("\\", "/")
            call_sig = f"call:read_file:{raw_path}"
        elif norm_tool == "write_file":
            raw_path = str(args_dict.get("path", "")).strip().replace("\\", "/")
            content = str(args_dict.get("content", ""))
            call_sig = f"call:write_file:{raw_path}:{len(content)}:{content[:60]}"
        elif norm_tool == "run_python":
            code_norm = re.sub(r"\s+", " ", str(args_dict.get("code", ""))).strip()[:120]
            call_sig = f"call:run_python:{code_norm}"
        elif norm_tool == "run_command":
            cmd_norm = re.sub(r"\s+", " ", str(args_dict.get("command", ""))).strip()[:120]
            call_sig = f"call:run_command:{cmd_norm}"
        else:
            args_sig = json.dumps(args_dict, sort_keys=True, ensure_ascii=False)[:140]
            call_sig = f"call:{norm_tool}:{args_sig}"

        # Normalize result / error signature
        if norm_tool == "list_dir" and "ディレクトリは空です" in res_clean:
            res_sig = f"res:empty_dir:{norm_path}"
            reason = f"{tool_name}('{norm_path}') で空ディレクトリの確認を繰り返している"
            return call_sig, res_sig, reason

        if norm_tool == "read_file" and ("エラー:" in res_clean or "見つかりません" in res_clean):
            err_head = res_clean.splitlines()[0][:80] if res_clean else "読込エラー"
            res_sig = f"res:read_err:{err_head}"
            reason = f"read_file('{raw_path}') で同じエラー ({err_head}) が続いている"
            return call_sig, res_sig, reason

        if norm_tool == "run_python":
            ok, err_excerpt, _ = cls._inspect_python_result(result_text or "")
            if not ok:
                err_norm = re.sub(r'File\s+"[^"]*"', 'File "tmp.py"', err_excerpt or res_clean)
                err_norm = re.sub(r"line\s+\d+", "line N", err_norm)
                err_lines = [ln.strip() for ln in err_norm.splitlines() if ln.strip()]
                core_err = err_lines[-1][:100] if err_lines else res_clean[:100]
                res_sig = f"res:python_err:{core_err}"
                reason = f"run_python で同じエラー ({core_err}) が連続している"
                return call_sig, res_sig, reason

        if "エラー:" in res_clean:
            err_line = res_clean.splitlines()[0][:100]
            res_sig = f"res:err:{norm_tool}:{err_line}"
            reason = f"{tool_name} で同じエラー ({err_line[:60]}) が続いている"
            return call_sig, res_sig, reason

        res_head = res_clean.splitlines()[0][:110] if res_clean else "(empty)"
        res_sig = f"res:ok:{norm_tool}:{res_head}"
        reason = f"{tool_name} で同じ操作・同じ出力 ({res_head[:60]}) を繰り返している"
        return call_sig, res_sig, reason

    class _LoopStagnationTracker:
        """Tracks 2-consecutive stagnation across tool calls, tool results, and plan progress."""

        def __init__(self, threshold: int = CIRCUIT_BREAKER_THRESHOLD) -> None:
            self.threshold = max(2, int(threshold))
            self.call_counts: dict[str, int] = {}
            self.res_counts: dict[str, int] = {}
            self.last_call_sig: Optional[str] = None
            self.consec_call_count: int = 0
            self.last_res_sig: Optional[str] = None
            self.consec_res_count: int = 0
            self.no_progress_steps: int = 0
            self.recent_steps: list[str] = []

        def reset(self, keep_recent: bool = False) -> None:
            self.call_counts.clear()
            self.res_counts.clear()
            self.last_call_sig = None
            self.consec_call_count = 0
            self.last_res_sig = None
            self.consec_res_count = 0
            self.no_progress_steps = 0
            if not keep_recent:
                self.recent_steps.clear()

        def record_action(self, call_sig: str, res_sig: str, history_entry: str, reason: str) -> Optional[str]:
            self.recent_steps.append(history_entry)
            if len(self.recent_steps) > 8:
                self.recent_steps = self.recent_steps[-8:]

            self.call_counts[call_sig] = self.call_counts.get(call_sig, 0) + 1
            if call_sig == self.last_call_sig:
                self.consec_call_count += 1
            else:
                self.last_call_sig = call_sig
                self.consec_call_count = 1

            self.res_counts[res_sig] = self.res_counts.get(res_sig, 0) + 1
            if res_sig == self.last_res_sig:
                self.consec_res_count += 1
            else:
                self.last_res_sig = res_sig
                self.consec_res_count = 1

            if self.consec_call_count >= self.threshold:
                return f"同一ツール+引数が{self.consec_call_count}回連続で重複停滞: {reason}"
            if self.consec_res_count >= self.threshold:
                return f"同一結果/エラーが{self.consec_res_count}回連続で反復停滞: {reason}"
            if self.call_counts[call_sig] >= max(4, self.threshold * 2):
                return f"同一ツール+引数の累積停滞 ({self.call_counts[call_sig]}回): {reason}"
            if self.res_counts[res_sig] >= max(4, self.threshold * 2):
                return f"同一結果/エラーの累積停滞 ({self.res_counts[res_sig]}回): {reason}"
            return None

        def record_step_progress(self, progress_changed: bool, step_summary: str) -> Optional[str]:
            if progress_changed:
                self.reset(keep_recent=True)
                return None
            self.no_progress_steps += 1
            self.recent_steps.append(step_summary)
            if len(self.recent_steps) > 8:
                self.recent_steps = self.recent_steps[-8:]
            if self.no_progress_steps >= self.threshold:
                return f"計画の進捗が{self.no_progress_steps}ステップ連続で変化せず停滞 ({step_summary[:80]})"
            return None

    _CODING_HINT_RE = re.compile(
        r"(python|スクリプト|コード|プログラム|実装|関数|クラス|モジュール|バグ|エラー|修正|デバッグ|テスト|"
        r"計算|シミュレーション|集計|解析|グラフ|描画|ファイル|作成|記述|生成|検算|\.py\b|"
        r"\bscript\b|\bcode\b|\bprogram\b|\bbug\b|\bfix\b|\bdebug\b|\btest\b|\bcalc\b|\bbuild\b|"
        r"\bwrite\b|\bcreate\b|\bjson\b|\bhtml\b|\bcss\b|\bjs\b|\bsql\b|\bapi\b)",
        re.IGNORECASE,
    )

    @classmethod
    def _step_involves_coding(
        cls,
        current_state: str,
        task_text: str,
        active_tools: Optional[list[dict[str, Any]]] = None,
    ) -> bool:
        if current_state in (STATE_INIT_PLAN, STATE_REPLAN, STATE_FINAL_REPORT):
            return False
        if current_state in (STATE_FIX_ERROR, STATE_VERIFY_RESULT):
            return True
        if active_tools is not None:
            tool_names = {t.get("function", {}).get("name", "") for t in active_tools}
            if not (tool_names & {"run_python", "write_file", "run_command"}):
                return False
        return bool(cls._CODING_HINT_RE.search(task_text or ""))

    @staticmethod
    def _is_non_json_pure_coder_choice(coder_choice: str) -> bool:
        return (coder_choice or "").strip() in ("lfm3_coder_claude", "lfm3_coder")

    def _resolve_route_descriptor(self, choice: str) -> tuple[str, str, str, bool]:
        """Map canonical route choice to (choice_key, model_id, role_label, use_json_tools)."""
        main_model = (self.state.model or "").strip()
        coder_model = (self.state.coder_model or DEFAULT_CODER_MODEL).strip() or main_model
        std_model = (self.state.standard_model or DEFAULT_STANDARD_MODEL).strip() or main_model
        norm = (choice or "").strip().lower()
        if "coder" in norm or "distill" in norm or "claude" in norm:
            return ("lfm3_coder_claude", coder_model, "LFM3-Coder", False)
        if "standard" in norm or "admin" in norm or "tool" in norm or "1.7" in norm or "17b" in norm:
            return ("lfm3_standard", std_model, "LFM3 1.7B (Admin/Tool)", True)
        return ("main_model", main_model, "Main Model", True)

    @staticmethod
    def _apply_pure_coder_constraints(
        messages: list[dict[str, Any]],
        active_tools: Optional[list[dict[str, Any]]],
        route_res: Optional[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], Optional[list[dict[str, Any]]], bool]:
        """Strip JSON tools and enforce non-JSON [WRITE_FILE: ...] formatting for lfm3_coder_claude."""
        use_json_tools = bool((route_res or {}).get("use_json_tools", True))
        if use_json_tools:
            return messages, active_tools, True
        constrained = [dict(m) for m in messages]
        if constrained and constrained[0].get("role") == "system":
            sys_body = str(constrained[0].get("content") or "")
            if "PURE_CODER_NON_JSON" not in sys_body and "JSONツール呼び出し厳禁" not in sys_body:
                constrained[0]["content"] = f"{sys_body}\n\n{PURE_CODER_NON_JSON_DIRECTIVE}"
        else:
            constrained.insert(0, {"role": "system", "content": PURE_CODER_NON_JSON_DIRECTIVE})
        return constrained, None, False

    @staticmethod
    def _augment_pure_coder_write_calls(
        parsed_calls: list[dict[str, Any]],
        allowed_names: set[str],
    ) -> list[dict[str, Any]]:
        """Auto-append run_python verification when lfm3_coder_claude emits [WRITE_FILE: *.py] without [RUN_PYTHON]."""
        if not parsed_calls or "run_python" not in allowed_names:
            return parsed_calls
        if any(c.get("name") == "run_python" for c in parsed_calls):
            return parsed_calls
        py_writes = [
            c
            for c in parsed_calls
            if c.get("name") == "write_file"
            and isinstance(c.get("arguments"), dict)
            and str(c["arguments"].get("path", "")).strip().lower().endswith(".py")
            and str(c["arguments"].get("content", "")).strip()
        ]
        if not py_writes:
            return parsed_calls
        last_py = py_writes[-1]
        augmented = list(parsed_calls)
        augmented.append(
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "name": "run_python",
                "arguments": {"code": str(last_py["arguments"]["content"])},
                "_auto_verify_python": True,
            }
        )
        return augmented

    async def _emit_trash_isolation_log(
        self,
        session: dict[str, Any],
        thinking_log: list[dict[str, Any]],
        assistant_msg_id: str,
        session_id: str,
        step: int,
        iso_res: Optional[dict[str, Any]],
        allow_persist: bool = True,
    ) -> None:
        """Record & broadcast resolved artifact promotion and trash_bin isolation."""
        if not iso_res:
            return
        iso_entry = {
            "type": "trash_isolation",
            "step": step,
            "task_id": iso_res.get("task_id", ""),
            "promoted_files": iso_res.get("promoted_files", []),
            "promoted_dest": iso_res.get("promoted_dest", ""),
            "trash_path": iso_res.get("trash_path", ""),
            "trashed_files": iso_res.get("trashed_files", []),
            "trashed_count": iso_res.get("trashed_count", 0),
            "message": iso_res.get("message") or "🗑️ 仮ファイルを trash_bin へ隔離完了",
        }
        if allow_persist:
            async with self._persist_lock:
                thinking_log.append(iso_entry)
                self.live_thinking_log.append(iso_entry)
                self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
        await self.broker.publish("trash_isolation", {**iso_entry, "session_id": session_id})
        await self._on_files_update()

    async def _route_coding_model_via_d1(
        self,
        client: AsyncOpenAI,
        step: int,
        session_id: str,
        assistant_msg_id: str,
        session: dict[str, Any],
        thinking_log: list[dict[str, Any]],
        task_summary: str,
        context_label: str = "",
        allow_persist: bool = True,
        forced_choice: Optional[str] = None,
        forced_reason: Optional[str] = None,
    ) -> dict[str, Any]:
        """Consult VRAM-resident d1-2B to route step among LFM3-Coder (pure code), LFM3-1.7B (tool/admin), and Main Model."""
        main_model = (self.state.model or "").strip()
        coder_model = (self.state.coder_model or DEFAULT_CODER_MODEL).strip()
        std_model = (self.state.standard_model or DEFAULT_STANDARD_MODEL).strip()
        sup_model = (self.state.supervisor_model or DEFAULT_SUPERVISOR_MODEL).strip()

        if forced_choice:
            choice_key, selected_model, role_label, use_json_tools = self._resolve_route_descriptor(forced_choice)
            reason_text = (forced_reason or f"{role_label} の担当領域のため自動ルーティング").strip()
            route_entry = {
                "type": "d1_coder_route",
                "step": step,
                "duration": 0.0,
                "supervisor_model": sup_model,
                "coder_choice": choice_key,
                "selected_model": selected_model,
                "role_label": role_label,
                "use_json_tools": use_json_tools,
                "reason": reason_text,
                "raw_output": f"[AUTO ROUTE: {role_label} ({selected_model})]\nReason: {reason_text}",
                "context_label": context_label,
                "message": f"🧭 [D1-2B Router] 担当 -> {role_label} [{selected_model}] ({reason_text})",
            }
            if allow_persist:
                async with self._persist_lock:
                    thinking_log.append(route_entry)
                    self.live_thinking_log.append(route_entry)
                    self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
            await self.broker.publish("d1_coder_route", {**route_entry, "session_id": session_id})
            return {
                "coder_choice": choice_key,
                "selected_model": selected_model,
                "role_label": role_label,
                "use_json_tools": use_json_tools,
                "reason": reason_text,
                "supervisor_model": sup_model,
            }

        if (not coder_model or coder_model == main_model) and (not std_model or std_model == main_model):
            return {
                "coder_choice": "main_model",
                "selected_model": main_model,
                "role_label": "Main Model",
                "use_json_tools": True,
                "reason": "メインモデル単一構成のためメインモデルを使用",
                "supervisor_model": sup_model,
            }

        prev_phase_state = self.state.phase_state
        prev_phase_label = self.state.phase_label
        prev_phase_started = self.state.phase_started_at
        route_start = time.monotonic()

        self.state.phase_state = "working"
        self.state.phase_label = f"🧭 [D1-2B Router] {sup_model} が4モデル分業の担当を判定中..."
        self.state.phase_started_at = route_start
        await self.broker.publish(
            "status",
            {
                **self._status_payload(),
                "state": "working",
                "step": step,
                "label": self.state.phase_label,
            },
        )

        sys_prompt = (
            "あなたは4モデル協調環境のタスクルーティング専任ルーター (D1-2B Router) です。\n"
            "次の作業ステップを、以下の3モデルのいずれか最適な1体へ割り振ってください：\n"
            "- \"lfm3_coder_claude\": 純粋プログラミング専任。Pythonスクリプトの新規アルゴリズム実装・コード生成のみを担当（JSONツール呼び出し不可のため `[WRITE_FILE: ...]` 直接記述でコード生成する作業向け）。\n"
            "- \"lfm3_standard\": 事務・JSONツール呼び出し専任（LFM3 1.7B）。JSON形式のツール呼び出し（コマンド実行・ファイル読込/検索・ディレクトリ整理）、`GLOBAL_INDEX.md`・`fact_XXXX.md`・`plan.md` の更新やデータ構造化などの事務処理向け。\n"
            "- \"main_model\": 深い論理推論・数学的解析・エラー発生直後の高度な原因分析と戦略変更（リプラン）・最終統合が必要なステップ向け。\n\n"
            "必ず以下のJSONオブジェクト1つだけを出力してください：\n"
            '{"target": "lfm3_coder_claude" | "lfm3_standard" | "main_model", "reason": "判定理由（1文）"}'
        )
        usr_prompt = (
            f"【文脈】{context_label or '実行ステップ'}\n"
            f"【タスク要件】\n{(task_summary or '').strip()[:900]}\n\n"
            f"候補1 (`lfm3_coder_claude` / 純粋コード生成・非JSON): {coder_model}\n"
            f"候補2 (`lfm3_standard` / JSONツール・事務構造化): {std_model}\n"
            f"候補3 (`main_model` / 深層推論・原因分析・統合): {main_model}\n"
            "最適な1つを JSON で回答してください。"
        )
        raw_route = ""
        used_sup = sup_model
        for cand in [sup_model, main_model]:
            c_id = (cand or "").strip()
            if not c_id:
                continue
            try:
                resp = await client.chat.completions.create(
                    model=c_id,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": usr_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=160,
                    timeout=30.0,
                )
                r_msg = resp.choices[0].message
                r_extra = getattr(r_msg, "reasoning_content", None) or getattr(r_msg, "reasoning", None) or ""
                raw_route = _compose_raw_model_output(r_msg.content or "", r_extra).strip()
                used_sup = c_id
                if raw_route:
                    break
            except Exception as exc:
                print(f"[D1-2B Router] model='{c_id}' 呼び出しフォールバック ({exc})", flush=True)
                continue

        parsed_obj: dict[str, Any] = {}
        if raw_route:
            _, vis_route = _extract_think_and_content(raw_route)
            cand_text = vis_route or raw_route
            for obj in _extract_json_objects_from_text(cand_text):
                if isinstance(obj, dict) and ("target" in obj or "coder" in obj or "model" in obj):
                    parsed_obj = obj
                    break

        raw_choice = str(parsed_obj.get("target") or parsed_obj.get("coder") or parsed_obj.get("model") or "").strip().lower()
        task_lower = (task_summary or "").lower()
        if "main" in raw_choice:
            inferred_choice = "main_model"
        elif "standard" in raw_choice or "1.7" in raw_choice or "admin" in raw_choice or "tool" in raw_choice:
            inferred_choice = "lfm3_standard"
        elif "coder" in raw_choice or "distill" in raw_choice or "claude" in raw_choice:
            inferred_choice = "lfm3_coder_claude"
        elif any(k in task_lower for k in ("python", "スクリプト", "実装", "コード", "アルゴリズム", "src_", ".py")):
            inferred_choice = "lfm3_coder_claude"
        elif any(k in task_lower for k in ("global_index", "fact_", "整理", "確認", "検索", "plan.md", "コマンド", "work_log")):
            inferred_choice = "lfm3_standard"
        else:
            inferred_choice = "main_model"

        choice_key, selected_model, role_label, use_json_tools = self._resolve_route_descriptor(inferred_choice)
        default_reasons = {
            "lfm3_coder_claude": "純粋Pythonコード実装・生成のため非JSON専任 LFM3-Coder を選択",
            "lfm3_standard": "JSONツール実行・事務整理・構造化のため LFM3 1.7B を選択",
            "main_model": "深い論理推論・解析・統合判断のためメインモデルを選択",
        }
        reason_text = str(parsed_obj.get("reason") or default_reasons[choice_key]).strip()

        route_dur = round(time.monotonic() - route_start, 1)
        route_entry = {
            "type": "d1_coder_route",
            "step": step,
            "duration": route_dur,
            "supervisor_model": used_sup,
            "coder_choice": choice_key,
            "selected_model": selected_model,
            "role_label": role_label,
            "use_json_tools": use_json_tools,
            "reason": reason_text,
            "raw_output": raw_route or reason_text,
            "context_label": context_label,
            "message": f"🧭 [D1-2B Router] 担当 -> {role_label} [{selected_model}] ({reason_text})",
        }
        if allow_persist:
            async with self._persist_lock:
                thinking_log.append(route_entry)
                self.live_thinking_log.append(route_entry)
                self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
        await self.broker.publish("d1_coder_route", {**route_entry, "session_id": session_id})

        self.state.phase_state = prev_phase_state
        self.state.phase_label = prev_phase_label
        self.state.phase_started_at = prev_phase_started
        return {
            "coder_choice": choice_key,
            "selected_model": selected_model,
            "role_label": role_label,
            "use_json_tools": use_json_tools,
            "reason": reason_text,
            "supervisor_model": used_sup,
        }

    async def _invoke_d1_supervisor(
        self,
        client: AsyncOpenAI,
        step: int,
        session_id: str,
        assistant_msg_id: str,
        session: dict[str, Any],
        thinking_log: list[dict[str, Any]],
        recent_history_items: list[str],
        plan_text: str,
        fallback_reason: str,
        context_label: str = "",
        intervention_count: int = 1,
        think_limit_unlocked: bool = False,
    ) -> dict[str, Any]:
        """Consult VRAM-resident d1-2B supervisor model on 2-step stagnation, unlock/replan, or escalate to user."""
        sup_model = (self.state.supervisor_model or DEFAULT_SUPERVISOR_MODEL).strip()
        sup_start = time.monotonic()
        prev_phase_state = self.state.phase_state
        prev_phase_label = self.state.phase_label
        prev_phase_started = self.state.phase_started_at

        self.state.phase_state = "working"
        self.state.phase_label = (
            f"⚡ [D1-2B Supervisor] {sup_model} が2連続停滞の原因を診断中 (同一箇所 {intervention_count}回目)..."
        )
        self.state.phase_started_at = sup_start
        await self.broker.publish(
            "status",
            {
                **self._status_payload(),
                "state": "working",
                "step": step,
                "label": self.state.phase_label,
            },
        )

        recent_block = "\n".join(recent_history_items[-6:]) if recent_history_items else f"- {fallback_reason}"
        plan_excerpt = (plan_text or "").strip()[:1500] or "(plan.md なし)"
        escalation_hint = (
            "\n注意: 同一サブタスクで既に5回以上のリプランと思考上限解除を行っても突破できていません。"
            "解決不能な環境制約・仕様不明・堂々巡りと判断した場合は `\"escalate_to_user\": true` とし、"
            "`user_consultation_message` にユーザーへの具体的な相談・確認事項を出力してループを安全停止させてください。"
            if intervention_count >= D1_ESCALATE_THRESHOLD
            else (
                "\n今回の介入で同一箇所5回目のため、この後思考トークン上限（3000）を無制限へ自動解除します。"
                if intervention_count == D1_UNLOCK_THINK_THRESHOLD
                else ""
            )
        )
        sup_system = (
            "あなたは自律エージェントを監視・救済する軽量メタ監査モデル (D1-2B Supervisor) です。\n"
            "現在メインエージェントで同じ失敗・停滞・重複操作が2回連続したため、即座に原因を断定し方針を指示するか、"
            "どうしても突破できない完全停滞ならユーザーへ相談・判断を仰ぐエスカレーション判定を行ってください。\n"
            f"{escalation_hint}\n"
            "余計な説明は出力せず、必ず以下のキーを持つ単一の JSON オブジェクトのみを出力してください：\n"
            '{\n'
            '  "stuck_reason": "何が詰まりの原因か（簡潔な1〜2文）",\n'
            '  "forbidden_action": "再実行を禁止する具体的アクション・手法",\n'
            '  "new_strategy": "次に取るべき具体的で全く別のアプローチ（1〜2文）",\n'
            '  "escalate_to_user": false,\n'
            '  "user_consultation_message": "escalate_to_user が true の場合のみ、ユーザーに解決の判断・情報提供を仰ぐ具体的報告文"\n'
            '}'
        )
        sup_user = (
            f"【2連続サーキットブレーカー検知 (同一箇所での累計D1介入: {intervention_count}回目 / 思考制限解除済: {think_limit_unlocked})】\n"
            f"{fallback_reason}\n"
            + (f"対象コンテキスト: {context_label}\n\n" if context_label else "\n")
            + f"【直近の操作履歴】\n{recent_block}\n\n"
            + f"【現在の plan.md / 進捗】\n```markdown\n{plan_excerpt}\n```\n\n"
            + "上記を基に、JSON形式で診断結果（突破不能なら `escalate_to_user: true` と相談メッセージ）を返してください。"
        )
        sup_messages = [
            {"role": "system", "content": sup_system},
            {"role": "user", "content": sup_user},
        ]

        raw_sup = ""
        used_model = sup_model
        for candidate_model in [sup_model, self.state.model]:
            cand = (candidate_model or "").strip()
            if not cand:
                continue
            try:
                resp = await client.chat.completions.create(
                    model=cand,
                    messages=sup_messages,
                    temperature=0.1,
                    max_tokens=480,
                    timeout=45.0,
                )
                s_msg = resp.choices[0].message
                s_extra = getattr(s_msg, "reasoning_content", None) or getattr(s_msg, "reasoning", None) or ""
                raw_sup = _compose_raw_model_output(s_msg.content or "", s_extra).strip()
                used_model = cand
                if raw_sup:
                    break
            except Exception as exc:
                print(f"[D1-2B Supervisor] model='{cand}' 呼び出しフォールバック ({exc})", flush=True)
                continue

        parsed_obj: dict[str, Any] = {}
        if raw_sup:
            _, visible_sup = _extract_think_and_content(raw_sup)
            cand_text = visible_sup or raw_sup
            for obj in _extract_json_objects_from_text(cand_text):
                if isinstance(obj, dict) and any(
                    k in obj
                    for k in ("stuck_reason", "forbidden_action", "new_strategy", "escalate_to_user")
                ):
                    parsed_obj = obj
                    break

        stuck_reason = str(parsed_obj.get("stuck_reason") or fallback_reason).strip()
        forbidden_action = str(parsed_obj.get("forbidden_action") or fallback_reason).strip()
        new_strategy = str(
            parsed_obj.get("new_strategy")
            or "直前の確認や失敗パターンを中止し、必要なスクリプト・解決策を別手段で直接新規作成・検証してください。"
        ).strip()
        raw_escalate = parsed_obj.get("escalate_to_user", False)
        escalate_to_user = bool(
            (isinstance(raw_escalate, bool) and raw_escalate)
            or (isinstance(raw_escalate, str) and raw_escalate.strip().lower() in ("true", "yes", "1"))
            or intervention_count >= D1_ESCALATE_THRESHOLD
        )
        unlocked_now = (not think_limit_unlocked) and (intervention_count >= D1_UNLOCK_THINK_THRESHOLD)
        consult_msg = str(parsed_obj.get("user_consultation_message") or "").strip()
        if escalate_to_user and not consult_msg:
            consult_msg = (
                f"同一箇所で {intervention_count} 回の D1-2B 介入（思考上限無制限化を含む）を行いましたが、"
                f"以下の原因により自律ループでは突破できませんでした。\n\n"
                f"- **特定された根本原因**: {stuck_reason}\n"
                f"- **試行しても回避できなかった手段**: {forbidden_action}\n"
                f"- **推奨する次の対策 / ユーザーへの確認事項**: {new_strategy}\n\n"
                "前提条件の調整・追加情報のご提供・手動確認などをご指示ください。"
            )

        sup_dur = round(time.monotonic() - sup_start, 1)
        if escalate_to_user:
            summary_msg = f"🛑 [D1-2B Supervisor] 解決不能・完全停滞を検知 → ユーザー相談のため中断 ({stuck_reason})"
        elif unlocked_now:
            summary_msg = (
                f"🔓 [D1-2B Supervisor] 累計{intervention_count}回目介入 → 難所判定により思考トークン上限を無制限解除 ({stuck_reason})"
            )
        else:
            summary_msg = (
                f"⚡ [D1-2B Supervisor] 2連続停滞を検知（{intervention_count}回目）→ 戦略強制変更 ({stuck_reason})"
            )

        sup_entry = {
            "type": "d1_supervisor",
            "step": step,
            "duration": sup_dur,
            "supervisor_model": used_model,
            "context_label": context_label,
            "intervention_count": intervention_count,
            "think_unlocked": bool(unlocked_now or think_limit_unlocked),
            "escalate_to_user": escalate_to_user,
            "user_consultation_message": consult_msg,
            "stuck_reason": stuck_reason,
            "forbidden_action": forbidden_action,
            "new_strategy": new_strategy,
            "raw_output": raw_sup,
            "message": summary_msg,
        }
        async with self._persist_lock:
            thinking_log.append(sup_entry)
            self.live_thinking_log.append(sup_entry)
            self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
        await self.broker.publish("d1_supervisor", {**sup_entry, "session_id": session_id})

        self.state.phase_state = prev_phase_state
        self.state.phase_label = prev_phase_label
        self.state.phase_started_at = prev_phase_started
        return {
            "stuck_reason": stuck_reason,
            "forbidden_action": forbidden_action,
            "new_strategy": new_strategy,
            "supervisor_model": used_model,
            "escalate_to_user": escalate_to_user,
            "user_consultation_message": consult_msg,
            "think_unlocked": bool(unlocked_now or think_limit_unlocked),
        }

    @staticmethod
    def _format_d1_escalation_report(diag: dict[str, Any], intervention_count: int, mode_label: str) -> str:
        """Build final Markdown consultation report when D1 escalates deadlock to user."""
        stuck = diag.get("stuck_reason") or "停滞原因の詳細不明"
        forbid = diag.get("forbidden_action") or "(記録なし)"
        strategy = diag.get("new_strategy") or "(記録なし)"
        consult = diag.get("user_consultation_message") or ""
        return (
            f"🛑 **D1-2B Supervisor 最終エスカレーション（ユーザー判断・相談のため一時停止）**\n\n"
            f"- **実行モード**: `{mode_label}`\n"
            f"- **監査モデル**: `{diag.get('supervisor_model') or DEFAULT_SUPERVISOR_MODEL}`\n"
            f"- **同一箇所での累積 D1 介入回数**: {intervention_count} 回\n"
            f"- **D1 が特定した停滞原因**: {stuck}\n"
            f"- **限界に達した / 禁止されたアプローチ**: {forbid}\n\n"
            f"### ユーザーへの相談・確認事項\n{consult}\n\n"
            f"### 補足（D1 による推奨方針）\n{strategy}\n\n"
            "上記の点についてご確認・ご指示のうえ、追加メッセージをお送りください。"
        )

    def _build_state_machine_prompt_and_tools(
        self,
        current_state: str,
        user_message: str,
        plan_text: str,
        work_log_text: str,
        is_longthink: bool,
        last_python_error: Optional[str] = None,
        last_python_stdout: str = "",
        same_state_history: Optional[list[dict[str, Any]]] = None,
        replan_reason: Optional[str] = None,
        replan_strategy: Optional[str] = None,
        forbidden_approaches: Optional[list[str]] = None,
        handoff_text: str = "",
        sub_agents_summary: str = "",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Assemble minimal single-focus messages and state-filtered tools for the LLM."""
        unchecked, checked = self._parse_plan_checklist(plan_text)
        first_task = unchecked[0] if unchecked else user_message.strip().splitlines()[0][:100]
        work_summary = self._summarize_work_log(work_log_text)
        rule_md = self.memory.read_file("rule.md").strip()[:600]
        active_tools = self._filter_tools_for_state(current_state, is_longthink)
        forbidden_list = [f for f in (forbidden_approaches or []) if f]
        forbidden_block = ""
        if forbidden_list:
            forbidden_lines = "\n".join(f"- 禁止: {item}" for item in forbidden_list[-5:])
            forbidden_block = f"\n\n【禁止されたアプローチ（再試行厳禁）】\n{forbidden_lines}"
        strategy_block = ""
        if replan_strategy:
            strategy_block = f"\n\n【⚡ D1-2B Supervisor 強制新戦略】\n{replan_strategy}"

        if current_state == STATE_REPLAN:
            plan_file_label = "master_plan.md" if is_longthink else "plan.md"
            plan_tool = "update_master_plan" if is_longthink else "update_plan"
            reason_display = replan_reason or "同一エラー・同一確認の2回連続反復"
            strategy_display = replan_strategy or "直前のアプローチを中止し、全く別の手法で計画を立て直す"
            sys_msg = (
                "あなたは方針再構築担当 (`STATE: REPLAN`) です。\n"
                "監査モデル `d1-2B` の判断により、これまでの手法は強制終了・禁止されました。\n"
                "禁止されたアクションや同じファイルの空確認・同一コマンドは絶対に繰り返さず、"
                f"`d1-2B` が指示した新戦略に沿って `{plan_file_label}` を `- [ ]` チェックリストとして全面的に書き直してください。\n"
                "このステップでは計画の書き直しのみを行い、実装やコード実行は行わないでください。"
            )
            user_prompt = (
                f"同じ原因（例：{reason_display}）で2回連続の停滞・修復ループが発生したため、"
                "監査モデル (`d1-2B`) が介入しこれまでのアプローチを強制終了しました。**直前に試していた方法は禁止**とします。\n"
                f"【採用すべき新戦略 (`d1-2B`)】\n{strategy_display}\n"
                f"{forbidden_block}\n\n"
                f"【元のユーザー要望】\n{user_message}\n\n"
                f"【これまでの計画 ({plan_file_label})】\n```markdown\n{plan_text.strip()[:1200]}\n```\n\n"
                f"【直近の作業ログ (work_log.md)】\n{work_summary}\n\n"
                f"必ず `{plan_tool}` ツールを使って、上記新戦略に基づく新しい `- [ ] ...` 形式の計画へ全面的に書き直してください。"
            )
        elif current_state == STATE_INIT_PLAN:
            plan_tool = "update_master_plan" if is_longthink else "update_plan"
            sys_msg = (
                "あなたはステートマシン制御下の計画立案担当 (`STATE: INIT_PLAN`) です。\n"
                "今は計画作成のみを行い、実装やコード実行は行わないでください。\n"
                "新規開始直後の workspace/ が空なのは正常ですので、list_files で確認する必要はありません。\n"
                "複雑な考えは <think>...</think> 内に収め、ツール実行前の本文は簡潔な1文にしてください。"
                f"{forbidden_block}{strategy_block}"
            )
            user_prompt = (
                f"【ユーザー要望】\n{user_message}\n\n"
                "ユーザーの要望を達成するための具体的な手順を `- [ ]` 形式のチェックリストで "
                f"`plan_master.md` に書き出してください（まだ実装は行わないこと）。\n"
                f"必ず `{plan_tool}` ツールを使って `- [ ] ...` 形式で保存してください。"
                f"{forbidden_block}{strategy_block}"
            )
        elif current_state == STATE_FIX_ERROR:
            sys_msg = (
                "あなたは自己修復担当 (`STATE: FIX_ERROR`) です。\n"
                "他の未着手タスクは無視し、直前のエラー修復だけに集中してください。\n"
                "workspace/ が空の場合は正常な初期状態なので、`list_files` で空ディレクトリを再確認することは禁止です。\n"
                f"{DIRECT_WRITE_FORMAT_GUIDE}\n"
                "思考は <think>...</think> 内に収め、ファイル保存後に必ず `run_python` で再検証してください。"
                f"{forbidden_block}{strategy_block}"
            )
            user_prompt = (
                f"【実行中タスク】{first_task}\n\n"
                "直前の実行で以下のエラーが発生しました。原因を特定し、修正したコードを再保存・再実行してください：\n"
                f"```\n{last_python_error or '(詳細不明)'}\n```\n\n"
                "※ 長いコードを JSON 引数で `write_file` に渡さず、可能な限り `[WRITE_FILE: path/to/script.py]` + ```python ... ``` 形式で直接記述してください。\n"
                f"【直近の work_log.md 要約】\n{work_summary}"
                f"{forbidden_block}{strategy_block}"
            )
        elif current_state == STATE_VERIFY_RESULT:
            plan_tool = "update_master_plan" if is_longthink else "update_plan"
            sys_msg = (
                "あなたは独立検算・検証担当 (`STATE: VERIFY_RESULT`) です。\n"
                "先ほど実行したコードの計算式とは別の検算式やエッジケーステストを Python で実行し、結果を裏付けてください。\n"
                f"{DIRECT_WRITE_FORMAT_GUIDE}"
                f"{forbidden_block}{strategy_block}"
            )
            user_prompt = (
                f"【対象タスク】{first_task}\n\n"
                f"【直前の正常実行出力】\n```\n{last_python_stdout or '(出力なし)'}\n```\n\n"
                "計算が成功しました。その数値・出力が本当に正しいか、別の計算式やエッジケーステストをPythonで実行して検算し、"
                f"正しければ `plan_master.md`（ツール: `{plan_tool}` / `update_memory`）の該当項目を `[x]` に更新して "
                "`work_log.md` に結果を記録してください。"
                f"{forbidden_block}{strategy_block}"
            )
        elif current_state == STATE_FINAL_REPORT:
            sys_msg = (
                "あなたは最終報告担当 (`STATE: FINAL_REPORT`) です。\n"
                "ツールは一切呼び出さず、提供された確定結果のみを用いてユーザー向けの最終レポートを日本語で分かりやすく作成してください。"
            )
            user_prompt = (
                f"【元のユーザー要望】\n{user_message}\n\n"
                f"【完了済み計画 (plan_master.md)】\n```markdown\n{plan_text.strip()}\n```\n\n"
                f"【確定結果 (work_log.md)】\n```markdown\n{work_summary}\n```\n\n"
                "すべての検証が完了しました。最終レポートをわかりやすくまとめてください。"
            )
        elif is_longthink:  # STATE_EXECUTE_STEP in /longthink dynamic orchestration mode
            hb_excerpt = (handoff_text or "").strip()
            if len(hb_excerpt) > 2200:
                hb_excerpt = "... (省略)\n" + hb_excerpt[-2200:]
            if not hb_excerpt:
                hb_excerpt = "（まだ報告はありません）"
            sub_summary = sub_agents_summary or "（まだサブエージェントは生成されていません）"
            sys_msg = (
                "あなたは `/longthink` のマスター（Orchestrator / 動的サブエージェント統括）です。\n"
                f"{rule_md}\n"
                "課題の性質・残タスク数・handoff_board.md の報告を監視し、次のいずれかを動的に実行してください：\n"
                "1. 今何体のサブエージェント（1〜4体）を立ち上げるべきかを決定し、`delegate_parallel_workers`（1〜4体）"
                "または `delegate_sub_agent`（1体）で生成・委譲する。\n"
                "2. `[報告 (要方針変更・停滞)]` を出したサブエージェントがいれば、同じ `agent_id` を指定して"
                "別アプローチの方針変更指示を出して再実行させる。\n"
                "3. 報告が十分揃った項目があれば、必要に応じ `sub_agents/<id>/workspace/...` を `read_file` で確認・統合し、"
                "`save_final_artifact` で成果物を保存して `update_master_plan` で `[x]` に更新する。\n"
                "新規ワークスペースが空でも `list_files` の反復確認は不要です。"
                f"{forbidden_block}{strategy_block}"
            )
            user_prompt = (
                f"【ユーザー要望】\n{user_message[:600]}\n\n"
                f"【全体計画 (master_plan.md) 進捗: 完了 {len(checked)}件 / 未完了 {len(unchecked)}件】\n"
                f"次に進める未完了項目: {first_task}\n"
                "未完了リスト先頭:\n"
                + ("\n".join(f"- [ ] {u}" for u in unchecked[:5]) if unchecked else "- (なし)")
                + f"\n\n【サブエージェント状況 (sub_agents/)】\n{sub_summary}\n\n"
                f"【共有引き継ぎボード (shared_workspace/handoff_board.md)】\n```markdown\n{hb_excerpt}\n```\n\n"
                f"【直近の work_log.md 要約】\n{work_summary}\n\n"
                "上記の状況を踏まえ、今立ち上げるサブエージェント数（1〜4体）や統合・方針変更・計画 `[x]` 更新を決定してツールを実行してください。"
                f"{forbidden_block}{strategy_block}"
            )
        else:  # STATE_EXECUTE_STEP (/normal)
            sys_msg = (
                "あなたは単一タスク実行担当 (`STATE: EXECUTE_STEP`) です。\n"
                f"{rule_md}\n"
                "全体管理はハーネスが行います。あなたは指示された『次の1タスク』だけを実行してください。\n"
                "初期状態の workspace/ が空であることは正常なため、空の `list_files` 確認を繰り返さないでください。\n"
                f"{DIRECT_WRITE_FORMAT_GUIDE}\n"
                "思考は <think>...</think> 内に収め、本文には `[WRITE_FILE: ...]` コードブロックまたは簡潔な結論のみを記述してください。"
                f"{forbidden_block}{strategy_block}"
            )
            user_prompt = (
                f"【ユーザー要望の背景】{user_message[:600]}\n\n"
                f"【直近の work_log.md 要約】\n{work_summary}\n\n"
                f"今は次の1タスクだけを実行してください：{first_task}\n"
                "（Pythonスクリプトや長文ファイルを作成する際は `[WRITE_FILE: path/to/script.py]` + ```python ... ``` 形式を最優先で使用してください。"
                "単純な調査や質問回答のみで完了できる場合はツール完了後に `update_plan` で `[x]` にしてください）"
                f"{forbidden_block}{strategy_block}"
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_prompt},
        ]
        if same_state_history and current_state in (STATE_EXECUTE_STEP, STATE_FIX_ERROR, STATE_VERIFY_RESULT):
            messages.extend(same_state_history[-4:])
        return self._merge_consecutive_roles(messages), active_tools

    async def _call_state_llm(
        self,
        client: AsyncOpenAI,
        messages: list[dict[str, Any]],
        active_tools: list[dict[str, Any]],
        model_override: Optional[str] = None,
        max_think_tokens: int = THINK_TOKEN_LIMIT_DEFAULT,
        max_output_tokens: int = MAX_TOKENS_NORMAL_OUTPUT,
    ) -> Any:
        """Call LM Studio with pinpoint messages, state-filtered tools, optional model routing, and hard token caps."""
        return await self._create_chat_completion_resilient(
            client,
            messages,
            tools=active_tools or None,
            temperature=0.3,
            check_stop=True,
            allow_tool_drop=True,
            model_override=model_override,
            max_think_tokens=max_think_tokens,
            max_output_tokens=max_output_tokens,
        )

    async def _invoke_direct_burst_write(
        self,
        client: AsyncOpenAI,
        step: int,
        session_id: str,
        assistant_msg_id: str,
        session: dict[str, Any],
        thinking_log: list[dict[str, Any]],
        task_summary: str,
        last_thought_text: str,
        failure_diag: str,
        active_tools: Optional[list[dict[str, Any]]],
        allowed_names: Optional[set[str]] = None,
        model_override: Optional[str] = None,
        context_label: str = "",
        persist_logs: bool = True,
    ) -> tuple[list[dict[str, Any]], Optional[str], str]:
        """One-shot rescue mode (`DIRECT_BURST_WRITE`): skips <think> completely and sets max_tokens=-1."""
        burst_start = time.monotonic()
        target_model = (model_override or "").strip() or self.state.model
        use_burst_json_tools = not _is_coder_model_id(target_model, self.state.coder_model)
        burst_tools = (active_tools or None) if use_burst_json_tools else None
        self.state.harness_state = "DIRECT_BURST_WRITE"
        self.state.phase_state = "working"
        burst_label = "🚀 [Direct Burst] 思考スキップ＆出力無制限でコード生成を実行中..."
        self.state.phase_label = burst_label
        self.state.phase_started_at = burst_start
        await self.broker.publish(
            "status",
            {
                **self._status_payload(),
                "state": "working",
                "step": step,
                "label": burst_label,
            },
        )
        await self.broker.publish(
            "direct_burst",
            {
                "type": "direct_burst",
                "step": step,
                "status": "running",
                "model": target_model,
                "context_label": context_label,
                "failure_diag": failure_diag[:400],
                "message": burst_label,
                "session_id": session_id,
            },
        )

        thought_snip = _compress_thought_summary(last_thought_text or "")
        sys_prompt = (
            "あなたは `DIRECT_BURST_WRITE`（思考スキップ・コード直接出力専用）担当です。\n"
            "思考タグ（`<think>`）は直前のステップで既に完了しています。このターンでは `<think>` を一切出力せず、"
            "1トークン目から直ちにファイル書き込みブロックまたはツール呼び出しのみを出力してください。\n\n"
            f"{DIRECT_WRITE_FORMAT_GUIDE}\n\n"
            "注意: ツール引数JSONが途中切断・破損しないよう、長いPythonコードや設定ファイルは必ず "
            "`[WRITE_FILE: path/to/file.py]` + ```python ... ``` の形式で完全なコードをそのまま書き出してください。"
        )
        thought_line = f"【思考要点（圧縮）】{thought_snip}\n\n" if thought_snip else ""
        user_prompt = (
            f"【コンテキスト】{context_label or 'Direct Burst Rescue'}\n"
            f"【実行タスク】\n{(task_summary or '').strip()[:1200]}\n\n"
            f"{thought_line}"
            f"【直前のツール失敗・重複原因】\n{(failure_diag or '').strip()[:700]}\n\n"
            "すでに直前の思考は確定しています。`<think>` は完全にスキップし、"
            "今すぐ 1文字目から `[WRITE_FILE: path/to/script.py]` + ```python ... ``` "
            "（または必要なツール呼び出し）を一気に出力してください。"
        )
        base_messages: list[dict[str, Any]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prefer_write_prefill = (
            _is_coder_model_id(target_model, self.state.coder_model)
            or ("write_file" in (allowed_names or set()))
            or "write_file" in {
                str((t.get("function") or {}).get("name") or "").strip()
                for t in (active_tools or [])
                if isinstance(t, dict)
            }
            or any(kw in f"{task_summary} {context_label}".lower() for kw in ("python", "script", ".py", "ultrathink", "fix_error", "verify"))
        )
        prefill_content = (
            "<think>\n思考完了。直ちにファイルを作成する。\n</think>\n[WRITE_FILE: "
            if prefer_write_prefill
            else "<think>\n思考完了。直ちにファイルを作成する。\n</think>\n"
        )
        prefill_messages = base_messages + [{"role": "assistant", "content": prefill_content}]
        fallback_messages = self._merge_consecutive_roles(
            base_messages
            + [
                {
                    "role": "user",
                    "content": "思考は完全に終了しました。`<think>` を書かず今すぐ `[WRITE_FILE: path/to/file.py]` + コードブロックのみを出力してください。",
                },
            ]
        )

        resp = None
        used_assistant_prefill = False
        for cand_idx, msg_cand in enumerate((prefill_messages, fallback_messages)):
            try:
                resp = await self._create_chat_completion_resilient(
                    client,
                    msg_cand,
                    tools=burst_tools,
                    temperature=0.2,
                    check_stop=True,
                    allow_tool_drop=True,
                    model_override=target_model,
                    max_think_tokens=0 if cand_idx == 0 else 128,
                    max_output_tokens=MAX_TOKENS_DIRECT_BURST,
                    frequency_penalty=0.5,
                    presence_penalty=0.3,
                )
                used_assistant_prefill = cand_idx == 0
                break
            except BadRequestError:
                continue

        if resp is None:
            resp = await self._create_chat_completion_resilient(
                client,
                base_messages,
                tools=burst_tools,
                temperature=0.2,
                check_stop=True,
                allow_tool_drop=True,
                model_override=target_model,
                max_think_tokens=128,
                max_output_tokens=MAX_TOKENS_DIRECT_BURST,
                frequency_penalty=0.5,
                presence_penalty=0.3,
            )

        burst_dur = round(time.monotonic() - burst_start, 1)
        msg = resp.choices[0].message
        raw_content = msg.content or ""
        burst_is_ut = "ultrathink" in (context_label or "").lower() or "10_active_workspace" in (task_summary or "")
        burst_auto_path = "10_active_workspace/src_auto.py" if burst_is_ut else "workspace/script_auto.py"
        if used_assistant_prefill and prefer_write_prefill and not getattr(msg, "tool_calls", None):
            stripped_rc = raw_content.lstrip()
            if stripped_rc and not stripped_rc.lower().startswith(("[write_file:", "<think>")):
                if re.match(r"^[^\[\n]{1,200}\]", stripped_rc):
                    raw_content = f"[WRITE_FILE: {stripped_rc}"
                    msg.content = raw_content
                elif stripped_rc.startswith("```"):
                    raw_content = f"[WRITE_FILE: {burst_auto_path}]\n{stripped_rc}"
                    msg.content = raw_content
        _, visible_text = _extract_think_and_content(raw_content)
        burst_orphan_path = burst_auto_path if prefer_write_prefill else None
        burst_calls, burst_diag = _parse_response_tool_calls_with_diagnostics(
            msg,
            visible_text,
            raw_content,
            allowed_names=allowed_names,
            orphan_rescue_path=burst_orphan_path,
        )

        target_summary = ", ".join(
            str(c["arguments"].get("path") or c["name"]) for c in burst_calls
        ) if burst_calls else "(抽出なし)"
        burst_entry = {
            "type": "direct_burst",
            "step": step,
            "duration": burst_dur,
            "status": "completed" if burst_calls else "failed",
            "model": target_model,
            "context_label": context_label,
            "failure_diag": failure_diag[:400],
            "extracted": target_summary,
            "output_chars": len(raw_content),
            "diagnostic": burst_diag or "",
            "message": f"🚀 [Direct Burst] 思考スキップ＆出力無制限完了 ({burst_dur}s / {len(raw_content)}文字 -> {target_summary})",
        }
        if persist_logs:
            async with self._persist_lock:
                thinking_log.append(burst_entry)
                self.live_thinking_log.append(burst_entry)
                self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
        await self.broker.publish("direct_burst", {**burst_entry, "session_id": session_id})
        return burst_calls, burst_diag, (visible_text or raw_content)

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
            std_override = (self.state.standard_model or "").strip() or None
            resp = await self._create_chat_completion_resilient(
                client,
                [
                    {"role": "system", "content": "あなたは記憶整理担当です。\n" + self.memory.build_context_block()},
                    {"role": "user", "content": prompt},
                ],
                tools=[TOOL_SCHEMAS[0]],  # update_memory only
                temperature=0.2,
                check_stop=False,
                allow_tool_drop=True,
                model_override=std_override,
                max_think_tokens=1000,
                max_output_tokens=1500,
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
        if tool_name == "update_master_plan":
            return f"全体計画更新{suffix}: master_plan.md"
        if tool_name == "delegate_parallel_workers":
            raw_tasks = tool_args.get("tasks") if isinstance(tool_args, dict) else None
            cnt = min(4, len(raw_tasks)) if isinstance(raw_tasks, list) and raw_tasks else 4
            return f"4並列サブエージェント実行{suffix} ({cnt}体同時稼働)"
        if tool_name == "delegate_sub_agent":
            aid = str(tool_args.get("agent_id") or tool_args.get("agent_role") or "sub_agent").strip()
            title = str(tool_args.get("task_title", "")).strip()[:24]
            return f"サブエージェント委譲{suffix} ({aid}): {title}"
        if tool_name == "save_final_artifact":
            fname = str(tool_args.get("filename", "")).strip()
            return f"最終成果物保存{suffix}: {fname}"
        if tool_name == "upsert_hypothesis":
            hid = str(tool_args.get("hyp_id") or "HYP").strip()
            title = str(tool_args.get("title", "")).strip()[:24]
            return f"仮説記録{suffix} ({hid}): {title}"
        if tool_name == "set_hypothesis_status":
            hid = str(tool_args.get("hyp_id", "")).strip()
            st = str(tool_args.get("status", "")).strip()
            return f"ステータス確定{suffix} ({hid} -> {st})"
        if tool_name == "update_global_index":
            return f"目次更新{suffix}: GLOBAL_INDEX.md"
        if tool_name == "crystallize_knowledge":
            title = str(tool_args.get("title", "")).strip()[:24]
            return f"知識結晶化{suffix}: {title}"
        if tool_name == "finalize_ultrathink":
            return f"最終統合レポート生成{suffix}"
        return f"{tool_name} 実行{suffix}"

    @staticmethod
    def _sanitize_intermediate_narration(text: str) -> str:
        """Keep intermediate progress messages concise (1-2 sentences) without code blocks."""
        cleaned = re.sub(r"<tool_call>[\s\S]*?(?:</tool_call>|$)", "", text)
        cleaned = re.sub(r"\[WRITE_FILE:[^\]]*\]\s*\r?\n\s*```[\s\S]*?```", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\[RUN_PYTHON\]\s*\r?\n\s*```[\s\S]*?```", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'\{[^{}]*"name"\s*:\s*"[^"]+"[\s\S]*\}', "", cleaned)
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        joined = " ".join(lines).strip()
        if len(joined) > 140:
            joined = joined[:137] + "..."
        return joined

    @staticmethod
    def _load_session_forbidden(
        session: dict[str, Any],
        work_log_path: Optional[Path] = None,
        max_items: int = 8,
    ) -> list[str]:
        """Restore persisted D1 forbidden approaches from session metadata and work_log.md."""
        collected: list[str] = []
        raw_meta = session.get("forbidden_approaches")
        if isinstance(raw_meta, list):
            for item in raw_meta:
                val = str(item or "").strip()
                if val and val not in collected:
                    collected.append(val)
        if work_log_path and work_log_path.exists():
            try:
                for line in work_log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    m = re.search(r"\[D1_FORBIDDEN\]\s*(.+)$", line)
                    if m:
                        val = m.group(1).strip()
                        if val and val not in collected:
                            collected.append(val)
            except Exception:
                pass
        return collected[-max_items:]

    def _record_session_forbidden(
        self,
        session: dict[str, Any],
        action: str,
        work_log_path: Optional[Path] = None,
        tools_obj: Optional[WorkspaceTools] = None,
        max_items: int = 12,
    ) -> list[str]:
        """Persist a D1 forbidden approach into session metadata & work_log.md so PAUSED_FOR_USER resumes retain it."""
        cleaned = str(action or "").strip()
        existing = session.get("forbidden_approaches")
        items: list[str] = [
            str(x).strip()
            for x in (existing if isinstance(existing, list) else [])
            if str(x or "").strip()
        ]
        if cleaned and cleaned not in items:
            items.append(cleaned)
        items = items[-max_items:]
        session["forbidden_approaches"] = items
        try:
            self.sessions.save_session(session)
        except Exception:
            pass
        if cleaned:
            if tools_obj is not None:
                tools_obj._append_work_log(f"[D1_FORBIDDEN] {cleaned}")
            elif work_log_path is not None:
                try:
                    work_log_path.parent.mkdir(parents=True, exist_ok=True)
                    prev = work_log_path.read_text(encoding="utf-8", errors="replace") if work_log_path.exists() else ""
                    sep = "" if (not prev or prev.endswith("\n")) else "\n"
                    work_log_path.write_text(
                        f"{prev}{sep}- [{_now_time()}] [D1_FORBIDDEN] {cleaned}\n",
                        encoding="utf-8",
                    )
                except Exception:
                    pass
        return items

    async def _run_sub_agent(
        self,
        client: AsyncOpenAI,
        session: dict[str, Any],
        assistant_msg_id: str,
        thinking_log: list[dict[str, Any]],
        agent_id: str,
        agent_role: str,
        task_title: str,
        instruction: str,
        parent_step: int,
        on_worker_update: Optional[Callable[[str, str, Optional[float], Optional[str]], Awaitable[None]]] = None,
    ) -> str:
        """Execute an isolated sub-agent in sessions/<id>/longthink/sub_agents/<agent_id>/."""
        sid = session["id"]
        sdir = self.sessions._session_dir(sid)
        shared_dir = sdir / "longthink" / "shared_workspace"
        handoff_path = shared_dir / "handoff_board.md"
        sub_dir = self.sessions.create_sub_agent_folder(sid, agent_id, task_title, instruction)
        safe_agent_id = sub_dir.name
        sub_start = time.monotonic()

        # Record Orchestrator's instruction on handoff_board.md with exclusive lock
        async with self._handoff_lock:
            hb_prev = handoff_path.read_text(encoding="utf-8", errors="replace") if handoff_path.exists() else ""
            sep = "\n\n" if hb_prev.strip() else ""
            handoff_path.write_text(
                f"{hb_prev.rstrip()}{sep}## [指示] -> {safe_agent_id} ({task_title})\n{instruction.strip()}\n",
                encoding="utf-8",
            )

        sub_tools = WorkspaceTools(
            workspace_root=sub_dir / "workspace",
            memory_dir=self.memory.memory_dir,
            session_dir=sdir,
            plan_path=sub_dir / "plan.md",
            work_log_path=sub_dir / "work_log.md",
            shared_workspace_dir=None,
            on_memory_update=self._on_memory_update,
            on_files_update=self._on_files_update,
        )
        await self._on_files_update()

        role_desc = (
            "検算・品質検証担当 (Critic): 成果物や数値を独立検証し、問題点を指摘または合格判定を行ってください。"
            if agent_role == "critic"
            else "実作業・実装・調査担当 (Worker): 与えられたサブタスクを実行し、検証済みの結論と根拠数値を報告してください。"
        )
        rule_md = self.memory.read_file("rule.md").strip()

        conclusion = ""
        sub_history: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    f"【タスク名】{task_title}\n\n"
                    f"【マスターからの指示】\n{instruction}\n\n"
                    "あなた専用の `workspace/` と `plan.md` のみを使って作業し（`shared_workspace` 等へは直接書けません）、"
                    "完了したら `handoff_board.md` に記録する簡潔な結論・作成ファイル・確定数値を日本語で返してください。"
                ),
            }
        ]

        print(f"[Step {parent_step}] サブエージェント起動 -> {safe_agent_id} ({agent_role}): {task_title}", flush=True)
        final_status = "completed"
        sub_tracker = self._LoopStagnationTracker(threshold=CIRCUIT_BREAKER_THRESHOLD)
        sub_replan_count = 0
        sub_think_unlocked = False
        sub_forbidden: list[str] = self._load_session_forbidden(session, sub_dir / "work_log.md")
        sub_strategy = ""
        last_action_note = "作業着手"
        sub_last_call_key: Optional[str] = None
        sub_tool_fail_streak = 0
        sub_last_thought = ""
        sub_burst_used = False
        sub_iso_step_id: Optional[str] = None
        sub_just_errored = False

        try:
            for sub_step in itertools.count(1):
                if self._stop_event.is_set():
                    conclusion = conclusion or "（ユーザーにより中断されました）"
                    break

                if on_worker_update:
                    await on_worker_update(
                        "thinking",
                        f"Step {sub_step}: 推論中...",
                        round(time.monotonic() - sub_start, 1),
                        None,
                    )

                sub_plan_path = sub_dir / "plan.md"
                sub_plan_text = sub_plan_path.read_text(encoding="utf-8", errors="replace") if sub_plan_path.exists() else ""
                plan_snap_before = self._parse_plan_checklist(sub_plan_text)
                ws_files = [
                    p.relative_to(sub_dir / "workspace").as_posix()
                    for p in sorted((sub_dir / "workspace").rglob("*"))
                    if p.is_file() and "trash_bin" not in p.parts and "tool_isolated_ws" not in p.parts
                ]
                ws_files_str = (
                    ", ".join(ws_files)
                    if ws_files
                    else "（空 — 正常な初期状態です。list_files での再確認は不要です。必要なファイルは [WRITE_FILE: ...] で新規作成してください）"
                )
                sub_forbid_note = (
                    "\n【禁止アクション (d1-2B 指定)】\n" + "\n".join(f"- {x}" for x in sub_forbidden)
                    if sub_forbidden
                    else ""
                )
                sub_strat_note = (
                    f"\n【D1-2B 強制新戦略】\n{sub_strategy}\n"
                    if sub_strategy
                    else ""
                )
                sub_system = (
                    f"あなたは独立した擬似サブエージェント `{safe_agent_id}` です。\n"
                    f"役割: {role_desc}\n"
                    "厳守ルール: 他のサブエージェントの作業フォルダは見れません。ファイル操作は自分専用の `workspace/` と `plan.md` のみに限定してください。\n"
                    "初期状態で `workspace/` が空でも `list_files` で空確認を繰り返さず、すぐ `[WRITE_FILE: ...]` 等で実装・検証を進めてください。\n"
                    f"{DIRECT_WRITE_FORMAT_GUIDE}\n"
                    "作業が完了したら結果・確定数値だけを簡潔に出力してください（完了報告はハーネスが handoff_board.md へ自動集約します）。"
                    f"{sub_forbid_note}{sub_strat_note}\n\n"
                    f"### rule.md\n```markdown\n{rule_md}\n```\n\n"
                    f"### あなたの専用 plan.md (sub_agents/{safe_agent_id}/plan.md)\n```markdown\n{sub_plan_text}\n```\n\n"
                    f"### あなたの専用 workspace/ ファイル一覧\n{ws_files_str}"
                )
                messages = [{"role": "system", "content": sub_system}, *sub_history[-8:]]

                if sub_just_errored:
                    route_res = await self._route_coding_model_via_d1(
                        client=client,
                        step=parent_step,
                        session_id=sid,
                        assistant_msg_id=assistant_msg_id,
                        session=session,
                        thinking_log=thinking_log,
                        task_summary=f"[{safe_agent_id}] エラー直後の原因分析・戦略修正: {task_title}",
                        context_label=f"Sub-Agent Error Analysis: {safe_agent_id} (SubStep {sub_step})",
                        forced_choice="main_model",
                        forced_reason="エラー・停滞発生直後の原因分析と戦略修正のためメインモデルを担当指定",
                    )
                    sub_just_errored = False
                else:
                    route_res = await self._route_coding_model_via_d1(
                        client=client,
                        step=parent_step,
                        session_id=sid,
                        assistant_msg_id=assistant_msg_id,
                        session=session,
                        thinking_log=thinking_log,
                        task_summary=f"[{safe_agent_id}] {task_title}: {instruction[:500]}\n{sub_strategy}",
                        context_label=f"Sub-Agent: {safe_agent_id} (SubStep {sub_step})",
                    )
                sub_model_override = route_res.get("selected_model") or None
                sub_call_messages, sub_call_tools, sub_use_json_tools = self._apply_pure_coder_constraints(
                    messages, TOOL_SCHEMAS, route_res
                )

                sub_think_cap = MAX_TOKENS_UNLOCKED_CAP if sub_think_unlocked else THINK_TOKEN_LIMIT_DEFAULT
                sub_output_cap = MAX_TOKENS_UNLOCKED_CAP if sub_think_unlocked else MAX_TOKENS_NORMAL_OUTPUT
                resp = await self._create_chat_completion_resilient(
                    client,
                    sub_call_messages,
                    tools=sub_call_tools,
                    temperature=0.3,
                    check_stop=True,
                    allow_tool_drop=True,
                    model_override=sub_model_override,
                    max_think_tokens=sub_think_cap,
                    max_output_tokens=sub_output_cap,
                )

                msg = resp.choices[0].message
                raw_content = msg.content or ""
                reasoning_sub = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
                think_sub, visible_text = _extract_think_and_content(raw_content)
                sub_thought_comb = "\n".join(p for p in [reasoning_sub, think_sub] if p).strip()
                if sub_thought_comb:
                    sub_last_thought = sub_thought_comb

                sub_allowed_names = {t["function"]["name"] for t in TOOL_SCHEMAS}
                calls, sub_parse_diag = _parse_response_tool_calls_with_diagnostics(
                    msg,
                    visible_text,
                    raw_content,
                    allowed_names=sub_allowed_names,
                    orphan_rescue_path="workspace/script_auto.py" if not sub_use_json_tools else None,
                )
                if calls and (not sub_use_json_tools or any(c.get("_orphan_rescue") for c in calls)):
                    calls = self._augment_pure_coder_write_calls(calls, sub_allowed_names)

                if not calls and sub_parse_diag:
                    sub_tool_fail_streak += 1
                    if sub_tool_fail_streak >= 2 and not sub_burst_used:
                        sub_burst_used = True
                        burst_calls, burst_diag, burst_vis = await self._invoke_direct_burst_write(
                            client=client,
                            step=parent_step,
                            session_id=sid,
                            assistant_msg_id=assistant_msg_id,
                            session=session,
                            thinking_log=thinking_log,
                            task_summary=f"[{safe_agent_id}] {task_title}\n{instruction}",
                            last_thought_text=sub_last_thought,
                            failure_diag=sub_parse_diag,
                            active_tools=TOOL_SCHEMAS,
                            allowed_names=sub_allowed_names,
                            model_override=sub_model_override or self.state.model,
                            context_label=f"Sub-Agent: {safe_agent_id}",
                        )
                        if burst_calls:
                            calls = burst_calls
                            sub_parse_diag = burst_diag
                            visible_text = burst_vis
                        else:
                            sub_parse_diag = burst_diag or sub_parse_diag

                if not calls and sub_parse_diag:
                    sub_history.append({"role": "assistant", "content": visible_text or raw_content or "(不正形式)"})
                    sub_history.append(
                        {
                            "role": "user",
                            "content": f"【ツール形式エラー診断】\n{sub_parse_diag}\n\n{DIRECT_WRITE_FORMAT_GUIDE}",
                        }
                    )
                    cb_diag = sub_tracker.record_action(
                        f"call:parse_err:{sub_parse_diag[:40]}",
                        f"res:parse_err:{sub_parse_diag[:80]}",
                        f"Step {sub_step}: ツール引数エラー -> {sub_parse_diag[:100]}",
                        "JSON引数パースまたはコード出力の形式失敗が続いている",
                    )
                    if cb_diag:
                        # handled right below cleanly via stalled_reason
                        stalled_reason = cb_diag
                    else:
                        continue
                elif not calls:
                    conclusion = visible_text or raw_content or "（サブエージェント出力なし）"
                    break
                else:
                    stalled_reason = None

                if calls:
                    sub_history.append({"role": "assistant", "content": self._sanitize_intermediate_narration(visible_text) or "[ツール実行]"})
                    stalled_reason = None
                    sub_ran_ok = False
                    sub_ran_err = False
                    if any(c["name"] in ("write_file", "read_file", "list_files", "run_python", "run_command") for c in calls):
                        sub_iso_step_id = sub_iso_step_id or f"{safe_agent_id}_{parent_step}_{sub_step}"
                        sub_tools.enter_isolated_workspace(sub_iso_step_id)
                    for c in calls:
                        t_name = c["name"]
                        t_args = c["arguments"]
                        call_key = _canonical_call_key(t_name, t_args)
                        if sub_last_call_key is not None and call_key == sub_last_call_key:
                            sub_tool_fail_streak += 1
                            dup_diag = (
                                f"エラー（重複呼び出しブロック）: 直前と全く同一のツール名 `{t_name}`・同一引数での再実行はブロックされました。"
                                "JSON引数エスケープによる失敗やループを回避するため、`[WRITE_FILE: path/to/script.py]` + ```python ... ``` "
                                "形式で直接ファイル記述するか、処理を複数ファイルへ分割してください。"
                            )
                            if sub_tool_fail_streak >= 2 and not sub_burst_used:
                                sub_burst_used = True
                                burst_calls, _, _ = await self._invoke_direct_burst_write(
                                    client=client,
                                    step=parent_step,
                                    session_id=sid,
                                    assistant_msg_id=assistant_msg_id,
                                    session=session,
                                    thinking_log=thinking_log,
                                    task_summary=f"[{safe_agent_id}] {task_title}\n{instruction}",
                                    last_thought_text=sub_last_thought,
                                    failure_diag=dup_diag,
                                    active_tools=TOOL_SCHEMAS,
                                    allowed_names=sub_allowed_names,
                                    model_override=sub_model_override or self.state.model,
                                    context_label=f"Sub-Agent Duplicate Block: {safe_agent_id}",
                                )
                                if burst_calls:
                                    burst_calls = self._augment_pure_coder_write_calls(burst_calls, sub_allowed_names)
                                    sub_iso_step_id = sub_iso_step_id or f"{safe_agent_id}_{parent_step}_{sub_step}"
                                    sub_tools.enter_isolated_workspace(sub_iso_step_id)
                                for bc in burst_calls:
                                    b_res = await sub_tools.execute(bc["name"], bc["arguments"])
                                    sub_last_call_key = _canonical_call_key(bc["name"], bc["arguments"])
                                    sub_history.append(
                                        {"role": "user", "content": f"【Direct Burst 実行結果 ({bc['name']})】\n{b_res[:1600]}"}
                                    )
                                    if bc["name"] == "run_python":
                                        ok_py, _, _ = self._inspect_python_result(b_res)
                                        if ok_py:
                                            sub_ran_ok = True
                                        else:
                                            sub_ran_err = True
                                            sub_just_errored = True
                                sub_tool_fail_streak = 0
                                sub_tracker.reset(keep_recent=True)
                                break
                            t_res = dup_diag
                            sub_ran_err = True
                            sub_just_errored = True
                        else:
                            if on_worker_update:
                                await on_worker_update(
                                    "working",
                                    self._describe_tool_action(t_name, t_args, in_progress=True),
                                    round(time.monotonic() - sub_start, 1),
                                    None,
                                )
                            w_start = time.monotonic()
                            t_res = await sub_tools.execute(t_name, t_args)
                            w_dur = round(time.monotonic() - w_start, 1)
                            sub_last_call_key = call_key
                            if t_name == "run_python":
                                ok_py, _, _ = self._inspect_python_result(t_res)
                                if ok_py:
                                    sub_ran_ok = True
                                else:
                                    sub_ran_err = True
                                    sub_just_errored = True
                            elif t_name == "run_command":
                                first_ln = t_res.splitlines()[0] if t_res else ""
                                if re.search(r"終了コード:\s*0\b", first_ln) and not t_res.startswith("エラー"):
                                    sub_ran_ok = True
                                else:
                                    sub_ran_err = True
                                    sub_just_errored = True
                            elif t_res.startswith("エラー"):
                                sub_ran_err = True
                                sub_just_errored = True
                            if not t_res.startswith("エラー") and not sub_ran_err:
                                sub_tool_fail_streak = 0
                                sub_burst_used = False
                            sub_work_item = {
                                "type": "work",
                                "step": parent_step,
                                "tool": f"{safe_agent_id}:{t_name}",
                                "label": f"[{safe_agent_id}] {self._describe_tool_action(t_name, t_args, in_progress=False)}",
                                "duration": w_dur,
                                "arguments": t_args,
                                "result": t_res,
                            }
                            async with self._persist_lock:
                                thinking_log.append(sub_work_item)
                                self.live_thinking_log.append(sub_work_item)
                                self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
                            await self.broker.publish("work", {**sub_work_item, "session_id": sid})

                        call_sig, res_sig, reason = self._build_action_stagnation_signature(t_name, t_args, t_res)
                        last_action_note = f"{t_name}: {reason}"
                        cb_trigger = sub_tracker.record_action(
                            call_sig,
                            res_sig,
                            f"Step {sub_step}: {t_name}({json.dumps(t_args, ensure_ascii=False)[:80]}) -> {t_res.splitlines()[0][:100] if t_res else ''}",
                            reason,
                        )
                        if cb_trigger:
                            stalled_reason = cb_trigger
                            break

                        trimmed_res = t_res[:2000] + "\n... (省略)" if len(t_res) > 2000 else t_res
                        sub_history.append(
                            {
                                "role": "user",
                                "content": f"【ツール実行結果 ({t_name})】\n{trimmed_res}\n\nこの結果をもとに次の操作を行うか、完了なら最終報告（結論・確定数値）を返してください。",
                            }
                        )

                    if sub_iso_step_id and sub_ran_ok and not sub_ran_err and not stalled_reason:
                        iso_res = sub_tools.complete_isolated_workspace(sub_iso_step_id, mode="normal")
                        sub_iso_step_id = None
                        if iso_res:
                            await self._emit_trash_isolation_log(
                                session, thinking_log, assistant_msg_id, sid, parent_step, iso_res
                            )

                sub_plan_after = sub_plan_path.read_text(encoding="utf-8", errors="replace") if sub_plan_path.exists() else ""
                plan_snap_after = self._parse_plan_checklist(sub_plan_after)
                if len(plan_snap_after[0]) < len(plan_snap_before[0]):
                    sub_replan_count = 0
                    sub_think_unlocked = False
                if not stalled_reason:
                    plan_progressed = (plan_snap_after != plan_snap_before) or (sub_plan_after != sub_plan_text)
                    stalled_reason = sub_tracker.record_step_progress(
                        plan_progressed,
                        f"SubStep {sub_step}: {last_action_note}",
                    )

                if stalled_reason:
                    sub_replan_count += 1
                    sub_just_errored = True
                    diag = await self._invoke_d1_supervisor(
                        client=client,
                        step=parent_step,
                        session_id=sid,
                        assistant_msg_id=assistant_msg_id,
                        session=session,
                        thinking_log=thinking_log,
                        recent_history_items=sub_tracker.recent_steps,
                        plan_text=sub_plan_after,
                        fallback_reason=stalled_reason,
                        context_label=f"Sub-Agent: {safe_agent_id} ({task_title})",
                        intervention_count=sub_replan_count,
                        think_limit_unlocked=sub_think_unlocked,
                    )
                    sub_think_unlocked = bool(sub_think_unlocked or diag.get("think_unlocked"))
                    if diag["forbidden_action"]:
                        if diag["forbidden_action"] not in sub_forbidden:
                            sub_forbidden.append(diag["forbidden_action"])
                        self._record_session_forbidden(session, diag["forbidden_action"], tools_obj=sub_tools)
                    if not diag.get("escalate_to_user"):
                        sub_strategy = diag["new_strategy"]
                        new_sub_plan = (
                            f"# サブエージェント計画 (D1-2B リプラン済 #{sub_replan_count}): {safe_agent_id}\n\n"
                            f"- 詰まり原因: {diag['stuck_reason']}\n"
                            f"- 禁止アクション: {diag['forbidden_action']}\n\n"
                            f"## 新戦略チェックリスト\n"
                            f"- [ ] {sub_strategy}\n"
                            f"- [ ] 結果を検証して結論を報告する\n"
                        )
                        sub_plan_path.write_text(new_sub_plan, encoding="utf-8")
                        sub_tracker.reset(keep_recent=False)
                        unlock_msg = (
                            "\n🔓 難所判定（累計5回介入）により、このサブタスク完了まで思考トークン上限を無制限化しました。"
                            if sub_think_unlocked
                            else ""
                        )
                        sub_history = [
                            {
                                "role": "user",
                                "content": (
                                    f"⚡ 監査モデル (d1-2B) が2回連続停滞を検知し方針を強制変更しました（{sub_replan_count}回目）。\n"
                                    f"- 禁止: {diag['forbidden_action']}\n"
                                    f"- 新戦略: {sub_strategy}{unlock_msg}\n\n"
                                    "あなたの専用 `plan.md` は新戦略に書き換えました。直前の方法は禁止とし、別の手段で今すぐ実行してください。"
                                ),
                            }
                        ]
                        async with self._handoff_lock:
                            hb_rp = handoff_path.read_text(encoding="utf-8", errors="replace") if handoff_path.exists() else ""
                            sep_rp = "\n\n" if hb_rp.strip() else ""
                            handoff_path.write_text(
                                f"{hb_rp.rstrip()}{sep_rp}## [D1-2B リプラン #{sub_replan_count}] {safe_agent_id} ({task_title})\n"
                                f"- 原因: {diag['stuck_reason']}\n- 新戦略: {sub_strategy}\n",
                                encoding="utf-8",
                            )
                        await self._on_files_update()
                        continue

                    final_status = "stuck"
                    conclusion = (
                        f"[要ユーザー判断・完全停滞] 累計 {sub_replan_count} 回介入後も突破できず D1-2B Supervisor が停止判定しました。\n"
                        f"- 原因: {diag['stuck_reason']}\n"
                        f"- 禁止アクション: {diag['forbidden_action']}\n"
                        f"- 推奨・相談: {diag.get('user_consultation_message') or diag['new_strategy']}"
                    )
                    break

                # Periodic intermediate handoff report every 2 steps
                if sub_step % 2 == 0:
                    ws_now = [
                        p.relative_to(sub_dir / "workspace").as_posix()
                        for p in sorted((sub_dir / "workspace").rglob("*"))
                        if p.is_file() and "trash_bin" not in p.parts and "tool_isolated_ws" not in p.parts
                    ]
                    ws_now_str = ", ".join(ws_now) if ws_now else "(空)"
                    async with self._handoff_lock:
                        hb_mid = handoff_path.read_text(encoding="utf-8", errors="replace") if handoff_path.exists() else ""
                        sep_mid = "\n\n" if hb_mid.strip() else ""
                        handoff_path.write_text(
                            f"{hb_mid.rstrip()}{sep_mid}## [中間経過 Step {sub_step}] {safe_agent_id} ({task_title})\n"
                            f"- 直近状況: {last_action_note}\n"
                            f"- 専用 workspace/: {ws_now_str}\n",
                            encoding="utf-8",
                        )
                    if on_worker_update:
                        await on_worker_update(
                            "working",
                            f"Step {sub_step}: 中間報告記録済",
                            round(time.monotonic() - sub_start, 1),
                            None,
                        )
        except Exception as exc:
            final_status = "error"
            conclusion = f"サブエージェント実行エラー ({safe_agent_id}): {exc}"

        if sub_iso_step_id and final_status == "completed":
            iso_res = sub_tools.complete_isolated_workspace(sub_iso_step_id, mode="normal")
            sub_iso_step_id = None
            if iso_res:
                await self._emit_trash_isolation_log(
                    session, thinking_log, assistant_msg_id, sid, parent_step, iso_res
                )

        ws_final = [
            p.relative_to(sub_dir / "workspace").as_posix()
            for p in sorted((sub_dir / "workspace").rglob("*"))
            if p.is_file() and "trash_bin" not in p.parts and "tool_isolated_ws" not in p.parts
        ]
        ws_final_str = ", ".join(ws_final) if ws_final else "(なし)"

        # Mark sub-agent's own plan.md completed only when completed cleanly
        sub_plan_path = sub_dir / "plan.md"
        if sub_plan_path.exists() and final_status == "completed":
            plan_body = sub_plan_path.read_text(encoding="utf-8", errors="replace")
            updated_plan = re.sub(r"^(\s*[-*]\s*)\[\s\]", r"\1[x]", plan_body, flags=re.MULTILINE)
            sub_plan_path.write_text(updated_plan, encoding="utf-8")

        status_label = (
            "要方針変更・停滞"
            if final_status == "stuck"
            else ("エラー" if final_status == "error" else "完了")
        )
        # Append final conclusion / stuck notice to shared_workspace/handoff_board.md
        conc_trimmed = conclusion.strip()
        if len(conc_trimmed) > 1200:
            conc_trimmed = conc_trimmed[:1200] + "\n... (省略)"
        async with self._handoff_lock:
            hb_curr = handoff_path.read_text(encoding="utf-8", errors="replace") if handoff_path.exists() else ""
            sep2 = "\n\n" if hb_curr.strip() else ""
            handoff_path.write_text(
                f"{hb_curr.rstrip()}{sep2}## [報告 ({status_label})] {safe_agent_id} ({task_title})\n"
                f"- 専用 workspace/ ファイル: {ws_final_str}\n"
                f"{conc_trimmed}\n",
                encoding="utf-8",
            )

        total_dur = round(time.monotonic() - sub_start, 1)
        ui_status = "completed" if final_status == "completed" else "error"
        ui_detail = (
            "完了 (handoff_board 記録済)"
            if final_status == "completed"
            else ("⚠️ 停滞・要方針変更 (handoff_board 記録済)" if final_status == "stuck" else conc_trimmed[:120])
        )
        if on_worker_update:
            await on_worker_update(
                ui_status,
                ui_detail,
                total_dur,
                conc_trimmed,
            )
        await self._on_files_update()
        print(
            f"[Step {parent_step}] サブエージェント終了 ({status_label}) -> {safe_agent_id} ({total_dur}s, handoff_board.md に追記)",
            flush=True,
        )
        return f"[{status_label}] (workspace: {ws_final_str})\n{conc_trimmed}"

    @staticmethod
    def _format_worker_label(agent_id: str, agent_role: str, fallback_num: int) -> str:
        m = re.match(r"^(worker|critic)[_-]?(\d+)$", agent_id.strip().lower())
        if m:
            return f"{m.group(1).capitalize()} {int(m.group(2)):02d}"
        if agent_id.strip():
            return agent_id.strip()
        return f"{agent_role.capitalize()} {fallback_num:02d}"

    async def _run_parallel_sub_agents(
        self,
        client: AsyncOpenAI,
        session: dict[str, Any],
        assistant_msg_id: str,
        thinking_log: list[dict[str, Any]],
        tasks_spec: list[dict[str, Any]],
        parent_step: int,
        start_index: int = 1,
    ) -> str:
        """Execute Worker sub-agents sequentially (Parallel = 1 across all modes)."""
        sid = session["id"]
        specs = [t for t in tasks_spec if isinstance(t, dict)][:4]
        if not specs:
            return "サブタスクが指定されませんでした。"

        batch_start = time.monotonic()
        batch_id = f"batch_{parent_step}_{uuid.uuid4().hex[:6]}"
        used_ids: set[str] = set()
        resolved_specs: list[dict[str, Any]] = []
        worker_cards: list[dict[str, Any]] = []

        for idx, spec in enumerate(specs):
            role = str(spec.get("agent_role", "worker")).strip().lower()
            if role not in ("worker", "critic"):
                role = "worker"
            raw_aid = str(spec.get("agent_id", "")).strip()
            aid = raw_aid if raw_aid else f"{role}_{(start_index + idx):02d}"
            while aid in used_ids:
                aid = f"{role}_{len(used_ids) + start_index:02d}_{uuid.uuid4().hex[:3]}"
            used_ids.add(aid)

            title = str(spec.get("task_title") or f"サブタスク {idx + 1}").strip()
            instr = str(spec.get("instruction") or title).strip()
            display_name = self._format_worker_label(aid, role, idx + 1)

            resolved_specs.append(
                {
                    "agent_id": aid,
                    "agent_role": role,
                    "task_title": title,
                    "instruction": instr,
                }
            )
            worker_cards.append(
                {
                    "agent_id": aid,
                    "display_name": display_name,
                    "role": role,
                    "task_title": title,
                    "status": "thinking",
                    "detail": "並列起動中...",
                    "duration": 0.0,
                    "conclusion": "",
                }
            )

        batch_entry: dict[str, Any] = {
            "type": "parallel_workers",
            "batch_id": batch_id,
            "step": parent_step,
            "status": "running",
            "duration": 0.0,
            "workers": worker_cards,
        }
        async with self._persist_lock:
            thinking_log.append(batch_entry)
            self.live_thinking_log.append(batch_entry)
            self.active_parallel_batch = batch_entry
            self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
        await self.broker.publish("parallel_workers", {**batch_entry, "session_id": sid})

        print(
            f"[Step {parent_step}] Worker 順次実行開始 (Parallel = 1): "
            + ", ".join(s["agent_id"] for s in resolved_specs),
            flush=True,
        )

        async def _run_one(slot_idx: int, item: dict[str, Any]) -> tuple[dict[str, Any], str]:
            async def _on_update(
                status: str,
                detail: str,
                dur: Optional[float],
                conclusion_text: Optional[str],
            ) -> None:
                async with self._persist_lock:
                    card = batch_entry["workers"][slot_idx]
                    card["status"] = status
                    card["detail"] = detail
                    if dur is not None:
                        card["duration"] = dur
                    if conclusion_text is not None:
                        card["conclusion"] = conclusion_text
                    self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
                await self.broker.publish("parallel_workers", {**batch_entry, "session_id": sid})

            res = await self._run_sub_agent(
                client,
                session,
                assistant_msg_id,
                thinking_log,
                agent_id=item["agent_id"],
                agent_role=item["agent_role"],
                task_title=item["task_title"],
                instruction=item["instruction"],
                parent_step=parent_step,
                on_worker_update=_on_update,
            )
            return item, res

        pairs: list[tuple[dict[str, Any], str]] = []
        for i, spec in enumerate(resolved_specs):
            pairs.append(await _run_one(i, spec))

        batch_dur = round(time.monotonic() - batch_start, 1)
        async with self._persist_lock:
            batch_entry["status"] = "completed"
            batch_entry["duration"] = batch_dur
            self.active_parallel_batch = None
            self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
        await self.broker.publish("parallel_workers", {**batch_entry, "session_id": sid})
        await self._on_files_update()

        lines = [f"【並列サブエージェント {len(pairs)}体 完了 ({batch_dur}s) — handoff_board.md 更新済】"]
        for spec_item, res_text in pairs:
            lines.append(f"\n### {spec_item['agent_id']} ({spec_item['task_title']})\n{res_text}")
        return "\n".join(lines)

    async def _finish_turn_with_text(
        self,
        client: AsyncOpenAI,
        tools: WorkspaceTools,
        session: dict[str, Any],
        thinking_log: list[dict[str, Any]],
        assistant_msg_id: str,
        step: int,
        user_message: str,
        final_text: str,
        harness_state: str = "COMPLETED",
        consolidate: bool = True,
    ) -> None:
        """Persist final message, reset runtime state, and emit final SSE notifications."""
        session_id = session["id"]
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
        self.state.harness_state = harness_state
        self.active_parallel_batch = None
        self.last_completed_message = assistant_entry

        await self.broker.publish(
            "final_answer",
            {
                "step": step,
                "session_id": session_id,
                "turn_id": assistant_msg_id,
                "message": assistant_entry,
            },
        )
        if consolidate:
            await self._auto_consolidate_memory(client, tools, user_message, final_text, thinking_log)
        self.state.status = "paused" if harness_state == "PAUSED_FOR_USER" else "completed"
        await self.broker.publish("status", self._status_payload())
        self.broker.clear_history()

    @staticmethod
    def _format_step_tag(step: int) -> str:
        return f"STEP_{max(1, int(step)):04d}"

    @classmethod
    def _ensure_atomic_frontmatter(
        cls,
        raw_text: str,
        hyp_id: str,
        title: str,
        status: str,
        step: int,
    ) -> str:
        """Ensure Atomic Markdown YAML frontmatter is present at top of hyp_XXXX.md."""
        meta, body = SessionManager._parse_yaml_frontmatter(raw_text)
        norm_id = re.sub(r"[^A-Z0-9_]", "_", (hyp_id or meta.get("ID") or "HYP_0001").upper()).strip("_")
        if not norm_id.startswith("HYP_"):
            digits = re.search(r"(\d+)", norm_id)
            num = int(digits.group(1)) if digits else 1
            norm_id = f"HYP_{num:04d}"
        norm_title = (title or meta.get("TITLE") or norm_id).strip()
        norm_title = re.sub(r"[\r\n]+", " ", norm_title)
        norm_status = (status or meta.get("STATUS") or "UNVERIFIED").strip("[] ").upper()
        if norm_status not in ("UNVERIFIED", "VERIFIED", "REJECTED"):
            norm_status = "UNVERIFIED"
        step_tag = cls._format_step_tag(step)
        clean_body = (body if meta else raw_text).lstrip("\r\n")
        return (
            "---\n"
            f"ID: {norm_id}\n"
            f"TITLE: {norm_title}\n"
            f"STATUS: {norm_status}\n"
            f"LAST_UPDATE: {step_tag}\n"
            "---\n\n"
            f"{clean_body.rstrip()}\n"
        )

    @staticmethod
    def _next_ultrathink_id(ut_dir: Path, prefix: str) -> str:
        """Allocate next monotonic HYP_XXXX or FACT_XXXX id across active, KB, and archive."""
        max_num = 0
        pattern = re.compile(rf"^(?:{re.escape(prefix)}_|hyp_|fact_)(\d+)$", re.IGNORECASE)
        for folder in ("10_active_workspace", "20_knowledge_base", "90_archive"):
            d = ut_dir / folder
            if not d.exists():
                continue
            for p in d.iterdir():
                if not p.is_file():
                    continue
                m = pattern.match(p.stem)
                if m:
                    max_num = max(max_num, int(m.group(1)))
                if p.suffix.lower() == ".md":
                    meta, _ = SessionManager._parse_yaml_frontmatter(
                        p.read_text(encoding="utf-8", errors="replace")
                    )
                    mid = meta.get("ID", "")
                    m2 = pattern.match(mid)
                    if m2:
                        max_num = max(max_num, int(m2.group(1)))
        gi_path = ut_dir / "GLOBAL_INDEX.md"
        if gi_path.exists():
            for m3 in re.finditer(rf"\b{re.escape(prefix)}_(\d+)\b", gi_path.read_text(encoding="utf-8", errors="replace")):
                max_num = max(max_num, int(m3.group(1)))
        return f"{prefix}_{max_num + 1:04d}"

    def _sync_global_index_from_files(self, ut_dir: Path, session_id: str) -> str:
        """Rebuild GLOBAL_INDEX.md deterministically from Atomic Markdown headers and existing rows."""
        metrics = self.sessions.get_ultrathink_metrics(session_id)
        rows = [
            "| ID | TITLE | STATUS | FILE | LAST_UPDATE |",
            "|---|---|---|---|---|",
        ]
        for item in metrics["items"]:
            title_safe = item["title"].replace("|", "/")
            rows.append(
                f"| {item['id']} | {title_safe} | [{item['status']}] | {item['file']} | {item['last_update']} |"
            )
        content = (
            "# 全体仮説インデックス (GLOBAL_INDEX.md)\n\n"
            f"- 集計: UNVERIFIED={metrics['unverified']} | "
            f"VERIFIED={metrics['verified']} | "
            f"REJECTED={metrics['rejected']} | "
            f"KB={metrics['knowledge_facts']}\n\n"
            + "\n".join(rows)
            + "\n"
        )
        (ut_dir / "GLOBAL_INDEX.md").write_text(content, encoding="utf-8")
        return content

    @staticmethod
    def _select_active_focus_files(ut_dir: Path, focused_hyp_id: Optional[str] = None) -> list[Path]:
        """Strictly select at most 2 active workspace files for context shielding."""
        act_dir = ut_dir / "10_active_workspace"
        if not act_dir.exists():
            return []
        all_files = sorted(
            [p for p in act_dir.iterdir() if p.is_file() and "trash_bin" not in p.parts and "tool_isolated_ws" not in p.parts],
            key=lambda p: p.name.lower(),
        )
        if not all_files:
            return []

        selected: list[Path] = []
        if focused_hyp_id:
            digits = re.search(r"(\d+)", focused_hyp_id)
            num_str = f"{int(digits.group(1)):04d}" if digits else ""
            if num_str:
                hyp_match = act_dir / f"hyp_{num_str}.md"
                src_match = act_dir / f"src_{num_str}.py"
                if hyp_match.exists():
                    selected.append(hyp_match)
                if src_match.exists() and len(selected) < 2:
                    selected.append(src_match)

        if not selected:
            # Prefer UNVERIFIED hyp_*.md and its paired script
            for p in all_files:
                if p.name.startswith("hyp_") and p.suffix == ".md":
                    meta, _ = SessionManager._parse_yaml_frontmatter(
                        p.read_text(encoding="utf-8", errors="replace")
                    )
                    if meta.get("STATUS", "UNVERIFIED").upper() == "UNVERIFIED":
                        selected.append(p)
                        num_m = re.search(r"(\d+)", p.stem)
                        if num_m:
                            paired_py = act_dir / f"src_{int(num_m.group(1)):04d}.py"
                            if paired_py.exists():
                                selected.append(paired_py)
                        break

        if len(selected) < 2:
            for p in all_files:
                if p not in selected:
                    selected.append(p)
                    if len(selected) >= 2:
                        break
        return selected[:2]

    async def _publish_ultrathink_status(
        self,
        session_id: str,
        gear: str,
        temperature: float,
        focus_paths: list[Path],
        last_commit: Optional[str] = None,
    ) -> dict[str, Any]:
        metrics = self.sessions.get_ultrathink_metrics(session_id)
        sdir = self.sessions._session_dir(session_id)
        focus_rels = []
        for p in focus_paths[:2]:
            try:
                focus_rels.append(p.relative_to(sdir).as_posix())
            except ValueError:
                focus_rels.append(p.name)
        info = {
            "gear": gear,
            "temperature": temperature,
            "focus_files": focus_rels,
            "last_commit": last_commit,
            **metrics,
        }
        self.state.ultrathink_info = info
        await self.broker.publish("ultrathink_status", {"session_id": session_id, **info})
        return info

    async def _run_ultrathink_gc(
        self,
        client: AsyncOpenAI,
        ut_dir: Path,
        session_id: str,
        step: int,
        north_star_text: str,
    ) -> tuple[int, int, Optional[str], str]:
        """20-step GC: crystallize VERIFIED files into 20_knowledge_base/ and move done/rejected to 90_archive/."""
        act_dir = ut_dir / "10_active_workspace"
        kb_dir = ut_dir / "20_knowledge_base"
        arch_dir = ut_dir / "90_archive"
        verified_files: list[tuple[Path, dict[str, str], str]] = []
        rejected_files: list[Path] = []

        if act_dir.exists():
            for p in sorted(act_dir.glob("hyp_*.md")):
                raw = p.read_text(encoding="utf-8", errors="replace")
                meta, body = SessionManager._parse_yaml_frontmatter(raw)
                st = (meta.get("STATUS") or "UNVERIFIED").upper()
                if st == "VERIFIED":
                    verified_files.append((p, meta, body))
                elif st == "REJECTED":
                    rejected_files.append(p)

        crystallized_count = 0
        archived_count = 0
        crystallized_summary = ""

        if verified_files:
            snippets: list[str] = []
            for p, meta, body in verified_files[:4]:
                hid = meta.get("ID", p.stem.upper())
                htitle = meta.get("TITLE", p.stem)
                trimmed_body = body.strip()
                if len(trimmed_body) > 900:
                    trimmed_body = trimmed_body[:900] + "\n... (省略)"
                snippets.append(f"## {hid}: {htitle}\n{trimmed_body}")
            combined_src = "\n\n".join(snippets)

            fact_id = self._next_ultrathink_id(ut_dir, "FACT")
            num_m = re.search(r"(\d+)", fact_id)
            fact_num = int(num_m.group(1)) if num_m else 1
            fact_filename = f"fact_{fact_num:04d}.md"
            fact_title = ", ".join(m.get("TITLE", p.stem) for p, m, _ in verified_files[:3])[:80]
            crystallized_body = ""

            try:
                gc_sys = (
                    f"{north_star_text}\n\n"
                    "あなたは /ultrathink の 20ステップ周期・知識結晶化 (GC) 担当です。\n"
                    "提供された VERIFIED 仮説群から重複を削ぎ落とし、後続ステップで再利用可能な確定事実・数値・結論だけを簡潔な Markdown に統合してください。"
                )
                gc_user = (
                    f"【Step {step} 結晶化対象 (VERIFIED 仮説 {len(verified_files)}件)】\n\n"
                    f"{combined_src}\n\n"
                    "出力はそのまま `20_knowledge_base/` に保存できる簡潔な確定知識レポートにしてください。"
                )
                std_gc_model = (self.state.standard_model or "").strip() or None
                resp = await self._create_chat_completion_resilient(
                    client,
                    [{"role": "system", "content": gc_sys}, {"role": "user", "content": gc_user}],
                    tools=None,
                    temperature=0.0,
                    check_stop=True,
                    allow_tool_drop=True,
                    model_override=std_gc_model,
                    max_think_tokens=MAX_TOKENS_LOW_GEAR,
                    max_output_tokens=1500,
                )
                raw_resp = resp.choices[0].message.content or ""
                _, vis_resp = _extract_think_and_content(raw_resp)
                crystallized_body = (vis_resp or raw_resp).strip()
            except Exception:
                crystallized_body = ""

            if not crystallized_body:
                crystallized_body = combined_src

            fact_md = (
                "---\n"
                f"ID: {fact_id}\n"
                f"TITLE: {fact_title}\n"
                f"STATUS: VERIFIED\n"
                f"LAST_UPDATE: {self._format_step_tag(step)}\n"
                "---\n\n"
                f"{crystallized_body.rstrip()}\n"
            )
            (kb_dir / fact_filename).write_text(fact_md, encoding="utf-8")

            kb_index_path = kb_dir / "INDEX_KB.md"
            kb_prev = kb_index_path.read_text(encoding="utf-8", errors="replace") if kb_index_path.exists() else (
                "# 知識ベース目次 (INDEX_KB.md)\n\n| ID | TITLE | FILE | STEP |\n|---|---|---|---|\n"
            )
            safe_title = fact_title.replace("|", "/")
            sep = "" if (not kb_prev or kb_prev.endswith("\n")) else "\n"
            kb_index_path.write_text(
                f"{kb_prev}{sep}| {fact_id} | {safe_title} | 20_knowledge_base/{fact_filename} | {self._format_step_tag(step)} |\n",
                encoding="utf-8",
            )
            crystallized_count = len(verified_files)
            crystallized_summary = f"{fact_filename} ({crystallized_count}件結晶化)"

        def _move_with_paired_src(hyp_path: Path) -> int:
            moved = 0
            if hyp_path.exists():
                dst = arch_dir / hyp_path.name
                if dst.exists():
                    dst = arch_dir / f"{hyp_path.stem}_s{step}{hyp_path.suffix}"
                shutil.move(str(hyp_path), str(dst))
                moved += 1
            num_match = re.search(r"(\d+)", hyp_path.stem)
            if num_match:
                src_path = act_dir / f"src_{int(num_match.group(1)):04d}.py"
                if src_path.exists():
                    dst_src = arch_dir / src_path.name
                    if dst_src.exists():
                        dst_src = arch_dir / f"{src_path.stem}_s{step}{src_path.suffix}"
                    shutil.move(str(src_path), str(dst_src))
                    moved += 1
            return moved

        for p, _, _ in verified_files:
            archived_count += _move_with_paired_src(p)
        for rp in rejected_files:
            archived_count += _move_with_paired_src(rp)

        gi_content = self._sync_global_index_from_files(ut_dir, session_id)
        await self._on_memory_update("plan.md", gi_content)
        commit_info = self.sessions.git_commit_snapshot(
            session_id, f"Step {step}: GC crystallize={crystallized_count}, archived={archived_count}"
        )
        await self._on_files_update()
        return crystallized_count, archived_count, commit_info, crystallized_summary

    async def _execute_ultrathink_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        ut_dir: Path,
        tools: WorkspaceTools,
        session_id: str,
        step: int,
    ) -> tuple[str, Optional[str], bool, Optional[str], Optional[str]]:
        """Execute an /ultrathink tool; returns (result_text, touched_hyp_id, status_changed, final_report, git_msg)."""
        act_dir = ut_dir / "10_active_workspace"
        kb_dir = ut_dir / "20_knowledge_base"
        args = tool_args if isinstance(tool_args, dict) else {}

        if tool_name == "upsert_hypothesis":
            raw_hid = str(args.get("hyp_id", "")).strip().upper()
            if not raw_hid or not re.match(r"^HYP_\d+$", raw_hid):
                digits = re.search(r"(\d+)", raw_hid)
                raw_hid = f"HYP_{int(digits.group(1)):04d}" if digits else self._next_ultrathink_id(ut_dir, "HYP")
            digits2 = re.search(r"(\d+)", raw_hid)
            hyp_num = int(digits2.group(1)) if digits2 else 1
            hyp_file = act_dir / f"hyp_{hyp_num:04d}.md"
            title = str(args.get("title") or raw_hid).strip()
            status = str(args.get("status") or "UNVERIFIED").strip().upper()
            if status not in ("UNVERIFIED", "VERIFIED", "REJECTED"):
                status = "UNVERIFIED"
            body = str(args.get("content") or "").strip()
            formatted = self._ensure_atomic_frontmatter(body, raw_hid, title, status, step)
            hyp_file.write_text(formatted, encoding="utf-8")
            gi_content = self._sync_global_index_from_files(ut_dir, session_id)
            await self._on_memory_update("plan.md", gi_content)
            await self._on_files_update()
            changed = status in ("VERIFIED", "REJECTED")
            git_msg = f"Step {step}: {raw_hid} -> {status}" if changed else None
            return (
                f"仮説ファイル保存完了: 10_active_workspace/{hyp_file.name} (ID={raw_hid}, STATUS={status})",
                raw_hid,
                changed,
                None,
                git_msg,
            )

        if tool_name == "set_hypothesis_status":
            raw_hid = str(args.get("hyp_id", "")).strip().upper()
            digits = re.search(r"(\d+)", raw_hid)
            if digits:
                hyp_num = int(digits.group(1))
                norm_hid = f"HYP_{hyp_num:04d}"
                hyp_file = act_dir / f"hyp_{hyp_num:04d}.md"
            else:
                metrics = self.sessions.get_ultrathink_metrics(session_id)
                first_unverified = next((it for it in metrics["items"] if it["status"] == "UNVERIFIED"), None)
                if not first_unverified:
                    return "エラー: 更新可能な未検証仮説が見つかりません。", None, False, None, None
                norm_hid = first_unverified["id"]
                d2 = re.search(r"(\d+)", norm_hid)
                hyp_file = act_dir / f"hyp_{int(d2.group(1)):04d}.md" if d2 else act_dir / f"{norm_hid.lower()}.md"

            new_status = str(args.get("status", "VERIFIED")).strip("[] ").upper()
            if new_status not in ("UNVERIFIED", "VERIFIED", "REJECTED"):
                new_status = "VERIFIED"
            summary = str(args.get("summary", "")).strip()

            if not hyp_file.exists():
                arch_candidate = ut_dir / "90_archive" / hyp_file.name
                target_file = arch_candidate if arch_candidate.exists() else hyp_file
            else:
                target_file = hyp_file

            existing = target_file.read_text(encoding="utf-8", errors="replace") if target_file.exists() else ""
            meta, body = SessionManager._parse_yaml_frontmatter(existing)
            title = meta.get("TITLE") or norm_hid
            note_block = f"\n\n### 検証判定 ({self._format_step_tag(step)}: {new_status})\n{summary}\n" if summary else ""
            updated_text = self._ensure_atomic_frontmatter(
                f"{body.rstrip()}{note_block}", norm_hid, title, new_status, step
            )
            target_file.write_text(updated_text, encoding="utf-8")
            gi_content = self._sync_global_index_from_files(ut_dir, session_id)
            await self._on_memory_update("plan.md", gi_content)
            await self._on_files_update()
            changed = new_status in ("VERIFIED", "REJECTED")
            git_msg = f"Step {step}: {norm_hid} -> {new_status}" if changed else None
            return (
                f"ステータス確定: {norm_hid} -> [{new_status}] (GLOBAL_INDEX.md 更新済)",
                norm_hid,
                changed,
                None,
                git_msg,
            )

        if tool_name == "update_global_index":
            content = str(args.get("content", "")).strip()
            if content:
                (ut_dir / "GLOBAL_INDEX.md").write_text(content + "\n", encoding="utf-8")
                await self._on_memory_update("plan.md", content)
                await self._on_files_update()
            return "GLOBAL_INDEX.md を更新しました。", None, False, None, None

        if tool_name == "crystallize_knowledge":
            raw_fid = str(args.get("fact_id", "")).strip().upper()
            digits = re.search(r"(\d+)", raw_fid)
            fact_id = f"FACT_{int(digits.group(1)):04d}" if digits else self._next_ultrathink_id(ut_dir, "FACT")
            d2 = re.search(r"(\d+)", fact_id)
            fact_num = int(d2.group(1)) if d2 else 1
            fact_file = kb_dir / f"fact_{fact_num:04d}.md"
            title = str(args.get("title") or fact_id).strip()
            yaml_title = re.sub(r"[\r\n]+", " ", title)
            body = str(args.get("content") or "").strip()
            fact_md = (
                "---\n"
                f"ID: {fact_id}\n"
                f"TITLE: {yaml_title}\n"
                "STATUS: VERIFIED\n"
                f"LAST_UPDATE: {self._format_step_tag(step)}\n"
                "---\n\n"
                f"{body}\n"
            )
            fact_file.write_text(fact_md, encoding="utf-8")
            kb_idx = kb_dir / "INDEX_KB.md"
            prev = kb_idx.read_text(encoding="utf-8", errors="replace") if kb_idx.exists() else (
                "# 知識ベース目次 (INDEX_KB.md)\n\n| ID | TITLE | FILE | STEP |\n|---|---|---|---|\n"
            )
            sep = "" if (not prev or prev.endswith("\n")) else "\n"
            safe_title = title.replace("|", "/")
            kb_idx.write_text(
                f"{prev}{sep}| {fact_id} | {safe_title} | 20_knowledge_base/{fact_file.name} | {self._format_step_tag(step)} |\n",
                encoding="utf-8",
            )
            gi_content = self._sync_global_index_from_files(ut_dir, session_id)
            await self._on_memory_update("plan.md", gi_content)
            await self._on_files_update()
            return f"知識結晶化完了: 20_knowledge_base/{fact_file.name}", None, False, None, None

        if tool_name == "finalize_ultrathink":
            report = str(args.get("final_report") or "").strip()
            return "最終レポートを受け付けました。", None, False, report, f"Step {step}: Finalized"

        # Standard sandboxed tools inside 10_active_workspace / ultrathink
        res_text = await tools.execute(tool_name, args)
        touched_hid: Optional[str] = None
        if tool_name == "write_file":
            rel_p = str(args.get("path", "")).strip().replace("\\", "/")
            if "hyp_" in Path(rel_p).name.lower() and rel_p.lower().endswith(".md"):
                try:
                    resolved = tools.resolve_safe_path(rel_p)
                    if resolved.exists():
                        raw = resolved.read_text(encoding="utf-8", errors="replace")
                        num_m = re.search(r"(\d+)", resolved.stem)
                        hid = f"HYP_{int(num_m.group(1)):04d}" if num_m else "HYP_0001"
                        fixed = self._ensure_atomic_frontmatter(raw, hid, resolved.stem, "UNVERIFIED", step)
                        resolved.write_text(fixed, encoding="utf-8")
                        touched_hid = hid
                except Exception:
                    pass
            elif "src_" in Path(rel_p).name.lower():
                num_m = re.search(r"(\d+)", Path(rel_p).stem)
                if num_m:
                    touched_hid = f"HYP_{int(num_m.group(1)):04d}"
            gi_content = self._sync_global_index_from_files(ut_dir, session_id)
            await self._on_memory_update("plan.md", gi_content)
        elif tool_name == "read_file":
            rel_p = str(args.get("path", "")).strip()
            num_m = re.search(r"hyp_(\d+)", rel_p, flags=re.IGNORECASE)
            if num_m:
                touched_hid = f"HYP_{int(num_m.group(1)):04d}"
        return res_text, touched_hid, False, None, None

    @staticmethod
    def _extract_first_python_code(text: str) -> str:
        """Extract the first ```python ... ``` block (or orphan Python code) from coder output."""
        src = (text or "").strip()
        if not src:
            return ""
        match = re.search(r"```(?:python|py)?\s*\n(.*?)```", src, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            if code:
                return code
        orphan = _extract_orphan_python_block(src)
        return orphan.strip() if orphan else ""

    @staticmethod
    def _sanitize_script_filename(raw_name: str, step: int) -> str:
        """Normalize script_filename to a safe single-segment .py filename."""
        base = Path((raw_name or "").replace("\\", "/")).name
        cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base).strip("._")
        if not cleaned:
            return f"src_{step:04d}.py"
        if not cleaned.lower().endswith(".py"):
            cleaned = f"{cleaned}.py"
        return cleaned

    @staticmethod
    def _parse_python_run_result(py_result: str) -> tuple[bool, bool, Optional[int], str, str]:
        """Parse WorkspaceTools.run_python output into (exit_ok, has_effective_output, exit_code, stdout, stderr)."""
        text = (py_result or "").strip()
        if not text or text.startswith("エラー:"):
            return False, False, None, "", text or "エラー: 出力なし"
        lines = text.splitlines()
        exit_code: Optional[int] = None
        if lines:
            m = re.match(r"終了コード:\s*(-?\d+)", lines[0].strip())
            if m:
                exit_code = int(m.group(1))
        stdout_text = ""
        stderr_text = ""
        if "--- STDOUT ---" in text:
            after_out = text.split("--- STDOUT ---", 1)[1]
            if "--- STDERR ---" in after_out:
                stdout_text, stderr_text = after_out.split("--- STDERR ---", 1)
            else:
                stdout_text = after_out
        elif "--- STDERR ---" in text:
            stderr_text = text.split("--- STDERR ---", 1)[1]
        stdout_text = stdout_text.strip()
        stderr_text = stderr_text.strip()
        exit_ok = exit_code == 0
        has_effective_output = exit_ok and bool(stdout_text) and stdout_text != "(出力なし)"
        return exit_ok, has_effective_output, exit_code, stdout_text, stderr_text

    @staticmethod
    def _parse_json_object_from_thinking_response(visible_text: str, raw_content: str) -> Optional[dict[str, Any]]:
        """Extract a JSON dict from visible body text (after </think>), falling back to raw_content."""
        for candidate in (visible_text, raw_content):
            s = (candidate or "").strip()
            if not s:
                continue
            for obj in _extract_json_objects_from_text(s):
                if isinstance(obj, dict):
                    return obj
        return None

    async def _call_thinking_json(
        self,
        client: AsyncOpenAI,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.1,
        max_think_tokens: Optional[int] = THINK_TOKEN_LIMIT_DEFAULT,
        max_output_tokens: int = MAX_TOKENS_NORMAL_OUTPUT,
    ) -> tuple[Optional[dict[str, Any]], str, str, str, float]:
        """Run normal Thinking inference (no response_format / no prefill) and extract JSON after </think>."""
        started = time.monotonic()
        response = await self._invoke_chat_once(
            client,
            messages,
            tools=None,
            temperature=temperature,
            check_stop=True,
            allow_tool_drop=False,
            model_override=model,
            max_think_tokens=max_think_tokens,
            max_output_tokens=max_output_tokens,
        )
        duration = round(time.monotonic() - started, 1)
        msg = response.choices[0].message
        raw_content = msg.content or ""
        reasoning_extra = (
            getattr(msg, "reasoning_content", None)
            or getattr(msg, "reasoning", None)
            or ""
        )
        think_text, visible_text = _extract_think_and_content(raw_content)
        raw_output = _compose_raw_model_output(raw_content, reasoning_extra)
        parsed = self._parse_json_object_from_thinking_response(visible_text, raw_content)
        return parsed, raw_output, think_text, visible_text, duration

    def _collect_pipeline_knowledge(
        self,
        ut_dir: Optional[Path],
        sdir: Path,
        plan_file_path: Optional[Path],
        is_ultrathink: bool,
    ) -> str:
        """Build compact, context-shielded knowledge summary for Phase 1 commander."""
        parts: list[str] = []
        if is_ultrathink and ut_dir is not None:
            kb_dir = ut_dir / "20_knowledge_base"
            if kb_dir.exists():
                fact_files = sorted(kb_dir.glob("fact_*.md"))[-6:]
                fact_snippets: list[str] = []
                for fp in fact_files:
                    txt = fp.read_text(encoding="utf-8", errors="replace").strip()
                    if len(txt) > 350:
                        txt = txt[:350] + "\n..."
                    fact_snippets.append(f"[{fp.name}]\n{txt}")
                if fact_snippets:
                    parts.append("【確定知識 (20_knowledge_base)】\n" + "\n\n".join(fact_snippets))
            gi_path = ut_dir / "GLOBAL_INDEX.md"
            if gi_path.exists():
                gi_txt = gi_path.read_text(encoding="utf-8", errors="replace").strip()
                if gi_txt:
                    parts.append("【GLOBAL_INDEX.md】\n" + gi_txt[:800])
        elif plan_file_path is not None and plan_file_path.exists():
            plan_txt = plan_file_path.read_text(encoding="utf-8", errors="replace").strip()
            if plan_txt:
                plan_lines = plan_txt.splitlines()[-15:]
                parts.append("【現在の plan.md】\n" + "\n".join(plan_lines))

        work_log_path = sdir / "work_log.md"
        if work_log_path.exists():
            wl_lines = work_log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
            if wl_lines:
                parts.append("【直近の work_log.md】\n" + "\n".join(wl_lines))
        return "\n\n".join(parts).strip()

    async def _crystallize_pipeline_step(
        self,
        tools: WorkspaceTools,
        session_id: str,
        step: int,
        step_goal: str,
        script_filename: str,
        stdout_text: str,
        is_ultrathink: bool,
        ut_dir: Optional[Path],
        plan_file_path: Optional[Path],
    ) -> str:
        """Phase 3: Deterministically update work_log.md, plan.md, fact_XXXX.md, and GLOBAL_INDEX.md."""
        now_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        short_goal = " ".join(step_goal.strip().splitlines())[:100]
        short_out = " ".join(stdout_text.strip().splitlines())[:180]
        tools._append_work_log(f"Step {step} [{script_filename}] 完了: {short_goal} -> {short_out}")

        plan_target = (ut_dir / "plan.md") if (is_ultrathink and ut_dir is not None) else plan_file_path
        existing_plan = ""
        if plan_target is not None and plan_target.exists():
            existing_plan = plan_target.read_text(encoding="utf-8", errors="replace")
        if not existing_plan.strip() or "（現在進行中の長期タスクはありません）" in existing_plan:
            existing_plan = "# 進行中タスク計画 (plan.md)\n"
        new_plan = existing_plan.rstrip() + f"\n- [x] Step {step} (`{script_filename}`): {short_goal}\n"
        self.sessions.sync_plan_files(session_id, new_plan)
        if is_ultrathink and ut_dir is not None:
            (ut_dir / "plan.md").write_text(new_plan, encoding="utf-8")
        await self._on_memory_update("plan.md", new_plan)

        created_fact = ""
        if is_ultrathink and ut_dir is not None:
            kb_dir = ut_dir / "20_knowledge_base"
            kb_dir.mkdir(parents=True, exist_ok=True)
            fact_id = self._next_ultrathink_id(ut_dir, "FACT")
            fact_filename = f"{fact_id.lower()}.md"
            fact_path = kb_dir / fact_filename
            safe_title = short_goal.replace('"', "'")
            fact_content = (
                "---\n"
                f"ID: {fact_id}\n"
                f'TITLE: "{safe_title}"\n'
                "STATUS: VERIFIED\n"
                f"LAST_UPDATE: {now_stamp}\n"
                f"SOURCE_SCRIPT: 10_active_workspace/{script_filename}\n"
                "---\n\n"
                f"## Step {step} 検証内容\n{step_goal.strip()}\n\n"
                f"## 実行結果 (STDOUT)\n```\n{stdout_text.strip()[:2500]}\n```\n"
            )
            fact_path.write_text(fact_content, encoding="utf-8")
            created_fact = f"20_knowledge_base/{fact_filename}"
            self._sync_global_index_from_files(ut_dir, session_id)
            self.sessions.git_commit_snapshot(session_id, f"Step {step}: {script_filename} -> {fact_id}")
            await self._publish_ultrathink_status(session_id, UT_GEAR_LOW, 0.0, [], None)

        await self._on_files_update()
        return created_fact

    async def _run_pipeline_turn(
        self,
        client: AsyncOpenAI,
        session: dict[str, Any],
        user_message: str,
        assistant_msg_id: str,
        is_ultrathink: bool = False,
    ) -> None:
        """One-way deterministic 4-phase pipeline: Commander(Thinking+JSON) -> Coder(Python) -> Auto-Crystallize -> D1 Pivot."""
        session_id = session["id"]
        sdir = self.sessions.ensure_session_files(session_id)
        work_log_path = sdir / "work_log.md"
        ut_dir: Optional[Path] = None
        plan_file_path: Optional[Path] = None
        north_star_fixed = ""

        if is_ultrathink:
            ut_dir = self.sessions.ensure_ultrathink_structure(session_id, user_prompt=user_message)
            tools = WorkspaceTools(
                workspace_root=ut_dir / "10_active_workspace",
                memory_dir=self.memory.memory_dir,
                session_dir=sdir,
                work_log_path=work_log_path,
                ultrathink_dir=ut_dir,
                on_memory_update=self._on_memory_update,
                on_files_update=self._on_files_update,
            )
            north_star_path = ut_dir / "NORTH_STAR.md"
            if north_star_path.exists():
                north_star_fixed = north_star_path.read_text(encoding="utf-8", errors="replace").strip()[:2000]
            last_commit_info = self.sessions.git_commit_snapshot(session_id, "Step 0: Initialize /ultrathink pipeline")
            await self._publish_ultrathink_status(session_id, UT_GEAR_HIGH, 0.1, [], last_commit_info)
        else:
            is_longthink = session.get("mode") == "longthink"
            if is_longthink:
                lt_dir = self.sessions.ensure_longthink_structure(session_id)
                shared_dir = lt_dir / "shared_workspace"
                plan_file_path = shared_dir / "master_plan.md"
                tools = WorkspaceTools(
                    workspace_root=shared_dir,
                    memory_dir=self.memory.memory_dir,
                    session_dir=sdir,
                    plan_path=plan_file_path,
                    work_log_path=work_log_path,
                    shared_workspace_dir=shared_dir,
                    on_memory_update=self._on_memory_update,
                    on_files_update=self._on_files_update,
                )
            else:
                plan_file_path = sdir / "plan_master.md"
                tools = WorkspaceTools(
                    workspace_root=sdir / "workspace",
                    memory_dir=self.memory.memory_dir,
                    session_dir=sdir,
                    plan_path=plan_file_path,
                    work_log_path=work_log_path,
                    on_memory_update=self._on_memory_update,
                    on_files_update=self._on_files_update,
                )
            blank_plan = "# 進行中タスク計画 (plan_master.md)\n\n（現在進行中の長期タスクはありません）\n"
            self.sessions.sync_plan_files(session_id, blank_plan)
            await self._on_memory_update("plan.md", blank_plan)
            await self._on_files_update()

        self.live_thinking_log = []
        self.last_completed_message = None
        self.active_parallel_batch = None
        self.state.turn_id = assistant_msg_id
        self.broker.set_active_turn_id(assistant_msg_id)
        thinking_log: list[dict[str, Any]] = []

        raw_main = (self.state.model or "").strip()
        commander_model = (
            (self.state.standard_model or DEFAULT_STANDARD_MODEL).strip()
            if (not raw_main or raw_main == NO_MAIN_MODEL_SENTINEL)
            else raw_main
        )
        coder_model = (self.state.coder_model or DEFAULT_CODER_MODEL).strip()
        sup_model = (self.state.supervisor_model or DEFAULT_SUPERVISOR_MODEL).strip()

        forbidden_approaches: list[str] = self._load_session_forbidden(session, work_log_path)
        active_pivot_strategy = ""
        last_exec_report = ""
        last_stdout_full = ""
        prev_err_sig = ""
        prev_out_sig = ""
        prev_goal_sig = ""
        error_streak = 0
        stdout_streak = 0
        goal_streak = 0
        pivot_count = 0

        try:
            for step in itertools.count(1):
                if self._stop_event.is_set():
                    await self._finalize_stopped(session, thinking_log, assistant_msg_id)
                    return

                self.state.current_step = step
                await self.broker.publish(
                    "step",
                    {"current_step": step, "max_steps": self.state.max_steps, "session_id": session_id},
                )

                # ===== Phase 1: Commander (Thinking + trailing JSON) =====
                self.state.harness_state = "ULTRATHINK_HIGH" if is_ultrathink else STATE_INIT_PLAN
                self.state.phase_state = "thinking"
                self.state.phase_label = f"Step {step} Phase 1 [司令 ({commander_model})]: 計画・完了判定中"
                self.state.phase_started_at = time.monotonic()
                await self.broker.publish("status", self._status_payload())

                knowledge_summary = self._collect_pipeline_knowledge(ut_dir, sdir, plan_file_path, is_ultrathink)
                cmd_sys = (
                    "あなたは自律タスク実行パイプラインの司令塔です。ツール呼び出しやコード記述は一切行わないでください。\n"
                    "通常通り <think>...</think> 内で思考したあと、</think> の後ろの本文には以下のキーを持つ JSON オブジェクト1つだけを出力してください:\n"
                    '{"status": "continue" | "solved", "step_goal": "次にPythonスクリプト1本で実行すべき具体的な1つの処理内容", "script_filename": "src_0001.py", "final_answer": "statusがsolvedの場合のみユーザーへの最終回答(Markdown)"}\n'
                    "- 計算・検証・データ処理・確認が必要な場合は status='continue' とし、step_goal と単純なファイル名(例: src_0001.py)を指定してください。\n"
                    "- 直前の実行結果や確定知見からユーザー課題が完全に解決・検証済みと判断できる場合のみ status='solved' とし、final_answer を記述してください。"
                )
                cmd_user_parts = [f"【ユーザー課題】\n{user_message.strip()}"]
                if north_star_fixed:
                    cmd_user_parts.append(f"【NORTH_STAR】\n{north_star_fixed}")
                if knowledge_summary:
                    cmd_user_parts.append(knowledge_summary)
                if last_exec_report:
                    cmd_user_parts.append(f"【直前のPython実行結果】\n{last_exec_report}")
                if active_pivot_strategy:
                    cmd_user_parts.append(f"【D1-2B 強制ピボット方針】\n{active_pivot_strategy}")
                if forbidden_approaches:
                    cmd_user_parts.append(
                        "【禁止アプローチ】\n" + "\n".join(f"- {x}" for x in forbidden_approaches[-6:])
                    )
                cmd_user_parts.append(f"推奨ファイル名: src_{step:04d}.py")

                cmd_parsed, cmd_raw, cmd_think, cmd_vis, cmd_dur = await self._call_thinking_json(
                    client,
                    commander_model,
                    [
                        {"role": "system", "content": cmd_sys},
                        {"role": "user", "content": "\n\n".join(cmd_user_parts)},
                    ],
                    temperature=0.1,
                    max_think_tokens=THINK_TOKEN_LIMIT_DEFAULT,
                    max_output_tokens=MAX_TOKENS_NORMAL_OUTPUT,
                )

                if self._stop_event.is_set():
                    await self._finalize_stopped(session, thinking_log, assistant_msg_id)
                    return

                parsed_dict = cmd_parsed if isinstance(cmd_parsed, dict) else {}
                raw_status = str(parsed_dict.get("status") or "continue").strip().lower()
                cmd_status = raw_status if raw_status in ("continue", "solved") else "continue"
                step_goal = str(parsed_dict.get("step_goal") or "").strip()
                script_filename = self._sanitize_script_filename(
                    str(parsed_dict.get("script_filename") or "").strip(),
                    step,
                )
                final_answer = str(parsed_dict.get("final_answer") or "").strip()

                if cmd_status == "solved" and not final_answer:
                    if last_stdout_full:
                        final_answer = f"実行結果:\n```\n{last_stdout_full[:2500]}\n```"
                    else:
                        cmd_status = "continue"

                if cmd_status != "solved" and not step_goal:
                    vis_fallback = " ".join((cmd_vis or "").strip().splitlines())[:400]
                    step_goal = vis_fallback or (
                        f"課題「{user_message.strip()[:200]}」を検証・計算するPythonスクリプトを作成し結果をprint()する"
                    )

                cmd_thought_content = (
                    f"[Phase 1 司令 / {commander_model}] status={cmd_status}\n"
                    + (f"final_answer長={len(final_answer)}文字" if cmd_status == "solved" else f"step_goal: {step_goal}\nscript: {script_filename}")
                )
                if cmd_think:
                    cmd_thought_content = f"{cmd_think}\n\n{cmd_thought_content}"

                cmd_thought_item = {
                    "type": "thinking",
                    "step": step,
                    "duration": cmd_dur,
                    "content": cmd_thought_content,
                    "raw_output": cmd_raw,
                }
                thinking_log.append(cmd_thought_item)
                self.live_thinking_log.append(cmd_thought_item)
                self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
                await self.broker.publish("thinking", {**cmd_thought_item, "session_id": session_id})

                if cmd_status == "solved":
                    if is_ultrathink and ut_dir is not None:
                        self._sync_global_index_from_files(ut_dir, session_id)
                        self.sessions.git_commit_snapshot(session_id, f"Step {step}: Solved")
                        await self._publish_ultrathink_status(session_id, "IDLE", 0.0, [], None)
                    await self._finish_turn_with_text(
                        client,
                        tools,
                        session,
                        thinking_log,
                        assistant_msg_id,
                        step,
                        user_message,
                        final_answer,
                        harness_state="COMPLETED",
                        consolidate=False,
                    )
                    return

                # ===== Phase 2: LFM3-Coder (Pure Python) -> Auto-Save & Auto-Run =====
                route_entry = {
                    "type": "d1_coder_route",
                    "step": step,
                    "duration": 0.0,
                    "supervisor_model": sup_model,
                    "coder_choice": "coder",
                    "selected_model": coder_model,
                    "role_label": "LFM3-Coder (Pure Python)",
                    "use_json_tools": False,
                    "reason": f"Phase 2 実装: {script_filename}",
                    "raw_output": f"[PIPELINE ROUTE: {coder_model}]\nTarget: {script_filename}\nGoal: {step_goal}",
                    "context_label": "ULTRATHINK" if is_ultrathink else "NORMAL",
                    "message": f"🧭 [Pipeline] Phase 2 -> LFM3-Coder [{coder_model}] ({script_filename})",
                }
                async with self._persist_lock:
                    thinking_log.append(route_entry)
                    self.live_thinking_log.append(route_entry)
                    self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
                await self.broker.publish("d1_coder_route", {**route_entry, "session_id": session_id})

                self.state.harness_state = "ULTRATHINK_HIGH" if is_ultrathink else STATE_TOOL_EXEC
                self.state.phase_state = "working"
                self.state.phase_label = f"Step {step} Phase 2 [LFM3-Coder]: {script_filename} を生成・実行中"
                self.state.phase_started_at = time.monotonic()
                await self.broker.publish("status", self._status_payload())

                coder_sys = (
                    "前置きや解説は不要。渡された step_goal を実行して結果を print() する完全な Python コードを "
                    "```python ... ``` ブロック1つだけで出力せよ。"
                )
                coder_user_parts = [
                    f"【step_goal】\n{step_goal}",
                    f"【保存ファイル名】\n{script_filename}",
                    f"【タスク背景（要約）】\n{user_message.strip()[:600]}",
                ]
                if last_exec_report:
                    coder_user_parts.append(f"【直前の実行結果】\n{last_exec_report}")
                if active_pivot_strategy:
                    coder_user_parts.append(f"【D1-2B 方針転換指示】\n{active_pivot_strategy}")
                if forbidden_approaches:
                    coder_user_parts.append(
                        "【禁止パターン】\n" + "\n".join(f"- {x}" for x in forbidden_approaches[-5:])
                    )

                coder_started = time.monotonic()
                coder_resp = await self._invoke_chat_once(
                    client,
                    [
                        {"role": "system", "content": coder_sys},
                        {"role": "user", "content": "\n\n".join(coder_user_parts)},
                    ],
                    tools=None,
                    temperature=0.15,
                    check_stop=True,
                    allow_tool_drop=False,
                    model_override=coder_model,
                    max_think_tokens=THINK_TOKEN_LIMIT_DEFAULT,
                    max_output_tokens=MAX_TOKENS_NORMAL_OUTPUT,
                )
                coder_dur = round(time.monotonic() - coder_started, 1)
                if self._stop_event.is_set():
                    await self._finalize_stopped(session, thinking_log, assistant_msg_id)
                    return

                c_msg = coder_resp.choices[0].message
                c_raw_content = c_msg.content or ""
                c_reasoning = (
                    getattr(c_msg, "reasoning_content", None)
                    or getattr(c_msg, "reasoning", None)
                    or ""
                )
                c_think, c_vis = _extract_think_and_content(c_raw_content)
                code_text = self._extract_first_python_code(c_vis or c_raw_content)

                coder_thought_item = {
                    "type": "thinking",
                    "step": step,
                    "duration": coder_dur,
                    "content": c_think or f"[Phase 2 Coder / {coder_model}] {script_filename} コード生成 ({len(code_text)} chars)",
                    "raw_output": _compose_raw_model_output(c_raw_content, c_reasoning),
                }
                thinking_log.append(coder_thought_item)
                self.live_thinking_log.append(coder_thought_item)
                self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
                await self.broker.publish("thinking", {**coder_thought_item, "session_id": session_id})

                exit_ok = False
                has_effective_output = False
                stdout_text = ""
                stderr_text = ""

                if not code_text:
                    stderr_text = "LFM3-Coder の出力から Python コードブロックを抽出できませんでした。"
                    last_exec_report = f"ファイル: {script_filename}\n終了コード: N/A\nエラー: {stderr_text}"
                else:
                    write_started = time.monotonic()
                    write_res = await tools.write_file(script_filename, code_text)
                    write_item = {
                        "type": "work",
                        "step": step,
                        "tool": "write_file",
                        "label": f"自動保存: {script_filename}",
                        "duration": round(time.monotonic() - write_started, 2),
                        "arguments": {"path": script_filename},
                        "result": write_res[:1000],
                    }
                    thinking_log.append(write_item)
                    self.live_thinking_log.append(write_item)
                    self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
                    await self.broker.publish("work", {**write_item, "session_id": session_id})

                    run_started = time.monotonic()
                    py_result = await tools.run_python(code_text, timeout=60)
                    run_dur = round(time.monotonic() - run_started, 2)
                    exit_ok, has_effective_output, exit_code, stdout_text, stderr_text = self._parse_python_run_result(py_result)
                    trimmed_py = py_result[:2500] + "\n... (省略)" if len(py_result) > 2500 else py_result
                    run_item = {
                        "type": "work",
                        "step": step,
                        "tool": "run_python",
                        "label": f"自動実行: {script_filename} (exit={exit_code})",
                        "duration": run_dur,
                        "arguments": {"script": script_filename, "code": code_text},
                        "result": trimmed_py,
                    }
                    thinking_log.append(run_item)
                    self.live_thinking_log.append(run_item)
                    self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
                    await self.broker.publish("work", {**run_item, "session_id": session_id})

                    last_exec_report = f"ファイル: {script_filename}\n{trimmed_py[:1800]}"
                    if has_effective_output:
                        last_stdout_full = stdout_text

                # ===== Phase 3: Harness Auto-Crystallization =====
                if exit_ok and has_effective_output:
                    cryst_started = time.monotonic()
                    created_fact = await self._crystallize_pipeline_step(
                        tools=tools,
                        session_id=session_id,
                        step=step,
                        step_goal=step_goal,
                        script_filename=script_filename,
                        stdout_text=stdout_text,
                        is_ultrathink=is_ultrathink,
                        ut_dir=ut_dir,
                        plan_file_path=plan_file_path,
                    )
                    cryst_label = f"台帳自動更新 ({created_fact or 'plan.md / work_log.md'})"
                    cryst_item = {
                        "type": "work",
                        "step": step,
                        "tool": "auto_crystallize",
                        "label": cryst_label,
                        "duration": round(time.monotonic() - cryst_started, 2),
                        "arguments": {"script": script_filename, "fact": created_fact},
                        "result": f"自動結晶化完了: {cryst_label}",
                    }
                    thinking_log.append(cryst_item)
                    self.live_thinking_log.append(cryst_item)
                    self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
                    await self.broker.publish("work", {**cryst_item, "session_id": session_id})
                    active_pivot_strategy = ""

                # ===== Phase 4: Stagnation Detection & d1-2B Pivot =====
                err_sig = re.sub(r"\s+", " ", stderr_text.strip())[:240] if not exit_ok else ""
                out_sig = re.sub(r"\s+", " ", stdout_text.strip())[:240] if exit_ok else ""
                goal_sig = re.sub(r"\s+", " ", step_goal.strip())[:200]

                if err_sig and err_sig == prev_err_sig:
                    error_streak += 1
                else:
                    error_streak = 1 if err_sig else 0
                prev_err_sig = err_sig

                if out_sig and out_sig == prev_out_sig:
                    stdout_streak += 1
                else:
                    stdout_streak = 1 if out_sig else 0
                prev_out_sig = out_sig

                if goal_sig and goal_sig == prev_goal_sig:
                    goal_streak += 1
                else:
                    goal_streak = 1 if goal_sig else 0
                prev_goal_sig = goal_sig

                stuck_reason_trigger = ""
                if error_streak >= 2:
                    stuck_reason_trigger = f"同一Pythonエラーが2回連続: {err_sig[:120]}"
                elif stdout_streak >= 2:
                    stuck_reason_trigger = f"同一STDOUT結果が2回連続: {out_sig[:120]}"
                elif goal_streak >= 2:
                    stuck_reason_trigger = f"同一step_goalが2回連続: {goal_sig[:120]}"

                if stuck_reason_trigger:
                    pivot_count += 1
                    piv_sys = (
                        "あなたは停滞監視AI (d1-2B) です。2回連続で同じエラー・結果・目標が繰り返されたため、"
                        "従来とは異なるアプローチへの強制方針転換指示を作成してください。\n"
                        "通常通り <think>...</think> 内で思考したあと、</think> の後ろの本文には以下のキーを持つ JSON オブジェクト1つだけを出力してください:\n"
                        '{"stuck_reason": "停滞原因", "forbidden_action": "禁止すべき反復手法", "pivot_strategy": "次に取るべき具体的な方針転換指示"}'
                    )
                    piv_user = (
                        f"【停滞検知理由】\n{stuck_reason_trigger}\n\n"
                        f"【ユーザー課題】\n{user_message.strip()[:500]}\n\n"
                        f"【直前の step_goal】\n{step_goal}\n\n"
                        f"【直前の実行結果】\n{last_exec_report[:1000]}\n\n"
                        f"【既知の禁止事項】\n" + ("\n".join(f"- {x}" for x in forbidden_approaches[-5:]) or "(なし)")
                    )
                    piv_parsed, piv_raw, _, _, piv_dur = await self._call_thinking_json(
                        client,
                        sup_model,
                        [
                            {"role": "system", "content": piv_sys},
                            {"role": "user", "content": piv_user},
                        ],
                        temperature=0.1,
                        max_think_tokens=THINK_TOKEN_LIMIT_DEFAULT,
                        max_output_tokens=MAX_TOKENS_NORMAL_OUTPUT,
                    )
                    stuck_reason = str((piv_parsed or {}).get("stuck_reason") or stuck_reason_trigger).strip()
                    forbidden_action = str((piv_parsed or {}).get("forbidden_action") or f"同じ手法の反復 ({goal_sig[:60]})").strip()
                    new_strategy = str((piv_parsed or {}).get("pivot_strategy") or "別のアルゴリズム・検証手順に切り替えて実行する").strip()

                    if forbidden_action:
                        self._record_session_forbidden(session, work_log_path, forbidden_approaches, forbidden_action)
                    active_pivot_strategy = new_strategy

                    sup_entry = {
                        "type": "d1_supervisor",
                        "step": step,
                        "duration": piv_dur,
                        "supervisor_model": sup_model,
                        "context_label": "ULTRATHINK" if is_ultrathink else "NORMAL",
                        "intervention_count": pivot_count,
                        "think_unlocked": False,
                        "escalate_to_user": False,
                        "user_consultation_message": "",
                        "stuck_reason": stuck_reason,
                        "forbidden_action": forbidden_action,
                        "new_strategy": new_strategy,
                        "raw_output": piv_raw,
                        "message": f"⚡ [D1-2B Pivot] 2連続停滞を検知（{pivot_count}回目）→ 強制方針転換 ({stuck_reason[:80]})",
                    }
                    async with self._persist_lock:
                        thinking_log.append(sup_entry)
                        self.live_thinking_log.append(sup_entry)
                        self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
                    await self.broker.publish("d1_supervisor", {**sup_entry, "session_id": session_id})

                    error_streak = 0
                    stdout_streak = 0
                    goal_streak = 0

                    if pivot_count >= 5:
                        fallback_final = (
                            f"5回のピボット介入後も課題が収束しなかったため、現時点の最新実行結果をもって完了します。\n\n"
                            f"```\n{(last_stdout_full or last_exec_report)[:2000]}\n```"
                        )
                        await self._finish_turn_with_text(
                            client,
                            tools,
                            session,
                            thinking_log,
                            assistant_msg_id,
                            step,
                            user_message,
                            fallback_final,
                            harness_state="COMPLETED",
                            consolidate=False,
                        )
                        return
        except asyncio.CancelledError:
            await self._finalize_stopped(session, thinking_log, assistant_msg_id)
        except Exception as exc:
            self.state.status = "error"
            self.state.phase_state = "idle"
            self.state.phase_label = ""
            self.state.phase_started_at = 0.0
            self.state.harness_state = "ERROR"
            self.active_parallel_batch = None
            self.state.error_message = str(exc)
            self._persist_progress(
                session,
                assistant_msg_id,
                thinking_log,
                latest_content=f"実行エラー: {exc}",
                in_progress=False,
            )
            await self.broker.publish("error", {"message": f"実行エラー: {exc}", "session_id": session_id})
            await self.broker.publish("status", self._status_payload())

    async def _run_ultrathink_turn(
        self,
        client: AsyncOpenAI,
        session: dict[str, Any],
        user_message: str,
        assistant_msg_id: str,
    ) -> None:
        """LFM3-optimized one-way pipeline loop for /ultrathink."""
        await self._run_pipeline_turn(client, session, user_message, assistant_msg_id, is_ultrathink=True)
        return
        session_id = session["id"]
        sdir = self.sessions.ensure_session_files(session_id)
        ut_dir = self.sessions.ensure_ultrathink_structure(session_id, user_prompt=user_message)
        tools = WorkspaceTools(
            workspace_root=ut_dir / "10_active_workspace",
            memory_dir=self.memory.memory_dir,
            session_dir=sdir,
            work_log_path=sdir / "work_log.md",
            ultrathink_dir=ut_dir,
            on_memory_update=self._on_memory_update,
            on_files_update=self._on_files_update,
        )

        north_star_path = ut_dir / "NORTH_STAR.md"
        north_star_fixed = north_star_path.read_text(encoding="utf-8", errors="replace").strip()
        if len(north_star_fixed) > 2000:
            north_star_fixed = north_star_fixed[:2000] + "\n... (省略)"

        self.live_thinking_log = []
        self.last_completed_message = None
        self.active_parallel_batch = None
        self.state.turn_id = assistant_msg_id
        self.broker.set_active_turn_id(assistant_msg_id)
        thinking_log: list[dict[str, Any]] = []
        pending_low_gear = False
        focused_hyp_id: Optional[str] = None
        last_exec_summary = ""
        last_commit_info = self.sessions.git_commit_snapshot(session_id, "Step 0: Initialize /ultrathink")
        consecutive_timeout_count = 0
        consecutive_noop_count = 0
        same_gear_history: list[dict[str, Any]] = []
        ut_tracker = self._LoopStagnationTracker(threshold=CIRCUIT_BREAKER_THRESHOLD)
        ut_forbidden: list[str] = self._load_session_forbidden(session, sdir / "work_log.md")
        ut_strategy = ""
        ut_replan_count = 0
        ut_subtask_interventions = 0
        ut_think_unlocked = False
        ut_last_call_key: Optional[str] = None
        ut_tool_fail_streak = 0
        ut_last_thought = ""
        ut_burst_used = False
        ut_iso_step_id: Optional[str] = None
        ut_just_errored = False

        _ut_allowed_base_tools = {"write_file", "read_file", "list_files", "run_python", "web_search", "fetch_url"}
        high_gear_tools = [
            t for t in TOOL_SCHEMAS if t["function"]["name"] in _ut_allowed_base_tools
        ] + [
            t
            for t in ULTRATHINK_EXTRA_TOOL_SCHEMAS
            if t["function"]["name"] in ("upsert_hypothesis", "crystallize_knowledge", "finalize_ultrathink")
        ]
        low_gear_tools = [
            t
            for t in ULTRATHINK_EXTRA_TOOL_SCHEMAS
            if t["function"]["name"] in ("set_hypothesis_status", "update_global_index")
        ]

        try:
            for step in itertools.count(1):
                if self._stop_event.is_set():
                    await self._finalize_stopped(session, thinking_log, assistant_msg_id)
                    return

                self.state.current_step = step
                await self.broker.publish(
                    "step",
                    {"current_step": step, "max_steps": self.state.max_steps, "session_id": session_id},
                )

                # Periodic 20-step GC & Knowledge Crystallization phase
                if step % 20 == 0:
                    gc_label = f"[ULTRATHINK GC] Step {step}: 確定知識の結晶化 & 90_archive 整理中"
                    self.state.harness_state = "ULTRATHINK_GC"
                    self.state.phase_state = "working"
                    self.state.phase_label = gc_label
                    self.state.phase_started_at = time.monotonic()
                    await self._publish_ultrathink_status(session_id, UT_GEAR_GC, 0.0, [], last_commit_info)
                    await self.broker.publish(
                        "status",
                        {**self._status_payload(), "state": "working", "step": step, "label": gc_label},
                    )
                    gc_start = self.state.phase_started_at
                    c_cnt, a_cnt, c_hash, c_sum = await self._run_ultrathink_gc(
                        client, ut_dir, session_id, step, north_star_fixed
                    )
                    if c_hash:
                        last_commit_info = c_hash
                    gc_dur = round(time.monotonic() - gc_start, 1)
                    gc_item = {
                        "type": "work",
                        "step": step,
                        "tool": "ultrathink_gc",
                        "label": f"[GC 結晶化] 確定={c_cnt}件 / 隔離={a_cnt}ファイル",
                        "duration": gc_dur,
                        "arguments": {"step": step},
                        "result": f"知識結晶化: {c_sum or '新規対象なし'}\n90_archive 移動数: {a_cnt}\nGit: {c_hash or '変更なし'}",
                    }
                    thinking_log.append(gc_item)
                    self.live_thinking_log.append(gc_item)
                    self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
                    await self.broker.publish("work", {**gc_item, "session_id": session_id})
                    same_gear_history = []
                    pending_low_gear = False
                    focused_hyp_id = None
                    ut_tracker.reset(keep_recent=True)
                    continue

                metrics = self.sessions.get_ultrathink_metrics(session_id)
                ut_progress_before = (
                    metrics["unverified"],
                    metrics["verified"],
                    metrics["rejected"],
                    metrics["knowledge_facts"],
                    tuple((it["id"], it["status"]) for it in metrics["items"]),
                )
                global_index_text = (ut_dir / "GLOBAL_INDEX.md").read_text(encoding="utf-8", errors="replace").strip()
                if len(global_index_text) > 2500:
                    global_index_text = global_index_text[:2500] + "\n... (省略)"

                # Strictly load ONLY NORTH_STAR.md, GLOBAL_INDEX.md, and at most 2 files in 10_active_workspace/
                focus_paths = self._select_active_focus_files(ut_dir, focused_hyp_id=focused_hyp_id)[:2]
                focus_blocks: list[str] = []
                for fp in focus_paths:
                    rel_fp = fp.relative_to(ut_dir).as_posix()
                    body_txt = fp.read_text(encoding="utf-8", errors="replace")
                    if len(body_txt) > 2000:
                        body_txt = body_txt[:2000] + "\n... (省略)"
                    focus_blocks.append(f"### 注目ファイル: `{rel_fp}`\n```\n{body_txt}\n```")

                is_low_gear = pending_low_gear and (metrics["unverified"] > 0 or bool(focused_hyp_id))
                gear = UT_GEAR_LOW if is_low_gear else UT_GEAR_HIGH
                temperature = 0.0 if is_low_gear else 0.3

                await self._publish_ultrathink_status(session_id, gear, temperature, focus_paths, last_commit_info)
                think_label = (
                    f"[LOW GEAR / temp=0.0] ステータス・GLOBAL_INDEX 即答更新中"
                    if is_low_gear
                    else f"[HIGH GEAR / temp=0.3] 原子仮説探索・Python検証中"
                )
                self.state.harness_state = f"ULTRATHINK_{gear}"
                self.state.phase_state = "thinking"
                self.state.phase_label = think_label
                self.state.phase_started_at = time.monotonic()
                await self.broker.publish(
                    "status",
                    {**self._status_payload(), "state": "thinking", "step": step, "label": think_label},
                )
                think_start = self.state.phase_started_at

                # Fixed NORTH_STAR.md always placed at the top of system message for KV Prefix Caching
                if is_low_gear:
                    sys_msg = (
                        f"{north_star_fixed}\n\n"
                        "## 現在モード: LOW GEAR（事務処理フェーズ / temperature=0.0 / 最大1000トークン）\n"
                        "- 思考タグ（`<think>` や長い説明文）を一切出力せず、即座に指定フォーマットのツール呼び出しのみを行ってください。\n"
                        "- 対象仮説が検証成功なら `VERIFIED`、誤り・矛盾・再検証要なら `REJECTED` または `UNVERIFIED` として `set_hypothesis_status` を実行してください。"
                    )
                    active_tools = low_gear_tools
                    target_hid = focused_hyp_id or next(
                        (it["id"] for it in metrics["items"] if it["status"] == "UNVERIFIED"),
                        "HYP_0001",
                    )
                    exec_snip = (last_exec_summary or "(実行出力あり)")[:600]
                    user_prompt = (
                        f"### GLOBAL_INDEX.md (目次のみ)\n```markdown\n{global_index_text}\n```\n\n"
                        + ("\n\n".join(focus_blocks) + "\n\n" if focus_blocks else "")
                        + f"### 直前の実行結果\n```\n{exec_snip}\n```\n\n"
                        f"思考なしで即答してください: `{target_hid}` に対し `set_hypothesis_status` ツールを呼び出し、`status` と1文要約を記録してください。"
                    )
                else:
                    ut_forbid_block = (
                        "\n\n### D1-2B Supervisor 強制指示（再試行禁止）\n"
                        + "\n".join(f"- 禁止: {x}" for x in ut_forbidden[-4:])
                        if ut_forbidden
                        else ""
                    )
                    ut_strat_block = (
                        f"\n### D1-2B Supervisor 新戦略\n{ut_strategy}\n"
                        if ut_strategy
                        else ""
                    )
                    sys_msg = (
                        f"{north_star_fixed}\n\n"
                        "## 現在モード: HIGH GEAR（推論・反証フェーズ / temperature=0.3）\n"
                        "- 参照ファイルは隔離領域 `ultrathink/` 内の `NORTH_STAR.md`・`GLOBAL_INDEX.md`・`10_active_workspace/`（最大2ファイル）のみに厳格限定し、他ディレクトリを走査・読込しないでください。\n"
                        "- 新仮説の立案には `upsert_hypothesis`、検証計算や反証テストには `10_active_workspace/src_XXXX.py`（例: `[WRITE_FILE: 10_active_workspace/src_0001.py]`）の保存および `run_python` を使ってください。\n"
                        "- 【パス禁止ルール】`10_active_workspace/workspace/...` や `workspace/10_active_workspace/...` のように `workspace/` を二重ネストさせないでください（保存先は常に `10_active_workspace/<ファイル名>` 直下に指定すること）。\n"
                        f"{DIRECT_WRITE_FORMAT_GUIDE}\n"
                        "- すべての疑問・制約が `VERIFIED` 事実によって解決された場合のみ `finalize_ultrathink` を呼んで最終報告を完了してください。"
                        f"{ut_forbid_block}{ut_strat_block}"
                    )
                    active_tools = high_gear_tools
                    if metrics["unverified"] == 0 and (metrics["verified"] > 0 or metrics["knowledge_facts"] > 0):
                        next_directive = (
                            "現在未検証仮説はありません。`NORTH_STAR.md` の全ゴールと絶対制約が検証済み知識だけで完全に満たされていれば `finalize_ultrathink` を呼び出してください。"
                            "追加検証・反証が必要なら `upsert_hypothesis` で新しい仮説を作成してください。"
                        )
                    elif metrics["unverified"] == 0:
                        next_directive = (
                            "最初の探索サブタスク・仮説を `upsert_hypothesis`（`HYP_0001`）で作成し、検証計画を立ててください。"
                        )
                    else:
                        next_hid = next(
                            (it["id"] for it in metrics["items"] if it["status"] == "UNVERIFIED"),
                            "HYP_0001",
                        )
                        next_directive = (
                            f"現在フォーカス中の未検証仮説 `{next_hid}` を Python コード（例: `10_active_workspace/src_0001.py` へ保存して `run_python` 実行）やツールで厳密に検証・反証してください。"
                        )

                    exec_hi_snip = last_exec_summary[:800] if last_exec_summary else ""
                    user_prompt = (
                        f"### GLOBAL_INDEX.md (目次のみ)\n```markdown\n{global_index_text}\n```\n\n"
                        + ("\n\n".join(focus_blocks) + "\n\n" if focus_blocks else "")
                        + (f"### 直前の実行結果\n```\n{exec_hi_snip}\n```\n\n" if exec_hi_snip else "")
                        + f"{ut_strat_block}"
                        + f"### Step {step} (∞) 指示\n{next_directive}"
                    )

                messages: list[dict[str, Any]] = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_prompt},
                ]
                if not is_low_gear and same_gear_history:
                    messages.extend(same_gear_history[-2:])
                messages = self._merge_consecutive_roles(messages)

                if is_low_gear:
                    route_res = await self._route_coding_model_via_d1(
                        client=client,
                        step=step,
                        session_id=session_id,
                        assistant_msg_id=assistant_msg_id,
                        session=session,
                        thinking_log=thinking_log,
                        task_summary="GLOBAL_INDEX.md / ステータス更新 (LOW GEAR JSONツール事務)",
                        context_label=f"/ultrathink LOW GEAR (Step {step})",
                        forced_choice="lfm3_standard",
                        forced_reason="LOW GEAR 事務処理・GLOBAL_INDEX 更新・JSONツール専任のため LFM3 1.7B を使用",
                    )
                elif ut_just_errored:
                    route_res = await self._route_coding_model_via_d1(
                        client=client,
                        step=step,
                        session_id=session_id,
                        assistant_msg_id=assistant_msg_id,
                        session=session,
                        thinking_log=thinking_log,
                        task_summary=user_prompt[:800],
                        context_label=f"/ultrathink HIGH GEAR Error Analysis (Step {step})",
                        forced_choice="main_model",
                        forced_reason="エラー・停滞直後の原因分析と戦略修正のためメインモデルを担当指定",
                    )
                    ut_just_errored = False
                else:
                    route_res = await self._route_coding_model_via_d1(
                        client=client,
                        step=step,
                        session_id=session_id,
                        assistant_msg_id=assistant_msg_id,
                        session=session,
                        thinking_log=thinking_log,
                        task_summary=f"{ut_strategy}\n{user_prompt[:800]}",
                        context_label=f"/ultrathink {gear} GEAR (Step {step})",
                    )

                ut_model_override = route_res.get("selected_model") or None
                ut_call_messages, ut_call_tools, ut_use_json_tools = self._apply_pure_coder_constraints(
                    messages, active_tools, route_res
                )

                if is_low_gear:
                    ut_max_think = MAX_TOKENS_LOW_GEAR
                    ut_max_output = MAX_TOKENS_LOW_GEAR
                elif ut_think_unlocked:
                    ut_max_think = MAX_TOKENS_UNLOCKED_CAP
                    ut_max_output = MAX_TOKENS_UNLOCKED_CAP
                else:
                    ut_max_think = THINK_TOKEN_LIMIT_DEFAULT
                    ut_max_output = MAX_TOKENS_NORMAL_OUTPUT

                try:
                    response = await self._create_chat_completion_resilient(
                        client,
                        ut_call_messages,
                        tools=ut_call_tools,
                        temperature=temperature,
                        check_stop=True,
                        allow_tool_drop=True,
                        model_override=ut_model_override,
                        max_think_tokens=ut_max_think,
                        max_output_tokens=ut_max_output,
                    )
                    consecutive_timeout_count = 0
                except (
                    APITimeoutError,
                    APIConnectionError,
                    httpx.TimeoutException,
                    httpx.TransportError,
                ) as timeout_exc:
                    if self._stop_event.is_set():
                        await self._finalize_stopped(session, thinking_log, assistant_msg_id)
                        return
                    consecutive_timeout_count += 1
                    warn_item = {
                        "type": "thinking",
                        "step": step,
                        "duration": round(time.monotonic() - think_start, 1),
                        "content": f"[ULTRATHINK/{gear}] Step {step} タイムアウト ({timeout_exc}); コンテキストを初期化して続行します。",
                    }
                    thinking_log.append(warn_item)
                    self.live_thinking_log.append(warn_item)
                    same_gear_history = []
                    self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
                    await self.broker.publish("thinking", {**warn_item, "session_id": session_id})
                    if consecutive_timeout_count <= 2:
                        continue
                    raise

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

                allowed_names = {t["function"]["name"] for t in active_tools}
                parsed_calls, ut_parse_diag = _parse_response_tool_calls_with_diagnostics(
                    msg,
                    visible_text,
                    raw_content,
                    allowed_names=allowed_names,
                    orphan_rescue_path="10_active_workspace/src_auto.py" if not is_low_gear else None,
                )
                if parsed_calls and (not ut_use_json_tools or any(c.get("_orphan_rescue") for c in parsed_calls)):
                    parsed_calls = self._augment_pure_coder_write_calls(parsed_calls, allowed_names)

                # In Low Gear, strip thinking and log compact fast-pass notice
                if is_low_gear:
                    thought_content = f"[LOW GEAR / temp=0.0] 思考スキップ・即答フォーマット ({','.join(c['name'] for c in parsed_calls) or 'fallback'})"
                else:
                    combined_thinking = "\n\n".join(p for p in [reasoning_extra, think_text] if p).strip()
                    if combined_thinking:
                        ut_last_thought = combined_thinking
                    thought_content = combined_thinking or visible_text or f"[HIGH GEAR] Step {step} 推論"

                thought_item = {
                    "type": "thinking",
                    "step": step,
                    "duration": think_duration,
                    "content": thought_content,
                    "raw_output": _compose_raw_model_output(raw_content, reasoning_extra),
                }
                thinking_log.append(thought_item)
                self.live_thinking_log.append(thought_item)
                self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
                await self.broker.publish("thinking", {**thought_item, "session_id": session_id})

                if not is_low_gear and not parsed_calls and ut_parse_diag:
                    ut_tool_fail_streak += 1
                    if ut_tool_fail_streak >= 2 and not ut_burst_used:
                        ut_burst_used = True
                        burst_calls, burst_diag, burst_vis = await self._invoke_direct_burst_write(
                            client=client,
                            step=step,
                            session_id=session_id,
                            assistant_msg_id=assistant_msg_id,
                            session=session,
                            thinking_log=thinking_log,
                            task_summary=user_prompt[:1000],
                            last_thought_text=ut_last_thought,
                            failure_diag=ut_parse_diag,
                            active_tools=active_tools,
                            allowed_names=allowed_names,
                            model_override=ut_model_override or self.state.model,
                            context_label=f"/ultrathink {gear} GEAR (Step {step})",
                        )
                        if burst_calls:
                            parsed_calls = (
                                self._augment_pure_coder_write_calls(burst_calls, allowed_names)
                                if (not ut_use_json_tools or any(c.get("_orphan_rescue") for c in burst_calls))
                                else burst_calls
                            )
                            ut_parse_diag = burst_diag
                            visible_text = burst_vis
                        elif burst_diag:
                            ut_parse_diag = burst_diag

                # Low Gear deterministic fallback if model omitted tool call
                if is_low_gear and not parsed_calls:
                    verdict = "VERIFIED"
                    lower_vis = (visible_text or raw_content).upper()
                    if "REJECTED" in lower_vis or "FALSE" in lower_vis or "エラー" in lower_vis:
                        verdict = "REJECTED"
                    fallback_hid = focused_hyp_id or next(
                        (it["id"] for it in metrics["items"] if it["status"] == "UNVERIFIED"),
                        "HYP_0001",
                    )
                    parsed_calls = [
                        {
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "name": "set_hypothesis_status",
                            "arguments": {
                                "hyp_id": fallback_hid,
                                "status": verdict,
                                "summary": (visible_text or last_exec_summary or "検証完了").strip()[:200],
                            },
                        }
                    ]

                async def _apply_ut_replan(trigger_reason: str) -> bool:
                    nonlocal ut_replan_count, ut_subtask_interventions, ut_think_unlocked, ut_just_errored
                    nonlocal ut_strategy, pending_low_gear, same_gear_history, last_commit_info, focused_hyp_id, consecutive_noop_count
                    ut_replan_count += 1
                    ut_subtask_interventions += 1
                    ut_just_errored = True
                    diag = await self._invoke_d1_supervisor(
                        client=client,
                        step=step,
                        session_id=session_id,
                        assistant_msg_id=assistant_msg_id,
                        session=session,
                        thinking_log=thinking_log,
                        recent_history_items=ut_tracker.recent_steps,
                        plan_text=global_index_text,
                        fallback_reason=trigger_reason,
                        context_label="Mode: /ultrathink",
                        intervention_count=ut_subtask_interventions,
                        think_limit_unlocked=ut_think_unlocked,
                    )
                    ut_think_unlocked = bool(ut_think_unlocked or diag.get("think_unlocked"))
                    if diag["forbidden_action"]:
                        if diag["forbidden_action"] not in ut_forbidden:
                            ut_forbidden.append(diag["forbidden_action"])
                        self._record_session_forbidden(session, diag["forbidden_action"], tools_obj=tools)
                    ut_strategy = diag["new_strategy"]
                    same_gear_history = []
                    pending_low_gear = False
                    consecutive_noop_count = 0
                    ut_tracker.reset(keep_recent=False)

                    if diag.get("escalate_to_user"):
                        await self._publish_ultrathink_status(session_id, "IDLE", 0.0, [], last_commit_info)
                        abort_text = self._format_d1_escalation_report(
                            diag,
                            ut_subtask_interventions,
                            "/ultrathink",
                        )
                        await self._finish_turn_with_text(
                            client,
                            tools,
                            session,
                            thinking_log,
                            assistant_msg_id,
                            step,
                            user_message,
                            abort_text,
                            harness_state="PAUSED_FOR_USER",
                            consolidate=False,
                        )
                        return False

                    replan_hid = focused_hyp_id or self._next_ultrathink_id(ut_dir, "HYP")
                    _, new_hid, _, _, _ = await self._execute_ultrathink_tool(
                        "upsert_hypothesis",
                        {
                            "hyp_id": replan_hid,
                            "title": f"D1-2B戦略: {ut_strategy[:42]}",
                            "status": "UNVERIFIED",
                            "content": (
                                f"### D1-2B Supervisor 強制リプラン (Step {step})\n"
                                f"- 詰まり原因: {diag['stuck_reason']}\n"
                                f"- 禁止アクション: {diag['forbidden_action']}\n"
                                f"- 新戦略: {ut_strategy}\n"
                            ),
                        },
                        ut_dir,
                        tools,
                        session_id,
                        step,
                    )
                    if new_hid:
                        focused_hyp_id = new_hid
                    c_hash = self.sessions.git_commit_snapshot(session_id, f"Step {step}: D1-2B Replan")
                    if c_hash:
                        last_commit_info = c_hash
                    focus_now = self._select_active_focus_files(ut_dir, focused_hyp_id=focused_hyp_id)
                    await self._publish_ultrathink_status(session_id, UT_GEAR_HIGH, 0.3, focus_now, last_commit_info)
                    return True

                if not parsed_calls:
                    consecutive_noop_count += 1
                    if ut_parse_diag:
                        diag_notice = f"【ツール引数パース・記述エラー診断】\n{ut_parse_diag}\n\n{DIRECT_WRITE_FORMAT_GUIDE}"
                        same_gear_history.append({"role": "user", "content": diag_notice})
                        last_exec_summary = ut_parse_diag
                        cb_noop = ut_tracker.record_action(
                            f"call:parse_err:ut:{ut_parse_diag[:40]}",
                            f"res:parse_err:ut:{ut_parse_diag[:60]}",
                            f"Step {step} [{gear}]: ツール引数パースエラー ({ut_parse_diag[:80]})",
                            "ツール呼び出しのJSON引数パースまたはコード記述に失敗している",
                        )
                        if cb_noop:
                            keep_going = await _apply_ut_replan(cb_noop)
                            if not keep_going:
                                return
                        continue
                    metrics_now = self.sessions.get_ultrathink_metrics(session_id)
                    if visible_text and metrics_now["unverified"] == 0 and (
                        metrics_now["verified"] > 0 or metrics_now["knowledge_facts"] > 0 or step >= 3
                    ):
                        final_commit = self.sessions.git_commit_snapshot(session_id, f"Step {step}: Final completion")
                        if final_commit:
                            last_commit_info = final_commit
                        await self._publish_ultrathink_status(session_id, "IDLE", 0.0, [], last_commit_info)
                        await self._finish_turn_with_text(
                            client,
                            tools,
                            session,
                            thinking_log,
                            assistant_msg_id,
                            step,
                            user_message,
                            visible_text,
                            harness_state="COMPLETED",
                            consolidate=False,
                        )
                        return
                    if visible_text and metrics_now["unverified"] == 0 and metrics_now["verified"] == 0:
                        # Auto-record initial text into HYP_0001 so exploration progresses reliably
                        hid = self._next_ultrathink_id(ut_dir, "HYP")
                        res_t, t_hid, _, _, _ = await self._execute_ultrathink_tool(
                            "upsert_hypothesis",
                            {
                                "hyp_id": hid,
                                "title": user_message.strip().splitlines()[0][:40],
                                "status": "UNVERIFIED",
                                "content": visible_text,
                            },
                            ut_dir,
                            tools,
                            session_id,
                            step,
                        )
                        focused_hyp_id = t_hid
                        last_exec_summary = res_t
                    else:
                        same_gear_history.append(
                            {
                                "role": "user",
                                "content": "必ずツール (`upsert_hypothesis`, `write_file`, `run_python`, `finalize_ultrathink` 等) を1つ実行してください。",
                            }
                        )
                        cb_noop = ut_tracker.record_action(
                            f"call:noop:ut:{gear}",
                            f"res:noop:ut:{(visible_text or '')[:40]}",
                            f"Step {step} [{gear}]: ツール呼び出しなし",
                            "ツールを実行せず空振り応答を繰り返している",
                        )
                        if cb_noop:
                            keep_going = await _apply_ut_replan(cb_noop)
                            if not keep_going:
                                return
                    continue

                consecutive_noop_count = 0
                ran_verification_tool = False
                ran_status_update = False
                ut_ran_tool_ok = False
                ut_ran_tool_err = False
                ut_cb_trigger: Optional[str] = None
                ut_pending_final_report: Optional[str] = None

                if any(c["name"] in ("write_file", "read_file", "list_files", "run_python", "run_command") for c in parsed_calls):
                    ut_iso_step_id = ut_iso_step_id or str(step)
                    tools.enter_isolated_workspace(ut_iso_step_id)

                for call in parsed_calls:
                    if self._stop_event.is_set():
                        await self._finalize_stopped(session, thinking_log, assistant_msg_id)
                        return

                    tool_name = call["name"]
                    tool_args = call["arguments"]
                    call_key = _canonical_call_key(tool_name, tool_args)
                    if not is_low_gear and ut_last_call_key is not None and call_key == ut_last_call_key:
                        ut_tool_fail_streak += 1
                        dup_diag = (
                            f"エラー（重複呼び出しブロック）: 直前と全く同一のツール名 `{tool_name}`・同一引数での再呼び出しはブロックされました。"
                            "長いPythonコードを書く際は JSON の `write_file` ではなく必ず `[WRITE_FILE: 10_active_workspace/src_XXXX.py]` + "
                            "```python ... ``` 形式へ切り替えるか、ファイルを小さく分割してください。"
                        )
                        if ut_tool_fail_streak >= 2 and not ut_burst_used:
                            ut_burst_used = True
                            burst_calls, _, _ = await self._invoke_direct_burst_write(
                                client=client,
                                step=step,
                                session_id=session_id,
                                assistant_msg_id=assistant_msg_id,
                                session=session,
                                thinking_log=thinking_log,
                                task_summary=user_prompt[:1000],
                                last_thought_text=ut_last_thought,
                                failure_diag=dup_diag,
                                active_tools=active_tools,
                                allowed_names=allowed_names,
                                model_override=ut_model_override or self.state.model,
                                context_label=f"/ultrathink Duplicate Block (Step {step})",
                            )
                            if burst_calls:
                                burst_calls = self._augment_pure_coder_write_calls(burst_calls, allowed_names)
                                if any(bc["name"] in ("write_file", "read_file", "list_files", "run_python", "run_command") for bc in burst_calls):
                                    ut_iso_step_id = ut_iso_step_id or str(step)
                                    tools.enter_isolated_workspace(ut_iso_step_id)
                                for bc in burst_calls:
                                    b_res, b_hid, _, _, b_git = await self._execute_ultrathink_tool(
                                        bc["name"], bc["arguments"], ut_dir, tools, session_id, step
                                    )
                                    if b_hid:
                                        focused_hyp_id = b_hid
                                    if b_git:
                                        self.sessions.git_commit_snapshot(session_id, b_git)
                                    ut_last_call_key = _canonical_call_key(bc["name"], bc["arguments"])
                                    last_exec_summary = b_res[:2000]
                                    if bc["name"] == "run_python":
                                        ok_py, _, _ = self._inspect_python_result(b_res)
                                        if ok_py:
                                            ut_ran_tool_ok = True
                                        else:
                                            ut_ran_tool_err = True
                                            ut_just_errored = True
                                ut_tool_fail_streak = 0
                                ut_tracker.reset(keep_recent=True)
                                break
                        res_text, touched_hid, status_changed, final_report, git_msg = dup_diag, None, False, None, None
                        work_active_label = f"[{gear} GEAR] 重複呼び出しブロック ({tool_name})"
                        work_done_label = f"[{gear} GEAR] 重複呼び出しブロック ({tool_name})"
                        work_dur = 0.0
                        ut_ran_tool_err = True
                        ut_just_errored = True
                    else:
                        work_active_label = f"[{gear} GEAR] {self._describe_tool_action(tool_name, tool_args, in_progress=True)}"
                        work_done_label = f"[{gear} GEAR] {self._describe_tool_action(tool_name, tool_args, in_progress=False)}"
                        self.state.phase_state = "working"
                        self.state.phase_label = work_active_label
                        self.state.phase_started_at = time.monotonic()
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
                        work_start = self.state.phase_started_at
                        res_text, touched_hid, status_changed, final_report, git_msg = await self._execute_ultrathink_tool(
                            tool_name, tool_args, ut_dir, tools, session_id, step
                        )
                        work_dur = round(time.monotonic() - work_start, 1)
                        ut_last_call_key = call_key
                        if tool_name == "run_python":
                            ok_py, _, _ = self._inspect_python_result(res_text)
                            if ok_py:
                                ut_ran_tool_ok = True
                            else:
                                ut_ran_tool_err = True
                                ut_just_errored = True
                        elif tool_name == "run_command":
                            first_ln = res_text.splitlines()[0] if res_text else ""
                            if re.search(r"終了コード:\s*0\b", first_ln) and not res_text.startswith("エラー"):
                                ut_ran_tool_ok = True
                            else:
                                ut_ran_tool_err = True
                                ut_just_errored = True
                        elif res_text.startswith("エラー"):
                            ut_ran_tool_err = True
                            ut_just_errored = True
                        if not res_text.startswith("エラー") and not ut_ran_tool_err:
                            ut_tool_fail_streak = 0
                            ut_burst_used = False

                    if touched_hid:
                        focused_hyp_id = touched_hid
                    if status_changed or tool_name in ("set_hypothesis_status", "update_global_index"):
                        ran_status_update = True
                    if tool_name in ("run_python", "run_command", "web_search", "fetch_url"):
                        ran_verification_tool = True

                    if git_msg:
                        c_hash = self.sessions.git_commit_snapshot(session_id, git_msg)
                        if c_hash:
                            last_commit_info = c_hash

                    trimmed_res = res_text[:2500] + "\n... (省略)" if len(res_text) > 2500 else res_text
                    work_item = {
                        "type": "work",
                        "step": step,
                        "tool": tool_name,
                        "label": work_done_label,
                        "duration": work_dur,
                        "arguments": tool_args,
                        "result": trimmed_res,
                    }
                    thinking_log.append(work_item)
                    self.live_thinking_log.append(work_item)
                    self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
                    await self.broker.publish("work", {**work_item, "session_id": session_id})
                    last_exec_summary = trimmed_res

                    if final_report is not None:
                        ut_pending_final_report = final_report
                        break

                    call_sig, res_sig, reason = self._build_action_stagnation_signature(
                        tool_name, tool_args, res_text
                    )
                    ut_cb_trigger = ut_tracker.record_action(
                        call_sig,
                        res_sig,
                        f"Step {step} [{gear}]: {tool_name} -> {res_text.splitlines()[0][:100] if res_text else ''}",
                        reason,
                    )
                    if ut_cb_trigger:
                        break

                if ut_iso_step_id and (ut_ran_tool_ok or ut_pending_final_report is not None or ran_status_update) and not ut_ran_tool_err and not ut_cb_trigger:
                    iso_res = tools.complete_isolated_workspace(ut_iso_step_id, mode="ultrathink")
                    ut_iso_step_id = None
                    if iso_res:
                        self._sync_global_index_from_files(ut_dir, session_id)
                        await self._emit_trash_isolation_log(
                            session, thinking_log, assistant_msg_id, session_id, step, iso_res
                        )

                if ut_pending_final_report is not None:
                    _, _, gc_hash, _ = await self._run_ultrathink_gc(
                        client, ut_dir, session_id, step, north_star_fixed
                    )
                    if gc_hash:
                        last_commit_info = gc_hash
                    await self._publish_ultrathink_status(session_id, "IDLE", 0.0, [], last_commit_info)
                    await self._finish_turn_with_text(
                        client,
                        tools,
                        session,
                        thinking_log,
                        assistant_msg_id,
                        step,
                        user_message,
                        ut_pending_final_report,
                        harness_state="COMPLETED",
                        consolidate=False,
                    )
                    return

                # Periodic 10-step Git Time Machine snapshot
                if step % 10 == 0:
                    c_hash = self.sessions.git_commit_snapshot(session_id, f"Step {step}: Periodic checkpoint")
                    if c_hash:
                        last_commit_info = c_hash

                metrics_after = self.sessions.get_ultrathink_metrics(session_id)
                ut_progress_after = (
                    metrics_after["unverified"],
                    metrics_after["verified"],
                    metrics_after["rejected"],
                    metrics_after["knowledge_facts"],
                    tuple((it["id"], it["status"]) for it in metrics_after["items"]),
                )
                if not ut_cb_trigger:
                    ut_cb_trigger = ut_tracker.record_step_progress(
                        ut_progress_after != ut_progress_before,
                        f"UT Step {step} ({gear}): GLOBAL_INDEX 進捗なし",
                    )

                if ut_cb_trigger:
                    keep_going = await _apply_ut_replan(ut_cb_trigger)
                    if not keep_going:
                        return
                    continue

                # Dynamic gear transition: after verification execution, enter Low Gear for temp=0.0 status update
                if ran_status_update:
                    pending_low_gear = False
                    same_gear_history = []
                elif ran_verification_tool and not is_low_gear:
                    pending_low_gear = True
                    same_gear_history = []
                elif not is_low_gear:
                    same_gear_history.append(
                        {"role": "assistant", "content": f"[実行: {','.join(c['name'] for c in parsed_calls)}]"}
                    )
                    same_gear_history.append({"role": "user", "content": f"【直前結果】\n{last_exec_summary[:1000]}"})

                focus_after = self._select_active_focus_files(ut_dir, focused_hyp_id=focused_hyp_id)
                await self._publish_ultrathink_status(
                    session_id,
                    UT_GEAR_LOW if pending_low_gear else UT_GEAR_HIGH,
                    0.0 if pending_low_gear else 0.3,
                    focus_after,
                    last_commit_info,
                )
                if (metrics_after["verified"] + metrics_after["rejected"]) > (metrics["verified"] + metrics["rejected"]):
                    ut_subtask_interventions = 0
                    ut_think_unlocked = False
                if metrics_after["unverified"] == 0 and metrics_after["verified"] > 0:
                    focused_hyp_id = None
        except asyncio.CancelledError:
            await self._finalize_stopped(session, thinking_log, assistant_msg_id)
        except Exception as exc:
            self.state.status = "error"
            self.state.phase_state = "idle"
            self.state.phase_label = ""
            self.state.phase_started_at = 0.0
            self.state.harness_state = "ERROR"
            self.active_parallel_batch = None
            self.state.error_message = str(exc)
            self._persist_progress(
                session,
                assistant_msg_id,
                thinking_log,
                latest_content=f"実行エラー: {exc}",
                in_progress=False,
            )
            await self.broker.publish("error", {"message": f"実行エラー: {exc}", "session_id": session_id})
            await self.broker.publish("status", self._status_payload())

    async def _run_fastthink_turn(
        self,
        client: AsyncOpenAI,
        session: dict[str, Any],
        user_message: str,
        assistant_msg_id: str,
    ) -> None:
        """Fast single-turn mode (/fastthink): strict 3000-token thinking limit, no autonomous multi-step loop."""
        session_id = session["id"]
        sdir = self.sessions.ensure_session_files(session_id)
        tools = WorkspaceTools(
            workspace_root=sdir / "workspace",
            memory_dir=self.memory.memory_dir,
            session_dir=sdir,
            plan_path=sdir / "plan_master.md",
            work_log_path=sdir / "work_log.md",
            on_memory_update=self._on_memory_update,
            on_files_update=self._on_files_update,
        )
        self.live_thinking_log = []
        self.last_completed_message = None
        self.active_parallel_batch = None
        self.state.turn_id = assistant_msg_id
        self.broker.set_active_turn_id(assistant_msg_id)
        thinking_log: list[dict[str, Any]] = []
        step = 1
        try:
            self.state.current_step = step
            self.state.harness_state = "FASTTHINK"
            self.state.phase_state = "thinking"
            self.state.phase_label = "[FASTTHINK] 高速単一推論を実行中（3000トークン上限）"
            self.state.phase_started_at = time.monotonic()
            await self.broker.publish(
                "step",
                {"current_step": step, "max_steps": 1, "session_id": session_id},
            )
            await self.broker.publish(
                "status",
                {**self._status_payload(), "state": "thinking", "step": step, "label": self.state.phase_label},
            )

            rule_md = self.memory.read_file("rule.md").strip()[:800]
            prompt_body = re.sub(r"^/fastthink\b\s*", "", user_message.strip(), flags=re.IGNORECASE).strip() or user_message.strip()
            fast_forbidden = self._load_session_forbidden(session, sdir / "work_log.md")
            fast_forbid_block = (
                "\n\n【禁止されたアプローチ（再試行厳禁）】\n" + "\n".join(f"- 禁止: {x}" for x in fast_forbidden[-5:])
                if fast_forbidden
                else ""
            )
            sys_msg = (
                "あなたはクリスタルLMの高速応答担当 (`/fastthink` モード) です。\n"
                f"{rule_md}\n"
                "長期的な計画ループは行わず、思考トークン上限（3000トークン）以内で端的に思考し、"
                "必要な場合はツール（`run_python` 等）または `[WRITE_FILE: path/to/script.py]` を1回だけ使ってすぐ結論を返してください。"
                f"{fast_forbid_block}"
            )
            recent_hist: list[dict[str, Any]] = []
            for prev in session.get("messages", [])[-6:]:
                if prev.get("id") == assistant_msg_id or prev.get("in_progress"):
                    continue
                role = prev.get("role")
                content_val = str(prev.get("content") or "").strip()
                if role in ("user", "assistant") and content_val:
                    recent_hist.append({"role": role, "content": content_val[:1200]})
            if not recent_hist or recent_hist[-1].get("role") != "user":
                recent_hist.append({"role": "user", "content": prompt_body})

            messages = self._merge_consecutive_roles([{"role": "system", "content": sys_msg}, *recent_hist])

            fast_model_override: Optional[str] = None
            fast_call_messages = messages
            fast_call_tools: Optional[list[dict[str, Any]]] = TOOL_SCHEMAS
            fast_use_json_tools = True
            if self._step_involves_coding(STATE_EXECUTE_STEP, prompt_body, TOOL_SCHEMAS):
                route_res = await self._route_coding_model_via_d1(
                    client=client,
                    step=step,
                    session_id=session_id,
                    assistant_msg_id=assistant_msg_id,
                    session=session,
                    thinking_log=thinking_log,
                    task_summary=prompt_body[:800],
                    context_label="Mode: /fastthink",
                )
                fast_model_override = route_res.get("selected_model") or None
                fast_call_messages, fast_call_tools, fast_use_json_tools = self._apply_pure_coder_constraints(
                    messages, TOOL_SCHEMAS, route_res
                )

            think_start = self.state.phase_started_at
            response = await self._create_chat_completion_resilient(
                client,
                fast_call_messages,
                tools=fast_call_tools,
                temperature=0.3,
                check_stop=True,
                allow_tool_drop=True,
                model_override=fast_model_override,
                max_think_tokens=THINK_TOKEN_LIMIT_DEFAULT,
                max_output_tokens=MAX_TOKENS_NORMAL_OUTPUT,
            )
            think_dur = round(time.monotonic() - think_start, 1)
            msg = response.choices[0].message
            raw_content = msg.content or ""
            reasoning_extra = (
                getattr(msg, "reasoning_content", None)
                or getattr(msg, "reasoning", None)
                or ""
            )
            think_text, visible_text = _extract_think_and_content(raw_content)
            combined_thinking = "\n\n".join(p for p in [reasoning_extra, think_text] if p).strip()

            thought_item = {
                "type": "thinking",
                "step": step,
                "duration": think_dur,
                "content": combined_thinking or visible_text or "[FASTTHINK] 単一推論",
                "raw_output": _compose_raw_model_output(raw_content, reasoning_extra),
            }
            thinking_log.append(thought_item)
            self.live_thinking_log.append(thought_item)
            self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
            await self.broker.publish("thinking", {**thought_item, "session_id": session_id})

            fast_allowed = {t["function"]["name"] for t in TOOL_SCHEMAS}
            parsed_calls, fast_diag = _parse_response_tool_calls_with_diagnostics(
                msg,
                visible_text,
                raw_content,
                allowed_names=fast_allowed,
                orphan_rescue_path="workspace/script_auto.py" if not fast_use_json_tools else None,
            )
            if not parsed_calls and fast_diag:
                burst_calls, _, _ = await self._invoke_direct_burst_write(
                    client=client,
                    step=step,
                    session_id=session_id,
                    assistant_msg_id=assistant_msg_id,
                    session=session,
                    thinking_log=thinking_log,
                    task_summary=prompt_body[:1000],
                    last_thought_text=combined_thinking,
                    failure_diag=fast_diag,
                    active_tools=TOOL_SCHEMAS,
                    allowed_names=fast_allowed,
                    model_override=fast_model_override or self.state.model,
                    context_label="Mode: /fastthink",
                )
                if burst_calls:
                    parsed_calls = burst_calls
            if parsed_calls and (not fast_use_json_tools or any(c.get("_orphan_rescue") for c in parsed_calls)):
                parsed_calls = self._augment_pure_coder_write_calls(parsed_calls, fast_allowed)

            if parsed_calls:
                tool_results: list[str] = []
                fast_iso_entered = False
                fast_ran_err = False
                if any(c["name"] in ("write_file", "read_file", "list_files", "run_python", "run_command") for c in parsed_calls):
                    tools.enter_isolated_workspace("fast_1")
                    fast_iso_entered = True
                for call in parsed_calls[:4]:
                    t_name = call["name"]
                    t_args = call["arguments"]
                    work_start = time.monotonic()
                    self.state.phase_state = "working"
                    self.state.phase_label = f"[FASTTHINK] {self._describe_tool_action(t_name, t_args, in_progress=True)}"
                    self.state.phase_started_at = work_start
                    await self.broker.publish(
                        "status",
                        {**self._status_payload(), "state": "working", "step": step, "label": self.state.phase_label},
                    )
                    res_text = await tools.execute(t_name, t_args)
                    if t_name == "run_python":
                        ok_py, _, _ = self._inspect_python_result(res_text)
                        if not ok_py:
                            fast_ran_err = True
                    elif t_name == "run_command":
                        first_ln = res_text.splitlines()[0] if res_text else ""
                        if not re.search(r"終了コード:\s*0\b", first_ln) or res_text.startswith("エラー"):
                            fast_ran_err = True
                    elif res_text.startswith("エラー"):
                        fast_ran_err = True
                    w_dur = round(time.monotonic() - work_start, 1)
                    work_item = {
                        "type": "work",
                        "step": step,
                        "tool": t_name,
                        "label": f"[FASTTHINK] {self._describe_tool_action(t_name, t_args, in_progress=False)}",
                        "duration": w_dur,
                        "arguments": t_args,
                        "result": res_text,
                    }
                    thinking_log.append(work_item)
                    self.live_thinking_log.append(work_item)
                    self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
                    await self.broker.publish("work", {**work_item, "session_id": session_id})
                    tool_results.append(f"### {t_name}\n{res_text[:2000]}")

                if fast_iso_entered and not fast_ran_err:
                    iso_res = tools.complete_isolated_workspace("fast_1", mode="fastthink")
                    if iso_res:
                        await self._emit_trash_isolation_log(
                            session, thinking_log, assistant_msg_id, session_id, step, iso_res
                        )

                self.state.phase_state = "thinking"
                self.state.phase_label = "[FASTTHINK] ツール結果から最終回答を生成中"
                self.state.phase_started_at = time.monotonic()
                await self.broker.publish(
                    "status",
                    {**self._status_payload(), "state": "thinking", "step": step, "label": self.state.phase_label},
                )
                followup_msgs = self._merge_consecutive_roles(
                    messages
                    + [
                        {"role": "assistant", "content": visible_text or "ツールを実行しました。"},
                        {
                            "role": "user",
                            "content": "【ツール実行結果】\n" + "\n\n".join(tool_results) + "\n\n以上の結果をもとに最終回答を簡潔に出力してください。",
                        },
                    ]
                )
                final_resp = await self._create_chat_completion_resilient(
                    client,
                    followup_msgs,
                    tools=None,
                    temperature=0.3,
                    check_stop=True,
                    allow_tool_drop=True,
                    model_override=self.state.model,
                    max_think_tokens=THINK_TOKEN_LIMIT_DEFAULT,
                    max_output_tokens=MAX_TOKENS_NORMAL_OUTPUT,
                )
                f_raw = final_resp.choices[0].message.content or ""
                _, f_vis = _extract_think_and_content(f_raw)
                final_text = f_vis or f_raw or "\n\n".join(tool_results)
            else:
                final_text = visible_text or raw_content or "応答を生成しました。"

            await self._finish_turn_with_text(
                client,
                tools,
                session,
                thinking_log,
                assistant_msg_id,
                step,
                user_message,
                final_text,
                harness_state="COMPLETED",
                consolidate=False,
            )
        except asyncio.CancelledError:
            await self._finalize_stopped(session, thinking_log, assistant_msg_id)
        except Exception as exc:
            self.state.status = "error"
            self.state.phase_state = "idle"
            self.state.phase_label = ""
            self.state.phase_started_at = 0.0
            self.state.harness_state = "ERROR"
            self.state.error_message = str(exc)
            self._persist_progress(
                session,
                assistant_msg_id,
                thinking_log,
                latest_content=f"実行エラー: {exc}",
                in_progress=False,
            )
            await self.broker.publish("error", {"message": f"実行エラー: {exc}", "session_id": session_id})
            await self.broker.publish("status", self._status_payload())

    async def _run_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_msg_id: Optional[str] = None,
    ) -> None:
        assistant_msg_id = assistant_msg_id or f"msg_{uuid.uuid4().hex[:8]}"
        client = AsyncOpenAI(
            base_url=self.state.base_url,
            api_key="lm-studio",
            timeout=LLM_HTTP_TIMEOUT,
            max_retries=0,
        )

        sdir = self.sessions.ensure_session_files(session_id)
        session = self.sessions.get_session(session_id)
        if session.get("mode") == "fastthink":
            await self._run_fastthink_turn(client, session, user_message, assistant_msg_id)
            return
        if session.get("mode") == "ultrathink":
            await self._run_ultrathink_turn(client, session, user_message, assistant_msg_id)
            return
        await self._run_pipeline_turn(client, session, user_message, assistant_msg_id, is_ultrathink=False)
        return
        is_longthink = session.get("mode") == "longthink"
        shared_dir: Optional[Path] = None
        work_log_path = sdir / "work_log.md"
        if is_longthink:
            lt_dir = self.sessions.ensure_longthink_structure(session_id)
            shared_dir = lt_dir / "shared_workspace"
            plan_file_path = shared_dir / "master_plan.md"
            tools = WorkspaceTools(
                workspace_root=shared_dir,
                memory_dir=self.memory.memory_dir,
                session_dir=sdir,
                plan_path=plan_file_path,
                work_log_path=work_log_path,
                shared_workspace_dir=shared_dir,
                on_memory_update=self._on_memory_update,
                on_files_update=self._on_files_update,
            )
        else:
            plan_file_path = sdir / "plan_master.md"
            tools = WorkspaceTools(
                workspace_root=sdir / "workspace",
                memory_dir=self.memory.memory_dir,
                session_dir=sdir,
                plan_path=plan_file_path,
                work_log_path=work_log_path,
                on_memory_update=self._on_memory_update,
                on_files_update=self._on_files_update,
            )

        # New user turn: if no pending '- [ ]' checklist exists, reset to enter INIT_PLAN
        initial_plan = plan_file_path.read_text(encoding="utf-8", errors="replace") if plan_file_path.exists() else ""
        init_unchecked, _ = self._parse_plan_checklist(initial_plan)
        if not init_unchecked:
            blank_plan = "# 進行中タスク計画 (plan_master.md)\n\n（現在進行中の長期タスクはありません）\n"
            self.sessions.sync_plan_files(session_id, blank_plan)
            await self._on_memory_update("plan.md", blank_plan)
            await self._on_files_update()

        self.live_thinking_log = []
        self.last_completed_message = None
        self.active_parallel_batch = None
        self.state.turn_id = assistant_msg_id
        self.broker.set_active_turn_id(assistant_msg_id)
        thinking_log: list[dict[str, Any]] = []
        last_python_error: Optional[str] = None
        pending_verification = False
        last_python_stdout = ""
        prev_state: Optional[str] = None
        same_state_history: list[dict[str, Any]] = []
        sub_agent_counter = 0
        consecutive_timeout_count = 0
        main_tracker = self._LoopStagnationTracker(threshold=CIRCUIT_BREAKER_THRESHOLD)
        force_replan = False
        replan_reason = ""
        replan_strategy = ""
        forbidden_approaches: list[str] = self._load_session_forbidden(session, work_log_path)
        replan_count = 0
        subtask_d1_count = 0
        think_limit_unlocked = False
        main_last_call_key: Optional[str] = None
        main_tool_fail_streak = 0
        main_last_thought = ""
        main_burst_used = False
        active_iso_step_id: Optional[str] = None
        last_error_step_analyzed = False

        try:
            for step in itertools.count(1):
                if self._stop_event.is_set():
                    await self._finalize_stopped(session, thinking_log, assistant_msg_id)
                    return

                plan_text = plan_file_path.read_text(encoding="utf-8", errors="replace") if plan_file_path.exists() else ""
                plan_snap_before = self._parse_plan_checklist(plan_text)
                work_log_text = work_log_path.read_text(encoding="utf-8", errors="replace") if work_log_path.exists() else ""
                if force_replan:
                    current_state = STATE_REPLAN
                else:
                    current_state = self._determine_harness_state(plan_text, last_python_error, pending_verification)
                if current_state != prev_state:
                    same_state_history = []
                    prev_state = current_state

                handoff_text = ""
                sub_agents_summary = ""
                if is_longthink and shared_dir is not None:
                    hb_file = shared_dir / "handoff_board.md"
                    if hb_file.exists():
                        handoff_text = hb_file.read_text(encoding="utf-8", errors="replace")
                    sub_agents_summary = self.sessions.get_longthink_sub_agents_summary(session_id)

                async def _trigger_main_replan(trigger_reason: str, latest_plan_txt: str) -> bool:
                    nonlocal replan_count, subtask_d1_count, think_limit_unlocked
                    nonlocal replan_reason, replan_strategy, force_replan
                    nonlocal same_state_history, last_python_error, pending_verification, last_error_step_analyzed
                    replan_count += 1
                    subtask_d1_count += 1
                    diag = await self._invoke_d1_supervisor(
                        client=client,
                        step=step,
                        session_id=session_id,
                        assistant_msg_id=assistant_msg_id,
                        session=session,
                        thinking_log=thinking_log,
                        recent_history_items=main_tracker.recent_steps,
                        plan_text=latest_plan_txt,
                        fallback_reason=trigger_reason,
                        context_label=f"Mode: {'/longthink' if is_longthink else '/normal'} ({current_state})",
                        intervention_count=subtask_d1_count,
                        think_limit_unlocked=think_limit_unlocked,
                    )
                    think_limit_unlocked = bool(think_limit_unlocked or diag.get("think_unlocked"))
                    replan_reason = diag["stuck_reason"]
                    replan_strategy = diag["new_strategy"]
                    if diag["forbidden_action"]:
                        if diag["forbidden_action"] not in forbidden_approaches:
                            forbidden_approaches.append(diag["forbidden_action"])
                        self._record_session_forbidden(session, diag["forbidden_action"], tools_obj=tools)
                    same_state_history = []
                    last_python_error = None
                    pending_verification = False
                    last_error_step_analyzed = False
                    main_tracker.reset(keep_recent=False)
                    tools._append_work_log(
                        f"D1-2B_SUPERVISOR -> REPLAN #{replan_count} (subtask #{subtask_d1_count}, think_unlocked={think_limit_unlocked}): "
                        f"{replan_reason} | 新戦略: {replan_strategy}"
                    )
                    if diag.get("escalate_to_user"):
                        abort_text = self._format_d1_escalation_report(
                            diag,
                            subtask_d1_count,
                            "/longthink" if is_longthink else "/normal",
                        )
                        await self._finish_turn_with_text(
                            client,
                            tools,
                            session,
                            thinking_log,
                            assistant_msg_id,
                            step,
                            user_message,
                            abort_text,
                            harness_state="PAUSED_FOR_USER",
                            consolidate=False,
                        )
                        return False
                    force_replan = True
                    return True

                self.state.current_step = step
                self.state.harness_state = current_state
                await self.broker.publish(
                    "step",
                    {"current_step": step, "max_steps": self.state.max_steps, "session_id": session_id},
                )

                state_labels = {
                    STATE_INIT_PLAN: "手順計画を立案中 (plan_master.md)",
                    STATE_EXECUTE_STEP: "単一タスクを実行中",
                    STATE_FIX_ERROR: "エラー原因を特定・修復中",
                    STATE_VERIFY_RESULT: "計算結果の独立検算を実行中",
                    STATE_REPLAN: "D1-2B 診断に基づき別方針へ計画を再構築中 (REPLAN)",
                    STATE_FINAL_REPORT: "最終レポートを作成中",
                }
                think_label = f"[{current_state}] {state_labels.get(current_state, '推論中')}"
                self.state.phase_state = "thinking"
                self.state.phase_label = think_label
                self.state.phase_started_at = time.monotonic()
                await self.broker.publish(
                    "status",
                    {**self._status_payload(), "state": "thinking", "step": step, "label": think_label},
                )
                think_start = self.state.phase_started_at

                messages, active_tools = self._build_state_machine_prompt_and_tools(
                    current_state=current_state,
                    user_message=user_message,
                    plan_text=plan_text,
                    work_log_text=work_log_text,
                    is_longthink=is_longthink,
                    last_python_error=last_python_error,
                    last_python_stdout=last_python_stdout,
                    same_state_history=same_state_history,
                    replan_reason=replan_reason,
                    replan_strategy=replan_strategy,
                    forbidden_approaches=forbidden_approaches,
                    handoff_text=handoff_text,
                    sub_agents_summary=sub_agents_summary,
                )

                if current_state in (STATE_INIT_PLAN, STATE_REPLAN) or think_limit_unlocked:
                    step_max_think = MAX_TOKENS_UNLOCKED_CAP
                    step_max_output = MAX_TOKENS_UNLOCKED_CAP
                else:
                    step_max_think = THINK_TOKEN_LIMIT_DEFAULT
                    step_max_output = MAX_TOKENS_NORMAL_OUTPUT
                curr_task_hint = plan_snap_before[0][0] if plan_snap_before[0] else user_message[:400]
                routing_task_text = (
                    f"{curr_task_hint}\n{last_python_error or ''}"
                    if current_state == STATE_FIX_ERROR
                    else (
                        f"{curr_task_hint}\n{last_python_stdout or ''}"
                        if current_state == STATE_VERIFY_RESULT
                        else f"{curr_task_hint}\n{replan_strategy or ''}"
                    )
                )
                route_mode_label = f"Mode: {'/longthink' if is_longthink else '/normal'} ({current_state})"
                if current_state == STATE_INIT_PLAN:
                    route_res = await self._route_coding_model_via_d1(
                        client=client,
                        step=step,
                        session_id=session_id,
                        assistant_msg_id=assistant_msg_id,
                        session=session,
                        thinking_log=thinking_log,
                        task_summary=user_message[:600],
                        context_label=route_mode_label,
                        forced_choice="main_model",
                        forced_reason="初期計画立案 (INIT_PLAN) のためメインモデルを使用",
                    )
                elif current_state == STATE_REPLAN:
                    route_res = await self._route_coding_model_via_d1(
                        client=client,
                        step=step,
                        session_id=session_id,
                        assistant_msg_id=assistant_msg_id,
                        session=session,
                        thinking_log=thinking_log,
                        task_summary=routing_task_text[:900],
                        context_label=route_mode_label,
                        forced_choice="main_model",
                        forced_reason="停滞後の原因分析と戦略再構築 (REPLAN) のためメインモデルを使用",
                    )
                elif current_state == STATE_FINAL_REPORT:
                    route_res = await self._route_coding_model_via_d1(
                        client=client,
                        step=step,
                        session_id=session_id,
                        assistant_msg_id=assistant_msg_id,
                        session=session,
                        thinking_log=thinking_log,
                        task_summary=curr_task_hint[:600],
                        context_label=route_mode_label,
                        forced_choice="main_model",
                        forced_reason="最終統合レポート生成のためメインモデルを使用",
                    )
                elif current_state == STATE_FIX_ERROR and not last_error_step_analyzed:
                    route_res = await self._route_coding_model_via_d1(
                        client=client,
                        step=step,
                        session_id=session_id,
                        assistant_msg_id=assistant_msg_id,
                        session=session,
                        thinking_log=thinking_log,
                        task_summary=routing_task_text[:900],
                        context_label=route_mode_label,
                        forced_choice="main_model",
                        forced_reason="エラー発生直後の原因分析と修復方針決定のためメインモデルを使用",
                    )
                    last_error_step_analyzed = True
                elif is_longthink and current_state == STATE_EXECUTE_STEP:
                    route_res = await self._route_coding_model_via_d1(
                        client=client,
                        step=step,
                        session_id=session_id,
                        assistant_msg_id=assistant_msg_id,
                        session=session,
                        thinking_log=thinking_log,
                        task_summary=routing_task_text[:900],
                        context_label=route_mode_label,
                        forced_choice="main_model",
                        forced_reason="Master Orchestrator による動的統合・サブエージェント統括のためメインモデルを使用",
                    )
                else:
                    route_res = await self._route_coding_model_via_d1(
                        client=client,
                        step=step,
                        session_id=session_id,
                        assistant_msg_id=assistant_msg_id,
                        session=session,
                        thinking_log=thinking_log,
                        task_summary=routing_task_text[:900],
                        context_label=route_mode_label,
                    )
                step_model_override = route_res.get("selected_model") or None
                step_call_messages, step_call_tools, step_use_json_tools = self._apply_pure_coder_constraints(
                    messages, active_tools, route_res
                )

                try:
                    response = await self._call_state_llm(
                        client,
                        step_call_messages,
                        step_call_tools or [],
                        model_override=step_model_override,
                        max_think_tokens=step_max_think,
                        max_output_tokens=step_max_output,
                    )
                    consecutive_timeout_count = 0
                except (
                    APITimeoutError,
                    APIConnectionError,
                    httpx.TimeoutException,
                    httpx.TransportError,
                ) as timeout_exc:
                    if self._stop_event.is_set():
                        await self._finalize_stopped(session, thinking_log, assistant_msg_id)
                        return
                    consecutive_timeout_count += 1
                    warn_dur = round(time.monotonic() - think_start, 1)
                    warn_item = {
                        "type": "thinking",
                        "step": step,
                        "duration": warn_dur,
                        "content": (
                            f"[{current_state}] Step {step} で LLM 応答タイムアウトが発生したため"
                            f"（連続 {consecutive_timeout_count} 回目: {timeout_exc}）、"
                            "直近の途中履歴を圧縮して次のステップで推論を再開します。"
                        ),
                    }
                    thinking_log.append(warn_item)
                    self.live_thinking_log.append(warn_item)
                    same_state_history = []
                    self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
                    await self.broker.publish("thinking", {**warn_item, "session_id": session_id})
                    if consecutive_timeout_count <= 2:
                        continue
                    raise

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
                if combined_thinking:
                    main_last_thought = combined_thinking

                parsed_calls: list[dict[str, Any]] = []
                parse_diag: Optional[str] = None
                allowed_names: set[str] = set()
                if current_state != STATE_FINAL_REPORT:
                    allowed_names = {t["function"]["name"] for t in active_tools}
                    rescue_path = (
                        ("script_auto.py" if is_longthink else "workspace/script_auto.py")
                        if (
                            not step_use_json_tools
                            or current_state in (STATE_EXECUTE_STEP, STATE_FIX_ERROR, STATE_VERIFY_RESULT)
                        )
                        else None
                    )
                    parsed_calls, parse_diag = _parse_response_tool_calls_with_diagnostics(
                        msg,
                        visible_text,
                        raw_content,
                        allowed_names=allowed_names,
                        orphan_rescue_path=rescue_path,
                    )
                    if parsed_calls and (not step_use_json_tools or any(c.get("_orphan_rescue") for c in parsed_calls)):
                        parsed_calls = self._augment_pure_coder_write_calls(parsed_calls, allowed_names)

                print(
                    f"[Step {step}] State={current_state} -> tools={','.join(c['name'] for c in parsed_calls) or 'none'}",
                    flush=True,
                )

                thought_content = combined_thinking or (
                    visible_text
                    if parsed_calls
                    else (
                        f"[{current_state}] 応答生成"
                        if current_state == STATE_FINAL_REPORT
                        else f"[{current_state}] 単一タスク処理"
                    )
                )
                thought_item = {
                    "type": "thinking",
                    "step": step,
                    "duration": think_duration,
                    "content": thought_content,
                    "raw_output": _compose_raw_model_output(raw_content, reasoning_extra),
                }
                thinking_log.append(thought_item)
                self.live_thinking_log.append(thought_item)
                self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
                await self.broker.publish("thinking", {**thought_item, "session_id": session_id})

                # Immediate rescue if consecutive tool-call JSON failures / truncations happened
                if current_state not in (STATE_FINAL_REPORT, STATE_INIT_PLAN, STATE_REPLAN) and not parsed_calls and (
                    parse_diag or not visible_text
                ):
                    main_tool_fail_streak += 1
                    fail_reason = parse_diag or "思考のみで実行ツール呼び出しやコード出力が返されませんでした。"
                    if main_tool_fail_streak >= 2 and not main_burst_used:
                        main_burst_used = True
                        burst_calls, burst_diag, burst_vis = await self._invoke_direct_burst_write(
                            client=client,
                            step=step,
                            session_id=session_id,
                            assistant_msg_id=assistant_msg_id,
                            session=session,
                            thinking_log=thinking_log,
                            task_summary=routing_task_text[:1000] or user_message[:800],
                            last_thought_text=main_last_thought,
                            failure_diag=fail_reason,
                            active_tools=active_tools,
                            allowed_names=allowed_names,
                            model_override=step_model_override or self.state.model,
                            context_label=f"Mode: {'/longthink' if is_longthink else '/normal'} ({current_state})",
                        )
                        if burst_calls:
                            parsed_calls = (
                                self._augment_pure_coder_write_calls(burst_calls, allowed_names)
                                if (not step_use_json_tools or any(c.get("_orphan_rescue") for c in burst_calls))
                                else burst_calls
                            )
                            parse_diag = burst_diag
                            visible_text = burst_vis
                            main_tool_fail_streak = 0
                        elif burst_diag:
                            parse_diag = burst_diag

                # State 5: FINAL_REPORT (tools disabled -> flush any remaining isolated workspace & finish turn cleanly)
                if current_state == STATE_FINAL_REPORT:
                    if active_iso_step_id:
                        iso_flush = tools.complete_isolated_workspace(
                            active_iso_step_id, mode="longthink" if is_longthink else "normal"
                        )
                        active_iso_step_id = None
                        if iso_flush:
                            await self._emit_trash_isolation_log(
                                session, thinking_log, assistant_msg_id, session_id, step, iso_flush
                            )
                    final_text = visible_text or raw_content or "すべての検証が完了しました。"
                    await self._finish_turn_with_text(
                        client,
                        tools,
                        session,
                        thinking_log,
                        assistant_msg_id,
                        step,
                        user_message,
                        final_text,
                        harness_state="COMPLETED",
                        consolidate=True,
                    )
                    return

                if parsed_calls and visible_text:
                    narration = self._sanitize_intermediate_narration(visible_text)
                    if narration:
                        step_out_item = {"type": "step_output", "step": step, "content": narration}
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

                if not parsed_calls and parse_diag:
                    diag_text = f"【ツール引数パース・記述エラー診断】\n{parse_diag}\n\n{DIRECT_WRITE_FORMAT_GUIDE}"
                    same_state_history.append({"role": "assistant", "content": visible_text[:400] or "(不正なツール呼び出し)"})
                    same_state_history.append({"role": "user", "content": diag_text})
                    cb_parse = main_tracker.record_action(
                        f"call:parse_err:{parse_diag[:40]}",
                        f"res:parse_err:{parse_diag[:80]}",
                        f"Step {step} [{current_state}]: ツール形式エラー -> {parse_diag[:100]}",
                        "JSON引数パースまたはコード出力の形式失敗が続いている",
                    )
                    if cb_parse:
                        keep_going = await _trigger_main_replan(cb_parse, plan_text)
                        if not keep_going:
                            return
                    continue

                unchecked_before = plan_snap_before[0]
                ran_python_ok = False
                ran_python_err = False
                ran_command_ok = False
                ran_tool_err = False
                ran_parallel_count = 0
                triggered_circuit_breaker = False

                multi_delegates = [
                    c for c in parsed_calls if is_longthink and c["name"] == "delegate_sub_agent"
                ]
                coalesced_delegate_result: Optional[str] = None

                if any(c["name"] in ("write_file", "read_file", "list_files", "run_python", "run_command") for c in parsed_calls):
                    active_iso_step_id = active_iso_step_id or str(step)
                    tools.enter_isolated_workspace(active_iso_step_id)

                for call in parsed_calls:
                    if self._stop_event.is_set():
                        await self._finalize_stopped(session, thinking_log, assistant_msg_id)
                        return

                    tool_name = call["name"]
                    tool_args = call["arguments"]
                    call_key = _canonical_call_key(tool_name, tool_args)

                    if main_last_call_key is not None and call_key == main_last_call_key:
                        main_tool_fail_streak += 1
                        dup_diag = (
                            f"エラー（重複呼び出しブロック）: 直前と全く同一のツール名 `{tool_name}`・同一引数での再呼び出しはブロックされました。"
                            "JSON引数エスケープによる失敗やループを回避するため、必ず `[WRITE_FILE: path/to/script.py]` + ```python ... ``` "
                            "形式で直接ファイル出力するか、処理を複数ファイルへ分割してください。"
                        )
                        if main_tool_fail_streak >= 2 and not main_burst_used:
                            main_burst_used = True
                            burst_calls, _, _ = await self._invoke_direct_burst_write(
                                client=client,
                                step=step,
                                session_id=session_id,
                                assistant_msg_id=assistant_msg_id,
                                session=session,
                                thinking_log=thinking_log,
                                task_summary=routing_task_text[:1000] or user_message[:800],
                                last_thought_text=main_last_thought,
                                failure_diag=dup_diag,
                                active_tools=active_tools,
                                allowed_names=allowed_names,
                                model_override=step_model_override or self.state.model,
                                context_label=f"Duplicate Block Rescue ({current_state})",
                            )
                            if burst_calls:
                                if not step_use_json_tools:
                                    burst_calls = self._augment_pure_coder_write_calls(burst_calls, allowed_names)
                                if any(bc["name"] in ("write_file", "read_file", "list_files", "run_python", "run_command") for bc in burst_calls):
                                    active_iso_step_id = active_iso_step_id or str(step)
                                    tools.enter_isolated_workspace(active_iso_step_id)
                                for bc in burst_calls:
                                    b_res = await tools.execute(bc["name"], bc["arguments"])
                                    main_last_call_key = _canonical_call_key(bc["name"], bc["arguments"])
                                    b_item = {
                                        "type": "work",
                                        "step": step,
                                        "tool": bc["name"],
                                        "label": f"[{current_state}] {self._describe_tool_action(bc['name'], bc['arguments'], in_progress=False)}",
                                        "duration": 0.0,
                                        "arguments": bc["arguments"],
                                        "result": b_res,
                                    }
                                    thinking_log.append(b_item)
                                    self.live_thinking_log.append(b_item)
                                    self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
                                    await self.broker.publish("work", {**b_item, "session_id": session_id})
                                    same_state_history.append(
                                        {"role": "user", "content": f"【Direct Burst 実行結果 ({bc['name']})】\n{b_res[:1200]}"}
                                    )
                                    if bc["name"] == "run_python":
                                        ok, err_ex, out_ex = self._inspect_python_result(b_res)
                                        if not ok:
                                            ran_python_err = True
                                            last_python_error = err_ex
                                            last_error_step_analyzed = False
                                            pending_verification = False
                                        else:
                                            ran_python_ok = True
                                            last_python_error = None
                                            last_error_step_analyzed = False
                                            last_python_stdout = out_ex
                                main_tool_fail_streak = 0
                                main_tracker.reset(keep_recent=True)
                                triggered_circuit_breaker = True
                                break
                        result_text = dup_diag
                        ran_tool_err = True
                        work_active_label = f"[{current_state}] 重複呼び出しブロック ({tool_name})"
                        work_done_label = f"[{current_state}] 重複呼び出しブロック ({tool_name})"
                        work_duration = 0.0
                    else:
                        work_active_label = f"[{current_state}] {self._describe_tool_action(tool_name, tool_args, in_progress=True)}"
                        work_done_label = f"[{current_state}] {self._describe_tool_action(tool_name, tool_args, in_progress=False)}"
                        self.state.phase_state = "working"
                        self.state.phase_label = work_active_label
                        self.state.phase_started_at = time.monotonic()
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
                        work_start = self.state.phase_started_at

                        if is_longthink and tool_name == "update_master_plan":
                            content_str = str(tool_args.get("content", ""))
                            self.sessions.sync_plan_files(session_id, content_str)
                            await self._on_memory_update("plan.md", content_str)
                            await self._on_files_update()
                            result_text = "plan_master.md / master_plan.md を更新しました。"
                        elif is_longthink and tool_name == "delegate_parallel_workers":
                            raw_tasks = tool_args.get("tasks") if isinstance(tool_args, dict) else []
                            tasks_spec = raw_tasks if isinstance(raw_tasks, list) else []
                            if not tasks_spec:
                                result_text = "エラー: tasks 配列に少なくとも1件のサブタスクを指定してください。"
                            else:
                                batches = [tasks_spec[i : i + 4] for i in range(0, len(tasks_spec), 4)]
                                batch_outputs: list[str] = []
                                for b_items in batches:
                                    b_res = await self._run_parallel_sub_agents(
                                        client,
                                        session,
                                        assistant_msg_id,
                                        thinking_log,
                                        b_items,
                                        parent_step=step,
                                        start_index=sub_agent_counter + 1,
                                    )
                                    batch_count = len(b_items[:4])
                                    sub_agent_counter += batch_count
                                    ran_parallel_count += batch_count
                                    batch_outputs.append(b_res)
                                result_text = "\n\n".join(batch_outputs)
                        elif is_longthink and tool_name == "delegate_sub_agent":
                            if len(multi_delegates) > 1:
                                if coalesced_delegate_result is None:
                                    combined_specs = [
                                        c["arguments"] if isinstance(c["arguments"], dict) else {}
                                        for c in multi_delegates[:4]
                                    ]
                                    coalesced_delegate_result = await self._run_parallel_sub_agents(
                                        client,
                                        session,
                                        assistant_msg_id,
                                        thinking_log,
                                        combined_specs,
                                        parent_step=step,
                                        start_index=sub_agent_counter + 1,
                                    )
                                    sub_agent_counter += len(combined_specs)
                                    ran_parallel_count += len(combined_specs)
                                result_text = coalesced_delegate_result
                            else:
                                single_spec = tool_args if isinstance(tool_args, dict) else {}
                                result_text = await self._run_parallel_sub_agents(
                                    client,
                                    session,
                                    assistant_msg_id,
                                    thinking_log,
                                    [single_spec],
                                    parent_step=step,
                                    start_index=sub_agent_counter + 1,
                                )
                                sub_agent_counter += 1
                                ran_parallel_count += 1
                        elif is_longthink and tool_name == "save_final_artifact" and shared_dir is not None:
                            artifacts_dir = (shared_dir / "final_artifacts").resolve()
                            artifacts_dir.mkdir(parents=True, exist_ok=True)
                            raw_fname = str(tool_args.get("filename", "artifact.md")).strip().replace("\\", "/").lstrip("/")
                            target_art = (artifacts_dir / raw_fname).resolve()
                            if artifacts_dir not in target_art.parents and target_art != artifacts_dir:
                                result_text = f"エラー: final_artifacts 外への書き込みは禁止されています ({raw_fname})"
                            else:
                                target_art.parent.mkdir(parents=True, exist_ok=True)
                                content_str = str(tool_args.get("content", ""))
                                target_art.write_text(content_str, encoding="utf-8")
                                await self._on_files_update()
                                result_text = (
                                    f"最終成果物を保存しました: shared_workspace/final_artifacts/"
                                    f"{target_art.relative_to(artifacts_dir).as_posix()}"
                                )
                        else:
                            result_text = await tools.execute(tool_name, tool_args)

                        work_duration = round(time.monotonic() - work_start, 1)
                        main_last_call_key = call_key
                        if not result_text.startswith("エラー"):
                            main_tool_fail_streak = 0
                            main_burst_used = False
                        else:
                            ran_tool_err = True
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
                    self._persist_progress(session, assistant_msg_id, thinking_log, in_progress=True)
                    await self.broker.publish("work", {**work_item, "session_id": session_id})

                    same_state_history.append(
                        {"role": "assistant", "content": f"[実行: {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:300]})]"}
                    )
                    same_state_history.append(
                        {"role": "user", "content": f"【実行結果 ({tool_name})】\n{result_text[:1200]}"}
                    )

                    if tool_name == "run_python":
                        ok, err_excerpt, out_excerpt = self._inspect_python_result(result_text)
                        if not ok:
                            ran_python_err = True
                            last_python_error = err_excerpt
                            last_error_step_analyzed = False
                            pending_verification = False
                        else:
                            ran_python_ok = True
                            last_python_error = None
                            last_error_step_analyzed = False
                            last_python_stdout = out_excerpt
                    elif tool_name == "run_command":
                        first_ln = result_text.splitlines()[0] if result_text else ""
                        if re.search(r"終了コード:\s*0\b", first_ln) and not result_text.startswith("エラー"):
                            ran_command_ok = True
                        else:
                            ran_tool_err = True

                    if current_state != STATE_REPLAN:
                        call_sig, res_sig, reason = self._build_action_stagnation_signature(
                            tool_name, tool_args, result_text
                        )
                        cb_reason = main_tracker.record_action(
                            call_sig,
                            res_sig,
                            f"Step {step} [{current_state}]: {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:80]}) -> {result_text.splitlines()[0][:100] if result_text else ''}",
                            reason,
                        )
                        if cb_reason:
                            keep_going = await _trigger_main_replan(cb_reason, plan_text)
                            if not keep_going:
                                return
                            triggered_circuit_breaker = True
                            break

                    if tool_name == "run_python" and ran_python_err:
                        break

                if active_iso_step_id and (ran_python_ok or ran_command_ok) and not ran_python_err and not ran_tool_err and not triggered_circuit_breaker:
                    iso_res = tools.complete_isolated_workspace(
                        active_iso_step_id, mode="longthink" if is_longthink else "normal"
                    )
                    active_iso_step_id = None
                    if iso_res:
                        await self._emit_trash_isolation_log(
                            session, thinking_log, assistant_msg_id, session_id, step, iso_res
                        )

                if triggered_circuit_breaker:
                    continue

                # Deterministic Post-Step State Transitions
                plan_after_text = plan_file_path.read_text(encoding="utf-8", errors="replace") if plan_file_path.exists() else ""
                unchecked_after, _ = self._parse_plan_checklist(plan_after_text)
                marked_plan_by_model = len(unchecked_after) < len(unchecked_before)

                if active_iso_step_id and marked_plan_by_model and not ran_python_err and not ran_tool_err:
                    iso_res_plan = tools.complete_isolated_workspace(
                        active_iso_step_id, mode="longthink" if is_longthink else "normal"
                    )
                    active_iso_step_id = None
                    if iso_res_plan:
                        await self._emit_trash_isolation_log(
                            session, thinking_log, assistant_msg_id, session_id, step, iso_res_plan
                        )

                if current_state == STATE_REPLAN:
                    if not unchecked_after or plan_after_text == plan_text:
                        fallback_replan = self._build_fallback_plan_from_response(
                            visible_text or raw_content, user_message
                        )
                        replan_header = (
                            f"# D1-2B 再構築計画 (REPLAN #{replan_count})\n"
                            f"- 詰まり原因（禁止方針）: {replan_reason}\n"
                            f"- 強制新戦略: {replan_strategy}\n\n"
                        )
                        body_lines = "\n".join(
                            ln for ln in fallback_replan.splitlines() if not ln.startswith("# ")
                        ).strip()
                        strat_task = f"- [ ] {replan_strategy}\n" if replan_strategy else ""
                        combined_replan = f"{replan_header}{strat_task}{body_lines}\n"
                        self.sessions.sync_plan_files(session_id, combined_replan)
                        await self._on_memory_update("plan.md", combined_replan)
                    tools._append_work_log(f"REPLAN #{replan_count}: D1-2B 新戦略に基づき再開します")
                    force_replan = False
                    same_state_history = []
                    main_tracker.reset(keep_recent=True)
                    await self._on_files_update()
                    continue

                if current_state == STATE_INIT_PLAN:
                    if not unchecked_after:
                        fallback_plan = self._build_fallback_plan_from_response(
                            visible_text or raw_content, user_message
                        )
                        self.sessions.sync_plan_files(session_id, fallback_plan)
                        await self._on_memory_update("plan.md", fallback_plan)
                    tools._append_work_log("INIT_PLAN: plan_master.md を作成しました")
                    await self._on_files_update()
                    same_state_history = []
                    main_tracker.reset(keep_recent=True)
                    continue

                if ran_python_err:
                    pass
                elif current_state == STATE_VERIFY_RESULT:
                    if not marked_plan_by_model:
                        new_plan, done_item = self._mark_first_unchecked_done(plan_after_text)
                        if done_item:
                            self.sessions.sync_plan_files(session_id, new_plan)
                            await self._on_memory_update("plan.md", new_plan)
                    else:
                        done_item = unchecked_before[0] if unchecked_before else "検算完了"
                    verify_note = (last_python_stdout or visible_text or "OK").replace("\n", " ")[:160]
                    tools._append_work_log(f"VERIFY_RESULT ({done_item}): {verify_note}")
                    pending_verification = False
                    same_state_history = []
                    await self._on_files_update()
                elif ran_python_ok:
                    if is_longthink:
                        if marked_plan_by_model:
                            self.sessions.sync_plan_files(session_id, plan_text)
                            await self._on_memory_update("plan.md", plan_text)
                        pending_verification = True
                        same_state_history = []
                    else:
                        if not marked_plan_by_model:
                            new_plan, _ = self._mark_first_unchecked_done(plan_after_text)
                            self.sessions.sync_plan_files(session_id, new_plan)
                            await self._on_memory_update("plan.md", new_plan)
                        same_state_history = []
                        await self._on_files_update()
                elif ran_parallel_count > 0:
                    tools._append_work_log(
                        f"SUB_AGENTS: {ran_parallel_count}体実行終了 (handoff_board.md 更新済)"
                    )
                    if marked_plan_by_model:
                        same_state_history = []
                    else:
                        same_state_history = [
                            {
                                "role": "user",
                                "content": (
                                    "サブエージェントの結果が shared_workspace/handoff_board.md に記録されました。"
                                    "報告を確認・統合し、完了したタスクは `update_master_plan` で `[x]` に更新するか、"
                                    "追加のサブエージェント生成・方針変更指示を行ってください。"
                                ),
                            }
                        ]
                    await self._on_files_update()
                elif not parsed_calls:
                    if visible_text:
                        snippet = " ".join(visible_text.strip().splitlines())[:180]
                        curr_task_name = unchecked_before[0] if unchecked_before else "step"
                        tools._append_work_log(f"作業結果 ({curr_task_name}): {snippet}")
                        new_plan, _ = self._mark_first_unchecked_done(plan_after_text)
                        self.sessions.sync_plan_files(session_id, new_plan)
                        await self._on_memory_update("plan.md", new_plan)
                        same_state_history = []
                        await self._on_files_update()
                    else:
                        same_state_history.append(
                            {
                                "role": "user",
                                "content": "思考のみで出力が空でした。指示された1タスクだけを実行するか結果を返してください。",
                            }
                        )
                        main_tracker.recent_steps.append(f"Step {step} [{current_state}]: 思考のみ空出力")
                elif marked_plan_by_model:
                    same_state_history = []

                # Unified 2-consecutive no-plan-progress check across all execution states
                plan_final_after = plan_file_path.read_text(encoding="utf-8", errors="replace") if plan_file_path.exists() else ""
                plan_snap_after = self._parse_plan_checklist(plan_final_after)
                plan_progressed = plan_snap_after != plan_snap_before
                if plan_progressed and len(plan_snap_after[0]) < len(plan_snap_before[0]):
                    replan_strategy = ""
                    subtask_d1_count = 0
                    think_limit_unlocked = False
                cb_progress = main_tracker.record_step_progress(
                    plan_progressed,
                    f"Step {step} [{current_state}]: plan.md 変化なし (tools={','.join(c['name'] for c in parsed_calls) or 'none'})",
                )
                if cb_progress:
                    keep_going = await _trigger_main_replan(cb_progress, plan_final_after)
                    if not keep_going:
                        return
                    continue

        except asyncio.CancelledError:
            await self._finalize_stopped(session, thinking_log, assistant_msg_id)
        except Exception as exc:
            self.state.status = "error"
            self.state.phase_state = "idle"
            self.state.phase_label = ""
            self.state.phase_started_at = 0.0
            self.state.harness_state = "ERROR"
            self.active_parallel_batch = None
            self.state.error_message = str(exc)
            self._persist_progress(
                session,
                assistant_msg_id,
                thinking_log,
                latest_content=f"実行エラー: {exc}",
                in_progress=False,
            )
            await self.broker.publish("error", {"message": f"実行エラー: {exc}", "session_id": session_id})
            await self.broker.publish("status", self._status_payload())

    async def _finalize_stopped(
        self,
        session: dict[str, Any],
        thinking_log: list[dict[str, Any]],
        assistant_msg_id: Optional[str] = None,
    ) -> None:
        resolved_msg_id = assistant_msg_id or self.state.turn_id or f"msg_{uuid.uuid4().hex[:8]}"
        self.state.status = "stopped"
        self.state.phase_state = "idle"
        self.state.phase_label = ""
        self.state.phase_started_at = 0.0
        self.state.harness_state = "STOPPED"
        self.active_parallel_batch = None
        assistant_entry = self._persist_progress(
            session,
            resolved_msg_id,
            thinking_log,
            latest_content="（ユーザーにより停止されました）",
            in_progress=False,
        )
        self.last_completed_message = assistant_entry
        await self.broker.publish(
            "final_answer",
            {
                "step": self.state.current_step,
                "session_id": session["id"],
                "turn_id": resolved_msg_id,
                "message": assistant_entry,
            },
        )
        await self.broker.publish("status", self._status_payload())
        self.broker.clear_history()
