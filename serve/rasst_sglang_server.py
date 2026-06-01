#!/usr/bin/env python3
"""RASST demo wrapper backed by SGLang-Omni.

Topology:
- Qwen3-Omni generation runs in an external SGLang-Omni server.
- The SGLang thinker uses TP=2 across logical cuda:0,cuda:1.
- RASST MaxSim RAG is loaded in this wrapper on logical cuda:1.
- Browser/demo protocol stays compatible with the existing UI and stress test.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple

import aiohttp
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles


TARGET_SAMPLE_RATE = 16000
DEFAULT_SEGMENT_SEC = 1.92
DEFAULT_RAG_LOOKBACK_SEC = 1.92
DEFAULT_MAX_CACHE_CHUNKS = 16
DEFAULT_KEEP_CACHE_CHUNKS = 8
DEFAULT_MAX_IMPORTED_GLOSSARY_TERMS = 10000
DEFAULT_SYSTEM_PROMPT_STYLE = "given_chunks"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
RASST_ROOT = Path(os.environ.get("RASST_ROOT", "/mnt/taurus/data2/jiaxuanluo/RASST"))
RASST_CODE_ROOT = Path(os.environ.get("RASST_ACTIVE_CODE_ROOT", RASST_ROOT / "code/rasst"))
RASST_EVAL_ROOT = RASST_CODE_ROOT / "eval"
DEMO_DATA_ROOT = Path(os.environ.get("RASST_DEMO_DATA_ROOT", "/mnt/taurus/data2/jiaxuanluo/rasst-demo"))


LANGUAGE_PAIRS = {
    "English -> Chinese": {
        "source_lang": "English",
        "target_lang": "Chinese",
        "lang_code": "zh",
        "model_path": os.environ.get(
            "RASST_MODEL_ZH_CAP16_DENOISE",
            "/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/models/"
            "speech_llm_zh_cap16_denoise_budget_ttag_r32a32_ep1_taurus4_hf",
        ),
        "index_path": os.environ.get(
            "RASST_INDEX_ZH_ACL",
            str(
                RASST_ROOT
                / "outputs/main_result_eval/20260527T071109Z/index_cache/"
                "acl_tagged_raw__zh__lm2/"
                "maxsim_acl6060_tagged_gt_raw_min_norm2_ebc26806ed693f1a_tr128_ta256.pt"
            ),
        ),
    },
    "English -> Japanese": {
        "source_lang": "English",
        "target_lang": "Japanese",
        "lang_code": "ja",
        "model_path": os.environ.get(
            "RASST_MODEL_JA_CAP16_DENOISE",
            "/mnt/taurus/data1/jiaxuanluo/slm_local_cache/"
            "ja_tagged_acl_20260525/cap16_denoise_ttag/v2-20260525-235251-hf",
        ),
        "index_path": os.environ.get(
            "RASST_INDEX_JA_ACL",
            str(
                RASST_ROOT
                / "outputs/main_result_eval/20260527T071109Z/index_cache/"
                "acl_tagged_raw__ja__lm2/"
                "maxsim_acl6060_tagged_gt_raw_min_norm2_ebc26806ed693f1a_tr128_ta256.pt"
            ),
        ),
    },
    "English -> German": {
        "source_lang": "English",
        "target_lang": "German",
        "lang_code": "de",
        "model_path": os.environ.get(
            "RASST_MODEL_DE_CAP16_DENOISE",
            "/mnt/taurus/data1/jiaxuanluo/slm_local_cache/"
            "de_tagged_acl_20260525/cap16_denoise_ttag/v0-20260525-203735-hf",
        ),
        "index_path": os.environ.get(
            "RASST_INDEX_DE_ACL",
            str(
                RASST_ROOT
                / "outputs/main_result_eval/20260527T071109Z/index_cache/"
                "acl_tagged_raw__de__lm2/"
                "maxsim_acl6060_tagged_gt_raw_min_norm2_ebc26806ed693f1a_tr128_ta256.pt"
            ),
        ),
    },
}

MODEL_PROFILES = [
    {
        "id": "RASST",
        "label": "RASST",
        "default": True,
        "backend": "qwen3_omni_sglang_tp2_maxsim_rag",
    },
    {
        "id": "InfiniSST",
        "label": "InfiniSST Legacy",
        "default": False,
        "backend": "legacy_infinisst_faster",
    },
]

INDEX_CACHE_DIR = DEMO_DATA_ROOT / "runtime/glossary_indexes"
GLOSSARY_RUNTIME_DIR = DEMO_DATA_ROOT / "runtime/glossaries"
MAIN_RESULT_INDEX_DIR = RASST_ROOT / "outputs/main_result_eval/20260527T071109Z/index_cache"
ACL_RAW_GLOSSARY = RASST_ROOT / "data/glossaries/acl6060_tagged_gt_raw_min_norm2.json"
MEDICINE_RAW_GLOSSARY = (
    RASST_ROOT / "data/glossaries/hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212.json"
)
MEDICINE_10K_GLOSSARY = (
    Path("/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only")
    / "medicine_glossary_gt_plus_medicine_wiki_gs10000_translated.json"
)
DEFAULT_GLOSSARY_PRESET = "none"
RAG_STARTUP_GLOSSARY_PRESET = "acl_tagged_raw"


def _main_result_index(domain: str, lang_code: str, latency_multiplier: int, filename: str) -> str:
    return str(MAIN_RESULT_INDEX_DIR / f"{domain}__{lang_code}__lm{latency_multiplier}" / filename)


GLOSSARY_PRESETS = {
    "none": {
        "id": "none",
        "label": "None",
        "path": "",
        "domain": "none",
        "index_path": "",
    },
    "acl_tagged_raw": {
        "id": "acl_tagged_raw",
        "label": "ACL tagged glossary raw",
        "path": str(ACL_RAW_GLOSSARY),
        "domain": "acl6060",
        "index_paths": {
            "zh": _main_result_index(
                "acl_tagged_raw",
                "zh",
                2,
                "maxsim_acl6060_tagged_gt_raw_min_norm2_ebc26806ed693f1a_tr128_ta256.pt",
            ),
            "ja": _main_result_index(
                "acl_tagged_raw",
                "ja",
                2,
                "maxsim_acl6060_tagged_gt_raw_min_norm2_ebc26806ed693f1a_tr128_ta256.pt",
            ),
            "de": _main_result_index(
                "acl_tagged_raw",
                "de",
                2,
                "maxsim_acl6060_tagged_gt_raw_min_norm2_ebc26806ed693f1a_tr128_ta256.pt",
            ),
        },
    },
    "acl_tagged_1k": {
        "id": "acl_tagged_1k",
        "label": "ACL tagged glossary 1k",
        "path": "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs1000.json",
        "domain": "acl6060",
        "index_path": str(INDEX_CACHE_DIR / "maxsim_acl6060_gt_union_gs1000_hn1024_tr128_ta256.pt"),
    },
    "acl_tagged_10k": {
        "id": "acl_tagged_10k",
        "label": "ACL tagged glossary 10k",
        "path": "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs10000.json",
        "domain": "acl6060",
        "index_path": str(INDEX_CACHE_DIR / "maxsim_acl6060_gt_union_gs10000_hn1024_tr128_ta256.pt"),
    },
    "medicine_raw": {
        "id": "medicine_raw",
        "label": "Medicine raw glossary",
        "path": str(MEDICINE_RAW_GLOSSARY),
        "domain": "medicine",
        "index_paths": {
            "zh": _main_result_index(
                "medicine_hardraw",
                "zh",
                2,
                "maxsim_hard_medicine_glossary_raw_llm_judge_manual_zh21_6d02fb5133b93f6d_tr128_ta256.pt",
            ),
            "de": _main_result_index(
                "medicine_hardraw",
                "de",
                1,
                "maxsim_hard_medicine_glossary_raw_llm_judge_manual_zh21_6d02fb5133b93f6d_tr128_ta256.pt",
            ),
            "ja": _main_result_index(
                "medicine_hardraw",
                "zh",
                2,
                "maxsim_hard_medicine_glossary_raw_llm_judge_manual_zh21_6d02fb5133b93f6d_tr128_ta256.pt",
            ),
        },
    },
    "medicine_1k": {
        "id": "medicine_1k",
        "label": "Medicine glossary 1k",
        "path": str(GLOSSARY_RUNTIME_DIR / "medicine_glossary_gt_plus_medicine_wiki_gs1000_translated.json"),
        "domain": "medicine",
        "index_path": str(INDEX_CACHE_DIR / "maxsim_medicine_gt_plus_wiki_gs1000_hn1024_tr128_ta256.pt"),
    },
    "medicine_10k": {
        "id": "medicine_10k",
        "label": "Medicine glossary 10k",
        "path": str(MEDICINE_10K_GLOSSARY),
        "domain": "medicine",
        "index_path": (
            "/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/medicine_gs10k_pr_sweep/"
            "maxsim_medicine_glossary_gt_plus_medicine_wiki_gs10000__0d6ee2097a706d9c_tr128_ta256.pt"
        ),
    },
}


@dataclass
class StreamState:
    session_id: str
    language_pair: str
    source_lang: str
    target_lang: str
    lang_code: str
    samplerate: int = TARGET_SAMPLE_RATE
    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    cursor_samples: int = 0
    last_llm_samples: int = 0
    segment_idx: int = 0
    inflight: bool = False
    messages: List[Dict[str, Any]] = field(default_factory=list)
    history: List[str] = field(default_factory=list)
    audio_paths: List[Path] = field(default_factory=list)
    imported_glossary: List[Dict[str, Any]] = field(default_factory=list)
    glossary_preset: str = DEFAULT_GLOSSARY_PRESET
    glossary_index_path: str = ""
    pending_since_s: Optional[float] = None


def _add_rasst_paths() -> None:
    for path in (RASST_EVAL_ROOT, RASST_CODE_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _format_term_map(references: Sequence[Dict[str, Any]], mode: str) -> str:
    lines: List[str] = []
    for ref in references:
        term = str(ref.get("term") or "").replace("\n", " ").strip()
        translation = str(ref.get("translation") or "").replace("\n", " ").strip()
        if not term or not translation:
            continue
        if mode == "xml_tagged":
            lines.append(f"<term>{term} => {translation}</term>")
        elif mode == "tagged":
            lines.append(f"[TERM] {term} => {translation} [/TERM]")
        else:
            lines.append(f"{term}={translation}")
    return "\n".join(lines)


def _translation_for_lang(item: Dict[str, Any], lang_code: str) -> str:
    translations = item.get("target_translations")
    if isinstance(translations, dict):
        value = translations.get(lang_code)
        if value:
            return str(value)
    for key in ("translation", "target", "value"):
        value = item.get(key)
        if value:
            return str(value)
    return ""


def _normalize_reference(term: str, translation: str, source: str) -> Optional[Dict[str, Any]]:
    clean_term = term.replace("\n", " ").strip()
    clean_translation = translation.replace("\n", " ").strip()
    if not clean_term or not clean_translation:
        return None
    return {"term": clean_term, "translation": clean_translation, "source": source}


def _parse_glossary_text(text: str) -> List[Dict[str, Any]]:
    references: List[Dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=>" in line:
            term, translation = line.split("=>", 1)
        elif "\t" in line:
            term, translation = line.split("\t", 1)
        elif "=" in line:
            term, translation = line.split("=", 1)
        else:
            continue
        ref = _normalize_reference(term, translation, "manual")
        if ref:
            references.append(ref)
    return references


def _merge_references(*groups: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for group in groups:
        for ref in group:
            term = str(ref.get("term") or "").strip().lower()
            if not term or term in seen:
                continue
            translation = str(ref.get("translation") or "").strip()
            if not translation:
                continue
            seen.add(term)
            merged.append(ref)
    return merged


def _use_chinese_training_prompt(source_lang: str, target_lang: str) -> bool:
    return source_lang.strip().lower() in {"english", "en"} and target_lang.strip().lower() in {
        "chinese",
        "zh",
        "zh-cn",
        "中文",
    }


def _build_system_prompt(
    source_lang: str,
    target_lang: str,
    system_prompt_style: str,
    rag_enabled: bool,
) -> str:
    if _use_chinese_training_prompt(source_lang, target_lang):
        return (
            "You are a professional simultaneous interpreter. "
            "Your task is to translate English audio chunks into accurate and fluent "
            "Chinese. Use the ‘term_map’ as a reference for terminology if provided."
        )
    if system_prompt_style == "given_chunks":
        system_text = (
            f"You are a professional simultaneous interpreter. "
            f"You will be given chunks of {source_lang} audio and you need to "
            f"translate the audio into {target_lang} text."
        )
    elif system_prompt_style == "translate_task":
        system_text = (
            f"You are a professional simultaneous interpreter. "
            f"Your task is to translate {source_lang} audio chunks into accurate and fluent "
            f"{target_lang}."
        )
    else:
        raise ValueError(f"Unsupported system_prompt_style={system_prompt_style!r}")
    if rag_enabled:
        system_text += " Use the 'term_map' as a reference for terminology if provided."
    return system_text


def _append_system_if_needed(
    state: StreamState,
    system_prompt_style: str,
    rag_enabled: bool,
) -> None:
    if state.messages:
        return
    state.messages.append(
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": _build_system_prompt(
                        state.source_lang,
                        state.target_lang,
                        system_prompt_style,
                        rag_enabled,
                    ),
                }
            ],
        }
    )


def _trim_messages(state: StreamState, max_cache_chunks: int, keep_cache_chunks: int) -> None:
    if len(state.messages) >= 2 * max_cache_chunks + 1:
        state.messages = [state.messages[0]] + state.messages[-2 * keep_cache_chunks :]


class RasstSglangRuntime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lang_cfg = LANGUAGE_PAIRS[args.language_pair]
        self.states: Dict[str, StreamState] = {}
        self.session_queues: Dict[str, asyncio.Queue] = {}
        self.pending: Deque[str] = deque()
        self.pending_set: Set[str] = set()
        self.scheduler_task: Optional[asyncio.Task] = None
        self.http: Optional[aiohttp.ClientSession] = None
        self.retriever: Any = None
        self.rag_status: Dict[str, Any] = {"status": "disabled"}
        self.glossary_cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        self.glossary_count_cache: Dict[str, int] = {}
        self.text_index_cache: Dict[str, Dict[str, Any]] = {}
        self.active_text_index_path = ""
        self.retriever_lock = asyncio.Lock()
        self.batch_seq = 0
        self.recent_batch_metrics: Deque[Dict[str, Any]] = deque(maxlen=64)
        self.tmp_dir = Path(args.tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        timeout = aiohttp.ClientTimeout(total=float(self.args.sglang_timeout_sec))
        self.http = aiohttp.ClientSession(timeout=timeout)
        if bool(self.args.rag_enabled):
            await asyncio.to_thread(self._load_retriever)
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self) -> None:
        if self.scheduler_task:
            self.scheduler_task.cancel()
            await asyncio.gather(self.scheduler_task, return_exceptions=True)
        if self.http:
            await self.http.close()
        for path in self.tmp_dir.glob("*.wav"):
            try:
                path.unlink()
            except OSError:
                pass

    def _load_retriever(self) -> None:
        _add_rasst_paths()
        from agents.streaming_maxsim_retriever import (  # noqa: WPS433
            MAXSIM_STRIDE,
            MAXSIM_WINDOWS,
            StreamingMaxSimRetriever,
        )

        index_path = self._index_path_for_preset(
            RAG_STARTUP_GLOSSARY_PRESET,
            self.lang_cfg["lang_code"],
        )
        self.rag_status = {
            "status": "loading",
            "device": self.args.rag_device,
            "model_path": self.args.rag_model_path,
            "index_path": index_path,
            "glossary_preset": RAG_STARTUP_GLOSSARY_PRESET,
        }
        self.retriever = StreamingMaxSimRetriever(
            model_path=self.args.rag_model_path,
            index_path=index_path,
            device=self.args.rag_device,
            top_k=int(self.args.rag_top_k),
            lora_rank=int(self.args.rag_lora_r),
            text_lora_rank=int(self.args.rag_text_lora_r),
            target_lang=self.lang_cfg["lang_code"],
            window_sec=0.0,
            score_threshold=float(self.args.rag_score_threshold),
            maxsim_windows=MAXSIM_WINDOWS,
            maxsim_stride=MAXSIM_STRIDE,
        )
        self.active_text_index_path = str(Path(index_path))
        self.text_index_cache[self.active_text_index_path] = {
            "text_embs": self.retriever.text_embs,
            "term_list": self.retriever.term_list,
        }
        self.rag_status["active_text_index_path"] = self.active_text_index_path
        self.rag_status["status"] = "ready"

    async def _sglang_health(self) -> Dict[str, Any]:
        if self.http is None:
            return {"status": "starting"}
        try:
            async with self.http.get(f"{self.args.sglang_base_url}/health") as resp:
                data = await resp.json()
                data["http_status"] = resp.status
                return data
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    async def _infinisst_health(self) -> Dict[str, Any]:
        if not self.args.infinisst_base_url:
            return {"status": "disabled"}
        if self.http is None:
            return {"status": "starting"}
        try:
            async with self.http.get(f"{self.args.infinisst_base_url}/health") as resp:
                data = await resp.json()
                data["http_status"] = resp.status
                return data
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def _normalize_preset_id(self, preset_id: Optional[str]) -> str:
        if not preset_id:
            return DEFAULT_GLOSSARY_PRESET
        if preset_id not in GLOSSARY_PRESETS:
            raise HTTPException(status_code=400, detail=f"Unknown glossary preset: {preset_id}")
        return preset_id

    def _index_path_for_preset(self, preset_id: Optional[str], lang_code: str) -> str:
        preset = GLOSSARY_PRESETS[self._normalize_preset_id(preset_id)]
        if preset["id"] == "none":
            return ""
        index_paths = preset.get("index_paths")
        if isinstance(index_paths, dict) and index_paths.get(lang_code):
            return str(index_paths[lang_code])
        index_path = str(preset.get("index_path") or "")
        if index_path:
            return index_path
        return str(self.lang_cfg["index_path"])

    def _glossary_path_for_preset(self, preset_id: Optional[str]) -> str:
        preset = GLOSSARY_PRESETS[self._normalize_preset_id(preset_id)]
        return str(preset.get("path") or "")

    def _count_glossary_rows(self, glossary_path: str) -> int:
        if not glossary_path:
            return 0
        if glossary_path in self.glossary_count_cache:
            return self.glossary_count_cache[glossary_path]
        path = Path(glossary_path)
        if not path.exists():
            self.glossary_count_cache[glossary_path] = 0
            return 0
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        count = len(raw) if isinstance(raw, (dict, list)) else 0
        self.glossary_count_cache[glossary_path] = count
        return count

    def _ensure_text_index(self, index_path: str) -> Dict[str, Any]:
        normalized_path = str(Path(index_path))
        if normalized_path in self.text_index_cache:
            return self.text_index_cache[normalized_path]
        if self.retriever is None:
            raise RuntimeError("RAG retriever is not initialized")
        path = Path(normalized_path)
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"RAG text index not found: {path}")
        import torch  # noqa: WPS433

        index_data = torch.load(str(path), map_location="cpu")
        text_embs = index_data["text_embs"].to(self.retriever.device)
        term_list = index_data["term_list"]
        if text_embs.shape[0] != len(term_list):
            raise RuntimeError(
                f"RAG text index mismatch: text_embs={text_embs.shape[0]} term_list={len(term_list)}"
            )
        self.text_index_cache[normalized_path] = {"text_embs": text_embs, "term_list": term_list}
        return self.text_index_cache[normalized_path]

    def _activate_text_index(self, index_path: str) -> None:
        if self.retriever is None:
            return
        normalized_path = str(Path(index_path))
        if self.active_text_index_path == normalized_path:
            return
        index_data = self._ensure_text_index(normalized_path)
        self.retriever.text_embs = index_data["text_embs"]
        self.retriever.term_list = index_data["term_list"]
        self.active_text_index_path = normalized_path
        self.rag_status["active_text_index_path"] = normalized_path
        self.rag_status["active_terms"] = len(index_data["term_list"])

    def _describe_glossary_selection(
        self,
        preset_id: Optional[str],
        glossary_text: str,
        lang_code: str,
    ) -> Dict[str, Any]:
        normalized_preset = self._normalize_preset_id(preset_id)
        glossary_path = self._glossary_path_for_preset(normalized_preset)
        index_path = self._index_path_for_preset(normalized_preset, lang_code)
        manual_refs = self._resolve_imported_glossary(glossary_text, lang_code)
        return {
            "glossary_preset": normalized_preset,
            "glossary_path": glossary_path,
            "preset_terms": self._count_glossary_rows(glossary_path),
            "manual_terms": len(manual_refs),
            "manual_refs": manual_refs,
            "index_path": index_path,
            "index_ready": (not index_path) or Path(index_path).is_file(),
        }

    async def build_glossary_selection(
        self,
        session_id: Optional[str],
        language_pair: str,
        glossary_preset: Optional[str],
        glossary_text: str,
    ) -> Dict[str, Any]:
        if language_pair != self.args.language_pair:
            raise HTTPException(
                status_code=400,
                detail=f"This process is loaded for {self.args.language_pair}, not {language_pair}",
            )
        selection = self._describe_glossary_selection(
            glossary_preset,
            glossary_text or "",
            self.lang_cfg["lang_code"],
        )
        if bool(self.args.rag_enabled) and selection["index_path"] and not selection["index_ready"]:
            raise HTTPException(
                status_code=400,
                detail=f"RAG text index is not ready: {selection['index_path']}",
            )
        if self.retriever is not None and selection["index_path"]:
            async with self.retriever_lock:
                await asyncio.to_thread(self._ensure_text_index, selection["index_path"])
        session_updated = False
        if session_id:
            state = self.states.get(session_id)
            if state is None:
                raise HTTPException(status_code=400, detail=f"Unknown RASST session: {session_id}")
            state.glossary_preset = selection["glossary_preset"]
            state.glossary_index_path = selection["index_path"]
            state.imported_glossary = selection["manual_refs"]
            session_updated = True
        return {
            "success": True,
            "session_updated": session_updated,
            "glossary_preset": selection["glossary_preset"],
            "glossary_path": selection["glossary_path"],
            "preset_terms": selection["preset_terms"],
            "manual_terms": selection["manual_terms"],
            "imported_glossary_terms": selection["manual_terms"],
            "index_path": selection["index_path"],
            "index_ready": selection["index_ready"],
        }

    def _batch_metric_summary(self) -> Dict[str, Any]:
        metrics = list(self.recent_batch_metrics)
        if not metrics:
            return {"count": 0, "recent": []}
        recent = metrics[-10:]
        return {
            "count": len(metrics),
            "recent": recent,
            "avg_total_s": round(sum(item["total_s"] for item in metrics) / len(metrics), 4),
            "avg_retrieve_s": round(sum(item["retrieve_s"] for item in metrics) / len(metrics), 4),
            "avg_generate_s": round(sum(item["generate_s"] for item in metrics) / len(metrics), 4),
            "max_total_s": round(max(item["total_s"] for item in metrics), 4),
            "max_queue_wait_s": round(max(item["queue_wait_max_s"] for item in metrics), 4),
        }

    async def health(self) -> Dict[str, Any]:
        sglang = await self._sglang_health()
        infinisst = await self._infinisst_health()
        sglang_ok = sglang.get("status") == "healthy"
        rag_ok = (not bool(self.args.rag_enabled)) or self.rag_status.get("status") == "ready"
        status = "healthy" if sglang_ok and rag_ok else "starting"
        if sglang.get("status") == "error" or self.rag_status.get("status") == "error":
            status = "error"
        return {
            "status": status,
            "backend": "rasst_qwen3_omni_sglang_tp2_maxsim_rag",
            "model": "RASST",
            "language_pair": self.args.language_pair,
            "supported_languages": list(LANGUAGE_PAIRS.keys()),
            "loaded_language_pair": self.args.language_pair,
            "active_sessions": len(self.states),
            "mock_mode": False,
            "tp_size": 2,
            "sglang_base_url": self.args.sglang_base_url,
            "sglang": sglang,
            "infinisst": infinisst,
            "infinisst_base_url": self.args.infinisst_base_url or None,
            "rag": self.rag_status,
            "rag_device": self.args.rag_device,
            "active_text_index_path": self.active_text_index_path or None,
            "cached_text_indexes": len(self.text_index_cache),
            "scheduler_batch_size": self.args.scheduler_batch_size,
            "segment_sec": self.args.segment_sec,
            "batch_metrics": self._batch_metric_summary(),
        }

    def _resolve_imported_glossary(
        self,
        glossary_text: str,
        lang_code: str,
    ) -> List[Dict[str, Any]]:
        manual_refs = _parse_glossary_text(glossary_text or "")
        merged = _merge_references(manual_refs)
        max_terms = max(0, int(self.args.max_imported_glossary_terms))
        if max_terms:
            merged = merged[:max_terms]
        return merged

    async def proxy_init_session(
        self,
        agent_type: str,
        language_pair: str,
        client_id: Optional[str],
        latency_multiplier: int,
    ) -> Dict[str, Any]:
        if not self.args.infinisst_base_url:
            raise HTTPException(status_code=400, detail="InfiniSST backend is not configured")
        if self.http is None:
            raise HTTPException(status_code=503, detail="HTTP client is not initialized")
        params = {
            "agent_type": agent_type,
            "language_pair": language_pair,
            "latency_multiplier": latency_multiplier,
        }
        if client_id is not None:
            params["client_id"] = client_id
        async with self.http.post(f"{self.args.infinisst_base_url}/init", params=params) as resp:
            data = await resp.json()
            if resp.status >= 400:
                raise HTTPException(status_code=resp.status, detail=data)
            return data

    async def proxy_post(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.args.infinisst_base_url:
            return {"success": False, "error": "InfiniSST backend is not configured"}
        if self.http is None:
            return {"success": False, "error": "HTTP client is not initialized"}
        async with self.http.post(f"{self.args.infinisst_base_url}{path}", params=params) as resp:
            data = await resp.json()
            if resp.status >= 400:
                data.setdefault("success", False)
                data.setdefault("http_status", resp.status)
            return data

    def init_session(
        self,
        agent_type: str,
        language_pair: str,
        client_id: Optional[str],
        latency_multiplier: int,
        glossary_preset: str = DEFAULT_GLOSSARY_PRESET,
        glossary_text: str = "",
    ) -> str:
        if agent_type != "RASST":
            raise HTTPException(status_code=400, detail="This server only serves the RASST model")
        if language_pair != self.args.language_pair:
            raise HTTPException(
                status_code=400,
                detail=f"This process is loaded for {self.args.language_pair}, not {language_pair}",
            )
        timestamp = int(time.time() * 1000)
        suffix = client_id or str(timestamp)
        safe_pair = language_pair.replace(" ", "").replace("->", "2")
        session_id = f"RASST_{safe_pair}_{suffix}_{timestamp}"
        selection = self._describe_glossary_selection(
            glossary_preset,
            glossary_text,
            self.lang_cfg["lang_code"],
        )
        if bool(self.args.rag_enabled) and selection["index_path"] and not selection["index_ready"]:
            raise HTTPException(
                status_code=400,
                detail=f"RAG text index is not ready: {selection['index_path']}",
            )
        state = StreamState(
            session_id=session_id,
            language_pair=language_pair,
            source_lang=self.lang_cfg["source_lang"],
            target_lang=self.lang_cfg["target_lang"],
            lang_code=self.lang_cfg["lang_code"],
            imported_glossary=selection["manual_refs"],
            glossary_preset=selection["glossary_preset"],
            glossary_index_path=selection["index_path"],
        )
        self.states[session_id] = state
        self.session_queues[session_id] = asyncio.Queue()
        return session_id

    def delete_session(self, session_id: str) -> bool:
        removed = self.states.pop(session_id, None)
        if removed is not None:
            self._cleanup_session_audio(removed)
        self.session_queues.pop(session_id, None)
        self.pending_set.discard(session_id)
        return removed is not None

    def _cleanup_session_audio(self, state: StreamState) -> None:
        for path in state.audio_paths:
            try:
                path.unlink()
            except OSError:
                pass
        state.audio_paths.clear()

    def reset_session(self, session_id: str) -> bool:
        state = self.states.get(session_id)
        if state is None:
            return False
        self._cleanup_session_audio(state)
        state.audio = np.zeros(0, dtype=np.float32)
        state.cursor_samples = 0
        state.last_llm_samples = 0
        state.segment_idx = 0
        state.inflight = False
        state.messages.clear()
        state.history.clear()
        state.pending_since_s = None
        self.pending_set.discard(session_id)
        self.pending = deque(item for item in self.pending if item != session_id)
        queue_for_session = self.session_queues.get(session_id)
        if queue_for_session is not None:
            while not queue_for_session.empty():
                try:
                    queue_for_session.get_nowait()
                except asyncio.QueueEmpty:
                    break
        return True

    def submit_audio(self, session_id: str, audio: np.ndarray) -> None:
        state = self.states[session_id]
        chunk = np.asarray(audio, dtype=np.float32).flatten()
        if chunk.size == 0:
            return
        state.audio = np.concatenate([state.audio, chunk])
        state.cursor_samples = int(state.audio.shape[0])
        self._mark_pending_if_ready(state)

    def _mark_pending_if_ready(self, state: StreamState) -> None:
        segment_samples = int(float(self.args.segment_sec) * TARGET_SAMPLE_RATE)
        if state.inflight:
            return
        if state.cursor_samples - state.last_llm_samples < segment_samples:
            return
        if state.session_id not in self.pending_set:
            self.pending.append(state.session_id)
            self.pending_set.add(state.session_id)
            state.pending_since_s = time.perf_counter()

    async def _scheduler_loop(self) -> None:
        while True:
            await asyncio.sleep(float(self.args.batch_timeout))
            batch: List[StreamState] = []
            while self.pending and len(batch) < int(self.args.scheduler_batch_size):
                session_id = self.pending.popleft()
                self.pending_set.discard(session_id)
                state = self.states.get(session_id)
                if state is None or state.inflight:
                    continue
                segment_samples = int(float(self.args.segment_sec) * TARGET_SAMPLE_RATE)
                if state.cursor_samples - state.last_llm_samples < segment_samples:
                    continue
                state.inflight = True
                batch.append(state)
            if batch:
                asyncio.create_task(self._process_batch(batch))

    async def _process_batch(self, batch: List[StreamState]) -> None:
        batch_t0 = time.perf_counter()
        self.batch_seq += 1
        batch_id = self.batch_seq
        queue_wait_values = [
            batch_t0 - state.pending_since_s
            for state in batch
            if state.pending_since_s is not None
        ]
        for state in batch:
            state.pending_since_s = None
        end_by_session = {state.session_id: state.cursor_samples for state in batch}
        start_by_session = {state.session_id: state.last_llm_samples for state in batch}
        increments = [
            np.asarray(state.audio[start_by_session[state.session_id] : end_by_session[state.session_id]], dtype=np.float32)
            for state in batch
        ]
        refs_by_state: List[List[Dict[str, Any]]] = [[] for _ in batch]
        results: List[Dict[str, Any]] = []
        retrieve_s = 0.0
        generate_s = 0.0
        batch_error: Optional[str] = None
        try:
            retrieve_t0 = time.perf_counter()
            refs_by_state = await self._retrieve_batch(batch, end_by_session)
            retrieve_s = time.perf_counter() - retrieve_t0
            generate_t0 = time.perf_counter()
            tasks = [
                self._generate_one(state, increment, refs, start_by_session[state.session_id], end_by_session[state.session_id])
                for state, increment, refs in zip(batch, increments, refs_by_state)
            ]
            results = await asyncio.gather(*tasks)
            generate_s = time.perf_counter() - generate_t0
        except Exception as exc:
            batch_error = str(exc)
            for state in batch:
                queue_for_session = self.session_queues.get(state.session_id)
                if queue_for_session:
                    await queue_for_session.put(
                        {
                            "type": "translation_error",
                            "session_id": state.session_id,
                            "error": batch_error,
                        }
                    )
        finally:
            for state in batch:
                if state.session_id in self.states:
                    state.inflight = False
                    self._mark_pending_if_ready(state)
        request_elapsed_values = [
            float(item["elapsed_s"])
            for item in results
            if item.get("elapsed_s") is not None
        ]
        metric = {
            "batch_id": batch_id,
            "batch_size": len(batch),
            "retrieve_s": round(retrieve_s, 4),
            "generate_s": round(generate_s, 4),
            "total_s": round(time.perf_counter() - batch_t0, 4),
            "queue_wait_avg_s": round(sum(queue_wait_values) / len(queue_wait_values), 4) if queue_wait_values else 0.0,
            "queue_wait_max_s": round(max(queue_wait_values), 4) if queue_wait_values else 0.0,
            "audio_increment_s_avg": round(
                sum(len(item) / TARGET_SAMPLE_RATE for item in increments) / len(increments),
                4,
            ) if increments else 0.0,
            "references_avg": round(sum(len(item) for item in refs_by_state) / len(refs_by_state), 4) if refs_by_state else 0.0,
            "generation_ok": sum(1 for item in results if item.get("ok")),
            "request_elapsed_avg_s": round(sum(request_elapsed_values) / len(request_elapsed_values), 4)
            if request_elapsed_values else 0.0,
            "request_elapsed_max_s": round(max(request_elapsed_values), 4) if request_elapsed_values else 0.0,
        }
        if batch_error:
            metric["error"] = batch_error[:500]
        self.recent_batch_metrics.append(metric)
        print("RASST_BATCH_METRIC " + json.dumps(metric, ensure_ascii=False), flush=True)

    async def _retrieve_batch(
        self,
        batch: Sequence[StreamState],
        end_by_session: Dict[str, int],
    ) -> List[List[Dict[str, Any]]]:
        if self.retriever is None:
            return [[] for _ in batch]
        outputs: List[List[Dict[str, Any]]] = [[] for _ in batch]
        grouped: Dict[str, List[Tuple[int, StreamState]]] = {}
        for idx, state in enumerate(batch):
            index_path = state.glossary_index_path or self._index_path_for_preset(
                state.glossary_preset,
                state.lang_code,
            )
            if not index_path:
                continue
            grouped.setdefault(index_path, []).append((idx, state))

        async with self.retriever_lock:
            for index_path, indexed_states in grouped.items():
                await asyncio.to_thread(self._activate_text_index, index_path)
                requests = [
                    {
                        "audio_buffer": state.audio[: end_by_session[state.session_id]],
                        "current_start_sec": float(state.last_llm_samples) / TARGET_SAMPLE_RATE,
                        "current_end_sec": float(end_by_session[state.session_id]) / TARGET_SAMPLE_RATE,
                        "lookback_sec": float(self.args.rag_timeline_lookback_sec),
                    }
                    for _, state in indexed_states
                ]
                group_results = await asyncio.to_thread(
                    self.retriever.retrieve_timeline_batch,
                    requests,
                    int(self.args.rag_top_k),
                    float(self.args.rag_timeline_lookback_sec),
                )
                for (original_idx, _), refs in zip(indexed_states, group_results):
                    outputs[original_idx] = refs
        return outputs

    async def _generate_one(
        self,
        state: StreamState,
        increment: np.ndarray,
        references: Sequence[Dict[str, Any]],
        start_sample: int,
        end_sample: int,
    ) -> Dict[str, Any]:
        chunk_rms = float(np.sqrt(np.mean(np.square(increment)))) if increment.size else 0.0
        if chunk_rms < float(self.args.min_audio_rms):
            state.last_llm_samples = end_sample
            state.segment_idx += 1
            print(
                "RASST_SKIPPED_SILENCE "
                + json.dumps(
                    {
                        "session_id": state.session_id,
                        "segment_idx": state.segment_idx,
                        "rms": round(chunk_rms, 6),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return {"ok": True, "elapsed_s": 0.0, "skipped": "silence"}
        wav_path = self.tmp_dir / f"{state.session_id}_{state.segment_idx + 1:05d}.wav"
        sf.write(str(wav_path), increment, TARGET_SAMPLE_RATE)
        state.audio_paths.append(wav_path)
        rag_enabled_for_prompt = bool(
            self.args.rag_enabled
            and (state.glossary_index_path or state.imported_glossary)
        )
        _append_system_if_needed(
            state,
            system_prompt_style=self.args.system_prompt_style,
            rag_enabled=rag_enabled_for_prompt,
        )
        merged_references = _merge_references(state.imported_glossary, references)
        user_content: List[Dict[str, Any]] = [{"type": "audio", "audio": str(wav_path)}]
        term_map = _format_term_map(merged_references, self.args.term_map_format)
        if term_map:
            user_content.append({"type": "text", "text": f"\n\nterm_map:\n{term_map}"})
        elif rag_enabled_for_prompt and self.args.empty_term_map_policy == "none_block":
            user_content.append({"type": "text", "text": "\n\nterm_map:\nNONE"})
        user_message = {"role": "user", "content": user_content}
        state.messages.append(user_message)
        payload = {
            "model": "rasst-qwen3-omni",
            "request_id": f"{state.session_id}-{state.segment_idx + 1}",
            "messages": state.messages,
            "modalities": ["text"],
            "max_tokens": int(self.args.max_new_tokens),
            "temperature": float(self.args.temperature),
            "top_p": float(self.args.top_p),
            "top_k": int(self.args.top_k),
            "seed": int(self.args.seed),
            "stream": False,
        }
        try:
            if self.http is None:
                raise RuntimeError("HTTP client is not initialized")
            t0 = time.perf_counter()
            async with self.http.post(
                f"{self.args.sglang_base_url}/v1/chat/completions",
                json=payload,
            ) as resp:
                data = await resp.json()
                if resp.status >= 400:
                    raise RuntimeError(f"SGLang status={resp.status} body={data}")
            elapsed = time.perf_counter() - t0
            text = str(data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
            state.last_llm_samples = end_sample
            state.segment_idx += 1
            state.messages.append({"role": "assistant", "content": text})
            _trim_messages(
                state,
                max_cache_chunks=int(self.args.max_cache_chunks),
                keep_cache_chunks=int(self.args.keep_cache_chunks),
            )
            if text:
                state.history.append(text)
                state.history = state.history[-int(self.args.keep_cache_chunks) :]
                print(
                    "RASST_TRANSLATION "
                    + json.dumps(
                        {
                            "session_id": state.session_id,
                            "segment_idx": state.segment_idx,
                            "text": text[:200],
                            "elapsed_s": round(elapsed, 4),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            queue_for_session = self.session_queues.get(state.session_id)
            if queue_for_session and text:
                await queue_for_session.put(
                    {
                        "type": "translation",
                        "session_id": state.session_id,
                        "text": text,
                        "segment_idx": state.segment_idx,
                        "cursor_samples": end_sample,
                        "start_sample": start_sample,
                        "elapsed_s": round(elapsed, 6),
                        "batch_size": 1,
                    }
                )
            return {"ok": True, "elapsed_s": elapsed}
        except Exception as exc:
            if state.messages and state.messages[-1] is user_message:
                state.messages.pop()
            queue_for_session = self.session_queues.get(state.session_id)
            if queue_for_session:
                await queue_for_session.put(
                    {
                        "type": "translation_error",
                        "session_id": state.session_id,
                        "error": str(exc),
                    }
                )
            return {"ok": False, "error": str(exc), "elapsed_s": None}


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
runtime: Optional[RasstSglangRuntime] = None


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        raise exc
    print(f"RASST SGLang server exception: {exc}", flush=True)
    return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})


@app.on_event("startup")
async def startup_event():
    if runtime is None:
        raise RuntimeError("runtime was not configured")
    await runtime.start()


@app.on_event("shutdown")
async def shutdown_event():
    if runtime is not None:
        await runtime.stop()


@app.get("/config")
async def get_config():
    loaded_lang_code = runtime.lang_cfg["lang_code"] if runtime else "zh"
    return {
        "models": MODEL_PROFILES,
        "language_pairs": [
            {"id": key, "label": key.replace("->", "->"), "available": runtime.args.language_pair == key}
            for key in LANGUAGE_PAIRS
        ],
        "glossary_presets": [
            {
                "id": preset["id"],
                "label": preset["label"],
                "domain": preset["domain"],
                "preset_terms": runtime._count_glossary_rows(str(preset.get("path") or "")) if runtime else 0,
                "index_path": runtime._index_path_for_preset(preset["id"], loaded_lang_code) if runtime else "",
                "available": (
                    preset["id"] == "none"
                    or Path(str(preset.get("path") or "")).exists()
                    and (
                        runtime is None
                        or Path(runtime._index_path_for_preset(preset["id"], loaded_lang_code)).is_file()
                    )
                ),
            }
            for preset in GLOSSARY_PRESETS.values()
        ],
        "default_model": "RASST",
        "default_glossary_preset": DEFAULT_GLOSSARY_PRESET,
        "loaded_language_pair": runtime.args.language_pair if runtime else None,
    }


@app.get("/health")
async def health_check():
    if runtime is None:
        return {"status": "error", "error": "runtime not configured"}
    return await runtime.health()


@app.post("/init")
async def initialize_translation(
    request: Request,
    agent_type: Optional[str] = None,
    language_pair: Optional[str] = None,
    latency_multiplier: int = 2,
    client_id: Optional[str] = None,
    glossary_preset: str = DEFAULT_GLOSSARY_PRESET,
    glossary_text: str = "",
):
    if runtime is None:
        raise HTTPException(status_code=503, detail="runtime not configured")
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            body = {}
        agent_type = body.get("agent_type", agent_type)
        language_pair = body.get("language_pair", language_pair)
        latency_multiplier = int(body.get("latency_multiplier", latency_multiplier))
        client_id = body.get("client_id", client_id)
        glossary_preset = body.get("glossary_preset", glossary_preset)
        glossary_text = body.get("glossary_text", glossary_text)
    if not agent_type or not language_pair:
        raise HTTPException(status_code=400, detail="agent_type and language_pair are required")
    health = await runtime.health()
    if health.get("status") != "healthy":
        raise HTTPException(status_code=503, detail=f"runtime not healthy: {health}")
    if agent_type != "RASST":
        return await runtime.proxy_init_session(agent_type, language_pair, client_id, latency_multiplier)
    session_id = runtime.init_session(
        agent_type,
        language_pair,
        client_id,
        latency_multiplier,
        glossary_preset,
        glossary_text,
    )
    state = runtime.states[session_id]
    preset_path = runtime._glossary_path_for_preset(state.glossary_preset)
    return {
        "session_id": session_id,
        "queued": False,
        "queue_position": 0,
        "scheduler_based": False,
        "rasst_backend": True,
        "sglang_backend": True,
        "glossary_preset": state.glossary_preset,
        "glossary_path": preset_path,
        "preset_terms": runtime._count_glossary_rows(preset_path),
        "manual_terms": len(state.imported_glossary),
        "glossary_terms": len(state.imported_glossary),
        "index_path": state.glossary_index_path,
        "index_ready": (not state.glossary_index_path) or Path(state.glossary_index_path).is_file(),
    }


@app.post("/glossary/build")
async def build_glossary(request: Request):
    if runtime is None:
        raise HTTPException(status_code=503, detail="runtime not configured")
    try:
        body = await request.json()
    except Exception:
        body = {}
    language_pair = body.get("language_pair") or runtime.args.language_pair
    return await runtime.build_glossary_selection(
        session_id=body.get("session_id"),
        language_pair=language_pair,
        glossary_preset=body.get("glossary_preset", DEFAULT_GLOSSARY_PRESET),
        glossary_text=body.get("glossary_text", ""),
    )


@app.post("/delete_session")
async def delete_session(session_id: str):
    if runtime is None:
        return {"success": False, "error": "runtime not configured"}
    if session_id not in runtime.states and not session_id.startswith("RASST_"):
        return await runtime.proxy_post("/delete_session", {"session_id": session_id})
    return {"success": runtime.delete_session(session_id)}


@app.post("/ping")
async def ping_session(session_id: str):
    if runtime is None:
        return {"success": False, "error": "runtime not configured"}
    if session_id not in runtime.states and not session_id.startswith("RASST_"):
        return await runtime.proxy_post("/ping", {"session_id": session_id})
    if session_id not in runtime.states:
        return {"success": False, "error": "Invalid session ID"}
    return {"success": True}


@app.post("/reset_translation")
async def reset_translation(session_id: str):
    if runtime is None:
        return {"success": False, "error": "runtime not configured"}
    if session_id not in runtime.states and not session_id.startswith("RASST_"):
        return await runtime.proxy_post("/reset_translation", {"session_id": session_id})
    if session_id not in runtime.states:
        return {"success": False, "error": "Invalid session ID"}
    return {
        "success": runtime.reset_session(session_id),
        "message": "Translation reset successfully.",
        "session_type": "rasst",
    }


@app.post("/download_youtube")
async def download_youtube(request: Request):
    if runtime is None:
        raise HTTPException(status_code=503, detail="runtime not configured")
    if not runtime.args.infinisst_base_url:
        raise HTTPException(status_code=400, detail="InfiniSST backend is not configured for YouTube downloads")
    if runtime.http is None:
        raise HTTPException(status_code=503, detail="HTTP client is not initialized")

    url = request.query_params.get("url")
    session_id = request.query_params.get("session_id")
    if not url:
        raise HTTPException(status_code=400, detail="Missing URL parameter")

    params = {"url": url}
    if session_id:
        params["session_id"] = session_id

    backend_resp = await runtime.http.post(
        f"{runtime.args.infinisst_base_url}/download_youtube",
        params=params,
    )
    content_type = backend_resp.headers.get("content-type", "")
    if backend_resp.status >= 400 or "application/json" in content_type:
        body = await backend_resp.text()
        backend_resp.release()
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = {"error": body}
        data.setdefault("success", False)
        data.setdefault("http_status", backend_resp.status)
        return JSONResponse(
            status_code=backend_resp.status if backend_resp.status >= 400 else 502,
            content=data,
        )

    headers = {}
    for header_name in ("content-length", "content-disposition", "accept-ranges"):
        if header_name in backend_resp.headers:
            headers[header_name] = backend_resp.headers[header_name]

    async def stream_backend_response():
        try:
            async for chunk in backend_resp.content.iter_chunked(1024 * 1024):
                yield chunk
        finally:
            backend_resp.release()

    return StreamingResponse(
        stream_backend_response(),
        status_code=backend_resp.status,
        media_type=content_type or "video/mp4",
        headers=headers,
    )


@app.websocket("/wss/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if runtime is None:
        await websocket.close(code=4000, reason="Invalid session ID")
        return
    if session_id not in runtime.states and not session_id.startswith("RASST_"):
        await proxy_infinisst_websocket(websocket, session_id)
        return
    if session_id not in runtime.states:
        await websocket.close(code=4000, reason="Invalid session ID")
        return
    result_queue = runtime.session_queues[session_id]
    await websocket.send_text("READY: RASST SGLang workers ready")

    async def sender():
        while True:
            item = await result_queue.get()
            if item.get("type") == "translation_error":
                await websocket.send_text(f"ERROR: {item.get('error')}")
            else:
                await websocket.send_text(str(item.get("text", "")))

    sender_task = asyncio.create_task(sender())
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message:
                audio = np.frombuffer(message["bytes"], dtype=np.float32)
                runtime.submit_audio(session_id, audio)
            elif "text" in message and message["text"] == "EOF":
                await websocket.send_text("PROCESSING_COMPLETE: File processing finished")
    except Exception:
        pass
    finally:
        sender_task.cancel()


async def proxy_infinisst_websocket(websocket: WebSocket, session_id: str) -> None:
    if runtime is None or not runtime.args.infinisst_base_url:
        await websocket.close(code=4000, reason="InfiniSST backend is not configured")
        return
    backend_base = runtime.args.infinisst_base_url.replace("http://", "ws://").replace("https://", "wss://")
    backend_uri = f"{backend_base}/wss/{session_id}"
    try:
        if runtime.http is None:
            raise RuntimeError("HTTP client is not initialized")
        async with runtime.http.ws_connect(backend_uri, max_msg_size=0) as backend_ws:
            async def client_to_backend():
                while True:
                    message = await websocket.receive()
                    if "bytes" in message:
                        await backend_ws.send_bytes(message["bytes"])
                    elif "text" in message:
                        await backend_ws.send_str(message["text"])
                    else:
                        break

            async def backend_to_client():
                async for message in backend_ws:
                    if message.type == aiohttp.WSMsgType.TEXT:
                        await websocket.send_text(str(message.data))
                    elif message.type == aiohttp.WSMsgType.BINARY:
                        await websocket.send_bytes(message.data)
                    elif message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break

            tasks = [
                asyncio.create_task(client_to_backend()),
                asyncio.create_task(backend_to_client()),
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                exc = task.exception()
                if exc is not None:
                    raise exc
    except Exception as exc:
        try:
            await websocket.send_text(f"ERROR: InfiniSST proxy failed: {exc}")
        except Exception:
            pass
        await websocket.close(code=4001, reason="InfiniSST proxy failed")


@app.get("/")
async def read_index():
    return FileResponse(STATIC_DIR / "index.html")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RASST SGLang-Omni demo wrapper")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--sglang-base-url", default=os.environ.get("RASST_SGLANG_BASE_URL", "http://127.0.0.1:8100"))
    parser.add_argument("--infinisst-base-url", default=os.environ.get("INFINISST_BASE_URL", ""))
    parser.add_argument("--sglang-timeout-sec", type=float, default=float(os.environ.get("RASST_SGLANG_TIMEOUT_SEC", "900")))
    parser.add_argument("--language-pair", default=os.environ.get("RASST_DEMO_LANGUAGE_PAIR", "English -> Chinese"))
    parser.add_argument(
        "--rag-model-path",
        default=os.environ.get("RASST_HN1024_RETRIEVER", str(PROJECT_ROOT / "checkpoints/retriever/rasst-hn1024.pt")),
    )
    parser.add_argument("--rag-enabled", type=int, default=int(os.environ.get("RASST_RAG_ENABLED", "1")))
    parser.add_argument("--rag-device", default=os.environ.get("RASST_RAG_DEVICE", "cuda:1"))
    parser.add_argument("--rag-top-k", type=int, default=int(os.environ.get("RASST_RAG_TOP_K", "10")))
    parser.add_argument("--rag-score-threshold", type=float, default=float(os.environ.get("RASST_RAG_SCORE_THRESHOLD", "0.78")))
    parser.add_argument("--rag-lora-r", type=int, default=int(os.environ.get("RASST_RAG_LORA_R", "128")))
    parser.add_argument("--rag-text-lora-r", type=int, default=int(os.environ.get("RASST_RAG_TEXT_LORA_R", "128")))
    parser.add_argument("--rag-timeline-lookback-sec", type=float, default=float(os.environ.get("RASST_RAG_LOOKBACK_SEC", str(DEFAULT_RAG_LOOKBACK_SEC))))
    parser.add_argument("--segment-sec", type=float, default=float(os.environ.get("RASST_SGLANG_SEGMENT_SEC", str(DEFAULT_SEGMENT_SEC))))
    parser.add_argument("--scheduler-batch-size", type=int, default=int(os.environ.get("RASST_SCHEDULER_BATCH_SIZE", "32")))
    parser.add_argument("--batch-timeout", type=float, default=float(os.environ.get("RASST_BATCH_TIMEOUT", "0.05")))
    parser.add_argument("--keep-cache-chunks", type=int, default=int(os.environ.get("RASST_KEEP_CACHE_CHUNKS", str(DEFAULT_KEEP_CACHE_CHUNKS))))
    parser.add_argument("--max-cache-chunks", type=int, default=int(os.environ.get("RASST_MAX_CACHE_CHUNKS", str(DEFAULT_MAX_CACHE_CHUNKS))))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("RASST_MAX_NEW_TOKENS", "40")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("RASST_TEMPERATURE", "0.0")))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("RASST_TOP_P", "0.9")))
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("RASST_TOP_K", "50")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("RASST_SEED", "998244353")))
    parser.add_argument("--term-map-format", default=os.environ.get("RASST_TERM_MAP_FORMAT", "plain"))
    parser.add_argument("--empty-term-map-policy", default=os.environ.get("RASST_EMPTY_TERM_MAP_POLICY", "none_block"))
    parser.add_argument(
        "--system-prompt-style",
        default=os.environ.get("RASST_SYSTEM_PROMPT_STYLE", DEFAULT_SYSTEM_PROMPT_STYLE),
        choices=["translate_task", "given_chunks"],
    )
    parser.add_argument("--max-imported-glossary-terms", type=int, default=int(os.environ.get("RASST_MAX_IMPORTED_GLOSSARY_TERMS", str(DEFAULT_MAX_IMPORTED_GLOSSARY_TERMS))))
    parser.add_argument("--min-audio-rms", type=float, default=float(os.environ.get("RASST_MIN_AUDIO_RMS", "0.001")))
    parser.add_argument("--tmp-dir", default=os.environ.get("RASST_TMP_DIR", f"/dev/shm/rasst_sglang_{os.getpid()}"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    global runtime
    args = parse_args(argv)
    if args.language_pair not in LANGUAGE_PAIRS:
        raise SystemExit(f"Unsupported language pair: {args.language_pair}")
    if bool(args.rag_enabled) and not Path(args.rag_model_path).is_file():
        raise SystemExit(f"RAG checkpoint not found: {args.rag_model_path}")
    lang_code = LANGUAGE_PAIRS[args.language_pair]["lang_code"]
    default_index = GLOSSARY_PRESETS[RAG_STARTUP_GLOSSARY_PRESET]["index_paths"][lang_code]
    index_path = Path(default_index)
    if bool(args.rag_enabled) and not index_path.is_file():
        raise SystemExit(f"RAG index not found: {index_path}")
    runtime = RasstSglangRuntime(args)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
