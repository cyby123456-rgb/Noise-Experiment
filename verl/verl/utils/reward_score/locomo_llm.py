# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import os
import re
import threading
from contextvars import ContextVar
from typing import Optional

from tenacity import AsyncRetrying, RetryCallState, stop_after_attempt
from tenacity import wait_exponential, wait_fixed

try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - defer error until used
    AsyncOpenAI = None


FALLBACK_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "Qwen/Qwen3-14B")
DEFAULT_EVALUATOR_BASE_URL = os.getenv("OPENAI_BASE_URL") or FALLBACK_BASE_URL
DEFAULT_EVALUATOR_API_KEY = os.getenv("OPENAI_API_KEY")

_evaluator_model = DEFAULT_EVALUATOR_MODEL
_evaluator_base_url = DEFAULT_EVALUATOR_BASE_URL
_evaluator_api_key = DEFAULT_EVALUATOR_API_KEY
_client: Optional["AsyncOpenAI"] = None
_client_lock = threading.Lock()

_event_loop: Optional[asyncio.AbstractEventLoop] = None
_event_loop_thread: Optional[threading.Thread] = None
_event_loop_lock = threading.Lock()
_event_loop_ready = threading.Event()

_retry_details: ContextVar[Optional[str]] = ContextVar("_retry_details", default=None)


ACCURACY_PROMPT = """Your task is to label an answer as "CORRECT" or "WRONG" given:
(1) a question,
(2) a gold (ground truth) answer,
(3) a generated answer.

Core principle -- Inclusion + Non-contradiction
- Be GENEROUS: if the generated answer clearly includes the gold's key content (or a clear paraphrase of the same content) and does not contradict it, mark CORRECT -- even if extra details are added.
- Mark WRONG only when the generated answer does not include the gold's content, changes it, or contradicts it.

TIME (strict granularity; relative form equivalence; no calendar math)
- Granularity must match exactly: HOUR<->HOUR, DAY<->DAY, MONTH<->MONTH, YEAR<->YEAR.
  Do not answer a gold at a different time unit -- even if the numeric value overlaps. Do not answer a month-level gold with a specific day, nor a year with a specific month/day/hour, etc.
  (e.g., gold = "July 26, 2019" [DAY]; generated = "2019-07-26 08:09:17" [includes Second] -> WRONG)
- Do NOT convert relative <-> absolute. If the gold uses a relative time expression, the generated answer must also use a relative form (or a clear paraphrase of that same form), not a computed date/range.
- Treat harmless modifiers in relative forms (e.g., "the/last/previous/just prior") as equivalent when both the anchor date and the time unit are the same.

- Lists of DISTINCT facts:
- If the gold answer lists multiple distinct facts (joined by "and", commas, or slashes), the generated answer must cover all of them.
- Extra non-contradictory items generally count as WRONG.
    - Example: gold = A, B, C ; gen = A, B, C -> CORRECT
    - Example: gold = A, B, C ; gen = A, B, C, D -> WRONG
- Exception: If a gold element is elaborated or split into finer details in the generated answer (e.g., C -> C, C'), it is still considered CORRECT.

Preference/Benefit Questions (e.g., "what X likes/values most")
- If gold lists multiple reasons/aspects, the generated answer only needs to include any one of them without contradiction to be CORRECT.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label":

```json
{{
    "label": "CORRECT" or "WRONG"
}}
```
"""


def _build_client() -> "AsyncOpenAI":
    if AsyncOpenAI is None:
        raise RuntimeError("openai package is required for llm_judge reward function.")
    client_kwargs = {}
    if _evaluator_base_url:
        client_kwargs["base_url"] = _evaluator_base_url
    if _evaluator_api_key:
        client_kwargs["api_key"] = _evaluator_api_key
    return AsyncOpenAI(**client_kwargs)


def configure_evaluator(model=None, base_url=None, api_key=None):
    """Update evaluator settings at runtime; None keeps current value."""
    global _client, _evaluator_model, _evaluator_base_url, _evaluator_api_key
    if _client is not None:
        _close_client()
    if model:
        _evaluator_model = model
    if base_url is not None:
        _evaluator_base_url = base_url or None
    if api_key is not None:
        _evaluator_api_key = api_key or None
    _client = None


def _close_client() -> None:
    global _client
    client = _client
    if not client:
        return
    _client = None
    closer = getattr(client, "aclose", None)
    if not callable(closer):
        return

    loop = _event_loop if _event_loop and _event_loop.is_running() else None
    if loop:
        future = asyncio.run_coroutine_threadsafe(closer(), loop)
        try:
            future.result()
        except Exception as exc:
            print(f"[llm_judge] client close failed: {exc}")
        return

    try:
        asyncio.run(closer())
    except RuntimeError:
        tmp_loop = asyncio.new_event_loop()
        try:
            tmp_loop.run_until_complete(closer())
        finally:
            tmp_loop.close()


def get_client() -> "AsyncOpenAI":
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = _build_client()
    return _client


def _ensure_event_loop() -> asyncio.AbstractEventLoop:
    global _event_loop, _event_loop_thread
    with _event_loop_lock:
        if _event_loop and not _event_loop.is_closed() and _event_loop.is_running():
            return _event_loop

        loop = asyncio.new_event_loop()
        _event_loop_ready.clear()

        def _run_loop() -> None:
            asyncio.set_event_loop(loop)
            _event_loop_ready.set()
            loop.run_forever()

        thread = threading.Thread(target=_run_loop, name="llm-judge-event-loop", daemon=True)
        thread.start()

        _event_loop = loop
        _event_loop_thread = thread

    _event_loop_ready.wait()
    return _event_loop


def shutdown_event_loop() -> None:
    global _event_loop, _event_loop_thread
    with _event_loop_lock:
        loop = _event_loop
        thread = _event_loop_thread
        _event_loop = None
        _event_loop_thread = None
    if not loop:
        return
    if loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
    if thread and thread.is_alive():
        thread.join(timeout=5)
    try:
        loop.close()
    except Exception:
        pass
    _event_loop_ready.clear()


def _log_retry(retry_state: RetryCallState):
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    wait_action = getattr(retry_state.next_action, "sleep", None)
    max_attempts = getattr(getattr(retry_state.retry_object, "stop", None), "max_attempt_number", None)
    attempt = retry_state.attempt_number

    message = f"[llm_judge] retry attempt {attempt}"
    if max_attempts:
        message += f"/{max_attempts}"
    if exception:
        message += f" due to: {exception}"
    if wait_action is not None:
        message += f"; waiting {wait_action:.2f}s before next attempt"
    context = _retry_details.get()
    if context:
        message += f" | context: {context}"
    # print(message, flush=True)


def _custom_wait(retry_state: RetryCallState):
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    if exception and any(x in str(exception).lower() for x in ("tpm", "limit")):
        base = wait_exponential(multiplier=1, min=40, max=80)
        return base(retry_state)
    return wait_fixed(1)(retry_state)


def _extract_json_fallback(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text.strip()


def _extract_json(text: str) -> str:
    try:
        local_mem0_path = os.getenv("LOCAL_MEM0_PATH")
        if local_mem0_path:
            import sys

            if local_mem0_path not in sys.path:
                sys.path.insert(0, local_mem0_path)
            from mem0.memory.utils import extract_json  # type: ignore

            return extract_json(text)
    except Exception:
        pass
    return _extract_json_fallback(text)


def _parse_label(content: str) -> Optional[int]:
    try:
        payload = json.loads(_extract_json(content))
        label = str(payload.get("label", "")).upper()
    except Exception:
        label = ""
    if label == "CORRECT":
        return 1
    if label == "WRONG":
        return 0
    upper = content.upper()
    if "CORRECT" in upper and "WRONG" not in upper:
        return 1
    if "WRONG" in upper and "CORRECT" not in upper:
        return 0
    return None


async def _evaluate_single_gold_async(question: str, gold_answer_str: str, generated_answer: str) -> int:
    context_token = _retry_details.set(
        f"question='{question[:60]}', gold='{gold_answer_str[:40]}', generated='{generated_answer[:40]}'"
    )
    try:
        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(10),
            wait=_custom_wait,
            before_sleep=_log_retry,
        ):
            with attempt:
                response = await get_client().chat.completions.create(
                    model=_evaluator_model or "Qwen/Qwen3-14B",
                    messages=[
                        {
                            "role": "user",
                            "content": ACCURACY_PROMPT.format(
                                question=question,
                                gold_answer=gold_answer_str,
                                generated_answer=generated_answer,
                            ),
                        }
                    ],
                    temperature=0.0,
                )
                content = response.choices[0].message.content
                label = _parse_label(content)
                if label is None:
                    raise RuntimeError(f"invalid evaluator response: {content}")
                return label
    finally:
        _retry_details.reset(context_token)


async def evaluate_llm_judge_async(question, gold_answer, generated_answer) -> int:
    try:
        if isinstance(gold_answer, (list, tuple)):
            tasks = [
                _evaluate_single_gold_async(question, str(gold), generated_answer)
                for gold in gold_answer
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            best = 0
            for outcome in results:
                if isinstance(outcome, Exception):
                    print(f"[llm_judge] single-gold error: {outcome}", flush=True)
                    continue
                value = int(outcome)
                if value == 1:
                    return 1
                best = max(best, value)
            return best

        return await _evaluate_single_gold_async(question, str(gold_answer), generated_answer)
    except Exception as exc:
        print(f"[llm_judge] evaluation failed: {exc}", flush=True)
        return 0


def submit_llm_judge(question, gold_answer, generated_answer):
    loop = _ensure_event_loop()
    try:
        return asyncio.run_coroutine_threadsafe(
            evaluate_llm_judge_async(question, gold_answer, generated_answer),
            loop,
        )
    except RuntimeError:
        loop = _ensure_event_loop()
        return asyncio.run_coroutine_threadsafe(
            evaluate_llm_judge_async(question, gold_answer, generated_answer),
            loop,
        )


def evaluate_llm_judge(question, gold_answer, generated_answer) -> int:
    future = submit_llm_judge(question, gold_answer, generated_answer)
    return future.result()


def _get_question(extra_info, fallback: str) -> str:
    if not extra_info:
        return fallback
    if isinstance(extra_info, dict):
        for key in ("question", "prompt", "problem"):
            value = extra_info.get(key)
            if value:
                return str(value)
    return fallback

def _normalize_ground_truth(ground_truth):
    if not isinstance(ground_truth, dict):
        return ground_truth

    candidates = []
    for field in ("raw", "fixed"):
        value = ground_truth.get(field)
        if hasattr(value, "tolist"):
            value = value.tolist()
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            if item is None:
                continue
            text = str(item)
            if text and text not in candidates:
                candidates.append(text)
    return candidates


def _score_single(solution_str, ground_truth, extra_info) -> int:
    question = _get_question(extra_info, fallback="")
    return evaluate_llm_judge(question, _normalize_ground_truth(ground_truth), solution_str)


def compute_score(
    data_source=None,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    data_sources=None,
    solution_strs=None,
    ground_truths=None,
    extra_infos=None,
    **kwargs,
):
    """Return 1 or 0 from LLM judge; supports batch-style arguments."""
    if solution_strs is not None or ground_truths is not None:
        data_sources = data_sources or [None] * len(solution_strs or [])
        extra_infos = extra_infos or [None] * len(solution_strs or [])
        scores = []
        for sol, gold, extra in zip(solution_strs or [], ground_truths or [], extra_infos):
            scores.append(_score_single(sol, gold, extra))
        return scores
    return _score_single(solution_str, ground_truth, extra_info)