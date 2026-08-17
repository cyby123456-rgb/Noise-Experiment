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

import math
import re
import string
import warnings
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Tuple


def normalize_answer(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _extract_golden_answers(ground_truth, extra_info=None) -> List[str]:
    if isinstance(ground_truth, dict):
        for key in ("golden_answers", "answers", "answer"):
            if key in ground_truth:
                return _as_list(ground_truth[key])
    if extra_info and isinstance(extra_info, dict):
        choices = extra_info.get("choices")
        golden = extra_info.get("golden_answers")
        if choices is not None and golden is not None:
            try:
                return [choices[idx] for idx in golden]
            except Exception:
                pass
        for key in ("golden_answers", "answers", "answer"):
            if key in extra_info:
                return _as_list(extra_info[key])
    return _as_list(ground_truth)


def _token_level_scores(prediction: str, ground_truths: Sequence[str]) -> dict:
    final_metric = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = 1.0 * num_same / max(len(prediction_tokens), 1)
        recall = 1.0 * num_same / max(len(ground_truth_tokens), 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-8)
        final_metric["f1"] = max(f1, final_metric["f1"])
        final_metric["precision"] = max(precision, final_metric["precision"])
        final_metric["recall"] = max(recall, final_metric["recall"])
    return final_metric


def compute_f1_score(solution_str: str, ground_truth, extra_info=None) -> float:
    golden_answers = _extract_golden_answers(ground_truth, extra_info)
    return _token_level_scores(solution_str, golden_answers)["f1"]


def compute_precision_score(solution_str: str, ground_truth, extra_info=None) -> float:
    golden_answers = _extract_golden_answers(ground_truth, extra_info)
    return _token_level_scores(solution_str, golden_answers)["precision"]


def compute_recall_score(solution_str: str, ground_truth, extra_info=None) -> float:
    golden_answers = _extract_golden_answers(ground_truth, extra_info)
    return _token_level_scores(solution_str, golden_answers)["recall"]


def compute_exact_match(solution_str: str, ground_truth, is_regex: bool = False, extra_info=None) -> float:
    golden_answers = _extract_golden_answers(ground_truth, extra_info)
    normalized_prediction = normalize_answer(solution_str)
    for golden_answer in golden_answers:
        if is_regex:
            pattern = re.compile(golden_answer, re.IGNORECASE)
            if re.fullmatch(pattern, normalized_prediction) is not None:
                return 1.0
        else:
            if normalize_answer(golden_answer) == normalized_prediction:
                return 1.0
    return 0.0


def compute_sub_exact_match(solution_str: str, ground_truth, is_regex: bool = False, extra_info=None) -> float:
    golden_answers = _extract_golden_answers(ground_truth, extra_info)
    normalized_prediction = normalize_answer(solution_str)
    for golden_answer in golden_answers:
        if is_regex:
            pattern = re.compile(golden_answer, re.IGNORECASE)
            if re.search(pattern, normalized_prediction) is not None:
                return 1.0
        else:
            if normalize_answer(golden_answer) in normalized_prediction:
                return 1.0
    return 0.0


def _extract_docs(retrieval_result: Sequence, topk: int) -> List[str]:
    docs = []
    for item in retrieval_result[:topk]:
        if isinstance(item, dict) and "contents" in item:
            docs.append(item["contents"])
        else:
            docs.append(item)
    return docs


def compute_retrieval_recall(
    retrieval_result: Sequence,
    ground_truth,
    topk: int = 5,
    extra_info=None,
) -> float:
    golden_answers = _extract_golden_answers(ground_truth, extra_info)
    if len(retrieval_result) < topk:
        warnings.warn(f"Length of retrieved docs is smaller than topk ({topk})")
    doc_list = _extract_docs(retrieval_result, topk)
    hit_list = []
    for doc in doc_list:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(doc):
                hit_list.append(True)
                break
        else:
            hit_list.append(False)
    return 1.0 if any(hit_list) else 0.0


def compute_retrieval_precision(
    retrieval_result: Sequence,
    ground_truth,
    topk: int = 5,
    extra_info=None,
) -> float:
    golden_answers = _extract_golden_answers(ground_truth, extra_info)
    if len(retrieval_result) < topk:
        warnings.warn(f"Length of retrieved docs is smaller than topk ({topk})")
    doc_list = _extract_docs(retrieval_result, topk)
    hit_list = []
    for doc in doc_list:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(doc):
                hit_list.append(True)
                break
        else:
            hit_list.append(False)
    return sum(hit_list) / max(len(hit_list), 1)


_ROUGE_SCORER = None
_ROUGE_CACHE = {}


def _get_rouge_scorer():
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        try:
            from rouge import Rouge
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("rouge is required for Rouge reward computation.") from exc
        _ROUGE_SCORER = Rouge()
    return _ROUGE_SCORER


def _compute_rouge_scores(pred: str, golden_answers: Sequence[str]) -> dict:
    cache_key = (pred, tuple(golden_answers))
    if cache_key in _ROUGE_CACHE:
        return _ROUGE_CACHE[cache_key]
    scorer = _get_rouge_scorer()
    output = {"rouge-1": [], "rouge-2": [], "rouge-l": []}
    for answer in golden_answers:
        scores = scorer.get_scores(pred, answer)
        for key in output:
            output[key].append(scores[0][key]["f"])
    for key, values in output.items():
        output[key] = max(values) if values else 0.0
    _ROUGE_CACHE[cache_key] = output
    return output


def compute_rouge_1(solution_str: str, ground_truth, extra_info=None) -> float:
    golden_answers = _extract_golden_answers(ground_truth, extra_info)
    return _compute_rouge_scores(solution_str, golden_answers)["rouge-1"]


def compute_rouge_2(solution_str: str, ground_truth, extra_info=None) -> float:
    golden_answers = _extract_golden_answers(ground_truth, extra_info)
    return _compute_rouge_scores(solution_str, golden_answers)["rouge-2"]


def compute_rouge_l(solution_str: str, ground_truth, extra_info=None) -> float:
    golden_answers = _extract_golden_answers(ground_truth, extra_info)
    return _compute_rouge_scores(solution_str, golden_answers)["rouge-l"]


_ZH_ROUGE_SCORER = None
_ZH_ROUGE_CACHE = {}


def _get_zh_rouge_scorer():
    global _ZH_ROUGE_SCORER
    if _ZH_ROUGE_SCORER is None:
        try:
            from rouge_chinese import Rouge
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("rouge_chinese is required for zh_rouge reward computation.") from exc
        _ZH_ROUGE_SCORER = Rouge()
    return _ZH_ROUGE_SCORER


def _compute_zh_rouge_scores(pred: str, golden_answers: Sequence[str]) -> dict:
    cache_key = (pred, tuple(golden_answers))
    if cache_key in _ZH_ROUGE_CACHE:
        return _ZH_ROUGE_CACHE[cache_key]
    try:
        import jieba
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("jieba is required for zh_rouge reward computation.") from exc
    scorer = _get_zh_rouge_scorer()
    output = {"rouge-1": [], "rouge-2": [], "rouge-l": []}
    pred_cut = " ".join(jieba.cut(pred))
    for answer in golden_answers:
        answer_cut = " ".join(jieba.cut(answer))
        scores = scorer.get_scores(pred_cut, answer_cut)
        for key in output:
            output[key].append(scores[0][key]["f"])
    for key, values in output.items():
        output[key] = max(values) if values else 0.0
    _ZH_ROUGE_CACHE[cache_key] = output
    return output


def compute_zh_rouge_1(solution_str: str, ground_truth, extra_info=None) -> float:
    golden_answers = _extract_golden_answers(ground_truth, extra_info)
    return _compute_zh_rouge_scores(solution_str, golden_answers)["rouge-1"]


def compute_zh_rouge_2(solution_str: str, ground_truth, extra_info=None) -> float:
    golden_answers = _extract_golden_answers(ground_truth, extra_info)
    return _compute_zh_rouge_scores(solution_str, golden_answers)["rouge-2"]


def compute_zh_rouge_l(solution_str: str, ground_truth, extra_info=None) -> float:
    golden_answers = _extract_golden_answers(ground_truth, extra_info)
    return _compute_zh_rouge_scores(solution_str, golden_answers)["rouge-l"]


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _closest_reference_length(candidate_len: int, reference_lens: Sequence[int]) -> int:
    closest = min(reference_lens, key=lambda ref_len: (abs(ref_len - candidate_len), ref_len))
    return closest


def compute_bleu_score(
    solution_str: str,
    ground_truth,
    max_order: int = 4,
    smooth: bool = False,
    extra_info=None,
) -> float:
    references = _extract_golden_answers(ground_truth, extra_info)
    pred_tokens = solution_str.split()
    ref_tokens = [ref.split() for ref in references if ref]
    if not ref_tokens or not pred_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_order + 1):
        pred_counts = _ngram_counts(pred_tokens, n)
        max_ref_counts = Counter()
        for ref in ref_tokens:
            max_ref_counts |= _ngram_counts(ref, n)
        overlap = pred_counts & max_ref_counts
        numerator = sum(overlap.values())
        denominator = max(sum(pred_counts.values()), 1)
        if smooth:
            precisions.append((numerator + 1) / (denominator + 1))
        else:
            precisions.append(numerator / denominator)

    if min(precisions) == 0 and not smooth:
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)

    ref_lens = [len(ref) for ref in ref_tokens]
    cand_len = len(pred_tokens)
    ref_len = _closest_reference_length(cand_len, ref_lens)
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / max(cand_len, 1))
    return geo_mean * bp


JUDGE_PROMPT = """
You will be given a user_question and system_answer couple.
Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
Give your answer as a float on a scale of 0 to 10, where 0 means that the system_answer is not helpful at all, and 10 means that the answer completely and helpfully addresses the question.
Provide your feedback as follows:
Feedback:::
Total rating: (your rating, as a float between 0 and 10)
Now here are the question and answer.
Question: {question}
Answer: {answer}
Feedback:::
Total rating: """

_JUDGE_PIPELINE = None
_JUDGE_MODEL_PATH = None
_JUDGE_DEVICE = None


def _get_judge_pipeline(model_path: str, device: int = 0):
    global _JUDGE_PIPELINE, _JUDGE_MODEL_PATH, _JUDGE_DEVICE
    if _JUDGE_PIPELINE is None or _JUDGE_MODEL_PATH != model_path or _JUDGE_DEVICE != device:
        try:
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers is required for llm_judge reward computation.") from exc
        _JUDGE_PIPELINE = pipeline("text2text-generation", model=model_path, device=device)
        _JUDGE_MODEL_PATH = model_path
        _JUDGE_DEVICE = device
    return _JUDGE_PIPELINE


def _extract_judge_score(answer: str, split_str: str = "Total rating:") -> float:
    try:
        rating = answer.split(split_str, 1)[1] if split_str in answer else answer
        digit_groups = [el.strip() for el in re.findall(r"\d+(?:\.\d+)?", rating)]
        return float(digit_groups[0])
    except Exception:
        return 0.0


def compute_llm_judge_score(
    solution_str: str,
    ground_truth,
    model_path: Optional[str] = None,
    device: int = 0,
    extra_info=None,
) -> float:
    if not extra_info or not isinstance(extra_info, dict):
        raise ValueError("extra_info with 'question' is required for llm_judge reward computation.")
    question = extra_info.get("question")
    if question is None:
        raise ValueError("extra_info['question'] is required for llm_judge reward computation.")
    if model_path is None:
        raise ValueError("model_path is required for llm_judge reward computation.")
    judge_input = JUDGE_PROMPT.format(question=question, answer=solution_str)
    pipeline = _get_judge_pipeline(model_path, device=device)
    output = pipeline(judge_input, max_new_tokens=100, batch_size=1)[0]["generated_text"]
    score = _extract_judge_score(output)
    return score / 10 + 1


def compute_input_tokens(text: Optional[str], tokenizer_name: Optional[str] = None, extra_info=None) -> int:
    if extra_info and isinstance(extra_info, dict) and extra_info.get("prompt") is not None:
        text = extra_info["prompt"]
    if text is None:
        text = ""
    if tokenizer_name is None or tokenizer_name.startswith("gpt-") or tokenizer_name.startswith("text-"):
        try:
            import tiktoken
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("tiktoken is required for OpenAI tokenizer counting.") from exc
        if tokenizer_name is None:
            tokenizer_name = "gpt-4"
        tokenizer = tiktoken.encoding_for_model(tokenizer_name)
        return len(tokenizer.encode(text))
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("transformers is required for HF tokenizer counting.") from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return len(tokenizer.tokenize(text))


def compute_gaokao_acc(solution_str: str, ground_truth, extra_info=None) -> float:
    if extra_info is None or not isinstance(extra_info, dict):
        raise ValueError("extra_info with 'question_type' is required for gaokao_acc reward computation.")
    question_type = extra_info.get("question_type")
    golden_answers = _extract_golden_answers(ground_truth, extra_info)
    golden_answer = "".join([ans.lower() for ans in golden_answers])
    pred = solution_str.lower()
    if question_type == "single_choice":
        return 1.0 if pred == golden_answer else 0.0
    if pred == golden_answer:
        return 1.0
    if pred in golden_answer:
        return 0.5
    return 0.0


_METRIC_FNS = {
    "f1": compute_f1_score,
    "precision": compute_precision_score,
    "recall": compute_recall_score,
    "em": compute_exact_match,
    "acc": compute_sub_exact_match,
    "retrieval_recall": compute_retrieval_recall,
    "retrieval_precision": compute_retrieval_precision,
    "rouge-1": compute_rouge_1,
    "rouge-2": compute_rouge_2,
    "rouge-l": compute_rouge_l,
    "zh_rouge-1": compute_zh_rouge_1,
    "zh_rouge-2": compute_zh_rouge_2,
    "zh_rouge-l": compute_zh_rouge_l,
    "bleu": compute_bleu_score,
    "llm_judge": compute_llm_judge_score,
    "input_tokens": compute_input_tokens,
    "gaokao_acc": compute_gaokao_acc,
}


def compute_score(
    solution_str: str,
    ground_truth,
    metric: str = "f1",
    extra_info=None,
    **kwargs,
) -> float:
    extra_info = extra_info or {}
    if metric.startswith("retrieval_recall_top"):
        topk = int(metric.replace("retrieval_recall_top", ""))
        retrieval_result = kwargs.get("retrieval_result") or extra_info.get("retrieval_result", [])
        return compute_retrieval_recall(retrieval_result, ground_truth, topk=topk, extra_info=extra_info)
    if metric.startswith("retrieval_precision_top"):
        topk = int(metric.replace("retrieval_precision_top", ""))
        retrieval_result = kwargs.get("retrieval_result") or extra_info.get("retrieval_result", [])
        return compute_retrieval_precision(retrieval_result, ground_truth, topk=topk, extra_info=extra_info)
    if metric not in _METRIC_FNS:
        raise ValueError(f"Unsupported metric: {metric}")
    fn = _METRIC_FNS[metric]
    return fn(solution_str, ground_truth, extra_info=extra_info, **kwargs)