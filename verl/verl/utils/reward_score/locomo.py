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

from typing import Optional

from .rag import compute_f1_score


def _normalize_ground_truth(ground_truth):
    if isinstance(ground_truth, dict):
        raw = ground_truth.get("raw")
        fixed = ground_truth.get("fixed")
        merged = []
        if isinstance(raw, (list, tuple)):
            merged.extend(raw)
        elif raw:
            merged.append(raw)
        if isinstance(fixed, (list, tuple)):
            merged.extend(fixed)
        elif fixed:
            merged.append(fixed)
        return merged
    return ground_truth


def _score_single(solution_str: str, ground_truth, extra_info=None) -> float:
    normalized_ground_truth = _normalize_ground_truth(ground_truth)
    score=float(compute_f1_score(solution_str, normalized_ground_truth, extra_info))
    print(f"score:{score}")
    return score


def compute_score(
    data_source=None,
    solution_str: Optional[str] = None,
    ground_truth=None,
    extra_info=None,
    data_sources=None,
    solution_strs=None,
    ground_truths=None,
    extra_infos=None,
    **kwargs,
):
    """Return token-level F1; supports batch-style arguments."""
    if solution_strs is not None or ground_truths is not None:
        data_sources = data_sources or [None] * len(solution_strs or [])
        extra_infos = extra_infos or [None] * len(solution_strs or [])
        scores = []
        for sol, gold, extra in zip(solution_strs or [], ground_truths or [], extra_infos):
            scores.append(_score_single(sol, gold, extra))
        return scores
    return _score_single(solution_str, ground_truth, extra_info)