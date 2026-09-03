# Copyright 2024 PRIME team and/or its affiliates
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

import base64
import json
import pickle
import re
import traceback
import zlib

from .utils import check_correctness as apps_check_correctness


_PYTHON_FENCE = re.compile(r"```\s*(?:python|py)?\s*\n(.*?)```", flags=re.IGNORECASE | re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_PYTHON_START = re.compile(
    r"^\s*(?:from\s+[A-Za-z_]\w*|import\s+[A-Za-z_]\w*|def\s+\w+|class\s+\w+|"
    r"if\s+__name__\s*==|[A-Za-z_]\w*\s*=|#)",
)


def extract_solution(completion: str) -> str:
    """Extract Python from common LLM response formats.

    The executor must receive the model response only.  Still, models vary in
    whether they use `````python``, `````py``, an unlabeled fence, or bare code.
    Select the last fenced block when present; otherwise remove thinking text
    and trim leading prose when a recognisable Python line follows it.
    """
    completion = str(completion)
    fenced = _PYTHON_FENCE.findall(completion)
    if fenced:
        return fenced[-1].strip()

    candidate = _THINK_BLOCK.sub("", completion).replace("</think>", "").strip()
    lines = candidate.splitlines()
    for index, line in enumerate(lines):
        if _PYTHON_START.match(line):
            return "\n".join(lines[index:]).strip()
    return candidate


def load_test_cases(test_cases):
    """Load APPS JSON or the compressed LiveCodeBench test-case payload."""
    if isinstance(test_cases, dict):
        return test_cases

    try:
        return json.loads(test_cases)
    except (TypeError, json.JSONDecodeError):
        decoded = pickle.loads(zlib.decompress(base64.b64decode(test_cases.encode("utf-8"))))
        return decoded if isinstance(decoded, dict) else json.loads(decoded)


def compute_score(completion, test_cases, continuous=False):
    solution = extract_solution(completion)
    success = False
    metadata_list = None
    try:
        test_cases = load_test_cases(test_cases)

        # Complete check on all in-out pairs first. If there is no failure, per-sample test can be skipped.
        try:
            res, metadata = apps_check_correctness(in_outs=test_cases, generation=solution, timeout=5, debug=False)
            metadata_list = metadata[0] if len(metadata) > 0 else {}
            # ``all([])`` is True in Python, so require at least one actual
            # test result before declaring a program correct.
            success = bool(res) and all(result is True for result in res)
            if success or not continuous:
                return success, metadata_list
        except Exception:
            # Evaluation errors and global timeouts fail closed. In strict
            # pass-all mode there is no reason to execute the cases again.
            if not continuous:
                return False, metadata_list

        test_cases_list = []
        inputs = test_cases["inputs"]
        outputs = test_cases["outputs"]
        for i in range(len(inputs)):
            test_cases_list.append({"inputs": [inputs[i]], "outputs": [outputs[i]]})

        if continuous:
            # per sample test: if continuous score is needed, test first 10 samples regardless of failures
            # do not test all samples cuz some problems have enormous test cases
            metadata_list = []
            res_list = []
            for test_case_id, test_case in enumerate(test_cases_list):
                res, metadata = apps_check_correctness(in_outs=test_case, generation=solution, timeout=5, debug=False)
                try:
                    metadata = dict(enumerate(metadata))[0]  # metadata can be empty occasionally
                except Exception:
                    metadata = {}
                metadata["test_case"] = {}
                metadata["test_case"]["input"] = str(test_case["inputs"][0])
                metadata["test_case"]["output"] = str(test_case["outputs"][0])
                metadata["test_case"]["res"] = str(res)
                metadata_list.append(metadata)
                res_list.extend(res)

                if test_case_id >= 9:
                    break
            res_count = len(res_list) if len(res_list) > 0 else 1
            success = sum(map(lambda x: x is True, res_list)) / res_count
    except Exception:
        traceback.print_exc(10)
        success = False
        metadata_list = None
    return success, metadata_list
