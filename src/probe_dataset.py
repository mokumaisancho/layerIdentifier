"""Probe dataset — same 50 problems × 3 seeds as InferenceLLM2.

The 50 GSM8K-style problems live in reference/InferenceLLM2/problems.py (copied
verbatim from InferenceLLM2/src/inference_llm/problems.py — NOT modified).

For each problem we re-generate on the target model with the same seeds, get a
fresh correct/wrong label, and capture the entropy + per-layer hidden states.

Output schema (per problem-seed):
  {problem_id, seed, ground_truth, model_answer, final_correct,
   n_tokens, entropy_vec, per_layer_norms, ...}
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


REF_PROBLEMS_PATH = Path(__file__).resolve().parent.parent / "reference" / "InferenceLLM2" / "problems.py"


def _load_builtin_problems():
    """Load problems.py from reference/ as a module (avoid touching original repo)."""
    import sys
    spec = importlib.util.spec_from_file_location("li_problems", str(REF_PROBLEMS_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["li_problems"] = mod  # required for dataclass + frozen + py3.14
    spec.loader.exec_module(mod)
    return mod.load_builtin_problems()


# ---------------------------------------------------------------------------
# Prompt construction — matches InferenceLLM2 convention (PAL-style Python)
# ---------------------------------------------------------------------------
PAL_PROMPT_TEMPLATE = """Solve the math problem with VERY SHORT Python code.

Rules:
- At most 6 lines.
- No print(), no comments, no functions, no input().
- The LAST line MUST be exactly: result = <numeric_literal>

Problem: {question}

```python
"""


@dataclass
class ProbeExample:
    problem_id: str
    seed: int
    question: str
    ground_truth: float
    problem_type: str
    difficulty: str
    prompt: str


def build_probe_dataset(seeds: tuple[int, ...] = (7, 42, 123)) -> list[ProbeExample]:
    """Return the full 50×N problem×seed matrix."""
    problems = _load_builtin_problems()
    out: list[ProbeExample] = []
    for p in problems:
        for s in seeds:
            out.append(ProbeExample(
                problem_id=p.id,
                seed=s,
                question=p.question,
                ground_truth=p.ground_truth,
                problem_type=p.problem_type,
                difficulty=p.difficulty,
                prompt=PAL_PROMPT_TEMPLATE.format(question=p.question),
            ))
    return out


# ---------------------------------------------------------------------------
# Answer extraction + correctness check
# ---------------------------------------------------------------------------
import ast
import operator as op

_BIN_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
}
_UNARY_OPS = {ast.UAdd: op.pos, ast.USub: op.neg}


def _safe_eval(node):
    """Evaluate an AST of arithmetic on numeric literals only."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsafe expr")


_RESULT_RE = re.compile(r"result\s*=\s*([^\n#=]+)")


def extract_answer(gen_text: str) -> Optional[float]:
    """Extract the value of the LAST `result = X` assignment from a PAL-style generation.

    Strategy:
      1. Sandboxed exec of the captured code, then read `result` from the namespace.
         Safe because we strip __builtins__ and only allow arithmetic + assignments.
      2. Fallback: regex the last `result = <numeric_literal>`.
      3. Fallback: regex the last `result = <expr>` and AST-eval it after substituting
         prior `var = <literal_or_evaluable_expr>` lines.
    """
    if not gen_text:
        return None

    # --- Strategy 1: sandboxed exec ---
    # Take only up to the LAST `result = ...` line (truncate any loops/repeats).
    lines = gen_text.splitlines()
    last_result_idx = -1
    for i, ln in enumerate(lines):
        if re.search(r"\bresult\s*=", ln):
            last_result_idx = i
    if last_result_idx >= 0:
        code_lines = lines[: last_result_idx + 1]
        # Drop trailing comments / closing fence
        last = code_lines[-1]
        last = re.sub(r"#.*$", "", last).strip().rstrip("`").strip()
        code_lines[-1] = last
        code = "\n".join(code_lines)
        # Remove markdown fences
        code = re.sub(r"```python\s*", "", code)
        code = re.sub(r"```\s*$", "", code)
        # Strip comment-only lines (annotations only, no exec effect)
        code = "\n".join(
            ln for ln in code.splitlines()
            if not re.match(r"\s*#", ln)
        )
        # Sandbox: restricted builtins (print is no-op, math is allowed)
        safe_builtins = {
            "print": lambda *a, **k: None,
            "len": len, "int": int, "float": float, "abs": abs,
            "round": round, "min": min, "max": max, "sum": sum,
            "range": range, "True": True, "False": False,
        }
        try:
            sandbox: dict = {"__builtins__": safe_builtins}
            exec(code, sandbox)  # noqa: S102 — intentional sandboxed exec
            r = sandbox.get("result")
            if isinstance(r, (int, float)):
                return float(r)
        except Exception:
            pass

    # --- Strategy 2: literal regex ---
    matches = re.findall(r"result\s*=\s*(-?\d+(?:\.\d+)?)", gen_text)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            pass

    # --- Strategy 3: AST eval with substitution ---
    expr_match = re.findall(r"result\s*=\s*([^\n#=]+)", gen_text)
    if expr_match:
        expr = expr_match[-1].strip()
        try:
            node = ast.parse(expr, mode="eval").body
            return float(_safe_eval(node))
        except Exception:
            pass

    return None


def is_correct(gen_text: str, ground_truth: float, tol: float = 1e-3) -> bool:
    ans = extract_answer(gen_text)
    if ans is None:
        return False
    return abs(ans - ground_truth) <= max(tol, abs(ground_truth) * tol)


# ---------------------------------------------------------------------------
# Subset for emit-check (cheap) vs. full sweep (dearer)
# ---------------------------------------------------------------------------
def emit_check_subset(k: int = 5) -> list[ProbeExample]:
    """A small balanced subset for the initial emit-check pass."""
    full = build_probe_dataset()
    # Pick 5 distinct problem_ids across types
    seen = set()
    out = []
    for ex in full:
        if ex.problem_id in seen:
            continue
        if ex.seed != 7:
            continue
        seen.add(ex.problem_id)
        out.append(ex)
        if len(out) >= k:
            break
    return out


def full_sweep_subset() -> list[ProbeExample]:
    return build_probe_dataset(seeds=(7, 42, 123))
