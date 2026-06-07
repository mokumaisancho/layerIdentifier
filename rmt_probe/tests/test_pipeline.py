"""TDD tests for pipeline.py — Wave 1 MVP orchestration."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def small_captures(tmp_path):
    """Synthetic captures with embedded multi-feature signal.

    T=80, N=10. Signal spread across 4 features so the leading eigenvalue
    clearly exceeds the MP upper bound (1+sqrt(0.125))^2 ≈ 1.83 even after
    Ledoit-Wolf shrinkage at small T.
    """
    rng = np.random.default_rng(7)
    problems = []
    feature_names = [f"L{L}_{f}" for L in (0, 1) for f in ("mean_norm", "std_norm", "n_spikes", "delta_mean", "delta_std")]
    n_problems = 80
    for i in range(n_problems):
        correct = bool(i % 2 == 0)
        # Embed multi-feature signal: class shift across 4 features
        layer_features = {name: float(rng.normal(0, 1)) for name in feature_names}
        if correct:
            layer_features["L0_mean_norm"] += 4.0
            layer_features["L0_std_norm"] += 3.0
            layer_features["L1_mean_norm"] += 2.5
            layer_features["L1_std_norm"] += 2.0
        problems.append({
            "problem_id": i,
            "seed": 7,
            "ground_truth": str(i),
            "model_answer_text": str(i),
            "final_correct": correct,
            "n_tokens": 30,
            "tps": 50.0,
            "entropy_features": {},
            "layer_features": layer_features,
            "per_layer_norms": {},
        })
    path = tmp_path / "captures.json"
    path.write_text(json.dumps(problems))
    return path


def test_run_mvp_returns_results_dict(small_captures, tmp_path):
    from src.pipeline import run_mvp
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = run_mvp(
        arch_name="test_arch",
        captures_path=str(small_captures),
        out_dir=str(out_dir),
        n_folds=5,
        n_bootstrap=200,
        seed=42,
    )
    assert isinstance(result, dict)
    for key in ("arch", "T", "N", "k_signal", "mp_lambda_max", "shrinkage",
                "auroc_rmt", "auroc_raw", "ci_rmt", "ci_raw"):
        assert key in result, f"missing key: {key}"


def test_pipeline_writes_results_json(small_captures, tmp_path):
    from src.pipeline import run_mvp
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    run_mvp(
        arch_name="test_arch",
        captures_path=str(small_captures),
        out_dir=str(out_dir),
        n_folds=5,
        n_bootstrap=200,
        seed=42,
    )
    result_path = out_dir / "mvp_results.json"
    assert result_path.exists(), "must write mvp_results.json"
    data = json.loads(result_path.read_text())
    assert "arch" in data


def test_pipeline_writes_eigvec_cache(small_captures, tmp_path):
    from src.pipeline import run_mvp
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    run_mvp(
        arch_name="test_arch",
        captures_path=str(small_captures),
        out_dir=str(out_dir),
        n_folds=5,
        n_bootstrap=200,
        seed=42,
    )
    cache_path = out_dir / "eigvec_cache.npz"
    assert cache_path.exists(), "must write eigvec_cache.npz"


def test_pipeline_auroc_in_valid_range(small_captures, tmp_path):
    from src.pipeline import run_mvp
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = run_mvp(
        arch_name="test_arch",
        captures_path=str(small_captures),
        out_dir=str(out_dir),
        n_folds=5,
        n_bootstrap=200,
        seed=42,
    )
    assert 0.5 <= result["auroc_rmt"]["mean"] <= 1.0
    assert 0.5 <= result["auroc_raw"]["mean"] <= 1.0


def test_pipeline_ci_contains_point(small_captures, tmp_path):
    from src.pipeline import run_mvp
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = run_mvp(
        arch_name="test_arch",
        captures_path=str(small_captures),
        out_dir=str(out_dir),
        n_folds=5,
        n_bootstrap=200,
        seed=42,
    )
    for key in ("ci_rmt", "ci_raw"):
        ci = result[key]
        assert ci["lower"] <= ci["point"] <= ci["upper"], \
            f"{key}: point must lie in CI"


def test_pipeline_smoke_n20_real_gemma():
    """Smoke test on real Gemma captures: T=150. Should complete in <60s."""
    captures = Path("/Users/apple/Downloads/Py/layerIdentifier/results/gemma-4-E4B-nothinking/captures.json")
    if not captures.exists():
        pytest.skip("Gemma captures.json not available")
    from src.pipeline import run_mvp
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        result = run_mvp(
            arch_name="gemma_smoke",
            captures_path=str(captures),
            out_dir=tmp,
            n_folds=5,
            n_bootstrap=200,
            seed=42,
        )
    assert result["T"] == 150
    assert result["N"] == 588
    assert result["k_signal"] >= 5, "Gemma should have ≥5 signal modes"
    assert 0.5 <= result["auroc_rmt"]["mean"] <= 1.0
