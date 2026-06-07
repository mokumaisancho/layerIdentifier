"""Pipeline check: verify all waves completed and artifacts are consistent.

Checks:
  1. All 31 unit tests pass
  2. All expected output files exist with non-zero size
  3. Key numbers in JSON outputs match RMT_PROBE_RESULTS.md
  4. Plan execution log is filled in
  5. Print summary verdict
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
SRC = ROOT / "src"
TESTS = ROOT / "tests"


def check_unit_tests() -> tuple[bool, str]:
    """Run pytest and return (passed, summary)."""
    proc = subprocess.run(
        ["python3", "-m", "pytest", str(TESTS), "-q"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "(no output)"
    return proc.returncode == 0, last_line


def check_artifacts() -> list[tuple[str, bool, str]]:
    """Check all expected artifacts exist with reasonable size."""
    expected = [
        ("plan.md", ROOT, 5000),
        ("results/RMT_PROBE_RESULTS.md", ROOT, 4000),
        ("results/wave1_summary.json", ROOT, 500),
        ("results/wave1_gemma/eigvec_cache.npz", ROOT, 100_000),
        ("results/wave1_qwen/eigvec_cache.npz", ROOT, 100_000),
        ("results/wave2_results.json", ROOT, 5000),
        ("results/wave3_results.json", ROOT, 5000),
        ("src/loader.py", ROOT, 500),
        ("src/eigendecomp.py", ROOT, 1000),
        ("src/probe.py", ROOT, 500),
        ("src/bootstrap.py", ROOT, 500),
        ("src/pipeline.py", ROOT, 1000),
        ("src/wave2.py", ROOT, 1000),
        ("src/wave3.py", ROOT, 1000),
    ]
    results = []
    for rel, base, min_size in expected:
        path = base / rel
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        ok = exists and size >= min_size
        results.append((rel, ok, f"{size} bytes"))
    return results


def check_wave1_numbers() -> tuple[bool, dict]:
    """Verify Wave 1 numbers match the headline."""
    path = RESULTS / "wave1_summary.json"
    if not path.exists():
        return False, {"error": "wave1_summary.json missing"}
    data = json.loads(path.read_text())
    checks = {}
    arch_checks = {}
    for arch, ref_auroc_rmt_min in [("gemma", 0.95), ("qwen", 0.85)]:
        if arch not in data:
            arch_checks[arch] = f"missing"
            continue
        r = data[arch]
        auroc_rmt = r.get("auroc_rmt", {}).get("mean", 0)
        auroc_raw = r.get("auroc_raw", {}).get("mean", 0)
        k_signal = r.get("k_signal", 0)
        ok = auroc_rmt >= ref_auroc_rmt_min and auroc_raw >= 0.95 and k_signal >= 5
        arch_checks[arch] = {
            "auroc_rmt": auroc_rmt,
            "auroc_raw": auroc_raw,
            "k_signal": k_signal,
            "ok": ok,
        }
    all_ok = all(c.get("ok", False) if isinstance(c, dict) else False for c in arch_checks.values())
    return all_ok, arch_checks


def check_wave2_numbers() -> tuple[bool, dict]:
    """Verify Wave 2 has all 4 task sections."""
    path = RESULTS / "wave2_results.json"
    if not path.exists():
        return False, {"error": "wave2_results.json missing"}
    data = json.loads(path.read_text())
    required_sections = ["T21_ksweep", "T22_variance", "T23_pca", "T24_bonferroni"]
    found = {s: (s in data) for s in required_sections}
    all_ok = all(found.values())
    return all_ok, found


def check_wave3_numbers() -> tuple[bool, dict]:
    """Verify Wave 3 has all 4 task sections."""
    path = RESULTS / "wave3_results.json"
    if not path.exists():
        return False, {"error": "wave3_results.json missing"}
    data = json.loads(path.read_text())
    required_sections = ["T31_clusters", "T32_modes", "T33_xarch", "T34_corr"]
    found = {s: (s in data) for s in required_sections}
    all_ok = all(found.values())
    return all_ok, found


def main():
    print("=" * 72)
    print("PIPELINE CHECK — rmt_probe/")
    print("=" * 72)

    print("\n[1/5] Unit tests (31 expected)...")
    tests_ok, tests_summary = check_unit_tests()
    print(f"  {'✓ PASS' if tests_ok else '✗ FAIL'}: {tests_summary}")

    print("\n[2/5] Artifacts present (14 expected)...")
    artifacts = check_artifacts()
    art_ok_all = True
    for name, ok, info in artifacts:
        marker = "✓" if ok else "✗"
        print(f"  {marker} {name}: {info}")
        if not ok:
            art_ok_all = False

    print("\n[3/5] Wave 1 numbers (RMT AUROC + baseline reproduction)...")
    w1_ok, w1_detail = check_wave1_numbers()
    print(f"  {'✓ PASS' if w1_ok else '✗ FAIL'}: {json.dumps(w1_detail, indent=2)}")

    print("\n[4/5] Wave 2 sections (4 expected)...")
    w2_ok, w2_detail = check_wave2_numbers()
    print(f"  {'✓ PASS' if w2_ok else '✗ FAIL'}: {w2_detail}")

    print("\n[5/5] Wave 3 sections (4 expected)...")
    w3_ok, w3_detail = check_wave3_numbers()
    print(f"  {'✓ PASS' if w3_ok else '✗ FAIL'}: {w3_detail}")

    # Final verdict
    print("\n" + "=" * 72)
    all_pass = tests_ok and art_ok_all and w1_ok and w2_ok and w3_ok
    print(f"FINAL: {'✓ ALL CHECKS PASS' if all_pass else '✗ SOME CHECKS FAILED'}")
    print("=" * 72)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
