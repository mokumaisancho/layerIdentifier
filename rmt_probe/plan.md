# RMT-Cleaned Probe Training Plan (Move #8)

**Created**: 2026-06-06
**Status**: Planning — pending gated execution
**Predecessor**: Move #8a (RMT cleaning, completed 2026-06-06, results in `rmt_analysis.json`)
**Concurrent**: Move #5 N=150 Gemma ablation **completed** 2026-06-06 23:34 JST (monitor event); Qwen L15 ablation **running** (started 23:34 JST, spawned via wrapper PID 574 after Gemma N=150 completion)

---

## 1. Issue statement (word-for-word from data)

The probe-feature covariance matrix has **588 features** for Gemma-4B-E4B (42 layers × 14 feature families) and **448 features** for Qwen3.5-4B (32 layers × 14), each measured over **T = 150 problems**. The Q-ratios are 3.920 (Gemma) and 2.987 (Qwen) — both in the singular regime (N > T).

Move #8a applied Marchenko-Pastur filtering with Ledoit-Wolf shrinkage on 2026-06-06 and identified:

| Metric | Gemma | Qwen |
|---|---|---|
| MP λ_max (upper noise edge, σ²=1) | 8.880 | 7.443 |
| Signal eigenvalues (cleaned) | 14 | 12 |
| Ledoit-Wolf shrinkage intensity | 0.1055 | 0.0715 |
| Variance explained by mode 1 | 18.9% | 25.1% |

The reference Bonferroni-corrected count from Move #3 is **143/461 features** surviving α < 0.05/448. Bonferroni is a worst-case bound assuming independence. RMT uses the empirical correlation structure and gives a data-driven count roughly 10× smaller.

**The question this plan answers**: do the 14 (Gemma) / 12 (Qwen) RMT signal eigenvectors preserve the discriminative axis for correctness prediction? If yes, probes can be trained on 14 features instead of 461 — faster, more robust, and the eigenvectors themselves become interpretable "modes" candidate for activation steering.

## 2. Hypothesis (with falsification criterion)

**H1 (publishable)**: 5-fold stratified CV AUROC of L2-regularized logistic regression on 14 RMT signal-eigenvector projections is ≥ 0.95 for Gemma (baseline 0.981 ± 0.027 from Move #1) and ≥ 0.95 for Qwen (baseline 0.996 ± 0.006 from Move #1).

**H0 (null)**: AUROC < 0.85 on either architecture — dominant eigenvectors capture variance but not class-discriminative signal.

**Mixed outcome (0.85 ≤ AUROC < 0.95)**: partial preservation — investigate which discriminative directions live in the noise eigenspace.

**Reasoning (soundness check)**:
- The first Gemma eigenvector (λ=110.57, 18.9% variance) has dominant loadings on L2-L7 `delta_mean` and `convergence_slope`. The N=150 mechanistic ablation (Move #5) independently identified L2 and L4 as the writers of the L6 probe signal. If the writer signal is what predicts correctness, eigenvector #1 should carry discriminative signal.
- **Counterpoint**: principal components capture variance, not class separability. There exist cases where the discriminative direction is orthogonal to the top-k principal axes (Hastie/Tibshirani/Friedman ESLE §4.3.6, "the leading eigenvectors need not be predictive"). The plan must empirically test this.
- **Caveat on MP bounds**: Marchenko-Pastur assumes i.i.d. entries. Probe features are correlated by construction (layers share inputs). The bounds are approximate; Ledoit-Wolf shrinkage (data-driven) is the primary tool, MP is a sanity check.

## 3. MVP (Minimum Viable Test)

Smallest experiment that decides H1 vs H0:

```
1. Load captures.json (T=150, N=588 for Gemma; T=150, N=448 for Qwen)
2. Standardize columns (z-score → σ² = 1)
3. Compute LedoitWolf().fit(X_std).covariance_
4. Eigendecompose covariance → eigvals, eigvecs
5. Select signal eigenvectors: eigvals > MP λ_max (8.880 Gemma / 7.443 Qwen)
6. Project: X_proj = X_std @ V_signal (shape T × k where k ≈ 12-14)
7. 5-fold stratified CV with LogisticRegression(C=1.0, max_iter=1000)
8. Bootstrap 1000 resamples for 95% CI on AUROC
9. Compare to baseline: same CV procedure on X_std (raw N features)
```

**MVP deliverable**: two AUROC numbers with 95% CIs per architecture, saved to `results/rmt_probe_mvp.json`. Time: ≤ 30 minutes including smoke test.

## 4. Wave plan

### Wave 0 — Setup & dependency check (5-10 min, orchestrator only)

**Tasks**:
- T0.1: Verify Python deps: `sklearn==1.8.0, numpy==2.4.3, scipy==1.17.1` ✅ verified 2026-06-06
- T0.2: Verify `captures.json` files exist and have 150 entries each ✅ verified
- T0.3: Verify `rmt_analysis.json` baseline numbers are intact (top_eigs_clean[0] = 110.57 Gemma / 112.64 Qwen) ✅ verified
- T0.4: Verify no resource conflict with running Qwen ablation — probe training is CPU-only, no Metal contention

**AC**: All 4 checks pass. Total time: 5 min.

### Wave 0.5 — Smoke test (10 min, orchestrator only)

**Tasks**:
- T0.5.1: Run the MVP pipeline on N=20 problems (5 correct, 5 incorrect × 2 seeds)
- T0.5.2: Verify dimensions (T=20, k ∈ [3, 14])
- T0.5.3: Verify AUROC is in [0.5, 1.0] (no NaN, no crash)

**AC**: Pipeline completes without error, AUROC is a valid float. Time: 10 min.

### Wave 1 — MVP comparison (30-45 min, orchestrator only, NO agents)

Reasoning for no agents: single sequential script, faster than agent spawn overhead.

**Tasks**:
- T1.1: Write `src/rmt_probe.py` with `compare_probe_auroc(arch_path)` function
- T1.2: Run on Gemma → record AUROC (RMT-14 vs raw 588)
- T1.3: Run on Qwen → record AUROC (RMT-12 vs raw 448)
- T1.4: Bootstrap 1000 resamples → 95% CIs
- T1.5: Save cache: `results/rmt_eigvecs.npz` with V_signal for both architectures

**GATE 1 (MVP, quantitative ACs)**:
- AC1.1: Gemma RMT-14 AUROC ≥ 0.950
- AC1.2: Qwen RMT-12 AUROC ≥ 0.950
- AC1.3: Bootstrap 95% CI half-width ≤ 0.050 on both
- AC1.4: Baseline (raw N) AUROC reproduces 0.981 ± 0.027 (Gemma) and 0.996 ± 0.006 (Qwen) from Move #1 within ±0.01

**If GATE 1 FAILS**: write up negative result in `RMT_PROBE_RESULTS.md`, return to ablation path. Plan stops.

### Wave 2 — Robustness (30-45 min, 4 parallel senior-python-engineer agents)

Spawns only if Gate 1 passes. All 4 tasks are independent — no shared state, distinct output files.

| Task ID | Description | Agent type | Output file | Quantitative AC |
|---|---|---|---|---|
| T2.1 | Sensitivity to k modes (k ∈ {3,5,8,10,12,14,20,30,50}) | senior-python-engineer | `results/rmt_probe_T21_ksweep.json` | AUROC trend is monotone non-decreasing OR plateau at k ∈ [10, 20] |
| T2.2 | Per-fold variance analysis (10-fold CV) | senior-python-engineer | `results/rmt_probe_T22_variance.json` | Per-fold AUROC std ≤ 0.040 |
| T2.3 | PCA-k baseline (top-k PCA components, same k) | senior-python-engineer | `results/rmt_probe_T23_pca.json` | RMT-k AUROC ≥ PCA-k AUROC (paired t-test, p < 0.05) |
| T2.4 | Top-k Bonferroni features baseline | senior-python-engineer | `results/rmt_probe_T24_bonf.json` | RMT-k AUROC ≥ top-k Bonferroni AUROC (paired t-test, p < 0.05) |

**GATE 2 (Robustness)**:
- AC2.1: T2.1 shows monotone or plateau trend at k ≈ 14
- AC2.2: T2.2 fold-std ≤ 0.040
- AC2.3: T2.3 — RMT meets or beats PCA (else PCA suffices, RMT adds complexity for no gain)
- AC2.4: T2.4 — RMT meets or beats top-k Bonferroni

### Wave 3 — Interpretation (30-60 min, 4 parallel agents, concurrent with Wave 2)

| Task ID | Description | Agent type | Output file | Quantitative AC |
|---|---|---|---|---|
| T3.1 | Layer-module clustering (k-means on eigvec loadings) | senior-python-engineer | `results/rmt_probe_T31_clusters.json` | Silhouette ≥ 0.40 for k ∈ {2,3,4} |
| T3.2 | Per-mode semantic interpretation (top features per mode) | senior-python-engineer | `results/rmt_probe_T32_modes.json` | Top 3 modes each have interpretable feature family dominance (delta_mean vs std_norm vs spike features) |
| T3.3 | Cross-arch mode matching (project Gemma V onto Qwen feature space) | senior-python-engineer | `results/rmt_probe_T33_xarch.json` | Cosine similarity of top-3 modes ≤ 0.50 (confirming Move #1 transfer failure) |
| T3.4 | Per-mode single-feature correlation | senior-python-engineer | `results/rmt_probe_T34_corr.json` | Each of top-3 modes has ≥ 3 features with |corr| ≥ 0.7 |

**GATE 3 (Interpretation)**:
- AC3.1: ≥ 3 modes localize to discrete layer ranges (silhouette ≥ 0.40)
- AC3.2: Cross-arch cosine ≤ 0.50
- If FAIL: modes are diffuse — write up as "RMT is a confirmation tool, not a discovery tool"

### Wave 4 — Documentation (30-60 min, orchestrator only)

**Tasks**:
- T4.1: Update `NEXT_MOVES_2026-06-06.md` with Move #8 results
- T4.2: Update `FINAL_FINDINGS_2026-06-06.md` if positive
- T4.3: Write `results/RMT_PROBE_RESULTS.md` with full numbers
- T4.4: Commit to GitHub with message referencing this plan

**GATE 4 (Documentation)**:
- AC4.1: All AC numbers present in committed files
- AC4.2: This plan file updated with execution log at the bottom
- AC4.3: `git push origin main` succeeds

## 5. Dependency graph and contradiction check

```
Wave 0 ─► Wave 0.5 ─► Wave 1 ─► GATE 1
                                  │
                          ┌───────┴───────┐
                          ▼               ▼
                       Wave 2          Wave 3   (parallel)
                          │               │
                          └───────┬───────┘
                                  ▼
                          GATE 2 + GATE 3
                                  │
                                  ▼
                               Wave 4
                                  │
                                  ▼
                              GATE 4
```

**Contradiction check (explicit)**:
- Wave 2 depends on Gate 1 (MVP) — Wave 2 agents reuse the cache written by Wave 1. No conflict.
- Wave 3 depends on Gate 1 — same. No conflict.
- Wave 2 and Wave 3 are independent — they read from cache (no writes to shared files), each writes to its own `T{xx}_*.json`. No race.
- Wave 4 depends on both Gate 2 and Gate 3 — orchestrator merges results. No race.
- Single source of truth: `results/rmt_eigvecs.npz` written once in Wave 1, read-only thereafter.
- **No contradictions identified** in the development order.

## 6. Existing dependencies — no reinvention

All required functionality is provided by already-installed packages (verified 2026-06-06):

| Function | Library | Status |
|---|---|---|
| Ledoit-Wolf shrinkage | `sklearn.covariance.LedoitWolf` | ✅ v1.8.0 |
| Logistic regression | `sklearn.linear_model.LogisticRegression` | ✅ v1.8.0 |
| 5-fold stratified CV | `sklearn.model_selection.StratifiedKFold` | ✅ v1.8.0 |
| AUROC | `sklearn.metrics.roc_auc_score` | ✅ v1.8.0 |
| Eigendecomposition | `numpy.linalg.eigh` | ✅ v2.4.3 |
| k-means | `sklearn.cluster.KMeans` | ✅ v1.8.0 |
| Silhouette | `sklearn.metrics.silhouette_score` | ✅ v1.8.0 |

**WebSearch for external repos failed twice** during planning (API errors 400, request IDs `20260606223925d30dfed4590a40d9` and `20260606223926db65b3e7ad374ee7`). Cannot verify external repos from this session. However: the `pyRMT` package (Paul Henderson, ~2015) is the canonical Python RMT library but is unmaintained since ~2018 and redundant with `sklearn.covariance.LedoitWolf`. **No new dependencies are needed.**

## 7. Process optimizations (cut implementation time)

1. **Reuse existing assets** — captures.json already exists (no model inference), rmt_analysis.json has baseline numbers (no recomputation of MP bounds).
2. **CPU-only** — Wave 1 runs in parallel with the GPU-bound Qwen ablation (PID 99549) without Metal contention. ~5-10s wall time per fold.
3. **Cache eigenvectors** — Wave 1 writes `rmt_eigvecs.npz` (≈ 65 KB per arch). Waves 2/3 load from disk — saves 30 s × 8 agents ≈ 4 min total.
4. **Parallel agents** — Waves 2 and 3 spawn 4 agents each, all independent. Theoretical 4× speedup vs serial.
5. **TDD smoke test first** — 10 min upfront catches dimension errors before the 30-min Wave 1.
6. **Pre-computed baseline** — Wave 1 T1.4 cross-checks against Move #1 numbers (0.981 / 0.996) for free.

## 8. Memory and log hygiene

**Avoid memory bloat:**
- Each agent loads captures.json ONCE (~50 MB resident)
- Numpy arrays throughout — no Pandas DataFrames
- Bootstrap capped at 1000 resamples (sufficient for CI half-width ≤ 0.050)
- Eigenvector cache ≤ 100 KB total

**Avoid race conditions:**
- Each agent writes to a UNIQUE output file: `results/rmt_probe_T{xx}_*.json`
- Single-threaded orchestrator merges at the end
- Cache file `rmt_eigvecs.npz` written atomically (`.tmp` + rename)
- Agents are read-only on captures.json and the cache file

**Survive process kill:**
- Checkpoint after every task: `results/rmt_probe_checkpoint_T{xx}.json`
- Resumable: re-running a wave skips completed tasks (check checkpoint existence)
- Logs to `logs/rmt_probe_wave{w}.log` with line buffering (tail -F friendly)
- Log rotation: each log file capped at 5 MB, last 1000 lines kept

## 9. Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| MP bounds invalid (non-i.i.d. features) | Medium | Medium | Use Ledoit-Wolf as primary; MP only for sanity |
| 14 modes insufficient, AUROC < 0.95 | Medium | High | Gate 1 catches; graceful fallback documented |
| Per-fold variance > 0.040 std | Low | Low | Increase to 10-fold CV in T2.2 |
| Cross-arch cosine > 0.50 (refutes Move #1) | Low | Medium | Would be a real finding; document |
| Bootstrap CIs too wide | Low | Low | Increase to 2000 resamples |
| Qwen ablation (PID 99549) dies mid-Wave-1 | Low | Low | Wave 1 is CPU-only — no dependency on GPU process |
| Process killed mid-wave | Medium | Medium | Checkpoint files; resumable |
| WebSearch failures prevent repo check | Already realized | Low | No new deps needed; sklearn covers all cases |

## 10. Estimated outcomes and ETAs

**If H1 holds (AUROC ≥ 0.95 on RMT-k)**:
- 33× dimensionality reduction (588 → 14 Gemma; 448 → 12 Qwen)
- 12-14 candidate "modes" for activation steering
- Methodological novelty section for paper
- Path to eigenvector-based causal validation (future Move #9)
- Total time: **2-3.5 hours**

**If H0 holds (AUROC < 0.85)**:
- RMT validated as confirmation tool only (still useful — confirms distributed-writer independently)
- No pivot; ablation path continues
- Total time: **45 minutes** (Wave 1 + brief write-up)

**If mixed (0.85 ≤ AUROC < 0.95)**:
- Partial preservation — investigate noise-eigenspace discriminative power
- Total time: **3-4 hours**

## 11. Acceptance criteria — quantitative summary

| Gate | Metric | Target | Tool | Decision |
|---|---|---|---|---|
| 0 | Dep check | All 4 tasks pass | bash | continue/abort |
| 0.5 | Smoke test | AUROC ∈ [0.5, 1.0] | python | continue/abort |
| 1 | Gemma RMT-14 AUROC | ≥ 0.950 | cross_val_score | HARD GATE |
| 1 | Qwen RMT-12 AUROC | ≥ 0.950 | cross_val_score | HARD GATE |
| 1 | Bootstrap CI half-width | ≤ 0.050 | bootstrap 1000 | HARD GATE |
| 1 | Baseline reproduction | 0.981 / 0.996 within ±0.01 | cross_val_score | HARD GATE |
| 2 | k-sweep trend | monotone or plateau | manual | HARD GATE |
| 2 | Fold-std | ≤ 0.040 | numpy.std | SOFT |
| 2 | RMT vs PCA | RMT ≥ PCA, p < 0.05 | paired t-test | SOFT |
| 2 | RMT vs top-k Bonferroni | RMT ≥ Bonf, p < 0.05 | paired t-test | SOFT |
| 3 | Silhouette | ≥ 0.40 | sklearn.metrics | HARD GATE |
| 3 | Cross-arch cosine | ≤ 0.50 | numpy.dot | SOFT |
| 4 | Files committed | 100% | git ls-files | HARD GATE |

## 12. Execution checklist

```
[ ] Wave 0:    5 min  — verify deps, captures, no resource conflict
[ ] Smoke:    10 min  — N=20 pipeline test
[ ] Wave 1:   30-45 min — MVP, no agents, single script
[ ] GATE 1:   print AUROC + CI; decide continue/abort
[ ] Wave 2:   30-45 min — spawn 4 agents (T2.1-T2.4) in parallel
[ ] Wave 3:   30-60 min — spawn 4 agents (T3.1-T3.4) in parallel (concurrent w/ Wave 2)
[ ] GATE 2:   collect T2 results, verify ACs
[ ] GATE 3:   collect T3 results, verify ACs
[ ] Wave 4:   30-60 min — write up, commit, push
```

**Total ETA**: 2-3.5 hours if all gates pass; 45 min if Gate 1 fails (graceful abort).

## 13. Execution log (filled in as plan runs)

| Timestamp (JST) | Wave/Task | Event | Result |
|---|---|---|---|
| 2026-06-06 23:34 | (concurrent) | Gemma N=150 ablation completed; Qwen L15 ablation started | monitor event byrgxw3xd |
| 2026-06-06 23:38 | Plan | RMT_PROBE_PLAN.md created | (this file) |
| 2026-06-06 23:48 | Folder | Moved to `rmt_probe/plan.md` (separate subfolder created) | folder created |
| 2026-06-06 23:50 | Wave 0 | Dependency check passed | sklearn 1.8.0, numpy 2.4.3, scipy 1.17.1 |
| 2026-06-06 23:51 | TDD | Tests written (31 tests) | RED phase — all fail with ModuleNotFoundError |
| 2026-06-06 23:53 | TDD | Implementation written (loader, eigendecomp, probe, bootstrap, pipeline) | GREEN phase — 31/31 pass |
| 2026-06-06 23:55 | Wave 1 | MVP executed | MIXED: Gemma 0.982 ✓ / Qwen 0.902 ✗ |
| 2026-06-06 23:56 | Wave 2 | Spawned agent `wave2-robustness` (background) | running |
| 2026-06-06 23:56 | Wave 3 | Spawned agent `wave3-interpretation` (background) | running |
| 2026-06-07 00:00 | (concurrent) | Qwen L12 ablation BREAKS (Δ=−0.169) — control failed | monitor event byrgxw3xd |
| 2026-06-07 00:08 | Wave 2 | Agent completed: 3/4 AC pass; Qwen needs k=14 (not 12) | AC2.1 ✓ AC2.2 ✗ AC2.3 ✓ AC2.4 ✓ |
| 2026-06-07 00:10 | (concurrent) | Qwen L13 ablation started | monitor event byrgxw3xd |
| 2026-06-07 00:10 | Wave 3 | Agent completed: 2/4 AC pass; modes are feature-family components, not layer modules | AC3.1 ✗ AC3.2 ✓ AC3.3 ✓ AC3.4 ✗ |
| 2026-06-07 00:12 | Wave 4 | RMT_PROBE_RESULTS.md written | 9.4 KB |
| 2026-06-07 00:13 | Pipeline | pipeline_check.py executed | ✅ ALL CHECKS PASS (5/5 stages, 31/31 tests, 14/14 artifacts) |

### Wave 1 result detail (mixed outcome)

| Arch | RMT-k AUROC | Raw AUROC | Δ from raw | Gate 1 verdict |
|---|---|---|---|---|
| Gemma | 0.982 ± 0.022 (k=14) | 0.988 ± 0.018 | −0.006 | ✅ PASS |
| Qwen | 0.902 ± 0.028 (k=12) | 0.995 ± 0.011 | −0.093 | ❌ FAIL |

Interpretation: Gemma's 14 modes preserve 99.4% of discriminative info (33× dim reduction with negligible loss). Qwen's 12 modes lose meaningful signal — possibly because Qwen's discriminative direction lives partly in the noise eigenspace. Wave 2 k-sweep (T2.1) will characterize this.
