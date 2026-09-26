# IRE CS4.406 — Assignment 2: Learning from Click-Logs on EB-NeRD and MIND

Two-stage news recommendation: lexical and dense candidate generation feeding a
learned re-ranker over 28 behavioural features, with a sliced evaluation harness,
paired-bootstrap ablations, and a serving/latency profile.


---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. Runs on CPU, CUDA, or Apple MPS. The dense retriever uses
`all-MiniLM-L6-v2`, so the first run downloads roughly 90 MB of model weights.

### Datasets

Deafults at `src/config.py`

```
EB-NeRD   data/ebnerd_small/{articles.parquet, train/, validation/}
          data/ebnerd_testset/{articles.parquet, test/}
MIND      MIND_data/MINDsmall_train/{news.tsv, behaviors.tsv}
          MIND_data/MINDsmall_dev/{news.tsv, behaviors.tsv}
          MIND_data/MINDlarge_test/{news.tsv, behaviors.tsv}
```

Only the `small`/`dev` bundles are needed for everything in this README. The large
test sets are required for Codabench submission.

---

## How to run things

### Stage 1 retrievers on their own

```bash
python3 src/retriever.py --dataset ebnerd --method hybrid --sample_size 200
python3 src/retriever.py --dataset mind   --method bm25   --sample_size 200
```

`--method` is `bm25`, `dense`, or `hybrid`.

### Loading data

```bash
python3 src/data_loader.py --dataset ebnerd --split val --sample_size 100
python3 src/data_loader.py --dataset mind   --split val --sample_size 100
```

### Features

```bash
python3 src/feature_engineering.py --dataset ebnerd --split train --sample_size 500
```

Prints the feature matrix shape and a NaN/range sanity report.

### Training a re-ranker

```bash
python3 src/reranker.py --model lgbm --dataset ebnerd --sample_size 1000
python3 src/reranker.py --model mlp  --dataset ebnerd --sample_size 1000 --epochs 10
```

### Evaluating

```bash
python3 src/evaluator.py --dataset ebnerd --model lgbm --sample_size 500
```

Reports AUC, MRR, nDCG@5, nDCG@10, ILD@5, Novelty@5 and Catalog Coverage@5, each
with a bootstrap CI, split four ways: cold-start vs warm users, head vs tail items.

### Baselines

```bash
python3 src/baseline_official.py --dataset ebnerd --sample_size 1000
python3 src/nrms_baseline.py     --dataset ebnerd --epochs 3 --sample_size 1000
```

### All local experiments at once

```bash
python3 run_all_experiments.py --dataset both --mode val --sample_size 5000
```

Writes a markdown results table. Useful flags:

| Flag | Effect |
|:--|:--|
| `--models lgbm mlp` | only score these; dependencies are still built |
| `--list_models` | print the model registry and exit |
| `--n_bootstrap N` | bootstrap resamples (default 100) |
| `--mode` | `val`, `test`, `ablation`, `two_stage`, `serving_ablation` |

### Ablations

```bash
# Q3 — M0..M3 with paired bootstrap CIs
python3 run_all_experiments.py --dataset both --mode ablation \
    --sample_size 5000 --n_bootstrap 1000

# Q9 — metrics with and without serving-unavailable columns
python3 run_all_experiments.py --dataset both --mode serving_ablation \
    --sample_size 500

# Q2 — retrieve-then-rank cascade, Stage 1 vs Stage 1+2
python3 run_all_experiments.py --dataset both --mode two_stage \
    --top_k 200 --sample_size 500
```

### Serving profile

```bash
python3 src/serving_analysis.py --dataset ebnerd --num_trials 100 --save --compare_stages
```

Per-stage p50/p95/p99, memory footprint, and the cost-per-1,000-queries model. Every
price, the SLA and the volume are CLI flags, so the model can be re-costed without
touching code:

```bash
python3 src/serving_analysis.py --sla_ms 50 --queries_per_day 100000000 \
    --price_vm_usd_per_hour 0.85
```

### Temporal boundary check

```bash
python3 src/temporal_guard.py --dataset both
```

Asserts the splits are time-ordered and non-overlapping, and prints the timestamp
envelope of each. Run this before trusting any evaluation number.

### Predictions for Codabench

```bash
python3 src/predictor.py --dataset ebnerd --model lgbm --split test
python3 src/predictor.py --dataset mind   --model lgbm --split test
```

Writes submission zips. This path is chunked, because the EB-NeRD test set is
13.5M impressions and the MIND large test set 2.37M.

---

## Where each part of the assignment lives

Line numbers refer to this repo.

### Q1 — Click-history and session features

| Requirement | Where |
|:--|:--|
| Click history, click count, exponential recency decay | `src/feature_engineering.py:115` `extract_impression_features` |
| Within-session click patterns, dwell time, prior CTR | `src/feature_engineering.py:127` `session_features` |
| Position bias | `position_in_impression`, `impression_size` in the same builder |
| Popularity, freshness, category match with history | `freshness_days`, `popularity_log`, `category_affinity` in the same builder |
| Behaviour-window boundary | `src/feature_engineering.py:194` `apply_behaviour_window` |
| Split-level boundary verification | `src/temporal_guard.py:139` `verify_dataset`, `:212` `assert_temporal_boundaries` |

The full 28-column schema is `FEATURE_NAMES` in `src/config.py`. Twenty are
behavioural or lexical; the last eight are within-session.

`apply_behaviour_window` runs *before* Stage 1 builds its retrieval query, not just
before feature extraction — otherwise the candidate set itself would be conditioned
on future clicks.

### Q2 — Re-ranker

| Requirement | Where |
|:--|:--|
| Stage 1 top-K candidates (K≈200) | `src/retriever.py:63` `BM25Retriever.retrieve_topk`, `:173` `DenseRetriever` |
| GBDT re-ranker | `src/reranker.py:62` `LightGBMReranker` (LambdaRank, listwise) |
| Neural re-ranker | `src/reranker.py:264` `PyTorchMLP`, `MLPReranker` |
| Score and re-rank | `src/feature_engineering.py:115`, then `predict_scores` on either model |
| Recall@K and before/after metrics | `src/evaluator.py:113` `recall_at_k`; `--mode two_stage` |

Stage 1 fuses BM25 and dense retrieval with reciprocal rank fusion,
`1/(60 + r_bm25) + 1/(60 + r_dense)`.

### Q3 — Baseline reproduced, then beaten

| Requirement | Where |
|:--|:--|
| Official popularity baseline | `src/baseline_official.py:85` `OfficialBaselineRanker` |
| Official NRMS baseline (PyTorch) | `src/nrms_baseline.py:337` `NRMSModel` |
| The improvement | `src/baseline_official.py:193` `CategoryAwareFreshnessRanker` |
| Ablation M0–M3 | `src/baseline_official.py:295` `run_ablation_study` |
| Paired bootstrap 95% CIs | `src/evaluator.py:158` `bootstrap_ci` |

The improvement scores a candidate by
`0.30·base + 0.45·category_affinity + 0.25·category_recency + 0.10·freshness`.
The paired CIs resample the per-impression *difference* array, so they are genuinely
paired rather than two independent intervals subtracted.

### Q4 — Serving and scale

| Requirement | Where |
|:--|:--|
| Index and feature-store memory | `src/serving_analysis.py:84` `measure_memory_footprint` |
| p99 retrieval latency | `src/serving_analysis.py:144` `benchmark_component_latencies` |
| Cost per 1,000 queries at an SLA | `src/serving_analysis.py:300` `compute_cost_model`, `:280` `workers_for_sla` |
| 10x scaling argument | same, projection section of the printed report |

### Q5 — Extended evaluation

| Requirement | Where |
|:--|:--|
| AUC, MRR, nDCG@5, nDCG@10 | `src/evaluator.py:88`–`:113` |
| Diversity, novelty, coverage | `src/evaluator.py:126`, `:134`, `:146` |
| Four slices | `src/evaluator.py:188` `evaluate_all_slices` |
| Bootstrap CIs | `src/evaluator.py:158` |

### Q7 — Reproducibility

`python3 run_all_experiments.py` is the single entry point. Every run writes to a
timestamped folder with a manifest recording the git SHA, package versions, the
28-column feature schema and the dense backend, because a number means very little
without the state that produced it. Datasets, checkpoints and prediction files are
gitignored.

### Q9 — Anti-gaming

| Requirement | Where |
|:--|:--|
| Metrics without serving-unavailable features | `src/serving_ablation.py:112` `run_serving_ablation` |
| Leakage assertions | `src/sanity_checks.py:37` `assert_no_future_leakage`, `:65` `assert_feature_matrix_clean` |
| No future-click leakage | `src/feature_engineering.py:194`, `src/temporal_guard.py:212` |

Four of the 28 columns are catalog-level period aggregates that do not exist when a
live ranker scores an impression: `popularity_log`, `inview_rate`, `readtime_rate`,
`sentiment_score`. The ablation retrains a matched re-ranker without them and reports
the paired delta.

---

## Files

```
run_all_experiments.py     entry point; --mode val | test | ablation | two_stage | serving_ablation
src/
  config.py                paths, hyperparameters, FEATURE_NAMES, device selection
  data_loader.py           unified EB-NeRD + MIND loading, temporal split handling
  retriever.py             BM25 index, MiniLM dense retrieval, RRF hybrid
  feature_engineering.py   28-feature builder, behaviour window, stage-1 wiring
  reranker.py              LightGBM LambdaRank and PyTorch MLP
  evaluator.py             metrics, four slices, bootstrap CIs, submission writer
  baseline_official.py     popularity baseline, the Q3 improvement, ablation study
  nrms_baseline.py         pure-PyTorch NRMS
  serving_analysis.py      latency profile, memory, cost model
  serving_ablation.py      serving-feature-subset ablation
  temporal_guard.py        split-level boundary checks
  predictor.py             chunked test-set prediction and zip generation
  sanity_checks.py         leakage and matrix assertions
  run_dir.py               timestamped run folders and manifests
  logger.py                stdout plus rotating file log
```
