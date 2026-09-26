# Two-Stage News Recommendation on EB-NeRD and MIND

**Course**: Information Retrieval & Extraction (IRE CS4.406) — Assignment 2

**Authors**: Khooshi Asmi 2022114006, Navya Shrivastava 2023114019 

**Date**: September 2026
---

## 1. Introduction and Task Formulation

Reading behaviour is dominated by events. A user's interests move with the news cycle,
an article's click probability collapses within 24 to 48 hours of publication, and much
of any given impression consists of cold or nearly-expired items. Any pipeline that
treats interaction data as static will leak, and any pipeline that leaks will look good
on the leaderboard while failing in production.

Let $\mathcal{U}$ be users and $\mathcal{A}$ the article corpus. An impression is
$i = (u, t_i, \mathcal{H}_u, \mathcal{C}_i)$ where $u \in \mathcal{U}$ is the active
user, $t_i$ the impression timestamp,
$\mathcal{H}_u = \{(a_1, t_1), \dots, (a_m, t_m)\}$ the click history with
$t_k < t_i$, $\mathcal{C}_i = \{c_1, \dots, c_n\} \subset \mathcal{A}$ the displayed
candidates, and $\mathbf{y}_i \in \{0,1\}^n$ the click vector.

We learn a ranking function $f(u, c_j, \mathcal{H}_u \mid t_i) \to \mathbb{R}$ that
orders $\mathcal{C}_i$ to maximise ranking accuracy while keeping the slate
diverse. The strict inequality $t_k < t_i$ is the whole problem. Everything in §3.4
and §5.4 exists to enforce it.

---

## 2. System Architecture

```
              [ User history H_u, impression time t_i ]
                              |
             +----------------+----------------+
             v                                 v
   BM25 inverted index              MiniLM dense retrieval
   k1=1.5, b=0.75                   384-d, unit-normalised
             |                                 |
             +----------------+----------------+
                              v
                  RRF fusion: 1/(60+r_bm25) + 1/(60+r_dense)
                              |
                              v
                 Top-K pool (K=200)  --  or the inview slate on test
                              |
                              v
              28-feature matrix, float32
              behaviour window applied first
                              |
             +----------------+----------------+
             v                                 v
    LightGBM LambdaRank                 PyTorch MLP
    listwise, early stopping            3 hidden layers, weighted BCE
```

### 2.1 Stage 1

**BM25** builds an inverted index over titles and abstracts. The user query is the
concatenation of titles from recent history, scored with $k_1 = 1.5$, $b = 0.75$.

**Dense retrieval** encodes every article with `all-MiniLM-L6-v2` into 384 dimensions,
unit-normalised, with the user represented as the mean of their history embeddings.
Scoring goes through a NumPy mirror of the embedding matrix rather than per-impression
torch kernels. That is not a micro-optimisation: at 13.5M test rows the kernel launches
dominated everything else, and switching the hot path cut the full EB-NeRD prediction
run from 17.06 h to 3.17 h at 0.6 to 1.0 GB peak RSS. A FAISS `IndexFlatIP` is built
alongside for global retrieval.

> A 128-dimensional TF-IDF plus TruncatedSVD backend is retained for comparison with
> Assignment 1, but it is not the default. Note that the benchmark tables in §5.1 label
> this arm "Dense SVD", a leftover from when that was the active backend; the figures
> are MiniLM.

**Hybrid fusion** combines the two rank lists, which are not on a common scale, with
reciprocal rank fusion at $k = 60$:
$$\text{RRF}(c) = \frac{1}{60 + r_{\text{bm25}}(c)} + \frac{1}{60 + r_{\text{dense}}(c)}$$

### 2.2 Stage 2

LightGBM LambdaRank optimises pairwise gradients weighted by $\Delta$NDCG$, grouped by
impression. The MLP is $28 \to 256 \to 128 \to 64 \to 1$ with BatchNorm, ReLU and
dropout $0.3 / 0.2 / 0.0$, trained on class-imbalance-weighted BCE:
$$\mathcal{L} = -\left[w_{\text{pos}}\, y \log \sigma(\hat{y}) + (1-y)\log(1 - \sigma(\hat{y}))\right], \quad w_{\text{pos}} = \frac{N_{\text{neg}}}{N_{\text{pos}}}$$

---

## 3. Feature Specification

Twenty behavioural and lexical columns, then eight within-session.

| # | Feature | Definition |
|---|:--------|:-----------|
| 1 | `bm25_score` | $\text{BM25}(\text{Query}(\mathcal{H}_u), c_j)$ |
| 2 | `dense_score` | $\cos(\mathbf{e}_u, \mathbf{e}_{c_j})$ |
| 3 | `hybrid_score` | RRF fusion of the two |
| 4 | `category_match` | $\mathbb{I}(\text{cat}(c_j) = \text{mode}(\text{cats}(\mathcal{H}_u)))$ |
| 5 | `category_affinity` | $\frac{\text{count}(\text{cat}(c_j) \in \mathcal{H}_u)}{\lvert \mathcal{H}_u \rvert}$ |
| 6 | `history_len` | $\min(\lvert\mathcal{H}_u\rvert, 50)$ |
| 7 | `recency_score` | $\sum_{h} \exp(-0.1\,\Delta t_{\text{hours}})$ — the Q1.1 decay |
| 8 | `avg_read_time_hist` | mean historical read time |
| 9 | `avg_scroll_hist` | mean historical scroll depth |
| 10 | `click_overlap` | $\mathbb{I}(c_j \in \mathcal{H}_u)$ |
| 11 | `freshness_days` | $\text{clip}\big(\frac{t_i - \text{pub}(c_j)}{86400}, 0, 365\big)$ |
| 12 | `freshness_log` | $\log(1 + \text{freshness\_days})$ |
| 13 | `popularity_log` | $\log(1 + \text{pageviews}(c_j))$ |
| 14 | `inview_rate` | $\text{inviews} / (\text{pageviews} + 1)$ |
| 15 | `readtime_rate` | $\text{read\_time} / (\text{inviews} + 1)$ |
| 16 | `sentiment_score` | editorial sentiment; $0$ on MIND |
| 17 | `position_in_impression` | 0-indexed slate position |
| 18 | `impression_size` | $\lvert\mathcal{C}_i\rvert$ |
| 19 | `is_subscriber` | subscription tier |
| 20 | `user_cat_entropy` | $-\sum p(c)\log_2 p(c)$ over history categories |

Features 21 to 28 describe what the same session did earlier, over a strict prefix of
impressions ordered by $(\text{impression\_time}, \text{impression\_id})$:
`session_impression_index`, `session_prior_impressions`, `session_prior_clicks`,
`session_prior_ctr`, `session_prior_read_time`, `session_prior_scroll`,
`session_candidate_overlap`, `session_prior_cat_match`.

EB-NeRD ships `session_id`; MIND does not, so all eight are identically zero there.
EB-NeRD sessions average 1.93 impressions with a maximum of 24, so roughly 44% are
single-impression and contribute an empty prefix.

The session context is built in two passes on purpose. The EB-NeRD test file is not
grouped by session — 200k rows contain 98,704 sessions across 102,329 contiguous runs
— and time is not ascending within a session in 66,783 of those 98,704. A single
streaming fold would read impressions out of order and fold future context backwards, so
the builder sorts explicitly and emits each impression's context before folding that
impression into the running state.

### 3.4 The behaviour window

`apply_behaviour_window()` filters every history event with timestamp greater than the
impression time, and it runs *before* Stage 1 builds its retrieval query. That ordering
is the point. Filtering only at feature-extraction time still lets the candidate set be
conditioned on future clicks, which is leakage by a different route.

The check is counterfactual invariance: a feature vector computed at time $t$ must be
bit-identical whether or not clicks after $t$ exist in the raw record. Disabling the
guard makes five of the anti-gaming tests fail, so the tests are not vacuous.

At the split level, `temporal_guard.py` verifies the time envelopes. EB-NeRD passes
6 of 6:

| Split | Rows | Envelope |
|:--|--:|:--|
| train | 232,887 | 2023-05-18 07:00:01 → 2023-05-25 06:59:58 |
| val | 244,647 | 2023-05-25 07:00:02 → 2023-06-01 06:59:59 |
| test | 13,536,710 | 2023-06-01 07:00:00 → 2023-06-08 06:59:59 |

`max(train) ≤ min(val)` and `max(val) ≤ min(test)` both hold. On MIND the boundary is
partly unenforceable: the dataset ships impression times but no per-click timestamps,
so the per-event filter has nothing to filter on. We say so rather than claiming
coverage we do not have.

---

## 4. Experimental Setup

**EB-NeRD small**: 20,738 articles, Danish, with continuous read times, scroll
percentages and inview counts. **MIND small**: 42,416 articles, English, with category
and subcategory tags.

Metrics are AUC (per impression), MRR, nDCG@5 and nDCG@10, plus three beyond-accuracy
measures. Intra-list diversity is the mean pairwise category disagreement within the
top 5. Novelty is mean $-\log_2 P(r_j)$ under global interaction frequency. Coverage is
the fraction of the catalogue appearing in any top-5 list.

Slicing is four-way: cold-start users ($\lvert\mathcal{H}_u\rvert \le 5$) against warm,
and head against tail slates split at median catalogue popularity. All headline
intervals use $B = 100$ bootstrap resamples; the ablation in §5.3 uses $B = 1{,}000$
and the serving ablation in §5.4 uses $B = 300$ on 400 impressions.

The benchmark figures come from two runs produced ten minutes apart by the same code
state, differing only in the dataset argument, so the datasets are directly comparable.
The ablation is a third run at the same 5,000 impressions and 28-feature schema, and its
final arm reproduces the benchmark figures exactly, which is what lets us treat them as
one set of evidence.

---

## 5. Results

### 5.1 Benchmark

5,000 impressions per dataset, 100 bootstrap resamples, 28 features.

| Dataset | Model | AUC | MRR | nDCG@5 | nDCG@10 | ILD@5 | Novelty@5 | Cov@5 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| **EB-NeRD** | BM25 | 0.4986 ± 0.0084 | 0.3165 | 0.3461 | 0.4331 | 0.6857 | 16.9417 | 0.0773 |
| **EB-NeRD** | Dense (MiniLM) | 0.5041 ± 0.0089 | 0.3222 | 0.3530 | 0.4354 | 0.6855 | 16.9885 | 0.0733 |
| **EB-NeRD** | Hybrid RRF | 0.4997 ± 0.0085 | 0.3204 | 0.3531 | 0.4334 | 0.6873 | 16.9362 | 0.0764 |
| **EB-NeRD** | NRMS (official) | 0.5297 ± 0.0084 | 0.3401 | 0.3741 | 0.4532 | 0.6866 | 17.0357 | 0.0759 |
| **EB-NeRD** | **LightGBM** | **0.7347 ± 0.0077** | **0.5049** | **0.5697** | **0.6092** | 0.6630 | 17.3183 | 0.0664 |
| **EB-NeRD** | PyTorch MLP | 0.6432 ± 0.0075 | 0.4106 | 0.4598 | 0.5228 | 0.6569 | 17.2323 | 0.0665 |
| **EB-NeRD** | Q3 improved | 0.5988 ± 0.0071 | 0.3861 | 0.4268 | 0.4991 | 0.4994 | 17.0916 | 0.0729 |
| **MIND** | BM25 | 0.5549 ± 0.0080 | 0.2936 | 0.2711 | 0.3325 | 0.7589 | 17.2125 | 0.0189 |
| **MIND** | Dense (MiniLM) | 0.6429 ± 0.0079 | 0.3543 | 0.3356 | 0.3974 | 0.7083 | 17.1912 | 0.0177 |
| **MIND** | Hybrid RRF | 0.6129 ± 0.0075 | 0.3350 | 0.3108 | 0.3744 | 0.7216 | 17.1921 | 0.0186 |
| **MIND** | NRMS (official) | 0.5107 ± 0.0070 | 0.2571 | 0.2368 | 0.2976 | 0.7912 | 17.2883 | 0.0141 |
| **MIND** | LightGBM | 0.6394 ± 0.0078 | 0.3559 | 0.3368 | 0.3997 | 0.6495 | 17.2350 | 0.0190 |
| **MIND** | **PyTorch MLP** | **0.6491 ± 0.0086** | **0.3656** | **0.3472** | **0.4066** | 0.6308 | 17.2222 | 0.0190 |
| **MIND** | Q3 improved | 0.6058 ± 0.0090 | 0.3090 | 0.2920 | 0.3591 | 0.4727 | 17.1843 | 0.0167 |

The best re-ranker differs by dataset. LightGBM wins EB-NeRD by a wide margin, 0.7347
against 0.6432. The MLP edges MIND, 0.6491 against 0.6394, but that gap is not
statistically resolved — the intervals overlap almost entirely. On MIND the defensible
claim is that GBDT and MLP are equivalent, not that the MLP is better. Neither beats
the dense retriever by a resolved margin on MIND.

The Q3 improvement beats the official NRMS baseline on both datasets, by 0.0691 AUC on
EB-NeRD and 0.0951 on MIND, but loses to both learned re-rankers. Its real advantage is
diversity: ILD@5 of 0.4994 and 0.4727 against 0.6630 and 0.6495, at essentially equal
novelty. It is an interpretable ranker, and the GBDT subsumes its accuracy while giving
up some of its spread.


### 5.2 Q3: baseline, improvement, ablation

The official baseline follows `ebnerd-benchmark/examples/baseline/ebnerd_feat_baselines.py`,
which emits four separate unweighted popularity scores — pageviews, inviews,
inview-frequency, readtime — with no temporal term. We combine them into a single score
for use as an ablation base:
$$S_{\text{base}}(a) = 0.5\,\text{norm}(\text{pageviews}) + 0.3\,\text{norm}(\text{inviews}) + 0.2\,\text{norm}(\text{readtime})$$
That blend is our construction, not a reproduction of the official one. The official
NRMS baseline is reimplemented in PyTorch from the reference, with masking and
`padding_idx` deviations corrected to match ebrec's actual behaviour; twenty tests pin
the correspondence, including reading ebrec's `nrms.py` to assert it is mask-free.

The improvement adds three signals to the baseline:
$$\text{Affinity}(u,a) = \frac{\sum_{h}\mathbb{I}(\text{cat}(h) = \text{cat}(a))}{\lvert\mathcal{H}_u\rvert + \epsilon}, \quad \text{Recency}(a) = \sum_j \mathbb{I}(\cdot)\,e^{-\lambda j}, \quad \text{Freshness}(a) = e^{-\gamma \Delta_{\text{days}}}$$
$$S_{\text{improved}} = 0.30\,S_{\text{base}} + 0.45\,\text{Affinity} + 0.25\,\text{Recency} + 0.10\,\text{Freshness}$$

Ablation at 5,000 impressions, 1,000 resamples, paired on the per-impression
difference array:

| Variant | EB-NeRD Δ AUC [95% CI] | | MIND Δ AUC [95% CI] | |
|:--|:--|:--|:--|:--|
| M1 category affinity | +0.0149 [+0.0044, +0.0255] | yes | +0.1076 [+0.0971, +0.1173] | yes |
| M2 category recency | +0.0036 [−0.0068, +0.0149] | partial | +0.0933 [+0.0828, +0.1030] | yes |
| **M3 full** | **+0.0323 [+0.0217, +0.0433]** | **yes** | **+0.1054 [+0.0946, +0.1156]** | **yes** |

Five of six intervals exclude zero. Category affinity is the load-bearing ingredient on
both datasets. M2 is the one unresolved arm, and only on EB-NeRD.

M3 is not additive. On MIND it scores 0.6058 against M1's 0.6080, with fully
overlapping intervals, because `full` moves 0.25 of weight from affinity onto recency
and recency is the weaker MIND signal. Nearly all of the MIND improvement is category
affinity.

The paired test is against the popularity M0 that the improvement builds on, not
against NRMS. The 0.0691 and 0.0951 gaps in §5.1 are differences of two independent
intervals and should not be read as paired.


---

## 6. Slices

**Cold-start against warm.** The pattern holds on both datasets. MIND MLP scores AUC
0.5989 on cold-start users against 0.6598 on warm ones; EB-NeRD LightGBM scores 0.7309
against 0.7348. The EB-NeRD cold-start slice is small enough that its intervals are wide
(±0.0838 for LightGBM, ±0.1587 for BM25), so those figures should be read as
indicative rather than precise. Cold-start users also get more diverse slates
throughout, which is the expected trade: with little history to go on, the ranker falls
back on priors that are broad rather than sharp.

**Head against tail is counter-intuitive, and consistently so.** Tail slates score
roughly 2.3 to 2.7 times higher than head slates on nDCG@5. MIND BM25 gets 0.1670 on
head and 0.4572 on tail; MIND MLP gets 0.2378 against 0.5428. AUC, by contrast, barely
moves, so this is a within-slate ordering effect rather than a difference in how well
the ranker separates clicks from non-clicks.

Our reading is that a head slate is a set of high-popularity candidates that are
genuinely hard to separate from one another — popularity is a weak discriminator
precisely because they are all popular — whereas in a tail slate the clicked article
tends to stand out on content. We would want to test that against a popularity-matched
control before treating it as settled.

---

## 7. Serving and Scale

Measured on Apple MPS over 100 impression trials, EB-NeRD:

| Stage | p50 | p95 | p99 | QPS |
|:--|--:|--:|--:|--:|
| BM25 retrieval | 0.21 ms | 0.56 ms | 0.65 ms | 3,973.0 |
| Dense retrieval | 1.60 ms | 6.17 ms | 20.56 ms | 281.6 |
| Hybrid RRF | 1.67 ms | 2.12 ms | 2.22 ms | 573.6 |
| Feature extraction (28) | 0.43 ms | 0.82 ms | 0.89 ms | 2,126.0 |
| LightGBM scoring | 0.63 ms | 0.73 ms | 0.84 ms | 1,590.1 |
| MLP scoring | 1.84 ms | 22.25 ms | 47.39 ms | 118.8 |
| **End-to-end (LGBM)** | **1.07 ms** | **1.51 ms** | **1.68 ms** | **909.7** |
| End-to-end (MLP) | 2.24 ms | 22.68 ms | 48.01 ms | 112.5 |

Memory totals 40.84 MB, of which the BM25 index is 21.30 MB, dense embeddings 10.13 MB,
the article corpus 6.72 MB, the LightGBM model 2.50 MB and the MLP 0.19 MB.

### 7.1 Cost at a stated SLA

Fleet sizing uses Little's Law with 1.30x headroom, priced from on-demand us-east-1
list rates (AWS `c7i.4xlarge` \$0.678/hr, NLB \$0.0225/hr, ElastiCache Redis
`r6g.large` \$0.156/hr, OpenSearch 2×`r6g.large` \$0.396/hr) at 2x replicas for N+1.

| Parameter | Value |
|:--|:--|
| Target SLA | p99 < 100 ms |
| Volume | 10,000,000 queries/day |
| Measured cascade | p50 1.07 ms, p99 1.68 ms (p99/p50 = 1.6x) |
| Meets SLA | yes |
| Workers | 1, at 9 for 10x traffic |
| Fleet cost | \$2.5050/hour |
| **Cost per 1,000 queries** | **\$0.0060** |
| At 10x traffic | \$0.0032 |

**Why the GBDT and not the MLP.** Tail latency is a property of the per-request
distribution, not of load, so replicas buy throughput and never shorten an individual
request's p99. The MLP path measures p99 48.01 ms against a 2.24 ms p50, a ratio of
21x. Had that exceeded the SLA, no fleet size would fix it, and the cost model says so
explicitly and attributes the remedy to a structurally cheaper re-ranker.

### 7.2 What breaks at 10x

Dense retrieval degrades first. Flat dot products over a 400K catalogue exceed 15 ms,
so the fix is architectural rather than volumetric: FAISS HNSW or ScaNN IVF-PQ holds
sub-2 ms lookup. BM25 postings scale linearly to roughly 213 MB and want term-based
sharding with WAND early termination. The stateless tier is CPU-bound on feature
extraction at this catalogue size, not memory-bound.

The more useful observation is what actually broke first in our own pipeline, which was
not retrieval or ranking. The 13.5M-row EB-NeRD test set stalled on history
materialisation: test history averages 144.6 clicks per user across 807,677 users, or
116.8M entries, and turning those into Python strings took minutes and several GB before
a single impression was scored. Vectorising the `list.tail(50)` truncation and the
`List(Int32) → Utf8` cast, together with the NumPy dense path, took the run from 17.06 h
to 3.17 h. At that row count the binding constraint is per-row dispatch overhead, so
the first thing to optimise is any per-row kernel launch or string materialisation, not
model capacity.

---

## 8. Conclusion

The system meets its serving target comfortably, at \$0.0060 per 1,000 queries against
a p99 < 100 ms SLA, and the re-rankers add roughly 0.10 to 0.20 AUC over the official
NRMS baseline on EB-NeRD. Two results matter more than those gains.

Roughly a quarter of the model's measured gain comes from columns that do not exist at
serving time, and the intervals are too wide to call the effect significant. Global
catalogue retrieval is the wrong candidate generator for editorially curated slates, at
Recall@200 ≈ 0.03. Both were visible only because the ablations were built to be able
to fail, and both changed the design rather than the write-up.

A third point is more mundane and probably more useful to anyone extending this work.
The strongest single improvement we made was not modelling at all: unifying the
retrieval index across splits closed a train/serve distribution shift that had the
re-ranker scoring below one of its own input features. The fingerprint check added
alongside it treats a checkpoint whose recorded schema is incomplete as a mismatch
rather than a pass, because a partial record is exactly what let the stale model load
in the first place.

Where we would go next, in order: measure the MIND cascade so the negative result is
confirmed on both datasets rather than one; replace the period-aggregate popularity
columns with streaming counters a live system could actually maintain, which would
close most of the §5.4 gap at the source; and run online exploration for cold-article
discovery, where the cold-start slice in §6 is weakest.

**Known gaps.** The MIND cascade is unmeasured. Serving-ablation intervals are wide at
400 impressions. Bootstrap counts are 100 for the benchmark and 1,000 for the ablation.
Leaderboard screenshots and the AI-usage log are separate deliverables and are not
included here.

---

## References

1. Kruse, J., Lindskow, K., Kalloori, S., Polignano, M., Pomo, C., Srivastava, A.,
   Uppal, A., Riis Andersen, M., Frellsen, J. *EB-NeRD: A Large-Scale Dataset for News
   Recommendation.* Proc. ACM RecSys Challenge 2024.
2. Wu, C., Wu, F., Ge, S., Qi, T., Huang, Y., Xie, X. *Neural News Recommendation with
   Multi-Head Self-Attention.* EMNLP-IJCNLP 2019, 6389–6394.
3. Wu, F., Qiao, Y., Chen, J., Wu, C., Qi, T., Lian, J., Liu, D., Xie, X., Gao, J.,
   Wu, B., Zhou, M. *MIND: A Large-Scale Dataset for News Recommendation.* Proc. ACL
   2020, 3597–3606.
4. Robertson, S., Zaragoza, H. *The Probabilistic Relevance Framework: BM25 and Beyond.*
   Foundations and Trends in Information Retrieval, 2009.
7. Cormack, G., Clarke, C., Buettcher, S. *Reciprocal Rank Fusion Outperforms Condorcet
   and Individual Rank Learning Methods.* SIGIR 2009.
