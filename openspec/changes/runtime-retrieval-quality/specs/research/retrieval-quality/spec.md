## Purpose

Provides production-grounded retrieval-quality assessment for the runtime research-agent batches: it quantifies each route's (Dense vs BM25) contribution and rescue of actually-cited evidence, and evaluates RRF weight choices over real queries, so that hybrid-retrieval value and weight tuning rest on real query distributions rather than template eval sets.

## ADDED Requirements

### Requirement: Batch derives retrieval-quality metrics from run traces
The runtime batch evaluation SHALL derive retrieval-quality metrics for each question and for the batch from the run's persisted evidence traces—using the single-route ranks (`bm25_rank`, `dense_rank`) of evidence actually cited by the agent as a weak gold signal. Metrics SHALL cover route contribution (share where Dense vs BM25 ranks cited evidence better), single-route rescue rate (cited evidence that only one route would surface within production depth), and two-route agreement (cited evidence both routes surface vs evidence both miss outside production depth).

#### Scenario: treat a route as excluding a cited piece below depth
- **WHEN** a cited evidence has `bm25_rank` beyond production depth and `dense_rank` within production depth
- **THEN** that evidence is counted under `rescue_by_dense` (BM25 alone would have excluded it) and not under `rescue_by_bm25`

#### Scenario: evidence both routes rank beyond weight-sweep depth
- **WHEN** a cited evidence has both `bm25_rank` and `dense_rank` beyond production depth yet was surfaced in the fused pool
- **THEN** it is counted under "fusion-only, both single routes outside depth" and reported separately, not claimed as a Dense or BM25 rescue

### Requirement: Retrieval-quality results are persisted into batch artifacts
The batch evaluation SHALL persist per-question and batch-level retrieval-quality results into the batch's existing artifacts (`q*.json` and `summary.md`) as additive fields, without altering or removing any existing field or outcome classification. A question with no usable evidence (failed run, or no cited evidence with single-route ranks) SHALL be excluded from retrieval-quality statistics, not counted as zero.

#### Scenario: quality block appears in the batch summary
- **WHEN** a batch completes
- **THEN** its summary contains a retrieval-quality aggregation with route-contribution, rescue, and agreement numbers, and all previously existing summary rows are unchanged

#### Scenario: failed question contributes nothing to quality
- **WHEN** a question outcome is `failed` or its evidence carries no usable single-route ranks
- **THEN** that question is omitted from retrieval-quality aggregation rather than scored as all-zero coverage

### Requirement: Offline weight sweep evaluates RRF weights over real queries
A read-only offline analysis SHALL evaluate RRF weight choices over a batch's real queries: it SHALL perform a single retrieval pass (Dense full candidates and BM25 full candidates, with no repeated embedding), scan a set of `dense_weight`/`bm25_weight` combinations, and report for each weight combination the weak-gold hit rate at production depth. The sweep SHALL NOT change production weights or any retrieval configuration; it produces decision evidence only.

#### Scenario: a single retrieval pass serves all weight combinations
- **WHEN** a sweep evaluates N weight combinations over a query set
- **THEN** each query is retrieved once per route and each combination is derived from the same candidates without re-embedding

#### Scenario: sweep output does not mutate configuration
- **WHEN** a weight sweep runs
- **THEN** retrieval options, indexes, and stored runs are unchanged, and its output is reported separately from any applied configuration

#### Scenario: unanswerable or unqualified queries are excluded
- **WHEN** a real query has no cited evidence within production depth
- **THEN** it is excluded from weight-sweep hit-rate statistics rather than counted as a miss