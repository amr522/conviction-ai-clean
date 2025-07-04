# Twitter & Alternative Sentiment Integration Plan

> **Purpose** — add high‑quality, leak‑proof sentiment features (5 min / 10 min / 60 min and daily) to the 46‑stock pipeline using Twitter, X (formerly Twitter), and optional FinGPT scoring while keeping the automation stack self‑healing and cost‑aware.

---

## 0 · When to Tackle This

| Milestone            | Prerequisite                                                                                                 | Target window                                               |
| -------------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| **Current session**  | Endpoint *conviction‑ensemble‑v4* must reach `InService` and leak‑proof retrain cycle finishes successfully. | **Tonight UTC 02‑04** (after existing cron jobs).           |
| **Sentiment sprint** | Create branch `` and run steps below.                                                                        | **Next working session (≈ tomorrow)**                       |
| **Full integration** | Sentiment features validated (AUC ≥ +0.02 uplift)                                                            | Merge & schedule into nightly set‑and‑forget orchestration. |

> **Rationale** — we avoid colliding with the live retrain window; sentiment ingestion can run in parallel once the endpoint is stable.

---

## 1 · Task Breakdown

| Phase   | Description                                                                                                                                 | Owner     | Status |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------ |
| **1‑A** | **Secrets** — store Twitter/X API keys in AWS Secrets Manager; update `aws_utils.get_secret()` usage.                                       | Dev Infra | ☐      |
| **1‑B** | **Ingest** — add `scripts/twitter_stream_ingest.py` (async Tweepy / v2 API) → raw JSON in `s3://…/raw/twitter/{symbol}/YYYY‑MM‑DD.json.gz`. | Data Eng  | ☐      |
| **2‑A** | **FinBERT local scoring** — `score_tweets_finbert.py` (ONNXRuntime CPU) → parquet with `timestamp`, `symbol`, `sent_score`.                 | ML Eng    | ☐      |
| **2‑B** | **FinGPT (optional)** — `score_tweets_fingpt_batch.py` launched on spot `g4dn.xlarge`; cache embeddings in S3.                              | ML Eng    | ☐      |
| **3**   | **Feature merge** — extend `create_intraday_features.py` to join aggregated sentiment (count/mean/std over 5/10/60 min) per symbol.         | FE Team   | ☐      |
| **4**   | **Pipeline hooks** — implement `TwitterSentimentTask` in `orchestrate_hpo_pipeline.py`; enable via `--twitter-sentiment`.                   | MLOps     | ☐      |
| **5**   | **Validation** — run mini‑HPO AAPL smoke test; verify AUC uplift ≥ 0.02.                                                                    | Research  | ☐      |
| **6**   | **Monitoring** — extend `setup_monitoring.py` to plot daily tweet‑volume & mean sentiment; alert on drift.                                  | MLOps     | ☐      |
| **7**   | **Deployment gate** — raise deployment threshold to AUC ≥ 0.60; add rollback logic.                                                         | MLOps     | ☐      |

---

## 2 · Data We Still Need

| Data                                | Source                                                      | Acquisition Script                        | Notes                             |
| ----------------------------------- | ----------------------------------------------------------- | ----------------------------------------- | --------------------------------- |
| Historical tweets (1 year backfill) | Twitter full‑archive search (academic) **or** paid XAI dump | `twitter_archive_backfill.py` (new)       | Pagination + rate‑limit handling. |
| Real‑time tweets (stream)           | Twitter filtered stream API v2                              | `twitter_stream_ingest.py`                | Filter on cashtags `$AAPL`, etc.  |
| FinBERT ONNX model                  | HuggingFace `ProsusAI/finbert`, ONNX export                 | download inside `score_tweets_finbert.py` | \~420 MB once.                    |
| FinGPT weights (optional)           | `FNGPT-small` HF checkpoint                                 | Spot batch script                         | GPU recommend.                    |

---

## 3 · Devin Execution Prompt

```devin_prompt
### Task: Integrate Twitter Sentiment Into 46‑Stock Pipeline

1. Checkout branch `feature/twitter-sentiment`.
2. Implement & test the tasks in *SENTIMENT_INTEGRATION_PLAN.md* phases 1‑4:
   - secrets retrieval, `twitter_stream_ingest.py`, `score_tweets_finbert.py`, feature merge, orchestrator hook.
3. Dry‑run AAPL mini‑HPO (`--algorithm xgb`) with `--include-sentiment` and confirm AUC uplift ≥ 0.02.
4. Push branch + open PR; update `omar.md` & `training.md` with results and next steps (phases 5‑7).
```

> Place this prompt in `` after merging.

---

## 4 · Acceptance Criteria

- Sentiment parquet files available for at least AAPL demo.
- Feature matrix gains new columns: `sent_mean_5m`, `sent_sum_10m`, `sent_pos_ratio_60m`.
- Mini‑HPO AUC improvement recorded in `docs/optimal_hyperparameters.md`.
- Dashboard shows tweet‑volume widget.
- All new code passes `pytest -q` and `scripts/orchestrate_hpo_pipeline.py --dry-run --twitter-sentiment`.

---

Created 2025‑07‑04

