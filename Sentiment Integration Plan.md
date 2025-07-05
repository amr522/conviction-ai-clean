# Twitter/X Sentiment Integration Plan

This document outlines the comprehensive 4-phase plan for integrating Twitter/X sentiment analysis into the conviction-ai trading system.

## Overview

The Twitter sentiment integration will extend the existing news sentiment capabilities by adding real-time social media sentiment analysis. This will provide additional alpha signals for the trading models through social sentiment aggregation at multiple timeframes.

**Current Baseline**: AAPL AUC = 0.9989 (target uplift ≥ +0.02)
**Existing Infrastructure**: `news_sentiment` and `news_volume` columns in feature schema

## Phase 1: Infrastructure & Data Ingestion

### Phase 1-A: Secrets Management → AWS Secrets Manager
**Objective**: Migrate Twitter API credentials to AWS Secrets Manager for secure access

**Implementation**:
- Move Twitter API keys from environment variables to AWS Secrets Manager
- Create `utils/twitter_secrets_manager.py` utility for secure credential retrieval
- Follow existing patterns from `aws_direct_access.py` (lines 23-34)
- Secrets to migrate:
  - `TWITTER_BEARER_TOKEN`
  - `TWITTER_API_KEY` 
  - `TWITTER_API_SECRET`
  - `TWITTER_ACCESS_TOKEN`
  - `TWITTER_ACCESS_TOKEN_SECRET`

**Files to create/modify**:
- `utils/twitter_secrets_manager.py` - New utility for secrets retrieval
- Update existing AWS secrets patterns following `aws_direct_access.py`

### Phase 1-B: Twitter Stream Ingestion
**Objective**: Implement async Twitter stream ingestion using Tweepy v2

**Implementation**:
- Create `twitter_stream_ingest.py` with async Tweepy v2 integration
- Stream tweets for target stock symbols ($AAPL, $AMZN, $GOOGL, $TSLA, $META, $MSFT, $NVDA, $JPM, $JNJ, $V, $MA)
- Filter by relevant financial keywords and cashtags
- Store raw tweets in S3 with timestamp and symbol association
- Implement rate limiting and error recovery

**Features**:
- Async streaming for high throughput
- Symbol-based filtering using cashtags
- Keyword filtering for financial relevance
- S3 storage with partitioning by date/symbol
- Graceful error handling and reconnection

**Files to create**:
- `twitter_stream_ingest.py` - Main streaming ingestion script
- `utils/twitter_stream_utils.py` - Helper utilities

## Phase 2: Sentiment Scoring

### Phase 2-A: FinBERT Sentiment Scoring (CPU)
**Objective**: Score tweets using FinBERT model with ONNX Runtime on CPU

**Implementation**:
- Create `score_tweets_finbert.py` for batch sentiment scoring
- Use ONNX Runtime for efficient CPU inference
- Process tweets from S3 storage
- Generate sentiment scores: positive, negative, neutral probabilities
- Store scored results back to S3 with sentiment metadata

**Features**:
- ONNX Runtime optimization for CPU efficiency
- Batch processing for throughput
- Financial domain-specific sentiment (FinBERT)
- Confidence scores and sentiment classification
- S3 integration for input/output

**Files to create**:
- `score_tweets_finbert.py` - FinBERT sentiment scoring
- `models/finbert_onnx/` - ONNX model artifacts directory

### Phase 2-B: FinGPT Batch Scoring (Optional GPU)
**Objective**: Optional GPU-accelerated sentiment scoring using FinGPT on spot instances

**Implementation**:
- Create `score_tweets_fingpt_batch.py` for GPU-accelerated scoring
- Use AWS Spot instances for cost-effective GPU compute
- Implement batch processing with higher throughput
- Fallback to FinBERT if GPU unavailable

**Features**:
- GPU acceleration for large-scale processing
- Spot instance cost optimization
- Advanced financial sentiment understanding
- Batch processing with queue management

**Files to create**:
- `score_tweets_fingpt_batch.py` - FinGPT GPU scoring (optional)
- `utils/spot_instance_manager.py` - Spot instance management

## Phase 3: Feature Engineering

### Phase 3: Extend Intraday Features
**Objective**: Merge Twitter sentiment into existing feature pipeline with multiple timeframe aggregations

**Current Feature Schema**: 67 features including `news_sentiment`, `news_volume`
**Target Schema**: Add `sent_5m`, `sent_10m`, `sent_60m`, `sent_daily` columns

**Implementation**:
- Create `create_intraday_features.py` to include Twitter sentiment
- Aggregate sentiment at multiple timeframes:
  - 5-minute sentiment aggregation (`sent_5m`)
  - 10-minute sentiment aggregation (`sent_10m`)
  - 60-minute sentiment aggregation (`sent_60m`)
  - Daily sentiment aggregation (`sent_daily`)
- Merge with existing OHLCV and news sentiment features
- Update `data/sagemaker/feature_metadata.json` with new columns

**Feature Engineering**:
- Volume-weighted sentiment scores
- Sentiment momentum indicators
- Sentiment volatility measures
- Cross-timeframe sentiment divergence
- Integration with existing `news_sentiment` column

**Files to create/modify**:
- `create_intraday_features.py` - Main feature engineering pipeline
- `utils/sentiment_aggregation.py` - Sentiment aggregation utilities
- `data/sagemaker/feature_metadata.json` - Update feature schema

## Phase 4: Pipeline Integration

### Phase 4: HPO Pipeline Integration
**Objective**: Add TwitterSentimentTask hook and --twitter-sentiment flag to orchestration pipeline

**Implementation**:
- Add `TwitterSentimentTask` class to orchestration pipeline
- Implement `--twitter-sentiment` flag in `scripts/orchestrate_hpo_pipeline.py`
- Create sentiment data validation and quality checks
- Integrate sentiment feature generation into HPO workflow
- Add sentiment-specific monitoring and logging

**Pipeline Integration**:
- Pre-HPO sentiment data validation
- Sentiment feature generation task
- Integration with existing feature pipeline
- Sentiment-aware model training
- Performance monitoring with sentiment metrics

**Files to modify**:
- `scripts/orchestrate_hpo_pipeline.py` - Add --twitter-sentiment flag and TwitterSentimentTask
- Add sentiment validation to existing data quality checks
- Update HPO job configurations to include sentiment features

## Testing & Validation

### Mini-HPO Smoke Test
**Objective**: Validate sentiment integration with AAPL XGBoost smoke test

**Requirements**:
- Run mini-HPO on AAPL with XGBoost algorithm
- Include sentiment features using `--include-sentiment` flag
- Measure AUC uplift ≥ +0.02 vs baseline (0.9989)
- Validate new `sent_*` columns in feature matrix
- Ensure parquet files exist for AAPL with sentiment data

**Test Command**:
```bash
python scripts/orchestrate_hpo_pipeline.py --algorithm xgboost --twitter-sentiment --include-sentiment --dry-run
```

**Success Criteria**:
- AAPL parquet file exists with sentiment features
- New `sent_*` columns present in feature matrix
- AUC uplift ≥ +0.02 achieved vs baseline
- Dashboard tweet-volume widget functional
- All tests and dry-runs pass

## Dependencies

### Python Packages
```bash
pip install tweepy>=4.0          # Twitter API v2 support
pip install onnxruntime          # CPU inference
pip install transformers         # FinBERT model
pip install torch               # FinGPT support (optional)
pip install asyncio            # Async streaming
pip install boto3              # AWS integration
```

### AWS Resources
- AWS Secrets Manager for Twitter API credentials
- S3 buckets for tweet storage and processed sentiment
- Spot instances for GPU processing (optional)
- CloudWatch for monitoring and logging

## Documentation Updates

### Required Documentation
- Update `omar.md` with sentiment integration results
- Update `training.md` with new feature descriptions
- Update `docs/optimal_hyperparameters.md` with sentiment-aware parameters
- Outline phases 5-7 for future sentiment enhancements

## Implementation Timeline

1. **Phase 1**: Infrastructure setup (secrets + ingestion)
2. **Phase 2**: Sentiment scoring implementation  
3. **Phase 3**: Feature engineering integration
4. **Phase 4**: Pipeline integration and testing
5. **Validation**: Mini-HPO smoke test and documentation

## Future Phases (5-7) Preview

- **Phase 5**: Real-time sentiment streaming and alerts
- **Phase 6**: Advanced sentiment features (emotion analysis, entity sentiment)
- **Phase 7**: Multi-source sentiment fusion (Twitter + Reddit + Discord)

## File Structure

```
conviction-ai-clean/
├── Sentiment Integration Plan.md          # This document
├── utils/
│   ├── twitter_secrets_manager.py         # AWS Secrets Manager integration
│   ├── twitter_stream_utils.py            # Twitter streaming utilities
│   ├── sentiment_aggregation.py           # Sentiment aggregation functions
│   └── spot_instance_manager.py           # Spot instance management (optional)
├── twitter_stream_ingest.py               # Main Twitter ingestion
├── score_tweets_finbert.py                # FinBERT sentiment scoring
├── score_tweets_fingpt_batch.py           # FinGPT GPU scoring (optional)
├── create_intraday_features.py            # Feature engineering pipeline
├── models/
│   └── finbert_onnx/                      # FinBERT ONNX model artifacts
├── scripts/
│   └── orchestrate_hpo_pipeline.py        # Modified with --twitter-sentiment
└── data/
    └── sagemaker/
        └── feature_metadata.json          # Updated with sent_* columns
```

## Risk Mitigation

1. **API Rate Limits**: Implement exponential backoff and connection pooling
2. **Data Quality**: Add validation checks for sentiment score ranges and completeness
3. **Cost Management**: Use spot instances and efficient batch processing
4. **Fallback Strategy**: Graceful degradation to news sentiment if Twitter unavailable
5. **Testing**: Comprehensive dry-run mode for all components

## Success Metrics

- ✅ All 4 phases implemented successfully
- ✅ AAPL AUC uplift ≥ +0.02 achieved
- ✅ New `sent_*` columns integrated into feature schema
- ✅ Dashboard tweet-volume widget functional
- ✅ All tests pass without regressions
- ✅ Documentation updated with results and future phases
