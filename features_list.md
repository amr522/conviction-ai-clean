# Macro/VIX/DXY Features
vix_value
vix_ma_divergence
dxy_value
iv_rank_30d

# News Features (NOW AVAILABLE)
news_count_lag1
avg_sentiment_lag1

# BLOCKED - no SPX GEX feed
# gex_spx_lag1
# Options Daily Features
optd_iv30
optd_hv30
optd_iv30_lag1
optd_hv30_lag1
optd_iv_percentile
optd_iv_percentile_lag1
optd_vrp_30d
optd_vrp_30d_lag1
optd_vrp_spike
optd_iv_skew_slope
optd_vol_surprise
optd_put_call_ratio
optd_volume
# Options 30-Minute Features
opt30_mid_price_return
opt30_bid_ask_spread
opt30_implied_volatility
opt30_delta
opt30_theta
opt30_volume_return
opt30_rolling_vol_5
opt30_flow_divergence
opt30_gamma_squeeze
# Stocks Daily Features
stockd_close
stockd_volume
stockd_return_1d
stockd_return_1d_lag1
stockd_vol_7d
stockd_vol_7d_lag1
stockd_volume_pct_change
stockd_beta_spy
stockd_days_to_earnings
stockd_earnings_flag
# Stocks 30-Minute Features
stock30_close_return
stock30_rolling_vol_5
stock30_is_last_30min

# Sector Features
sector
is_tech
is_financial
is_energy
market_beta_20d

# Market Direction Features
spy_return_1d
qqq_return_1d
spy_volume
qqq_volume

# Commodity Reference Features
gold_return_1d
oil_return_1d
gold_volatility_10d

# Training Universe: 36 Stocks
# Tech: AAPL, GOOGL, META, MSFT, NFLX, SMCI
# Tech (High IV): AMD, NVDA, PLTR
# Financial: BAC, GS, JPM, MA, MS, V
# Healthcare: ABBV, JNJ, MRK, PFE, UNH
# Consumer: DIS, NKE, SBUX, WMT, AMZN
# Energy: CVX, XOM
# Industrial: BA, CAT, GE
# Crypto/Fintech (High IV): COIN, HOOD, MSTR
# Auto (High IV): TSLA
# ETF Options Targets: QQQ, SPY
