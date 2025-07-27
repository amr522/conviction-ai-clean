# ✅ Task: **Audit ONLY** whether Conviction-AI's training dataset is present and usable.
#          Do NOT build any new Parquet files; just report what exists and its basic health.
#
# ────────────────  CONFIG  ────────────────
LOCAL_TRAIN_PATH = "data/Parquet_data/training/train_dataset.parquet"
S3_BUCKET       = "convictionai-data"
S3_KEY          = "conviction-ai/training/train_dataset.parquet"
# ───────────────────────────────────────────

import os, sys, boto3, pyarrow.parquet as pq, pandas as pd
from pathlib import Path
from datetime import datetime

def s3_exists(bucket: str, key: str) -> bool:
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False

def audit_local(path: str):
    p = Path(path)
    if not p.exists():
        print(f"❌ Local file **NOT** found: {p}")
        return False
    print(f"✅ Found **local** training Parquet: {p}")
    tbl = pq.read_table(p, columns=["timestamp", "symbol"])
    df_info = tbl.to_pandas()  # only two cols so it's light
    row_cnt = len(df_info)
    start, end = df_info["timestamp"].min(), df_info["timestamp"].max()
    print(f"   • Rows          : {row_cnt:,}")
    print(f"   • Date span     : {start}  →  {end}")
    return True

def audit_s3(bucket: str, key: str):
    if not s3_exists(bucket, key):
        print(f"❌ S3 object **NOT** found: s3://{bucket}/{key}")
        return False
    print(f"✅ Found **S3** training Parquet: s3://{bucket}/{key}")
    # lightweight metadata (size + last-modified)
    s3 = boto3.client("s3")
    obj = s3.head_object(Bucket=bucket, Key=key)
    size_mb = obj["ContentLength"] / 1_048_576
    mod_time = obj["LastModified"]
    print(f"   • Size          : {size_mb:,.2f} MB")
    print(f"   • Last modified : {mod_time}")
    return True

def main():
    print("🔍 Auditing Conviction-AI training dataset readiness …")
    local_ok = audit_local(LOCAL_TRAIN_PATH)
    s3_ok    = audit_s3(S3_BUCKET, S3_KEY)
    if not (local_ok or s3_ok):
        print("\n⚠️  Training dataset is missing in **both** locations.")
        print("   Next step: decide whether to trigger full feature build.")
        sys.exit(1)
    print("\n🎯 Audit complete. Review stats above and decide next action.")

if __name__ == "__main__":
    main()