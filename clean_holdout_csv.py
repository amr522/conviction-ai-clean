#!/usr/bin/env python3
"""
clean_holdout_csv.py - Clean the holdout dataset by removing NaN values
"""

import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_PATH = "data/holdout_2023.csv"
OUTPUT_PATH = "data/holdout_2023_clean.csv"
TARGET_COL = "return"

def main():
    try:
        # Load the holdout dataset
        logger.info(f"Loading holdout dataset from {INPUT_PATH}")
        df = pd.read_csv(INPUT_PATH)
        
        logger.info(f"Original dataset shape: {df.shape}")
        
        # Check for NaN values in target column
        nan_count_target = df[TARGET_COL].isna().sum()
        logger.info(f"NaN values in target column '{TARGET_COL}': {nan_count_target}")
        
        # Check for NaN values in entire dataset
        nan_count_total = df.isna().sum().sum()
        logger.info(f"Total NaN values in dataset: {nan_count_total}")
        
        # Remove rows with NaN values in target column
        df_clean = df.dropna(subset=[TARGET_COL])
        logger.info(f"Dataset shape after removing NaN values in target column: {df_clean.shape}")
        
        # Get total number of NaN values remaining
        remaining_nans = df_clean.isna().sum().sum()
        logger.info(f"Remaining NaN values in dataset: {remaining_nans}")
        
        # Replace remaining NaN values with column means
        if remaining_nans > 0:
            logger.info("Replacing remaining NaN values with column means")
            for col in df_clean.columns:
                if df_clean[col].isna().sum() > 0:
                    if df_clean[col].dtype in [np.float64, np.int64]:
                        mean_val = df_clean[col].mean()
                        df_clean[col].fillna(mean_val, inplace=True)
                    else:
                        # For non-numeric columns, use most frequent value
                        mode_val = df_clean[col].mode()[0]
                        df_clean[col].fillna(mode_val, inplace=True)
        
        # Verify no NaN values remain
        final_nan_count = df_clean.isna().sum().sum()
        logger.info(f"Final NaN count: {final_nan_count}")
        
        # Save cleaned dataset
        logger.info(f"Saving cleaned dataset to {OUTPUT_PATH}")
        df_clean.to_csv(OUTPUT_PATH, index=False)
        
        logger.info(f"Cleaned dataset saved successfully with {len(df_clean)} rows and {len(df_clean.columns)} columns")
        
        return True
        
    except Exception as e:
        logger.error(f"Error cleaning holdout dataset: {str(e)}")
        return False

if __name__ == "__main__":
    main()
