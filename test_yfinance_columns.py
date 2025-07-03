#!/usr/bin/env python3

import yfinance as yf
import pandas as pd

def test_yfinance_columns():
    """Test yfinance column structure to fix the cross-asset data issue"""
    print("Testing yfinance column structure...")
    
    try:
        spy = yf.download('SPY', start='2021-01-01', end='2021-01-05', progress=False)
        qqq = yf.download('QQQ', start='2021-01-01', end='2021-01-05', progress=False)
        
        print(f"SPY columns: {spy.columns}")
        print(f"SPY columns type: {type(spy.columns)}")
        print(f"SPY shape: {spy.shape}")
        
        if hasattr(spy.columns, 'levels'):
            print(f"SPY MultiIndex levels: {spy.columns.levels}")
            print(f"SPY MultiIndex names: {spy.columns.names}")
        
        print(f"\nQQQ columns: {qqq.columns}")
        print(f"QQQ columns type: {type(qqq.columns)}")
        print(f"QQQ shape: {qqq.shape}")
        
        spy_reset = spy.reset_index()
        qqq_reset = qqq.reset_index()
        
        print(f"\nAfter reset_index:")
        print(f"SPY columns: {spy_reset.columns}")
        print(f"QQQ columns: {qqq_reset.columns}")
        
        if isinstance(spy_reset.columns, pd.MultiIndex):
            print("SPY has MultiIndex columns")
            spy_cols = ['date'] + [f'spy_{col[1].lower()}' if isinstance(col, tuple) else f'spy_{str(col).lower()}' for col in spy_reset.columns[1:]]
        else:
            print("SPY has regular columns")
            spy_cols = ['date'] + [f'spy_{str(col).lower()}' for col in spy_reset.columns[1:]]
        
        print(f"Proposed SPY column names: {spy_cols}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_yfinance_columns()
