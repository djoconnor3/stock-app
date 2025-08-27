import streamlit as st
import pandas as pd

def display_regular_signals(chart_df: pd.DataFrame):
    """
    Displays trading signals based on traditional indicators like SMA, RSI, and MACD.

    Args:
        chart_df (pd.DataFrame): The DataFrame containing price and technical indicator data.
    """
    if len(chart_df) > 1:
        latest_price_data = chart_df.iloc[-1]
        
        st.subheader("Traditional Trading Signals")
        
        # SMA Crossover Signal (Only show if data is available)
        if 'SMA_50' in chart_df.columns and 'SMA_200' in chart_df.columns:
            latest_50_sma = latest_price_data['SMA_50']
            latest_200_sma = latest_price_data['SMA_200']
            if latest_50_sma > latest_200_sma:
                st.success("✅ **SMA BUY SIGNAL** - The 50-day SMA is above the 200-day SMA, indicating potential upward momentum.")
            else:
                st.error("❌ **SMA SELL SIGNAL** - The 50-day SMA is below the 200-day SMA, indicating potential downward momentum.")
        else:
            st.info("ℹ️ **SMA Signal Not Available** - SMA data is missing.")

        # RSI Signal (Only show if data is available)
        if 'RSI' in chart_df.columns:
            latest_rsi = latest_price_data['RSI']
            if latest_rsi < 30:
                st.success("✅ **RSI BUY SIGNAL** - The RSI is below 30, indicating the stock may be oversold.")
            elif latest_rsi > 70:
                st.error("❌ **RSI SELL SIGNAL** - The RSI is above 70, indicating the stock may be overbought.")
            else:
                st.info("ℹ️ **RSI NEUTRAL** - The RSI is between 30 and 70, with no strong signal.")
            
        # MACD Crossover Signal (Only show if data is available)
        if 'MACD' in chart_df.columns and 'MACD_Signal' in chart_df.columns:
            latest_macd = latest_price_data['MACD']
            latest_macd_signal = latest_price_data['MACD_Signal']
            if latest_macd > latest_macd_signal:
                st.success("✅ **MACD BUY SIGNAL** - The MACD line is above the signal line, suggesting potential bullish momentum.")
            else:
                st.error("❌ **MACD SELL SIGNAL** - The MACD line is below the signal line, suggesting potential bearish momentum.")
