# To run this app, use the following command in your terminal:
# pip install scikit-learn pandas streamlit requests
# python3 -m streamlit run main.py --server.port 8502 --server.address 0.0.0.0

import streamlit as st
import pandas as pd
import requests
from datetime import date, timedelta
import numpy as np
import json
import os

# Import the refactored signal and forecast modules
from regular_signals import display_regular_signals
from ai_signals import display_ai_forecasts

# =============================================================================
# Configuration
# =============================================================================

# Set the page title and favicon
st.set_page_config(
    page_title="Alpha Vantage Stock Analyzer",
    page_icon="📈"
)

# --- App Introduction and Title ---
st.title("📈 Alpha Vantage Stock Analyzer")
st.markdown("Enter a stock ticker to see its recent price performance and a buy/sell recommendation based on moving averages, RSI, MACD, and several simple AI forecasts.")

# --- API Key from Secrets ---
try:
    # This is the corrected line to access the nested key
    api_key = st.secrets["general"]["api_key"]
    # We will use this API key for the Gemini API call as well
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
except KeyError:
    st.error("API key not found. Please add it to your Streamlit secrets.")
    api_key = None # Set to None to prevent further execution

# =============================================================================
# AI-Powered Data Key Discovery
# =============================================================================

@st.cache_data(show_spinner=False)
def get_data_keys_with_ai(json_data):
    """
    Uses an LLM to dynamically determine the correct JSON keys for parsing the data.
    This makes the application robust to changes in the Alpha Vantage API format.
    
    Args:
        json_data (dict): The raw JSON data from the Alpha Vantage API.
    
    Returns:
        dict: A dictionary of key names to use for parsing, or None if the AI call fails.
    """
    try:
        # Construct the API payload with grounding and a structured response schema
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"Analyze the following JSON object to identify the data keys for stock metrics. \
                    The main time series data is likely in a key like 'Time Series (Daily)'. \
                    The metrics for SMA, RSI, and MACD will be in keys like 'Technical Analysis: SMA', etc. \
                    The close price is typically a key like '4. close'. \
                    Identify the exact key names for each of these. Return the result as a JSON object with the specified schema.\
                    JSON Data to analyze: {json.dumps(json_data)}"
                }]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "time_series_key": {"type": "STRING"},
                        "close_price_key": {"type": "STRING"},
                        "sma_key": {"type": "STRING"},
                        "rsi_key": {"type": "STRING"},
                        "macd_key": {"type": "STRING"}
                    },
                    "propertyOrdering": ["time_series_key", "close_price_key", "sma_key", "rsi_key", "macd_key"]
                }
            },
        }
        
        # Call the Gemini API to get the structured keys
        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={gemini_api_key}"
        response = requests.post(apiUrl, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if result and 'candidates' in result and len(result['candidates']) > 0:
            json_text = result['candidates'][0]['content']['parts'][0]['text']
            return json.loads(json_text)
        
    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
        st.error(f"An error occurred while using AI to determine data keys: {e}")
        return None

# =============================================================================
# Main Application Logic
# =============================================================================

# --- User Input Section ---
ticker_symbol = st.text_input(
    "Enter a stock ticker (e.g., AAPL, GOOGL, MSFT):",
    "GOOGL" # Default value
).upper()

# --- Data Fetching and Error Handling ---
# Function to fetch data from Alpha Vantage
def get_alpha_vantage_data(api_key, ticker, function, interval="daily", time_period=200):
    """
    Fetches data from the Alpha Vantage API for a specific function.
    
    Args:
        api_key (str): Your Alpha Vantage API key.
        ticker (str): The stock ticker symbol.
        function (str): The API function to call (e.g., TIME_SERIES_DAILY, SMA, RSI, MACD).
        interval (str): The time interval (e.g., 'daily').
        time_period (int): The number of data points for the technical indicator.
    
    Returns:
        dict: The JSON data from the API response.
    """
    url = f"https://www.alphavantage.co/query?function={function}&symbol={ticker}&interval={interval}&time_period={time_period}&series_type=close&apikey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while fetching data: {e}")
        return None

# Check if both API key and ticker symbol are provided
if api_key and ticker_symbol:
    # Use a loading spinner while fetching the data
    with st.spinner(f"Fetching data for {ticker_symbol}..."):
        # Fetch daily price data
        price_data = get_alpha_vantage_data(api_key, ticker_symbol, "TIME_SERIES_DAILY", interval="daily")
        
        # Fetch 50-day SMA data
        sma_50_data = get_alpha_vantage_data(api_key, ticker_symbol, "SMA", interval="daily", time_period=50)
        
        # Fetch 200-day SMA data
        sma_200_data = get_alpha_vantage_data(api_key, ticker_symbol, "SMA", interval="daily", time_period=200)

        # Fetch 14-day RSI data
        rsi_data = get_alpha_vantage_data(api_key, ticker_symbol, "RSI", interval="daily", time_period=14)
        
        # Fetch MACD data
        macd_data = get_alpha_vantage_data(api_key, ticker_symbol, "MACD", interval="daily")

    if price_data and sma_50_data and sma_200_data and rsi_data and macd_data:
        # Check for API errors in the response
        if "Error Message" in price_data:
            st.error(f"Alpha Vantage API error on price data: {price_data['Error Message']}")
        elif "Error Message" in sma_50_data:
            st.error(f"Alpha Vantage API error on 50-day SMA: {sma_50_data['Error Message']}")
        elif "Error Message" in sma_200_data:
            st.error(f"Alpha Vantage API error on 200-day SMA: {sma_200_data['Error Message']}")
        elif "Error Message" in rsi_data:
            st.error(f"Alpha Vantage API error on RSI: {rsi_data['Error Message']}")
        elif "Error Message" in macd_data:
            st.error(f"Alpha Vantage API error on MACD: {macd_data['Error Message']}")
        elif "Note" in price_data or "Note" in sma_50_data or "Note" in sma_200_data or "Note" in rsi_data or "Note" in macd_data:
            st.warning("Alpha Vantage API rate limit reached. Please wait a minute and try again.")
        else:
            # Dynamically get the data keys using AI
            with st.spinner("Analyzing data format with AI..."):
                data_keys = get_data_keys_with_ai(price_data)
            
            if not data_keys:
                st.error("Failed to determine data keys using AI. Cannot proceed with analysis.")
            else:
                # Initialize an empty DataFrame
                chart_df = pd.DataFrame()
                
                # --- Individual Data Parsing with Defensive Checks ---
                # Parse price data using the AI-identified key
                if data_keys['time_series_key'] in price_data:
                    price_history = price_data[data_keys['time_series_key']]
                    price_df = pd.DataFrame.from_dict(price_history, orient='index', dtype=float)
                    price_df.index = pd.to_datetime(price_df.index)
                    price_df = price_df.rename(columns={data_keys['close_price_key']: "Close"})
                    price_df = price_df.sort_index()
                    chart_df = pd.concat([chart_df, price_df['Close']], axis=1)
                else:
                    st.error("Failed to retrieve daily price data.")

                # Parse SMA data
                if data_keys['sma_key'] in sma_50_data:
                    sma_50_history = sma_50_data[data_keys['sma_key']]
                    sma_50_df = pd.DataFrame.from_dict(sma_50_history, orient='index', dtype=float)
                    sma_50_df.index = pd.to_datetime(sma_50_df.index)
                    sma_50_df = sma_50_df.rename(columns={"SMA": "SMA_50"})
                    sma_50_df = sma_50_df.sort_index()
                    chart_df = pd.concat([chart_df, sma_50_df['SMA_50']], axis=1)
                else:
                    st.error("Failed to retrieve 50-day SMA data.")

                if data_keys['sma_key'] in sma_200_data:
                    sma_200_history = sma_200_data[data_keys['sma_key']]
                    sma_200_df = pd.DataFrame.from_dict(sma_200_history, orient='index', dtype=float)
                    sma_200_df.index = pd.to_datetime(sma_200_df.index)
                    sma_200_df = sma_200_df.rename(columns={"SMA": "SMA_200"})
                    sma_200_df = sma_200_df.sort_index()
                    chart_df = pd.concat([chart_df, sma_200_df['SMA_200']], axis=1)
                else:
                    st.error("Failed to retrieve 200-day SMA data.")
                
                # Parse RSI data
                if data_keys['rsi_key'] in rsi_data:
                    rsi_history = rsi_data[data_keys['rsi_key']]
                    rsi_df = pd.DataFrame.from_dict(rsi_history, orient='index', dtype=float)
                    rsi_df.index = pd.to_datetime(rsi_df.index)
                    rsi_df = rsi_df.rename(columns={"RSI": "RSI"})
                    rsi_df = rsi_df.sort_index()
                    chart_df = pd.concat([chart_df, rsi_df['RSI']], axis=1)
                else:
                    st.error("Failed to retrieve RSI data.")
                
                # Parse MACD data
                if data_keys['macd_key'] in macd_data:
                    macd_history = macd_data[data_keys['macd_key']]
                    macd_df = pd.DataFrame.from_dict(macd_history, orient='index', dtype=float)
                    macd_df.index = pd.to_datetime(macd_df.index)
                    macd_df = macd_df.sort_index()
                    macd_df = macd_df.rename(columns={
                        "MACD": "MACD",
                        "MACD_Hist": "MACD_Hist",
                        "MACD_Signal": "MACD_Signal"
                    })
                    chart_df = pd.concat([chart_df, macd_df], axis=1)
                else:
                    st.error("Failed to retrieve MACD data.")

                # Filter data to the past 5 years and drop NaN rows
                if not chart_df.empty:
                    five_years_ago = date.today() - timedelta(days=5*365)
                    chart_df = chart_df[chart_df.index >= pd.to_datetime(five_years_ago)]
                    chart_df.dropna(inplace=True)
                
                # Check if chart_df is still not empty after filtering
                if not chart_df.empty:
                    # --- AI-Powered Forecast Logic ---
                    display_ai_forecasts(chart_df)
                    
                    # --- Displaying Data ---
                    st.subheader("Price and Moving Averages")
                    # Dynamically select and plot available columns
                    available_ma_cols = [col for col in ['Close', 'SMA_50', 'SMA_200'] if col in chart_df.columns]
                    if available_ma_cols:
                        st.line_chart(chart_df[available_ma_cols])
                    else:
                        st.warning("No price or moving average data is available to plot.")

                    st.subheader("Relative Strength Index (RSI)")
                    # Check for RSI before attempting to display
                    if 'RSI' in chart_df.columns:
                        st.line_chart(chart_df['RSI'])
                    else:
                        st.warning("RSI data is not available.")
                    
                    st.subheader("MACD (Moving Average Convergence Divergence)")
                    # Check for MACD before attempting to display
                    if 'MACD' in chart_df.columns and 'MACD_Signal' in chart_df.columns:
                        st.line_chart(chart_df[['MACD', 'MACD_Signal']])
                    else:
                        st.warning("MACD data is not available.")
                    
                    # --- Buy/Sell Recommendation Logic ---
                    display_regular_signals(chart_df)
                    
                    # --- Key Metrics ---
                    st.subheader("Key Metrics")
                    latest_price = chart_df.iloc[-1]
                    metrics = {}
                    if 'Close' in chart_df.columns:
                        metrics["Latest Close Price"] = f"${latest_price['Close']:.2f}"
                    if 'SMA_50' in chart_df.columns:
                        metrics["Latest 50-day SMA"] = f"${latest_price['SMA_50']:.2f}"
                    if 'SMA_200' in chart_df.columns:
                        metrics["Latest 200-day SMA"] = f"${latest_price['SMA_200']:.2f}"
                    if 'RSI' in chart_df.columns:
                        metrics["Latest 14-day RSI"] = f"{latest_price['RSI']:.2f}"
                    if 'MACD' in chart_df.columns:
                        metrics["Latest MACD"] = f"{latest_price['MACD']:.2f}"
                    if 'MACD_Signal' in chart_df.columns:
                        metrics["Latest MACD Signal"] = f"{latest_price['MACD_Signal']:.2f}"
                    
                    # Add the data timestamp if available
                    if not chart_df.empty:
                        metrics["Data as of"] = latest_price.name.strftime('%Y-%m-%d')
                    
                    st.json(metrics)

                    # --- Footer ---
                    st.markdown("---")
                    st.write(f"Data sourced from Alpha Vantage as of {date.today()}.")
                else:
                    st.error(f"Could not process the retrieved data for the ticker symbol: **{ticker_symbol}**. The data may be incomplete or formatted incorrectly.")
        
elif not api_key:
    st.info("Please enter your Alpha Vantage API key to get started.")
else:
    st.info("Please enter a stock ticker symbol to get started.") 
except KeyError:
    st.error("API key not found. Please add it to your Streamlit secrets.")
    api_key = None # Set to None to prevent further execution

# --- User Input Section ---
ticker_symbol = st.text_input(
    "Enter a stock ticker (e.g., AAPL, GOOGL, MSFT):",
    "GOOGL" # Default value
).upper()

# --- Data Fetching and Error Handling ---
# Function to fetch data from Alpha Vantage
def get_alpha_vantage_data(api_key, ticker, function, interval="daily", time_period=200):
    """
    Fetches data from the Alpha Vantage API for a specific function.
    
    Args:
        api_key (str): Your Alpha Vantage API key.
        ticker (str): The stock ticker symbol.
        function (str): The API function to call (e.g., TIME_SERIES_DAILY, SMA, RSI, MACD).
        interval (str): The time interval (e.g., 'daily').
        time_period (int): The number of data points for the technical indicator.
    
    Returns:
        dict: The JSON data from the API response.
    """
    url = f"https://www.alphavantage.co/query?function={function}&symbol={ticker}&interval={interval}&time_period={time_period}&series_type=close&apikey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while fetching data: {e}")
        return None

# Check if both API key and ticker symbol are provided
if api_key and ticker_symbol:
    # Use a loading spinner while fetching the data
    with st.spinner(f"Fetching data for {ticker_symbol}..."):
        # Fetch daily price data
        price_data = get_alpha_vantage_data(api_key, ticker_symbol, "TIME_SERIES_DAILY", interval="daily")
        
        # Fetch 50-day SMA data
        sma_50_data = get_alpha_vantage_data(api_key, ticker_symbol, "SMA", interval="daily", time_period=50)
        
        # Fetch 200-day SMA data
        sma_200_data = get_alpha_vantage_data(api_key, ticker_symbol, "SMA", interval="daily", time_period=200)

        # Fetch 14-day RSI data
        rsi_data = get_alpha_vantage_data(api_key, ticker_symbol, "RSI", interval="daily", time_period=14)
        
        # Fetch MACD data
        macd_data = get_alpha_vantage_data(api_key, ticker_symbol, "MACD", interval="daily")

    if price_data and sma_50_data and sma_200_data and rsi_data and macd_data:
        # Check for API errors in the response
        if "Error Message" in price_data:
            st.error(f"Alpha Vantage API error on price data: {price_data['Error Message']}")
        elif "Error Message" in sma_50_data:
            st.error(f"Alpha Vantage API error on 50-day SMA: {sma_50_data['Error Message']}")
        elif "Error Message" in sma_200_data:
            st.error(f"Alpha Vantage API error on 200-day SMA: {sma_200_data['Error Message']}")
        elif "Error Message" in rsi_data:
            st.error(f"Alpha Vantage API error on RSI: {rsi_data['Error Message']}")
        elif "Error Message" in macd_data:
            st.error(f"Alpha Vantage API error on MACD: {macd_data['Error Message']}")
        elif "Note" in price_data or "Note" in sma_50_data or "Note" in sma_200_data or "Note" in rsi_data or "Note" in macd_data:
            st.warning("Alpha Vantage API rate limit reached. Please wait a minute and try again.")
        else:
            # Initialize an empty DataFrame
            chart_df = pd.DataFrame()
            
            # --- Individual Data Parsing with Defensive Checks ---
            # Parse price data
            if "Time Series (Daily)" in price_data:
                price_history = price_data["Time Series (Daily)"]
                price_df = pd.DataFrame.from_dict(price_history, orient='index', dtype=float)
                price_df.index = pd.to_datetime(price_df.index)
                price_df = price_df.rename(columns={"4. close": "Close"})
                price_df = price_df.sort_index()
                chart_df = pd.concat([chart_df, price_df['Close']], axis=1)
            else:
                st.error("Failed to retrieve daily price data. Please check your ticker symbol or API key.")

            # Parse SMA data
            if "Technical Analysis: SMA" in sma_50_data:
                sma_50_history = sma_50_data["Technical Analysis: SMA"]
                sma_50_df = pd.DataFrame.from_dict(sma_50_history, orient='index', dtype=float)
                sma_50_df.index = pd.to_datetime(sma_50_df.index)
                sma_50_df = sma_50_df.rename(columns={"SMA": "SMA_50"})
                sma_50_df = sma_50_df.sort_index()
                chart_df = pd.concat([chart_df, sma_50_df['SMA_50']], axis=1)
            else:
                st.error("Failed to retrieve 50-day SMA data.")

            if "Technical Analysis: SMA" in sma_200_data:
                sma_200_history = sma_200_data["Technical Analysis: SMA"]
                sma_200_df = pd.DataFrame.from_dict(sma_200_history, orient='index', dtype=float)
                sma_200_df.index = pd.to_datetime(sma_200_df.index)
                sma_200_df = sma_200_df.rename(columns={"SMA": "SMA_200"})
                sma_200_df = sma_200_df.sort_index()
                chart_df = pd.concat([chart_df, sma_200_df['SMA_200']], axis=1)
            else:
                st.error("Failed to retrieve 200-day SMA data.")
            
            # Parse RSI data
            if "Technical Analysis: RSI" in rsi_data:
                rsi_history = rsi_data["Technical Analysis: RSI"]
                rsi_df = pd.DataFrame.from_dict(rsi_history, orient='index', dtype=float)
                rsi_df.index = pd.to_datetime(rsi_df.index)
                rsi_df = rsi_df.rename(columns={"RSI": "RSI"})
                rsi_df = rsi_df.sort_index()
                chart_df = pd.concat([chart_df, rsi_df['RSI']], axis=1)
            else:
                st.error("Failed to retrieve RSI data.")
            
            # Parse MACD data
            if "Technical Analysis: MACD" in macd_data:
                macd_history = macd_data["Technical Analysis: MACD"]
                macd_df = pd.DataFrame.from_dict(macd_history, orient='index', dtype=float)
                macd_df.index = pd.to_datetime(macd_df.index)
                macd_df = macd_df.sort_index()
                macd_df = macd_df.rename(columns={
                    "MACD": "MACD",
                    "MACD_Hist": "MACD_Hist",
                    "MACD_Signal": "MACD_Signal"
                })
                chart_df = pd.concat([chart_df, macd_df], axis=1)
            else:
                st.error("Failed to retrieve MACD data.")

            # Filter data to the past 5 years and drop NaN rows
            if not chart_df.empty:
                five_years_ago = date.today() - timedelta(days=5*365)
                chart_df = chart_df[chart_df.index >= pd.to_datetime(five_years_ago)]
                chart_df.dropna(inplace=True)
            
            # Check if chart_df is still not empty after filtering
            if not chart_df.empty:
                # --- AI-Powered Forecast Logic ---
                display_ai_forecasts(chart_df)
                
                # --- Displaying Data ---
                st.subheader("Price and Moving Averages")
                # Dynamically select and plot available columns
                available_ma_cols = [col for col in ['Close', 'SMA_50', 'SMA_200'] if col in chart_df.columns]
                if available_ma_cols:
                    st.line_chart(chart_df[available_ma_cols])
                else:
                    st.warning("No price or moving average data is available to plot.")

                st.subheader("Relative Strength Index (RSI)")
                # Check for RSI before attempting to display
                if 'RSI' in chart_df.columns:
                    st.line_chart(chart_df['RSI'])
                else:
                    st.warning("RSI data is not available.")
                
                st.subheader("MACD (Moving Average Convergence Divergence)")
                # Check for MACD before attempting to display
                if 'MACD' in chart_df.columns and 'MACD_Signal' in chart_df.columns:
                    st.line_chart(chart_df[['MACD', 'MACD_Signal']])
                else:
                    st.warning("MACD data is not available.")
                
                # --- Buy/Sell Recommendation Logic ---
                display_regular_signals(chart_df)
                
                # --- Key Metrics ---
                st.subheader("Key Metrics")
                latest_price = chart_df.iloc[-1]
                metrics = {}
                if 'Close' in chart_df.columns:
                    metrics["Latest Close Price"] = f"${latest_price['Close']:.2f}"
                if 'SMA_50' in chart_df.columns:
                    metrics["Latest 50-day SMA"] = f"${latest_price['SMA_50']:.2f}"
                if 'SMA_200' in chart_df.columns:
                    metrics["Latest 200-day SMA"] = f"${latest_price['SMA_200']:.2f}"
                if 'RSI' in chart_df.columns:
                    metrics["Latest 14-day RSI"] = f"{latest_price['RSI']:.2f}"
                if 'MACD' in chart_df.columns:
                    metrics["Latest MACD"] = f"{latest_price['MACD']:.2f}"
                if 'MACD_Signal' in chart_df.columns:
                    metrics["Latest MACD Signal"] = f"{latest_price['MACD_Signal']:.2f}"
                
                # Add the data timestamp if available
                if not chart_df.empty:
                    metrics["Data as of"] = latest_price.name.strftime('%Y-%m-%d')
                
                st.json(metrics)

                # --- Footer ---
                st.markdown("---")
                st.write(f"Data sourced from Alpha Vantage as of {date.today()}.")
            else:
                st.error(f"Could not process the retrieved data for the ticker symbol: **{ticker_symbol}**. The data may be incomplete or formatted incorrectly.")
    
elif not api_key:
    st.info("Please enter your Alpha Vantage API key to get started.")
else:
    st.info("Please enter a stock ticker symbol to get started.")
