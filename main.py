import streamlit as st
import pandas as pd
import requests
from datetime import date, timedelta

# Set the page title and favicon
st.set_page_config(
    page_title="Alpha Vantage Stock Analyzer",
    page_icon="📈"
)

# --- App Introduction and Title ---
st.title("📈 Alpha Vantage Stock Analyzer")
st.markdown("Enter a stock ticker to see its recent price performance and a buy/sell recommendation based on moving averages.")

# --- API Key Input ---
api_key = st.text_input(
    "Enter your Alpha Vantage API Key:",
    type="password" # This hides the key as the user types
)

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
        function (str): The API function to call (e.g., TIME_SERIES_DAILY, SMA).
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

    if price_data and sma_50_data and sma_200_data:
        # Check for API errors in the response
        if "Error Message" in price_data or "Error Message" in sma_50_data or "Error Message" in sma_200_data:
            st.error("Alpha Vantage API error: One of the data requests failed. Please check your API key and ticker symbol.")
        elif "Note" in price_data or "Note" in sma_50_data or "Note" in sma_200_data:
            st.warning("Alpha Vantage API rate limit reached. Please wait a minute and try again.")
        elif "Time Series (Daily)" in price_data and "Technical Analysis: SMA" in sma_50_data and "Technical Analysis: SMA" in sma_200_data:
            # Parse price data
            price_history = price_data["Time Series (Daily)"]
            price_df = pd.DataFrame.from_dict(price_history, orient='index', dtype=float)
            price_df.index = pd.to_datetime(price_df.index)
            price_df = price_df.rename(columns={"4. close": "Close"})
            price_df = price_df.sort_index()

            # Parse SMA data
            sma_50_history = sma_50_data["Technical Analysis: SMA"]
            sma_50_df = pd.DataFrame.from_dict(sma_50_history, orient='index', dtype=float)
            sma_50_df.index = pd.to_datetime(sma_50_df.index)
            sma_50_df = sma_50_df.rename(columns={"SMA": "SMA_50"})
            sma_50_df = sma_50_df.sort_index()

            sma_200_history = sma_200_data["Technical Analysis: SMA"]
            sma_200_df = pd.DataFrame.from_dict(sma_200_history, orient='index', dtype=float)
            sma_200_df.index = pd.to_datetime(sma_200_df.index)
            sma_200_df = sma_200_df.rename(columns={"SMA": "SMA_200"})
            sma_200_df = sma_200_df.sort_index()

            # Merge data for charting and analysis
            chart_df = pd.concat([price_df['Close'], sma_50_df['SMA_50'], sma_200_df['SMA_200']], axis=1)

            # Filter data to the past 5 years
            five_years_ago = date.today() - timedelta(days=5*365)
            chart_df = chart_df[chart_df.index >= pd.to_datetime(five_years_ago)]

            # --- Displaying Data ---
            st.subheader("Price and Moving Averages")
            st.line_chart(chart_df)
            
            # --- Buy/Sell Recommendation Logic ---
            if not chart_df.empty and len(chart_df) > 1:
                latest_close = chart_df['Close'].iloc[-1]
                latest_50_sma = chart_df['SMA_50'].iloc[-1]
                latest_200_sma = chart_df['SMA_200'].iloc[-1]

                # Compare the latest SMA values to generate a signal
                if latest_50_sma > latest_200_sma:
                    st.success("✅ **BUY SIGNAL** - The 50-day SMA is above the 200-day SMA, indicating potential upward momentum.")
                else:
                    st.error("❌ **SELL SIGNAL** - The 50-day SMA is below the 200-day SMA, indicating potential downward momentum.")
            
            # --- Key Metrics ---
            st.subheader("Key Metrics")
            latest_price = chart_df.iloc[-1]
            st.json({
                "Latest Close Price": f"${latest_price['Close']:.2f}",
                "Latest 50-day SMA": f"${latest_price['SMA_50']:.2f}",
                "Latest 200-day SMA": f"${latest_price['SMA_200']:.2f}",
                "Data as of": latest_price.name.strftime('%Y-%m-%d')
            })

            # --- Footer ---
            st.markdown("---")
            st.write(f"Data sourced from Alpha Vantage as of {date.today()}.")
        else:
            st.error(f"Could not retrieve all necessary data for the ticker symbol: **{ticker_symbol}**. The API returned an unknown response format.")
    
elif not api_key:
    st.info("Please enter your Alpha Vantage API key to get started.")
else:
    st.info("Please enter a stock ticker symbol to get started.")
