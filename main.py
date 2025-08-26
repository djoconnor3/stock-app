import streamlit as st
import pandas as pd
import requests
from datetime import date

# Set the page title and favicon
st.set_page_config(
    page_title="Alpha Vantage Stock Analyzer",
    page_icon="📈"
)

# --- App Introduction and Title ---
st.title("📈 Alpha Vantage Stock Analyzer")
st.markdown("Enter a stock ticker to see its recent price performance and volume trends.")

# --- API Key Input ---
# Get the Alpha Vantage API key from the user. It is highly recommended to store this
# in Streamlit secrets for production apps. For this example, we take it as text input.
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
def get_stock_data(api_key, ticker, function="TIME_SERIES_DAILY"):
    """
    Fetches daily stock data from the Alpha Vantage API.
    
    Args:
        api_key (str): Your Alpha Vantage API key.
        ticker (str): The stock ticker symbol.
        function (str): The API function to call (e.g., TIME_SERIES_DAILY).
    
    Returns:
        dict: The JSON data from the API response.
    """
    url = f"https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={api_key}&outputsize=compact"
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
        data = get_stock_data(api_key, ticker_symbol)

    if data:
        # Check for API errors in the response
        if "Error Message" in data:
            st.error(f"Alpha Vantage API error: {data['Error Message']}")
        elif "Note" in data:
            st.warning("Alpha Vantage API rate limit reached. Please wait a minute and try again.")
        elif "Time Series (Daily)" in data:
            # Parse the time series data into a pandas DataFrame
            history_data = data["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(history_data, orient='index', dtype=float)
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume"
            })
            # Sort by date
            df = df.sort_index()

            # --- Displaying Data ---
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Price Chart")
                st.line_chart(df['Close'])

            with col2:
                st.subheader("Volume Chart")
                st.bar_chart(df['Volume'])
                
            # --- Key Metrics ---
            # Alpha Vantage does not provide a summary for the free API.
            # We'll display a basic summary from the data itself.
            st.subheader("Key Metrics")
            latest_data = df.iloc[-1]
            st.json({
                "Latest Close Price": f"${latest_data['Close']:.2f}",
                "Latest Volume": f"{latest_data['Volume']:,}",
                "Data as of": latest_data.name.strftime('%Y-%m-%d')
            })

            # --- Footer ---
            st.markdown("---")
            st.write(f"Data sourced from Alpha Vantage as of {date.today()}.")
        else:
            st.error(f"Could not retrieve data for the ticker symbol: **{ticker_symbol}**. The API returned an unknown response format.")
    
elif not api_key:
    st.info("Please enter your Alpha Vantage API key to get started.")
else:
    st.info("Please enter a stock ticker symbol to get started.")
