# To run this app, use the following command in your terminal:
# pip install scikit-learn pandas streamlit requests
# python3 -m streamlit run main.py --server.port 8502 --server.address 0.0.0.0

import streamlit as st
import pandas as pd
import requests
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Set the page title and favicon
st.set_page_config(
    page_title="Alpha Vantage Stock Analyzer",
    page_icon="📈"
)

# --- App Introduction and Title ---
st.title("📈 Alpha Vantage Stock Analyzer")
st.markdown("Enter a stock ticker to see its recent price performance and a buy/sell recommendation based on moving averages, RSI, MACD, and several simple AI forecasts.")

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
            five_years_ago = date.today() - timedelta(days=5*365)
            chart_df = chart_df[chart_df.index >= pd.to_datetime(five_years_ago)]
            chart_df.dropna(inplace=True)

            if not chart_df.empty:
                # --- AI-Powered Forecast Logic ---
                if len(chart_df) > 30:
                    st.subheader("AI-Powered Forecasts")
                    st.info("ℹ️ Note: These are simple machine learning models for demonstration. They should not be used for real-world financial decisions.")
                    
                    # Create a copy and fill missing values for the model
                    forecast_df = chart_df.copy()
                    
                    # Create a target variable (y) and features (X)
                    # We'll predict the next day's closing price change direction (up/down)
                    forecast_df['Price_Change'] = np.where(forecast_df['Close'].shift(-1) > forecast_df['Close'], 1, 0)
                    
                    # Drop the last row which has a NaN target
                    forecast_df.dropna(inplace=True)
                    
                    # Define features and target
                    features = ['Close', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_Signal']
                    
                    # Check if all features exist in the DataFrame
                    if all(f in forecast_df.columns for f in features):
                        X = forecast_df[features]
                        y = forecast_df['Price_Change']
                        
                        # --- Linear Regression Model (Price Prediction) ---
                        price_model = LinearRegression()
                        price_model.fit(X, forecast_df['Close'].shift(-1).dropna())
                        
                        latest_data = forecast_df.iloc[-1][features].values.reshape(1, -1)
                        next_day_price_pred = price_model.predict(latest_data)[0]
                        
                        st.write(f"The **Linear Regression** model predicts the next closing price will be: **${next_day_price_pred:.2f}**")
                        if next_day_price_pred > latest_data[0][0]:
                            st.success("✅ **AI BUY SIGNAL (Linear Regression)** - The model predicts the price will increase.")
                        else:
                            st.error("❌ **AI SELL SIGNAL (Linear Regression)** - The model predicts the price will decrease.")

                        # --- SVC Model (Buy/Sell Prediction) ---
                        svc_model = SVC(kernel='linear', C=1)
                        svc_model.fit(X, y)
                        svc_pred = svc_model.predict(latest_data)[0]
                        
                        if svc_pred == 1:
                            st.success("✅ **AI BUY SIGNAL (SVC)** - The SVC model predicts an upward price movement.")
                        else:
                            st.error("❌ **AI SELL SIGNAL (SVC)** - The SVC model predicts a downward price movement.")

                        # --- KNN Model (Buy/Sell Prediction) ---
                        knn_model = KNeighborsClassifier(n_neighbors=5)
                        knn_model.fit(X, y)
                        knn_pred = knn_model.predict(latest_data)[0]
                        
                        if knn_pred == 1:
                            st.success("✅ **AI BUY SIGNAL (KNN)** - The KNN model predicts an upward price movement.")
                        else:
                            st.error("❌ **AI SELL SIGNAL (KNN)** - The KNN model predicts a downward price movement.")

                    else:
                        st.warning("Could not train the AI models. Missing necessary data.")
                else:
                    st.warning("Not enough data to train the AI models.")
                
                # --- Displaying Data ---
                st.subheader("Price and Moving Averages")
                st.line_chart(chart_df[['Close', 'SMA_50', 'SMA_200']])
                
                st.subheader("Relative Strength Index (RSI)")
                st.line_chart(chart_df['RSI'])
                
                st.subheader("MACD (Moving Average Convergence Divergence)")
                st.line_chart(chart_df[['MACD', 'MACD_Signal']])
                
                # --- Buy/Sell Recommendation Logic ---
                if len(chart_df) > 1:
                    latest_price_data = chart_df.iloc[-1]
                    latest_50_sma = latest_price_data['SMA_50']
                    latest_200_sma = latest_price_data['SMA_200']
                    latest_rsi = latest_price_data['RSI']
                    latest_macd = latest_price_data['MACD']
                    latest_macd_signal = latest_price_data['MACD_Signal']

                    st.subheader("Traditional Trading Signals")
                    
                    # SMA Crossover Signal
                    if latest_50_sma > latest_200_sma:
                        st.success("✅ **SMA BUY SIGNAL** - The 50-day SMA is above the 200-day SMA, indicating potential upward momentum.")
                    else:
                        st.error("❌ **SMA SELL SIGNAL** - The 50-day SMA is below the 200-day SMA, indicating potential downward momentum.")
                    
                    # RSI Signal
                    if latest_rsi < 30:
                        st.success("✅ **RSI BUY SIGNAL** - The RSI is below 30, indicating the stock may be oversold.")
                    elif latest_rsi > 70:
                        st.error("❌ **RSI SELL SIGNAL** - The RSI is above 70, indicating the stock may be overbought.")
                    else:
                        st.info("ℹ️ **RSI NEUTRAL** - The RSI is between 30 and 70, with no strong signal.")
                        
                    # MACD Crossover Signal
                    if latest_macd > latest_macd_signal:
                        st.success("✅ **MACD BUY SIGNAL** - The MACD line is above the signal line, suggesting potential bullish momentum.")
                    else:
                        st.error("❌ **MACD SELL SIGNAL** - The MACD line is below the signal line, suggesting potential bearish momentum.")
                
                # --- Key Metrics ---
                st.subheader("Key Metrics")
                latest_price = chart_df.iloc[-1]
                st.json({
                    "Latest Close Price": f"${latest_price['Close']:.2f}",
                    "Latest 50-day SMA": f"${latest_price['SMA_50']:.2f}",
                    "Latest 200-day SMA": f"${latest_price['SMA_200']:.2f}",
                    "Latest 14-day RSI": f"{latest_price['RSI']:.2f}",
                    "Latest MACD": f"{latest_price['MACD']:.2f}",
                    "Latest MACD Signal": f"{latest_price['MACD_Signal']:.2f}",
                    "Data as of": latest_price.name.strftime('%Y-%m-%d')
                })

                # --- Footer ---
                st.markdown("---")
                st.write(f"Data sourced from Alpha Vantage as of {date.today()}.")
            else:
                st.error(f"Could not process the retrieved data for the ticker symbol: **{ticker_symbol}**. The data may be incomplete or formatted incorrectly.")
    
elif not api_key:
    st.info("Please enter your Alpha Vantage API key to get started.")
else:
    st.info("Please enter a stock ticker symbol to get started.")# Function to fetch data from Alpha Vantage
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
        # Check if all necessary data keys exist before proceeding
        elif "Time Series (Daily)" in price_data and "Technical Analysis: SMA" in sma_50_data and "Technical Analysis: SMA" in sma_200_data and "Technical Analysis: RSI" in rsi_data and "Technical Analysis: MACD" in macd_data:
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
            
            # Parse RSI data
            rsi_history = rsi_data["Technical Analysis: RSI"]
            rsi_df = pd.DataFrame.from_dict(rsi_history, orient='index', dtype=float)
            rsi_df.index = pd.to_datetime(rsi_df.index)
            rsi_df = rsi_df.rename(columns={"RSI": "RSI"})
            rsi_df = rsi_df.sort_index()
            
            # Parse MACD data
            macd_history = macd_data["Technical Analysis: MACD"]
            macd_df = pd.DataFrame.from_dict(macd_history, orient='index', dtype=float)
            macd_df.index = pd.to_datetime(macd_df.index)
            macd_df = macd_df.sort_index()
            macd_df = macd_df.rename(columns={
                "MACD": "MACD",
                "MACD_Hist": "MACD_Hist",
                "MACD_Signal": "MACD_Signal"
            })

            # Merge data for charting and analysis
            chart_df = pd.concat([price_df['Close'], sma_50_df['SMA_50'], sma_200_df['SMA_200'], rsi_df['RSI'], macd_df], axis=1)

            # Filter data to the past 5 years
            five_years_ago = date.today() - timedelta(days=5*365)
            chart_df = chart_df[chart_df.index >= pd.to_datetime(five_years_ago)]

            # --- AI-Powered Forecast Logic ---
            if not chart_df.empty and len(chart_df) > 30:
                st.subheader("AI-Powered Forecasts")
                st.info("ℹ️ Note: These are simple machine learning models for demonstration. They should not be used for real-world financial decisions.")
                
                # Create a copy and fill missing values for the model
                forecast_df = chart_df.copy()
                forecast_df.fillna(method='ffill', inplace=True)
                
                # Create a target variable (y) and features (X)
                # We'll predict the next day's closing price change direction (up/down)
                forecast_df['Price_Change'] = np.where(forecast_df['Close'].shift(-1) > forecast_df['Close'], 1, 0)
                
                # Drop the last row which has a NaN target
                forecast_df.dropna(inplace=True)
                
                # Define features and target
                features = ['Close', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_Signal']
                
                # Check if all features exist in the DataFrame
                if all(f in forecast_df.columns for f in features):
                    X = forecast_df[features]
                    y = forecast_df['Price_Change']
                    
                    # --- Linear Regression Model (Price Prediction) ---
                    price_model = LinearRegression()
                    price_model.fit(X, forecast_df['Close'].shift(-1).dropna())
                    
                    latest_data = forecast_df.iloc[-1][features].values.reshape(1, -1)
                    next_day_price_pred = price_model.predict(latest_data)[0]
                    
                    st.write(f"The **Linear Regression** model predicts the next closing price will be: **${next_day_price_pred:.2f}**")
                    if next_day_price_pred > latest_data[0][0]:
                        st.success("✅ **AI BUY SIGNAL (Linear Regression)** - The model predicts the price will increase.")
                    else:
                        st.error("❌ **AI SELL SIGNAL (Linear Regression)** - The model predicts the price will decrease.")

                    # --- SVC Model (Buy/Sell Prediction) ---
                    svc_model = SVC(kernel='linear', C=1)
                    svc_model.fit(X, y)
                    svc_pred = svc_model.predict(latest_data)[0]
                    
                    if svc_pred == 1:
                        st.success("✅ **AI BUY SIGNAL (SVC)** - The SVC model predicts an upward price movement.")
                    else:
                        st.error("❌ **AI SELL SIGNAL (SVC)** - The SVC model predicts a downward price movement.")

                    # --- KNN Model (Buy/Sell Prediction) ---
                    knn_model = KNeighborsClassifier(n_neighbors=5)
                    knn_model.fit(X, y)
                    knn_pred = knn_model.predict(latest_data)[0]
                    
                    if knn_pred == 1:
                        st.success("✅ **AI BUY SIGNAL (KNN)** - The KNN model predicts an upward price movement.")
                    else:
                        st.error("❌ **AI SELL SIGNAL (KNN)** - The KNN model predicts a downward price movement.")

                else:
                    st.warning("Could not train the AI models. Missing necessary data.")
            else:
                st.warning("Not enough data to train the AI models.")
            
            # --- Displaying Data ---
            st.subheader("Price and Moving Averages")
            st.line_chart(chart_df[['Close', 'SMA_50', 'SMA_200']])
            
            st.subheader("Relative Strength Index (RSI)")
            st.line_chart(chart_df['RSI'])
            
            st.subheader("MACD (Moving Average Convergence Divergence)")
            st.line_chart(chart_df[['MACD', 'MACD_Signal']])
            
            # --- Buy/Sell Recommendation Logic ---
            if not chart_df.empty and len(chart_df) > 1:
                latest_price_data = chart_df.iloc[-1]
                latest_50_sma = latest_price_data['SMA_50']
                latest_200_sma = latest_price_data['SMA_200']
                latest_rsi = latest_price_data['RSI']
                latest_macd = latest_price_data['MACD']
                latest_macd_signal = latest_price_data['MACD_Signal']

                st.subheader("Traditional Trading Signals")
                
                # SMA Crossover Signal
                if latest_50_sma > latest_200_sma:
                    st.success("✅ **SMA BUY SIGNAL** - The 50-day SMA is above the 200-day SMA, indicating potential upward momentum.")
                else:
                    st.error("❌ **SMA SELL SIGNAL** - The 50-day SMA is below the 200-day SMA, indicating potential downward momentum.")
                
                # RSI Signal
                if latest_rsi < 30:
                    st.success("✅ **RSI BUY SIGNAL** - The RSI is below 30, indicating the stock may be oversold.")
                elif latest_rsi > 70:
                    st.error("❌ **RSI SELL SIGNAL** - The RSI is above 70, indicating the stock may be overbought.")
                else:
                    st.info("ℹ️ **RSI NEUTRAL** - The RSI is between 30 and 70, with no strong signal.")
                    
                # MACD Crossover Signal
                if latest_macd > latest_macd_signal:
                    st.success("✅ **MACD BUY SIGNAL** - The MACD line is above the signal line, suggesting potential bullish momentum.")
                else:
                    st.error("❌ **MACD SELL SIGNAL** - The MACD line is below the signal line, suggesting potential bearish momentum.")
            
            # --- Key Metrics ---
            st.subheader("Key Metrics")
            latest_price = chart_df.iloc[-1]
            st.json({
                "Latest Close Price": f"${latest_price['Close']:.2f}",
                "Latest 50-day SMA": f"${latest_price['SMA_50']:.2f}",
                "Latest 200-day SMA": f"${latest_price['SMA_200']:.2f}",
                "Latest 14-day RSI": f"{latest_price['RSI']:.2f}",
                "Latest MACD": f"{latest_price['MACD']:.2f}",
                "Latest MACD Signal": f"{latest_price['MACD_Signal']:.2f}",
                "Data as of": latest_price.name.strftime('%Y-%m-%d')
            })

            # --- Footer ---
            st.markdown("---")
            st.write(f"Data sourced from Alpha Vantage as of {date.today()}.")
        else:
            # Fallback for unexpected response structure.
            st.error(f"Could not retrieve all necessary data for the ticker symbol: **{ticker_symbol}**. The API returned an unknown response format. Please check your API key and ticker symbol.")
            st.write("---")
            st.write("For debugging, here is the raw API response:")
            st.json({
                "price_data": price_data,
                "sma_50_data": sma_50_data,
                "sma_200_data": sma_200_data,
                "rsi_data": rsi_data,
                "macd_data": macd_data
            })
    
elif not api_key:
    st.info("Please enter your Alpha Vantage API key to get started.")
else:
    st.info("Please enter a stock ticker symbol to get started.")
