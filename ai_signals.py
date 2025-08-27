import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def display_ai_forecasts(chart_df: pd.DataFrame):
    """
    Displays stock price forecasts based on simple AI models.

    Args:
        chart_df (pd.DataFrame): The DataFrame containing price and technical indicator data.
    """
    # Check if there's enough data for AI models
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
        features = [col for col in ['Close', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_Signal'] if col in forecast_df.columns]
        
        # Check if all necessary features for AI models exist
        if len(features) > 1 and 'Price_Change' in forecast_df.columns:
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
