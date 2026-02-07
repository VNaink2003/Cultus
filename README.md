# Advanced Time Series Forecasting with Attention Mechanism

## Project Overview
This project implements an advanced multivariate time series forecasting system using deep learning and attention mechanisms.  
The goal is to predict future values from complex non-stationary time series data containing trend, seasonality, and correlated features.

A baseline LSTM model is compared with an attention-based LSTM model to demonstrate performance improvements and interpretability.

---

## Problem Statement
Traditional time series models such as ARIMA or simple LSTM networks struggle with long-term dependencies and complex seasonal patterns.  
This project uses an attention mechanism to help the model focus on the most relevant historical timesteps when making predictions.

The system simulates real-world forecasting tasks such as:
- energy demand prediction  
- stock price forecasting  
- retail sales forecasting  
- weather prediction  

---

## Dataset Description
A synthetic multivariate dataset representing **5 years of daily data** was generated.

Features included:
1. Trend component  
2. Yearly seasonality  
3. Weekly seasonality  
4. Random noise  
5. Lag feature  
6. External correlated variable  

The dataset is intentionally non-stationary and seasonal to mimic real industrial forecasting scenarios.

---

## Data Preprocessing Pipeline
The following steps were applied:

1. Missing value handling using backfill  
2. Standardization using StandardScaler  
3. Feature engineering with lag and external variables  
4. Sequence windowing:
   - Input window: 30 days  
   - Forecast horizon: 7 days  
5. Train-test split:
   - 80% training  
   - 20% testing  

---

## Models Implemented

### 1. Baseline Model
A standard LSTM model without attention.

Architecture:
- LSTM encoder  
- Fully connected output layer  
- Predicts next 7 timesteps  

Purpose:
Provides a comparison benchmark for the attention model.

---

### 2. Attention-Based Model
An LSTM model enhanced with an attention mechanism.

Architecture:
- LSTM encoder  
- Attention layer computing timestep importance  
- Context vector from weighted encoder outputs  
- Fully connected layer for multi-step forecasting  

Advantages:
- Focuses on important historical timesteps  
- Improves prediction accuracy  
- Provides interpretability through attention weights  

---

## Hyperparameters

| Parameter | Value |
|----------|------|
Hidden units | 64  
Epochs | 20  
Learning rate | 0.001  
Optimizer | Adam  
Loss function | MSE  
Input window | 30 days  
Forecast horizon | 7 days  

---

## Evaluation Metrics
The models were evaluated using:

1. RMSE (Root Mean Squared Error)  
2. MAE (Mean Absolute Error)  
3. MASE (Mean Absolute Scaled Error)  
4. Directional Accuracy  

These metrics provide a comprehensive evaluation of forecasting performance.

---

## Results Summary
The attention-based model achieved better performance than the baseline LSTM across most evaluation metrics.

Key observations:
- Lower RMSE  
- Better directional accuracy  
- More stable predictions  
- Improved interpretability  

---

## Attention Weights Interpretation
The attention visualization shows that recent timesteps receive higher weights, indicating that short-term patterns are more influential in forecasting.

However, the model still considers longer seasonal cycles, demonstrating the effectiveness of attention in balancing short-term and long-term dependencies.

---

## Visualizations
The project generates:
- Forecast vs actual plot  
- Attention weight distribution plot  

These visualizations validate model performance and provide interpretability.

---

## How to Run

### Install dependencies
