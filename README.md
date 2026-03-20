# AAPL Stock Prediction

A complete end-to-end machine learning project 
predicting Apple Inc. (AAPL) daily stock direction.

##  Models Used
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 48.35% |
| Random Forest | 51.28% |
| XGBoost | 50.92% |

##  Features Used
- Moving Averages (MA50, MA200)
- RSI, MACD, Bollinger Bands, ATR
- Daily Returns and Lag Features
- Volatility Indicators
- Volume Change

## Project Structure
- aapl_stock_prediction.ipynb — main analysis
- aapl_dashboard.py — streamlit dashboard

##  How to Run

### Jupyter Notebook

jupyter notebook aapl_stock_prediction.ipynb


### Streamlit Dashboard

streamlit run aapl_dashboard.py


## Requirements

pip install yfinance ta xgboost scikit-learn
pip install seaborn matplotlib pandas numpy streamlit


## Disclaimer
For educational purposes only — Not financial advice
