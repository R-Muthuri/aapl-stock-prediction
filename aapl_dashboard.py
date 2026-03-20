# ============================================
# AAPL STOCK PREDICTION DASHBOARD
# ============================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ta
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Page Configuration ──────────────────────
st.set_page_config(
    page_title="AAPL Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# ── Title ────────────────────────────────────
st.title("📈 AAPL Stock Prediction Dashboard")
st.markdown("*Powered by Machine Learning | For educational purposes only*")
st.divider()

# ── Load & Cache Data ────────────────────────
@st.cache_data
def load_data():
    data = yf.download("AAPL", start="2020-01-01")
    
    # Fix MultiIndex
    if data.columns.nlevels > 1:
        data.columns = [col[0] for col in data.columns]
    
    # Indicators
    data['MA50']    = data['Close'].rolling(50).mean()
    data['MA200']   = data['Close'].rolling(200).mean()
    data['Returns'] = data['Close'].pct_change()

    # Technical indicators
    data['RSI']          = ta.momentum.RSIIndicator(data['Close']).rsi()
    macd                 = ta.trend.MACD(data['Close'])
    data['MACD']         = macd.macd()
    data['MACD_Signal']  = macd.macd_signal()
    data['MACD_Diff']    = macd.macd_diff()
    bb                   = ta.volatility.BollingerBands(data['Close'])
    data['BB_High']      = bb.bollinger_hband()
    data['BB_Low']       = bb.bollinger_lband()
    data['BB_Width']     = bb.bollinger_wband()
    data['ATR']          = ta.volatility.AverageTrueRange(
                           data['High'], data['Low'], data['Close']
                           ).average_true_range()

    # Feature engineering
    data['Target']         = (data['Close'].shift(-1) > data['Close']).astype(int)
    data['Daily_Range']    = data['High'] - data['Low']
    data['MA50_Distance']  = data['Close'] - data['MA50']
    data['MA200_Distance'] = data['Close'] - data['MA200']
    data['Volume_Change']  = data['Volume'].pct_change()
    data['Returns_Lag1']   = data['Returns'].shift(1)
    data['Returns_Lag2']   = data['Returns'].shift(2)
    data['Returns_Lag3']   = data['Returns'].shift(3)
    data['Volatility_5d']  = data['Returns'].rolling(5).std()
    data['Day_of_Week']    = data.index.dayofweek

    data = data.dropna()
    return data

# ── Train & Cache Models ─────────────────────
@st.cache_resource
def train_models(data):
    features = [
        'MA50', 'MA200', 'Returns',
        'Daily_Range', 'MA50_Distance', 'MA200_Distance',
        'Volume_Change', 'Returns_Lag1', 'Returns_Lag2',
        'Returns_Lag3', 'Volatility_5d',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff',
        'BB_High', 'BB_Low', 'BB_Width', 'ATR',
        'Day_of_Week'
    ]

    X = data[features]
    y = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Logistic Regression
    log_model = LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    )
    log_model.fit(X_train_scaled, y_train)
    log_preds = log_model.predict(X_test_scaled)

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        min_samples_split=20, min_samples_leaf=10,
        random_state=42
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_preds = rf_model.predict(X_test_scaled)

    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=100, max_depth=4,
        learning_rate=0.05, random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_preds = xgb_model.predict(X_test_scaled)

    return (log_model, rf_model, xgb_model,
            scaler, features,
            log_preds, rf_preds, xgb_preds,
            y_test, X_test)

# ── Load Data & Train ────────────────────────
with st.spinner("Loading data and training models..."):
    data    = load_data()
    results = train_models(data)

(log_model, rf_model, xgb_model,
 scaler, features,
 log_preds, rf_preds, xgb_preds,
 y_test, X_test) = results

# ════════════════════════════════════════════
# SECTION 1: KEY METRICS
# ════════════════════════════════════════════
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Last Close Price",
    value=f"${data['Close'].iloc[-1]:.2f}"
)
col2.metric(
    label="XGBoost Accuracy",
    value=f"{accuracy_score(y_test, xgb_preds):.2%}"
)
col3.metric(
    label="Total Trading Days",
    value=f"{len(data):,}"
)
col4.metric(
    label="Mean Daily Return",
    value=f"{data['Returns'].mean():.3%}"
)

st.divider()

# ════════════════════════════════════════════
# SECTION 2: PRICE CHART
# ════════════════════════════════════════════
st.subheader("📉 AAPL Price with Moving Averages")

fig1, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(data.index, data['Close'],
         label='Close Price', color='steelblue', linewidth=1)
ax1.plot(data.index, data['MA50'],
         label='MA50', color='orange', linewidth=1.5)
ax1.plot(data.index, data['MA200'],
         label='MA200', color='green', linewidth=1.5)
ax1.set_title('AAPL Price with MA50 & MA200')
ax1.legend()
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

st.divider()

# ════════════════════════════════════════════
# SECTION 3: RETURNS DISTRIBUTION
# ════════════════════════════════════════════
st.subheader("📊 Returns Distribution")

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.hist(data['Returns'], bins=50, color='steelblue',
         edgecolor='white', alpha=0.8)
ax2.set_title('Daily Returns Distribution')
ax2.set_xlabel('Daily Return')
ax2.set_ylabel('Frequency')
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

st.divider()

# ════════════════════════════════════════════
# SECTION 4: MODEL PERFORMANCE
# ════════════════════════════════════════════
st.subheader("🤖 Model Performance Comparison")

col1, col2, col3 = st.columns(3)

col1.metric(
    label="Logistic Regression",
    value=f"{accuracy_score(y_test, log_preds):.2%}"
)
col2.metric(
    label="Random Forest",
    value=f"{accuracy_score(y_test, rf_preds):.2%}"
)
col3.metric(
    label="XGBoost",
    value=f"{accuracy_score(y_test, xgb_preds):.2%}"
)

st.divider()

# ════════════════════════════════════════════
# SECTION 5: TOMORROW'S PREDICTION
# ════════════════════════════════════════════
st.subheader("🔮 Tomorrow's Prediction")

latest        = data[features].iloc[-1:]
latest_scaled = scaler.transform(latest)

log_pred  = log_model.predict(latest_scaled)[0]
rf_pred   = rf_model.predict(latest_scaled)[0]
xgb_pred  = xgb_model.predict(latest_scaled)[0]

log_prob  = log_model.predict_proba(latest_scaled)[0]
rf_prob   = rf_model.predict_proba(latest_scaled)[0]
xgb_prob  = xgb_model.predict_proba(latest_scaled)[0]

direction = {0: '🔴 DOWN', 1: '🟢 UP'}

col1, col2, col3 = st.columns(3)

col1.metric(
    label="Logistic Regression",
    value=direction[log_pred],
    delta=f"UP: {log_prob[1]:.2%} | DOWN: {log_prob[0]:.2%}"
)
col2.metric(
    label="Random Forest",
    value=direction[rf_pred],
    delta=f"UP: {rf_prob[1]:.2%} | DOWN: {rf_prob[0]:.2%}"
)
col3.metric(
    label="XGBoost",
    value=direction[xgb_pred],
    delta=f"UP: {xgb_prob[1]:.2%} | DOWN: {xgb_prob[0]:.2%}"
)

st.divider()

# ════════════════════════════════════════════
# SECTION 6: FEATURE IMPORTANCE
# ════════════════════════════════════════════
st.subheader("🔍 Feature Importance — XGBoost")

importance_df = pd.DataFrame({
    'Feature'   : features,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

fig3, ax3 = plt.subplots(figsize=(10, 7))
sns.barplot(data=importance_df, x='Importance',
            y='Feature', hue='Feature',
            legend=False, palette='magma', ax=ax3)
ax3.set_title('XGBoost Feature Importance')
st.pyplot(fig3)

st.divider()

# ════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════
st.markdown("""
---
*⚠️ For educational purposes only — Not financial advice*  
*📊 Data sourced from Yahoo Finance via yfinance*
""")