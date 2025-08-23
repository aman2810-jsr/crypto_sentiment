# 🪙 Sentiment Analysis in Cryptocurrency Trading

**Project Number:** 239  
**Level:** Master's / Doctoral  
**Keywords:** Cryptocurrency, Twitter Sentiment, MATLAB, NLP, Trading Strategy, Time Series

---

## 📌 Objective

To develop optimized cryptocurrency trading strategies using Twitter sentiment analysis, integrating Natural Language Processing (NLP), Machine Learning, and Time Series modeling within the MATLAB environment.

---

## 🚀 Goals

- Retrieve Twitter data for cryptocurrency sentiment.
- Classify sentiments using ML models and compare with VADER and rule-based methods.
- Use sentiment data as external variables in time series forecasting.
- Design, backtest, and evaluate trading strategies informed by sentiment signals.
- Optimize a crypto-cash portfolio under various risk levels.

---

## 🔧 Tools & Technologies

- **MATLAB Toolboxes:**
  - Datafeed Toolbox™
  - Text Analytics Toolbox™
  - Statistics and Machine Learning Toolbox™
  - Deep Learning Toolbox™
  - Econometrics Toolbox™
  - Financial Toolbox™

- **APIs & Libraries:**
  - Twitter API
  - VADER Sentiment (via Python Integration)

---

## 🧠 Workflow

1. **Data Retrieval**
   - Use MATLAB’s Twitter API connection to collect tweets with crypto-related hashtags.
2. **Sentiment Analysis**
   - Clean and tokenize text.
   - Train ML classifiers (e.g., Naive Bayes, SVM).
   - Compare with VADER and Ratio Rule sentiment scores.
3. **Feature Extraction**
   - Extract tweet volume, average sentiment score, and tweet volatility.
   - Integrate Large Language Model via MATLAB API (optional).
4. **Time Series Modeling**
   - Build ARIMA and LSTM models using sentiment as a regressor.
   - Evaluate with RMSE, MAPE, and other metrics.
5. **Trading Strategy & Backtesting**
   - Design strategy using sentiment-informed signals.
   - Optimize portfolio using Financial Toolbox.
   - Backtest using historical price data and evaluate performance.

---

## 🔁 Extensions

- Add Python integration to bypass Twitter API data range limitations.
- Use Reinforcement Learning for dynamic portfolio management.
- Build an interactive MATLAB app for live strategy simulation.
- Apply the same methodology to equity or fixed-income markets.

---

## 📚 References

- Mittal, A. “Stock Prediction Using Twitter Sentiment Analysis.” (2011)  
- Şaşmaz & Tek, “Tweet Sentiment Analysis for Cryptocurrencies,” IEEE (2021)  
- Bollen & Mao, “Twitter Mood as a Stock Market Predictor,” *Computer*, 44(10), 2011  

---

## 💡 Skills Gained

- Text Mining & Sentiment Classification
- Time Series Modeling with Exogenous Factors
- Portfolio Optimization
- Algorithmic Trading
- Integration of MATLAB with APIs and Python

---

## 🧾 Project Impact

Builds foundation for:
- AI-powered trading systems
- Social media-driven market analytics
- ESG-aware portfolio strategies

---

> *This project is part of the [Project Number 239: Sentiment Analysis in Cryptocurrency Trading]*  
> Powered by MATLAB & Twitter API  

---

## 🧪 Quickstart (Python Demo)

This repo includes a self-contained Python demo you can run without API keys. It synthesizes tweets and prices, applies VADER sentiment, builds a simple sentiment-z strategy, and saves outputs.

### Prerequisites
- Python 3.10+
- Install minimal packages:

```bash
python3 -m pip install --break-system-packages numpy pandas matplotlib vaderSentiment
```

### Run the demo

```bash
python src/demo_sentiment_strategy.py
```

Artifacts will be saved under `artifacts/`:
- `artifacts/equity_curves.png`: Strategy vs Buy&Hold cumulative returns
- `artifacts/demo_results.csv`: Merged daily features and performance

> If your system disallows system-wide pip installs, use a virtualenv: `python3 -m venv .venv && source .venv/bin/activate && pip install numpy pandas matplotlib vaderSentiment`.

---

## 🎤 Interview Talking Points

- **Problem framing**: Market microstructure is influenced by retail and narrative flow; Twitter sentiment can proxy short-term order flow imbalances.
- **NLP choices**: Baseline VADER for speed/interpretability; TextCNN with Word2Vec for richer semantics; discuss class imbalance and domain-specific lexicon drift.
- **Feature engineering**: Daily average sentiment, volume, sentiment volatility; transformations like rolling means and z-scores; leakage controls via lagging.
- **Modeling**: Price prediction notebooks (OHLCV with optional sentiment as exogenous); contrast classical ML vs deep CNNs for text.
- **Evaluation**: Separate metrics for NLP (F1, ROC-AUC) and trading (CAGR, Sharpe, drawdown); emphasize walk-forward validation.
- **Risk & robustness**: Regime changes, API sampling bias, survivorship bias in tickers, and hyperparameter overfitting; outline guardrails.
- **Extensions**: Portfolio optimization, RL position sizing, and real-time streaming with backtesting infra.  