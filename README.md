# 📈 Algorithmic Trading Signal Generator (NLP & Market Data)

This project is a quantitative analysis tool designed to correlate stock market price action with financial news sentiment using Large Language Models (LLMs).

## 🚀 Features
- **Data Collection:** Fetches historical market data (OHLCV) using `yfinance`.
- **Sentiment Analysis:** Utilizes **FinBERT** (ProsusAI), a model pre-trained on financial texts, to score news headlines (-1 to +1).
- **Visualization:** Generates an interactive dashboard comparing stock prices with "Risk-On/Risk-Off" sentiment signals using **Plotly**.
- **Automation:** Script configured for daily automated reporting via Task Scheduler.

## 🛠️ Tech Stack
- **Python 3.10+**
- **Libraries:** `yfinance`, `transformers`, `plotly`, `pandas`, `numpy`

## 📦 Installation

1. Clone the repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
