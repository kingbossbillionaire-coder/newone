import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# Streamlit config
st.set_page_config(
    page_title="Crypto Trading Scanner",
    page_icon="📈",
    layout="wide"
)

# --- Technical Indicators ---
class TechnicalIndicators:
    @staticmethod
    def rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line

    @staticmethod
    def support_resistance(prices, window=20):
        highs = prices.rolling(window).max()
        lows = prices.rolling(window).min()
        return lows.iloc[-1], highs.iloc[-1]

# --- Data Fetcher ---
class CryptoDataFetcher:
    BASE_URL = "https://api.coingecko.com/api/v3"

    @staticmethod
    def get_top_cryptos(limit=5):
        url = f"{CryptoDataFetcher.BASE_URL}/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            return r.json()
        except Exception as e:
            st.error(f"Error fetching top cryptos: {e}")
            return []

    @staticmethod
    def get_historical_data(coin_id, days=3):
        url = f"{CryptoDataFetcher.BASE_URL}/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": days, "interval": "hourly"}
        try:
            r = requests.get(url, params=params, timeout=10)
            data = r.json()
            if "prices" not in data:   # ✅ Safe check
                return pd.DataFrame()
            df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        except Exception:
            return pd.DataFrame()

# --- Analyzer ---
class TradingAnalyzer:
    def analyze(self, coin, data):
        if data.empty:
            return {
                "symbol": coin.get("symbol","").upper(),
                "name": coin.get("name",""),
                "price": None,    # ✅ Always exists
                "change": coin.get("price_change_percentage_24h", 0),
                "rsi": "N/A",
                "macd": "N/A",
                "signal": "N/A",
                "support": "N/A",
                "resistance": "N/A",
                "bot": "No Data"
            }

        prices = data["price"]
        rsi = TechnicalIndicators.rsi(prices).iloc[-1]
        macd, signal = TechnicalIndicators.macd(prices)
        macd_last, signal_last = macd.iloc[-1], signal.iloc[-1]
        support, resistance = TechnicalIndicators.support_resistance(prices)

        # Bot logic
        if rsi < 40 and macd_last > signal_last:
            bot = "🟢 Long"
        elif rsi > 60 and macd_last < signal_last:
            bot = "🔴 Short"
        else:
            bot = "🟡 Range"

        return {
            "name": coin.get("name",""),
            "symbol": coin.get("symbol","").upper(),
            "price": coin.get("current_price", None),
            "change": coin.get("price_change_percentage_24h", 0),
            "rsi": round(rsi, 2) if not pd.isna(rsi) else "N/A",
            "macd": round(macd_last, 6) if not pd.isna(macd_last) else "N/A",
            "signal": round(signal_last, 6) if not pd.isna(signal_last) else "N/A",
            "support": round(support, 6) if not pd.isna(support) else "N/A",
            "resistance": round(resistance, 6) if not pd.isna(resistance) else "N/A",
            "bot": bot
        }

# --- UI ---
st.title("🚀 Crypto Trading Scanner")

with st.sidebar:
    st.header("⚙️ Settings")
    top_n = st.slider("Number of Cryptos", 5, 30, 5)  # Default 5
    days = st.selectbox("History (days)", [1, 3, 7], index=1)
    st.caption("Data from CoinGecko (free API)")

fetcher = CryptoDataFetcher()
analyzer = TradingAnalyzer()

crypto_data = fetcher.get_top_cryptos(top_n)

if crypto_data:
    results = []
    placeholder = st.empty()
    progress = st.progress(0)
    status = st.empty()

    for i, coin in enumerate(crypto_data, 1):
        status.text(f"Analyzing {coin.get('name','Unknown')} ({i}/{len(crypto_data)}) ...")

        history = fetcher.get_historical_data(coin.get("id",""), days)
        analysis = analyzer.analyze(coin, history)
        results.append(analysis)

        # --- Safe rendering ---
        with placeholder.container():
            for r in results:
                price = r.get("price", None)
                change = r.get("change", 0)

                if price is None:
                    st.warning(f"⚠️ Skipping {r.get('symbol','???')} ({r.get('name','')}) - no price data")
                    continue

                st.markdown(f"""
                **{r.get('symbol','')} - {r.get('name','')}**
                - Price: ${float(price):.2f} ({float(change):.2f}% 24h)
                - RSI: {r.get('rsi','N/A')}
                - MACD: {r.get('macd','N/A')} vs Signal {r.get('signal','N/A')}
                - Support: {r.get('support','N/A')}
                - Resistance: {r.get('resistance','N/A')}
                - Bot Suggestion: {r.get('bot','N/A')}
                ---
                """)

        progress.progress(i / len(crypto_data))
        time.sleep(0.2)

    status.empty()
    progress.empty()
else:
    st.error("⚠️ Could not fetch crypto data. Please try again later.")
