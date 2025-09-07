import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Crypto Trading Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark/light mode
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 0.5rem 0;
    }

    .bot-recommendation {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.5rem 0;
    }

    .long-bot {
        background: linear-gradient(135deg, #00C851, #007E33);
        color: white;
    }

    .short-bot {
        background: linear-gradient(135deg, #FF4444, #CC0000);
        color: white;
    }

    .range-bot {
        background: linear-gradient(135deg, #FF8800, #FF6600);
        color: white;
    }

    .trend-indicator {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.1rem;
    }

    .upward { background: #00C851; color: white; }
    .downward { background: #FF4444; color: white; }
    .ranging { background: #FF8800; color: white; }

    .refresh-indicator {
        position: fixed;
        top: 10px;
        right: 10px;
        background: rgba(0, 200, 81, 0.8);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        z-index: 1000;
    }
    </style>
    """, unsafe_allow_html=True)

class TechnicalIndicators:
    """Custom implementation of technical indicators"""

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }

    @staticmethod
    def two_pole_oscillator(prices: pd.Series, period: int = 14) -> pd.Series:
        """Custom Two-Pole Oscillator implementation"""
        # This is a custom oscillator that combines momentum and trend
        momentum = prices.pct_change(period)
        trend = prices.rolling(window=period).mean().pct_change()

        # Normalize both components
        momentum_norm = (momentum - momentum.rolling(window=50).mean()) / momentum.rolling(window=50).std()
        trend_norm = (trend - trend.rolling(window=50).mean()) / trend.rolling(window=50).std()

        # Combine with weights
        oscillator = (momentum_norm * 0.7) + (trend_norm * 0.3)

        # Scale to -100 to 100
        oscillator = oscillator * 50

        return oscillator

    @staticmethod
    def support_resistance(prices: pd.Series, window: int = 20) -> Dict:
        """Calculate support and resistance levels"""
        highs = prices.rolling(window=window).max()
        lows = prices.rolling(window=window).min()

        # Find pivot points
        resistance = highs.rolling(window=5).max()
        support = lows.rolling(window=5).min()

        return {
            'resistance': resistance.iloc[-1] if not resistance.empty else prices.iloc[-1],
            'support': support.iloc[-1] if not support.empty else prices.iloc[-1]
        }

class CryptoDataFetcher:
    """Handles data fetching from CoinGecko API"""

    BASE_URL = "https://api.coingecko.com/api/v3"

    @staticmethod
    def get_top_cryptos(limit: int = 30) -> List[Dict]:
        """Get top cryptocurrencies by market cap"""
        try:
            url = f"{CryptoDataFetcher.BASE_URL}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': limit,
                'page': 1,
                'sparkline': False
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching crypto data: {e}")
            return []

    @staticmethod
    def get_historical_data(coin_id: str, days: int = 7) -> pd.DataFrame:
        """Get historical price data for a cryptocurrency"""
        try:
            url = f"{CryptoDataFetcher.BASE_URL}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly' if days <= 7 else 'daily'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Convert to DataFrame
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            return df
        except Exception as e:
            st.error(f"Error fetching historical data for {coin_id}: {e}")
            return pd.DataFrame()

class TradingAnalyzer:
    """Analyzes cryptocurrency data and provides trading recommendations"""

    def __init__(self):
        self.indicators = TechnicalIndicators()

    def analyze_asset(self, coin_data: Dict, historical_data: pd.DataFrame) -> Dict:
        """Comprehensive analysis of a cryptocurrency"""
        if historical_data.empty:
            return self._default_analysis(coin_data)

        prices = historical_data['price']
        current_price = coin_data['current_price']

        # Calculate indicators
        rsi = self.indicators.rsi(prices)
        macd_data = self.indicators.macd(prices)
        bb_data = self.indicators.bollinger_bands(prices)
        tpo = self.indicators.two_pole_oscillator(prices)
        sr_levels = self.indicators.support_resistance(prices)

        # Get latest values
        latest_rsi = rsi.iloc[-1] if not rsi.empty else 50
        latest_macd = macd_data['macd'].iloc[-1] if not macd_data['macd'].empty else 0
        latest_signal = macd_data['signal'].iloc[-1] if not macd_data['signal'].empty else 0
        latest_tpo = tpo.iloc[-1] if not tpo.empty else 0

        # Multi-timeframe analysis (simulated based on available data)
        timeframes = self._analyze_timeframes(prices)

        # Bot recommendation
        bot_rec = self._get_bot_recommendation(
            latest_rsi, latest_macd, latest_signal, latest_tpo, 
            current_price, sr_levels, bb_data
        )

        # Price ranges
        price_ranges = self._calculate_price_ranges(
            current_price, sr_levels, bb_data, bot_rec['type']
        )

        return {
            'symbol': coin_data['symbol'].upper(),
            'name': coin_data['name'],
            'current_price': current_price,
            'price_change_24h': coin_data.get('price_change_percentage_24h', 0),
            'market_cap_rank': coin_data.get('market_cap_rank', 0),
            'bot_recommendation': bot_rec,
            'price_ranges': price_ranges,
            'timeframes': timeframes,
            'indicators': {
                'rsi': latest_rsi,
                'macd': latest_macd,
                'signal': latest_signal,
                'tpo': latest_tpo
            },
            'support_resistance': sr_levels,
            'profitability_score': bot_rec['confidence']
        }

    def _analyze_timeframes(self, prices: pd.Series) -> Dict:
        """Analyze trends across different timeframes (simulated)"""
        if len(prices) < 60:
            return {
                '1h': 'ranging',
                '15m': 'ranging',
                '5m': 'ranging',
                '1m': 'ranging'
            }

        # Simulate different timeframes using available data
        recent_1h = prices.iloc[-60:] if len(prices) >= 60 else prices
        recent_15m = prices.iloc[-15:] if len(prices) >= 15 else prices
        recent_5m = prices.iloc[-5:] if len(prices) >= 5 else prices
        recent_1m = prices.iloc[-1:] if len(prices) >= 1 else prices

        def get_trend(data):
            if len(data) < 2:
                return 'ranging'
            slope = (data.iloc[-1] - data.iloc[0]) / len(data)
            if slope > data.std() * 0.1:
                return 'upward'
            elif slope < -data.std() * 0.1:
                return 'downward'
            else:
                return 'ranging'

        return {
            '1h': get_trend(recent_1h),
            '15m': get_trend(recent_15m),
            '5m': get_trend(recent_5m),
            '1m': get_trend(recent_1m)
        }

    def _get_bot_recommendation(self, rsi, macd, signal, tpo, price, sr_levels, bb_data):
        """Determine the best bot strategy"""
        confidence = 0
        bot_type = "range"

        # Long signals
        long_signals = 0
        if rsi < 30:  # Oversold
            long_signals += 2
            confidence += 20
        elif rsi < 50:
            long_signals += 1
            confidence += 10

        if macd > signal:  # MACD bullish
            long_signals += 2
            confidence += 15

        if tpo < -20:  # TPO oversold
            long_signals += 1
            confidence += 10

        if price < sr_levels['support'] * 1.02:  # Near support
            long_signals += 2
            confidence += 15

        # Short signals
        short_signals = 0
        if rsi > 70:  # Overbought
            short_signals += 2
            confidence += 20
        elif rsi > 50:
            short_signals += 1
            confidence += 10

        if macd < signal:  # MACD bearish
            short_signals += 2
            confidence += 15

        if tpo > 20:  # TPO overbought
            short_signals += 1
            confidence += 10

        if price > sr_levels['resistance'] * 0.98:  # Near resistance
            short_signals += 2
            confidence += 15

        # Determine bot type
        if long_signals >= 4:
            bot_type = "long"
            confidence = min(confidence, 95)
        elif short_signals >= 4:
            bot_type = "short"
            confidence = min(confidence, 95)
        else:
            bot_type = "range"
            confidence = max(30, min(confidence, 80))

        return {
            'type': bot_type,
            'confidence': confidence
        }

    def _calculate_price_ranges(self, current_price, sr_levels, bb_data, bot_type):
        """Calculate entry, upper, and lower price ranges"""
        support = sr_levels['support']
        resistance = sr_levels['resistance']

        if bot_type == "long":
            entry = current_price * 0.99  # Slight discount for entry
            upper = resistance * 0.95
            lower = support * 1.05
            stop_loss = current_price * 0.95
            expected_profit = ((upper - entry) / entry) * 100
        elif bot_type == "short":
            entry = current_price * 1.01  # Slight premium for short entry
            upper = resistance * 0.95
            lower = support * 1.05
            stop_loss = current_price * 1.05
            expected_profit = ((entry - lower) / entry) * 100
        else:  # range
            entry = current_price
            upper = resistance * 0.98
            lower = support * 1.02
            stop_loss = current_price * 0.93 if current_price > (upper + lower) / 2 else current_price * 1.07
            expected_profit = ((upper - lower) / current_price) * 100 * 0.5

        return {
            'entry': round(entry, 6),
            'upper': round(upper, 6),
            'lower': round(lower, 6),
            'stop_loss': round(stop_loss, 6),
            'expected_profit': round(expected_profit, 2)
        }

    def _default_analysis(self, coin_data):
        """Default analysis when historical data is unavailable"""
        current_price = coin_data['current_price']
        price_change = coin_data.get('price_change_percentage_24h', 0)

        # Simple analysis based on 24h change
        if price_change > 5:
            bot_type = "short"
            confidence = 60
        elif price_change < -5:
            bot_type = "long"
            confidence = 60
        else:
            bot_type = "range"
            confidence = 40

        return {
            'symbol': coin_data['symbol'].upper(),
            'name': coin_data['name'],
            'current_price': current_price,
            'price_change_24h': price_change,
            'market_cap_rank': coin_data.get('market_cap_rank', 0),
            'bot_recommendation': {'type': bot_type, 'confidence': confidence},
            'price_ranges': {
                'entry': current_price,
                'upper': current_price * 1.05,
                'lower': current_price * 0.95,
                'stop_loss': current_price * 0.9,
                'expected_profit': 5.0
            },
            'timeframes': {'1h': 'ranging', '15m': 'ranging', '5m': 'ranging', '1m': 'ranging'},
            'indicators': {'rsi': 50, 'macd': 0, 'signal': 0, 'tpo': 0},
            'support_resistance': {'support': current_price * 0.95, 'resistance': current_price * 1.05},
            'profitability_score': confidence
        }

def display_asset_card(analysis: Dict, rank: int):
    """Display a single asset analysis card"""
    bot_rec = analysis['bot_recommendation']
    price_ranges = analysis['price_ranges']
    timeframes = analysis['timeframes']

    # Bot recommendation styling
    bot_class = f"{bot_rec['type']}-bot"

    with st.container():
        col1, col2, col3, col4 = st.columns([1, 2, 2, 2])

        with col1:
            st.markdown(f"**#{rank}**")
            st.markdown(f"**{analysis['symbol']}**")
            st.markdown(f"${analysis['current_price']:.6f}")
            change_color = "🟢" if analysis['price_change_24h'] > 0 else "🔴"
            st.markdown(f"{change_color} {analysis['price_change_24h']:.2f}%")

        with col2:
            st.markdown(f"""
            <div class="bot-recommendation {bot_class}">
                {bot_rec['type'].upper()} BOT<br>
                Confidence: {bot_rec['confidence']:.0f}%
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"**Profitability Score:** {analysis['profitability_score']:.0f}")

        with col3:
            st.markdown("**Price Ranges:**")
            st.markdown(f"Entry: ${price_ranges['entry']:.6f}")
            st.markdown(f"Upper: ${price_ranges['upper']:.6f}")
            st.markdown(f"Lower: ${price_ranges['lower']:.6f}")
            st.markdown(f"Stop Loss: ${price_ranges['stop_loss']:.6f}")
            st.markdown(f"Expected Profit: {price_ranges['expected_profit']:.2f}%")

        with col4:
            st.markdown("**Multi-Timeframe Trends:**")
            for timeframe, trend in timeframes.items():
                trend_class = trend.lower()
                st.markdown(f"""
                <span class="trend-indicator {trend_class}">
                    {timeframe}: {trend.upper()}
                </span>
                """, unsafe_allow_html=True)

            st.markdown(f"**S/R Levels:**")
            st.markdown(f"Support: ${analysis['support_resistance']['support']:.6f}")
            st.markdown(f"Resistance: ${analysis['support_resistance']['resistance']:.6f}")

        st.markdown("---")

def main():
    load_css()

    # Header
    st.markdown('<h1 class="main-header">🚀 Crypto Trading Scanner</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Theme toggle
        theme = st.selectbox("🎨 Theme", ["Dark", "Light"])

        # Auto-refresh toggle
        auto_refresh = st.checkbox("🔄 Auto-refresh (60s)", value=True)

        # Manual refresh button
        if st.button("🔄 Refresh Now"):
            st.rerun()

        # Info
        st.markdown("---")
        st.markdown("### 📊 About")
        st.markdown("""
        This scanner analyzes the top 30 cryptocurrencies and recommends 
        the most profitable trading bot strategies based on:

        - **Technical Indicators**: RSI, MACD, Bollinger Bands, Two-Pole Oscillator
        - **Support/Resistance Levels**
        - **Multi-timeframe Analysis**
        - **Risk Management**: Stop-loss and profit targets
        """)

        st.markdown("---")
        st.markdown("**Data Source:** CoinGecko API")
        st.markdown("**Update Frequency:** Every 60 seconds")

    # Main content
    if 'last_update' not in st.session_state:
        st.session_state.last_update = 0

    # Check if we need to refresh data
    current_time = time.time()
    should_refresh = (current_time - st.session_state.last_update) > 60 or 'analysis_results' not in st.session_state

    if should_refresh:
        with st.spinner("🔍 Scanning cryptocurrencies..."):
            # Fetch data
            fetcher = CryptoDataFetcher()
            analyzer = TradingAnalyzer()

            crypto_data = fetcher.get_top_cryptos(30)

            if crypto_data:
                analysis_results = []

                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, coin in enumerate(crypto_data):
                    status_text.text(f"Analyzing {coin['name']} ({i+1}/30)...")

                    # Get historical data
                    historical_data = fetcher.get_historical_data(coin['id'], days=7)

                    # Analyze
                    analysis = analyzer.analyze_asset(coin, historical_data)
                    analysis_results.append(analysis)

                    # Update progress
                    progress_bar.progress((i + 1) / 30)

                    # Small delay to prevent API rate limiting
                    time.sleep(0.1)

                # Sort by profitability score
                analysis_results.sort(key=lambda x: x['profitability_score'], reverse=True)

                # Store in session state
                st.session_state.analysis_results = analysis_results
                st.session_state.last_update = current_time

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                # Show refresh indicator
                st.success("✅ Data updated successfully!")
            else:
                st.error("❌ Failed to fetch cryptocurrency data. Please try again.")
                return

    # Display results
    if 'analysis_results' in st.session_state:
        results = st.session_state.analysis_results

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            long_bots = sum(1 for r in results if r['bot_recommendation']['type'] == 'long')
            st.metric("🟢 Long Bots", long_bots)

        with col2:
            short_bots = sum(1 for r in results if r['bot_recommendation']['type'] == 'short')
            st.metric("🔴 Short Bots", short_bots)

        with col3:
            range_bots = sum(1 for r in results if r['bot_recommendation']['type'] == 'range')
            st.metric("🟡 Range Bots", range_bots)

        with col4:
            avg_confidence = np.mean([r['bot_recommendation']['confidence'] for r in results])
            st.metric("📊 Avg Confidence", f"{avg_confidence:.1f}%")

        st.markdown("---")

        # Display all assets
        st.markdown("### 📈 Trading Recommendations (Ranked by Profitability)")

        for i, analysis in enumerate(results, 1):
            display_asset_card(analysis, i)

        # Last update time
        last_update_time = datetime.fromtimestamp(st.session_state.last_update)
        st.markdown(f"**Last Updated:** {last_update_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Auto-refresh
    if auto_refresh and 'analysis_results' in st.session_state:
        time_since_update = time.time() - st.session_state.last_update
        if time_since_update >= 60:
            st.rerun()
        else:
            # Show countdown
            remaining = 60 - int(time_since_update)
            st.markdown(f"🔄 Next refresh in: {remaining} seconds")
            time.sleep(1)
            st.rerun()

if __name__ == "__main__":
    main()