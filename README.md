# 🚀 Crypto Trading Scanner

A comprehensive cryptocurrency trading scanner built with Streamlit that analyzes the top 30 cryptocurrencies and recommends the most profitable trading bot strategies.

## 🌟 Features

### Core Functionality
- **Real-time Analysis**: Continuously scans top 30 cryptocurrencies
- **Bot Recommendations**: Suggests Long, Short, or Range trading strategies
- **Confidence Scoring**: Each recommendation comes with a confidence percentage
- **Price Ranges**: Displays entry, upper, lower, stop-loss, and expected profit levels
- **Multi-timeframe Analysis**: Trend predictions for 1h, 15m, 5m, and 1m timeframes
- **Support & Resistance**: Automatic calculation of key price levels
- **Auto-refresh**: Updates data every 60 seconds
- **Profitability Ranking**: Assets sorted by potential profitability

### Technical Indicators
All indicators are implemented in pure Python/Pandas for deployment compatibility:
- **Custom Two-Pole Oscillator**: Proprietary momentum and trend indicator
- **RSI**: Relative Strength Index for overbought/oversold conditions
- **MACD**: Moving Average Convergence Divergence for trend changes
- **Bollinger Bands**: Volatility and price level indicator
- **Support/Resistance**: Dynamic price level calculations

### User Interface
- **Modern Design**: Clean, professional interface
- **Dark/Light Mode**: Toggle between themes
- **Responsive Layout**: Works on desktop and mobile
- **Real-time Updates**: Live progress indicators and status updates
- **Color-coded Recommendations**: Visual distinction between bot types

## 🚀 Quick Start

### Option 1: Deploy on Streamlit Community Cloud
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy from your forked repository
5. The app will be live in minutes!

### Option 2: Run Locally
```bash
# Clone the repository
git clone https://github.com/yourusername/crypto-trading-scanner.git
cd crypto-trading-scanner

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 📊 How It Works

### Data Source
- **CoinGecko API**: Free, reliable cryptocurrency data
- **No API Key Required**: Works out of the box
- **Rate Limit Friendly**: Built-in delays to prevent API throttling

### Analysis Process
1. **Data Collection**: Fetches top 30 cryptocurrencies by market cap
2. **Historical Analysis**: Retrieves 7 days of hourly price data
3. **Technical Analysis**: Calculates all indicators using custom algorithms
4. **Strategy Recommendation**: Determines optimal bot type based on multiple factors
5. **Risk Assessment**: Calculates stop-loss and profit targets
6. **Ranking**: Sorts assets by profitability potential

### Bot Strategy Logic
- **Long Bot**: Recommended when RSI < 50, MACD bullish, price near support
- **Short Bot**: Recommended when RSI > 50, MACD bearish, price near resistance  
- **Range Bot**: Recommended for sideways markets with clear support/resistance

## 🛠 Technical Details

### Architecture
- **Modular Design**: Separate classes for data fetching, analysis, and indicators
- **Error Handling**: Comprehensive exception handling and fallbacks
- **Performance Optimized**: Efficient data processing and caching
- **Deployment Ready**: No system-level dependencies

### Key Components
- `TechnicalIndicators`: Custom indicator implementations
- `CryptoDataFetcher`: API interaction and data management
- `TradingAnalyzer`: Core analysis and recommendation engine
- `main()`: Streamlit UI and application flow

## 📈 Understanding the Results

### Bot Recommendations
- **🟢 Long Bot**: Buy and hold strategy for upward trends
- **🔴 Short Bot**: Sell strategy for downward trends  
- **🟡 Range Bot**: Buy low, sell high within a range

### Confidence Scores
- **90-100%**: Very high confidence, strong signals
- **70-89%**: High confidence, good signals
- **50-69%**: Medium confidence, mixed signals
- **30-49%**: Low confidence, weak signals

### Price Ranges
- **Entry**: Recommended entry price for the strategy
- **Upper**: Target sell price (resistance level)
- **Lower**: Target buy price (support level)
- **Stop Loss**: Risk management exit price
- **Expected Profit**: Potential profit percentage

## 🔧 Configuration

### Customization Options
- Modify the number of cryptocurrencies analyzed (default: 30)
- Adjust technical indicator parameters
- Change refresh intervals
- Customize confidence scoring weights

### Environment Variables
No environment variables required - the app works out of the box!

## 📱 Mobile Compatibility

The application is fully responsive and works on:
- Desktop browsers
- Mobile phones
- Tablets
- All modern browsers

## 🚨 Disclaimer

**Important**: This tool is for educational and informational purposes only. 

- **Not Financial Advice**: Do not use this as the sole basis for trading decisions
- **Risk Warning**: Cryptocurrency trading involves significant risk
- **Do Your Research**: Always conduct your own analysis before trading
- **Test First**: Use paper trading to test strategies before risking real money

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
```bash
# Clone and setup
git clone https://github.com/yourusername/crypto-trading-scanner.git
cd crypto-trading-scanner
pip install -r requirements.txt

# Run in development mode
streamlit run app.py --server.runOnSave true
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CoinGecko](https://www.coingecko.com/) for providing free cryptocurrency data
- [Streamlit](https://streamlit.io/) for the amazing web app framework
- [Plotly](https://plotly.com/) for interactive charts

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/yourusername/crypto-trading-scanner/issues) page
2. Create a new issue with detailed information
3. Include error messages and steps to reproduce

---

**⭐ If you find this project useful, please give it a star!**