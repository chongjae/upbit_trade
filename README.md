# Upbit Volatility Breakout Trading Bot

A high-performance, automated crypto trading bot for Upbit, utilizing the **Volatility Breakout (VB)** strategy with **Volatility Targeting**. It includes a real-time web dashboard to monitor your portfolio and bot status.

## 🚀 Key Features

*   **Algorithmic Trading**:
    *   **Strategy**: Volatility Breakout (Larry Williams).
    *   **Dynamic K**: Adjusts breakout levels based on recent market noise.
    *   **Volatility Targeting**: Dynamically adjusts position size based on asset volatility (ATR) to manage risk.
    *   **Trailing Stop**: Protects profits by securing gains as price rises.
    *   **Protection**: Pump & Dump detection filter.
*   **Web Dashboard**:
    *   Real-time monitoring of Holdings, PnL, and Open Orders.
    *   **Market Watch**: Live strategy indicators (Target Price, MA5, Waiting/Buy Signal status).
    *   Auto-refreshing interface.
*   **Deployment**:
    *   Dockerized for 24/7 server deployment.
    *   Auto-restart mechanism for reliability.
*   **Backtesting**:
    *   Built-in simulation tool (`backtest.py`) to verify strategy performance against historical data (up to 30 days).

## 🛠 Prerequisites

*   Python 3.9+
*   [Upbit Account](https://upbit.com/) & API Keys (Access/Secret)
*   Docker (Optional, recommended for deployment)

## ⚙️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/upbit-trade.git
    cd upbit-trade
    ```

2.  **Configure Environment**
    Create a `.env` file in the root directory and add your Upbit API keys:
    ```bash
    # .env
    access_key=YOUR_UPBIT_ACCESS_KEY
    secret_key=YOUR_UPBIT_SECRET_KEY
    PAPER_MODE=True  # Set to False for Real Trading
    ```

## 🏃‍♂️ How to Run

### Method 1: Docker (Recommended)
The easiest way to run the bot 24/7.

1.  **Build the Image**
    ```bash
    docker build -t upbit-trade .
    ```

2.  **Run the Container**
    ```bash
    docker run -d --name upbit-trade \
      --env-file .env \
      -v $(pwd)/trades.db:/app/trades.db \
      -v $(pwd)/trade.log:/app/trade.log \
      -v $(pwd)/bot_status.json:/app/bot_status.json \
      -p 5000:5000 \
      upbit-trade
    ```

### Method 2: Local Execution
1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start the Bot**
    ```bash
    ./start.sh
    ```
    *   This script runs the trading bot in the background and the web dashboard in the foreground.
    *   Access the dashboard at `http://localhost:5000`.

## 📊 Dashboard

Once running, visit `http://localhost:5000` (or your server IP) to see:
*   **Asset Summary**: Total estimated equity and available KRW.
*   **Market Watch**: Real-time status of monitored coins (e.g., WAITING, BUY SIGNAL).
*   **Holdings**: Current positions and PnL.
*   **Trade History**: Log of recent buy/sell executions.

## 🧪 Backtesting

To simulate the strategy on past data:
```bash
python backtest.py
```
*   This will fetch historical candle data and simulate trading results for the last 30 days.

## 📁 Project Structure

*   `trade.py`: Core trading logic and bot engine.
*   `app.py`: Flask web server for the dashboard.
*   `backtest.py`: Backtesting simulation script.
*   `start.sh`: Shell script to manage processes.
*   `templates/`: HTML templates for the dashboard.

## ⚠️ Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.
