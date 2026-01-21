import os
import json
import time
import math
import uuid
from datetime import datetime, timedelta
import urllib.request
from urllib.parse import urlencode
import sqlite3
import jwt
import hashlib
import logging
import sys
from logging.handlers import TimedRotatingFileHandler
import asyncio
import aiohttp


# --- Logging Configuration ---
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File Handler (Daily Rotation)
    # interval=1, when='midnight', backupCount=7 (keep 7 days)
    handler = TimedRotatingFileHandler("trade.log", when="midnight", interval=1, backupCount=7, encoding='utf-8')
    handler.suffix = "%Y-%m-%d" # Suffix for rotated files: trade.log.2025-01-14
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Stream Handler (Print to Console as well, useful for Docker logs)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()

# Redirect print to logger
class StreamToLogger(object):
   def __init__(self, logger, level=logging.INFO):
      self.logger = logger
      self.level = level
      self.linebuf = ''

   def write(self, buf):
      for line in buf.rstrip().splitlines():
         self.logger.log(self.level, line.rstrip())

   def flush(self):
      pass

sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)

# --- Configuration ---
# Default to True (Paper Mode) unless set to False explicitly
IF_PAPER_MODE = os.environ.get('PAPER_MODE', 'True').lower() == 'true'
INITIAL_BALANCE = 1_000_000  # KRW
PAPER_ACCOUNT_FILE = "paper_account.json"
DB_FILE = "trades.db"
# ALLOCATION removed (Dynamic Volatility Targeting)
# ALLOCATION removed (Dynamic Volatility Targeting)
# STRATEGIES removed (Dynamic Selection)

# --- Risk Management Config ---
RISK_TARGET = 0.02 # 2% risk per trade
MAX_ALLOCATION_PCT = 0.2 # 20% max per coin (Target 5 coins)
TRAILING_STOP_PCT = 0.03 # 3% trailing stop
PUMP_VOL_MULTIPLIER = 10 # 10x volume spike
KIMCHI_PREMIUM_THRESHOLD = 0.05 # 5% limit for buying
BREAKOUT_BUFFER_PCT = 0.001 # 0.1% buffer for signals (Optimized)
COOLDOWN_MINUTES = 5 # 5 minutes cooldown between trades
STRICT_COOLDOWN_MINUTES = 10 # Force wait after any sell before re-buying
STOP_LOSS_COOLDOWN_MINUTES = 30 # 30 minutes cooldown after stop loss
LAST_TRADE_TIME = {} # {market: (datetime, reason)}

# --- Averaging Down Config ---
ENABLE_AVERAGING_DOWN = True
AVERAGING_DOWN_STEP_PCT = 0.03 # 3% drop from avg price
MAX_REBUYS = 1 # Allow 1 additional buy
RSI_OVERBOUGHT_THRESHOLD = 80 # Trigger early exit if in profit


# --- Binance Integration Config ---


# --- Exchange Rate / Premium Helpers ---


# --- Data Fetching (Public/Private API) ---
def send_request(method, endpoint, params=None, auth=False):
    server_url = 'https://api.upbit.com'
    url = server_url + endpoint

    headers = {'Content-Type': 'application/json'}

    if auth:
        access_key = os.environ.get('access_key')
        secret_key = os.environ.get('secret_key')

        if not access_key or not secret_key:
            print("Error: Missing API Keys")
            return None

        payload = {
            'access_key': access_key,
            'nonce': str(uuid.uuid4()),
        }

        if params:
            query_string = urlencode(params)
            m = hashlib.sha512()
            m.update(query_string.encode())
            query_hash = m.hexdigest()
            payload['query_hash'] = query_hash
            payload['query_hash_alg'] = 'SHA512'
            url += '?' + query_string

        encoded_jwt = jwt.encode(payload, secret_key, algorithm='HS256')
        headers['Authorization'] = f'Bearer {encoded_jwt}'
    elif params:
        url += '?' + urlencode(params)

    try:
        req = urllib.request.Request(url, headers=headers, method=method)
        with urllib.request.urlopen(req) as res:
            return json.loads(res.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print(f"Rate Limit Hit (429) on {endpoint}. Sleeping 2s...")
            time.sleep(2)
            # One retry
            try:
                with urllib.request.urlopen(req) as res:
                    return json.loads(res.read().decode())
            except: return None
        print(f"API Error ({endpoint}): {e}")
        return None
    except Exception as e:
        print(f"API Error ({endpoint}): {e}")
        return None

def get_current_price(market):
    return get_current_prices([market]).get(market)

# --- Candle Data Caching ---
CANDLE_CACHE = {} # { (market, type, unit): (timestamp, data) }
CACHE_TTL = 30 # 30 seconds cache for candles

def get_current_prices(markets):
    if not markets: return {}
    joined_markets = ",".join(markets)
    data = send_request('GET', '/v1/ticker', {'markets': joined_markets})
    prices = {}
    if data:
        for item in data:
            prices[item['market']] = float(item['trade_price'])
    return prices

def get_daily_ohlcv(market, count=21):
    cache_key = (market, 'days', 0)
    now = time.time()
    if cache_key in CANDLE_CACHE:
        ts, data = CANDLE_CACHE[cache_key]
        if now - ts < CACHE_TTL: return data

    data = send_request('GET', '/v1/candles/days', {'market': market, 'count': count})
    if data:
        CANDLE_CACHE[cache_key] = (now, data)
    return data

def get_minute_candles(market, unit=1, count=11):
    cache_key = (market, 'minutes', unit)
    now = time.time()
    if cache_key in CANDLE_CACHE:
        ts, data = CANDLE_CACHE[cache_key]
        if now - ts < CACHE_TTL: return data

    data = send_request('GET', f'/v1/candles/minutes/{unit}', {'market': market, 'count': count})
    if data:
        CANDLE_CACHE[cache_key] = (now, data)
    return data

class CoinSelector:
    def __init__(self):
        pass

    def get_market_candidates(self, limit=20):
        # Fetch all KRW markets
        url = "https://api.upbit.com/v1/market/all?isDetails=false"
        try:
            import requests # Helper since we use it only here or standard lib
            # We already have send_request, but it's hardcoded to /v1/ticker for prices
            # Let's use send_request for market list if possible?
            # send_request hardcodes endpoint prefix? Yes.
            res = send_request('GET', '/v1/market/all', {'isDetails': 'false'})
            if not res: return []

            krw_markets = [item['market'] for item in res if item['market'].startswith('KRW-')]

            # Get Ticker Data for all to sort by Volume
            # Upbit /ticker endpoint supports multiple markets string
            if not krw_markets: return []

            # Split into chunks if too many (Upbit limit usually 100?)
            # KRW markets are around 110-120.
            chunks = [krw_markets[i:i + 100] for i in range(0, len(krw_markets), 100)]

            candidates = []

            for chunk in chunks:
                joined = ",".join(chunk)
                tickers = send_request('GET', '/v1/ticker', {'markets': joined})
                if tickers:
                    candidates.extend(tickers)

            # Sort by acc_trade_price_24h (Volume)
            candidates.sort(key=lambda x: x['acc_trade_price_24h'], reverse=True)

            return candidates[:limit]

        except Exception as e:
            print(f"CoinSelector Error: {e}")
            return []

    def select_targets(self, top_n=15, select_count=5):
        candidates = self.get_market_candidates(limit=top_n)

        blacklist = ['USDT', 'USDE', 'USDC', 'DAI', 'FDUSD', 'BTCU']

        filtered = []
        for c in candidates:
            if any(word in c['market'] for word in blacklist): continue
            if c['signed_change_rate'] > 0: # Positive Momentum
                filtered.append(c)

        selected = filtered[:select_count]

        if len(selected) < select_count:
             remaining = select_count - len(selected)
             for c in candidates:
                 if c['market'] not in [s['market'] for s in selected] and not any(word in c['market'] for word in blacklist):
                     selected.append(c)
                     remaining -= 1
                     if remaining == 0: break

        return [c['market'] for c in selected]

def is_market_ok():
    """
    Check if BTC is relatively stable/bullish on 1h timeframe
    Returns True if Buy is allowed
    """
    try:
        btc_candles = get_minute_candles('KRW-BTC', 60, 20)
        if not btc_candles: return True

        closes = [c['trade_price'] for c in btc_candles]
        ma20 = sum(closes) / len(closes)
        current_price = closes[0]

        # If BTC is 2% below MA20, it's likely a crash/heavy dump
        if current_price < ma20 * 0.98:
            print(f"[MARKET SAFETY] KRW-BTC ({current_price}) < MA20 ({ma20:.0f}) * 0.98 -> HALT BUYS")
            return False
        return True
    except:
        return True

# --- Binance Async Fetcher ---


def get_atr(candles, period=14):
    if len(candles) < period + 1: return 0
    # Candles: Newest -> Oldest. Reverse for calculation
    data = candles[::-1] # Oldest -> Newest

    trs = []
    for i in range(1, len(data)):
        high = data[i]['high_price']
        low = data[i]['low_price']
        prev_close = data[i-1]['trade_price']

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)

    if not trs: return 0

    # Simple Moving Average for ATR or Wilder's Smoothing?
    # Using Wilder's Smoothing
    if len(trs) < period: return sum(trs)/len(trs)

    atr = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period

    return atr

def get_rsi(candles, period=14):
    if len(candles) < period + 1: return 50 # Not enough data
    # Candles are typically Newest -> Oldest. We need chronological order for calc.
    # But let's check input. get_daily_ohlcv returns Newest first.
    closes = [c['trade_price'] for c in candles]
    closes.reverse() # Oldest to Newest

    deltas = [closes[i+1] - closes[i] for i in range(len(closes)-1)]

    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Smooth RSI
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_bollinger_bands(candles, period=20, k=2):
    if len(candles) < period: return 0, 0, 0
    target = candles[:period]
    closes = [c['trade_price'] for c in target]

    ma = sum(closes) / len(closes)
    variance = sum([(c - ma) ** 2 for c in closes]) / len(closes)
    std_dev = math.sqrt(variance)

    upper = ma + (k * std_dev)
    lower = ma - (k * std_dev)

    return upper, ma, lower

def get_macd(candles, short=12, long=26, signal=9):
    if len(candles) < long + signal: return 0, 0, 0

    # Chronological: Old -> New
    closes = [c['trade_price'] for c in candles]
    closes.reverse()

    # EMA Helper
    def calc_ema(data, period):
        k = 2 / (period + 1)
        ema = [data[0]] # Initialize with SMA? Standard usually SMA first, then EMA.
        # But simple EMA start with first value is common approx.
        # Ideally SMA first.
        sma_first = sum(data[:period])/period
        ema = [sma_first]

        for i in range(period, len(data)):
            val = (data[i] * k) + (ema[-1] * (1 - k))
            ema.append(val)
        return ema

    # We need full series for MACD history (for divergence check)
    # But Divergence requires Swing Points.
    # Let's just calculate the EMA series properly.

    # Re-using Pandas would be easier but no pandas allowed?
    # Let's stick to simple iterative calculation.

    # 1. EMAs
    # Need sufficient history for EMA to converge.
    # If we pass 100~200 candles, it should be fine.

    # Standard MACD:
    # EMA12
    # EMA26
    # MACD Line = EMA12 - EMA26
    # Signal Line = EMA9 of MACD Line

    ema12_val = 0
    ema26_val = 0
    # ... This is getting complex to code efficiently in one function without Pandas.
    # Let's simple implementation:

    exp12 = []
    exp26 = []

    # EMA 12
    k12 = 2 / (12 + 1)
    if len(closes) > 12:
        val = sum(closes[:12]) / 12
        exp12.append(val)
        for i in range(12, len(closes)):
             val = (closes[i] * k12) + (exp12[-1] * (1 - k12))
             exp12.append(val)

    # EMA 26
    k26 = 2 / (26 + 1)
    if len(closes) > 26:
        val = sum(closes[:26]) / 26
        exp26.append(val)
        for i in range(26, len(closes)):
             val = (closes[i] * k26) + (exp26[-1] * (1 - k26))
             exp26.append(val)

    # Align lengths ??
    # We need the TAIL (Latest).
    # closes[i] corresponds to...
    # It's better to compute full arrays aligned to 'closes' index.

    # Let's just implement a robust stateful or full-array computation.

    macd_line = []

    # We can only compute MACD where both EMAs exist.
    # EMA26 starts at index 26 (0-indexed 25).
    # So MACD starts at index 26.

    # Actually, let's simplify. We only need the *current* and *recent* MACD values for divergence.
    # We need a list of MACD values.

    # Re-calc:
    ema12 = sum(closes[:12])/12
    ema26 = sum(closes[:26])/26

    # Current state EMAs
    # We need a list of EMAs corresponding to time.

    emas_12 = [ema12] * 26 # padding? No.
    # ...

    # Valid simplified approach:
    # Just compute iterative from start.
    e12 = closes[0]
    e26 = closes[0]

    macds = []

    for i in range(len(closes)):
        e12 = (closes[i] * k12) + (e12 * (1 - k12))
        e26 = (closes[i] * k26) + (e26 * (1 - k26))
        macds.append(e12 - e26)

    # Signal (EMA 9 of MACD)
    sig_k = 2 / (9 + 1)
    signal = macds[0]
    histograms = []

    macd_signals = []

    for i in range(len(macds)):
        signal = (macds[i] * sig_k) + (signal * (1 - sig_k))
        macd_signals.append(signal)
        histograms.append(macds[i] - signal)

    # Return lists for analysis
    return macds, macd_signals, histograms

def get_adx(candles, period=14):
    if len(candles) < period * 2: return 0 # Need history for smoothing

    # Chronological: Old -> New
    # candles argument is typically New->Old from API if not reversed yet?
    # get_daily_ohlcv returns Newest First (index 0 is today).
    # We need to reverse it.

    data = candles[::-1] # Old -> New

    highs = [c['high_price'] for c in data]
    lows = [c['low_price'] for c in data]
    closes = [c['trade_price'] for c in data]

    # TR, +DM, -DM
    trs = []
    plus_dms = []
    minus_dms = []

    # First value
    trs.append(0)
    plus_dms.append(0)
    minus_dms.append(0)

    for i in range(1, len(data)):
        h = highs[i]
        l = lows[i]
        prev_c = closes[i-1]
        prev_h = highs[i-1]
        prev_l = lows[i-1]

        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)

        up_move = h - prev_h
        down_move = prev_l - l

        if up_move > down_move and up_move > 0:
            plus_dms.append(up_move)
        else:
            plus_dms.append(0)

        if down_move > up_move and down_move > 0:
            minus_dms.append(down_move)
        else:
            minus_dms.append(0)

    # Smoothing (Wilder's)
    # Start at index 'period'
    # First smoothed value = sum of first 'period' values

    smooth_tr = []
    smooth_plus = []
    smooth_minus = []

    # Initialize with SMA of first 'period'
    # Note: trs[0] is garbage (0), but usually ignored or handled in Wilder?
    # Wilder starts calc from period+1.

    if len(trs) < period + 1: return 0

    # First smoothed value (Sum of periods)
    # Exclude index 0?
    # Standard: Sum of first N true ranges.

    current_tr = sum(trs[1:period+1])
    current_plus = sum(plus_dms[1:period+1])
    current_minus = sum(minus_dms[1:period+1])

    # Compute subsequent values
    dxs = []

    for i in range(period + 1, len(trs)):
        current_tr = current_tr - (current_tr / period) + trs[i]
        current_plus = current_plus - (current_plus / period) + plus_dms[i]
        current_minus = current_minus - (current_minus / period) + minus_dms[i]

        di_plus = (current_plus / current_tr * 100) if current_tr != 0 else 0
        di_minus = (current_minus / current_tr * 100) if current_tr != 0 else 0

        dx_sum = di_plus + di_minus
        dx = (abs(di_plus - di_minus) / dx_sum * 100) if dx_sum != 0 else 0
        dxs.append(dx)

    # ADX is smoothed DX
    if len(dxs) < period: return 0

    # First ADX = Avg of first 'period' DXs
    adx = sum(dxs[:period]) / period

    # Smoothing ADX
    for i in range(period, len(dxs)):
        adx = (adx * (period - 1) + dxs[i]) / period

    return adx

# --- Paper Trading System ---
class PaperAccount:
    def __init__(self, filename=PAPER_ACCOUNT_FILE, db_file=DB_FILE):
        self.filename = filename
        self.db_file = db_file
        self.init_db()
        self.load()

    def init_db(self):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS trades
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      market TEXT,
                      side TEXT,
                      price REAL,
                      volume REAL,
                      total REAL)''')
        conn.commit()
        conn.close()

    def load(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                data = json.load(f)
                self.balance = data.get('balance', INITIAL_BALANCE)
                self.positions = data.get('positions', {}) # {market: {'volume': 0, 'avg_price': 0, 'rebuys': 0}}
                self.orders = data.get('orders', []) # List of open orders
                # Trades are now in SQLite, not loaded into memory
        else:
            self.balance = INITIAL_BALANCE
            self.positions = {}
            self.orders = []
            self.save()

    def save(self):
        with open(self.filename, 'w') as f:
            json.dump({
                'balance': self.balance,
                'positions': self.positions,
                'orders': self.orders,
                'last_updated': str(datetime.now())
            }, f, indent=4)

    def get_position(self, market):
        return self.positions.get(market, {'volume': 0, 'avg_price': 0, 'rebuys': 0})

    def update_position(self, market, volume, price, side):
        pos = self.get_position(market)
        curr_vol = pos['volume']
        curr_avg = pos['avg_price']

        if side == 'bid': # Buy
            new_vol = curr_vol + volume
            # Avg price weighted average
            new_avg = ((curr_vol * curr_avg) + (volume * price)) / new_vol if new_vol > 0 else 0

            # Rebuys tracking
            rebuys = pos.get('rebuys', 0)
            if curr_vol > 0: # This is a re-buy
                rebuys += 1

            self.positions[market] = {'volume': new_vol, 'avg_price': new_avg, 'rebuys': rebuys}
        elif side == 'ask': # Sell
            new_vol = curr_vol - volume
            if new_vol < 0: new_vol = 0 # Should not happen if logic correct
            self.positions[market] = {'volume': new_vol, 'avg_price': curr_avg} # Sell doesn't change avg buy price
            if new_vol == 0:
                if market in self.positions: del self.positions[market]

    def get_estimated_equity(self, current_prices=None):
        val = 0
        for m, pos in self.positions.items():
            price = pos['avg_price']
            if current_prices and m in current_prices:
                price = current_prices[m]
            val += pos['volume'] * price
        return self.balance + val

    def log_trade(self, market, side, price, volume, total):
        try:
            conn = sqlite3.connect(self.db_file)
            c = conn.cursor()
            c.execute("INSERT INTO trades (timestamp, market, side, price, volume, total) VALUES (?, ?, ?, ?, ?, ?)",
                      (str(datetime.now()), market, side, price, volume, total))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"DB Log Error: {e}")

    # --- Order Execution Simulation ---
    def buy_market(self, market, amount_krw, current_price):
        fee = 0.0005 # 0.05%
        if self.balance < amount_krw:
            print(f"[Paper] Insufficient KRW for {market}")
            return False

        real_invest = amount_krw * (1 - fee)
        volume = real_invest / current_price

        self.balance -= amount_krw
        import logging
        logging.info(f"[Paper] {market} BUY Executed. Spent: {amount_krw:,.0f} | New Balance: {self.balance:,.0f}")
        self.update_position(market, volume, current_price, 'bid')
        self.log_trade(market, 'bid', current_price, volume, amount_krw)
        self.save()  # Ensure state is saved
        print(f"[Paper] BUY MARKET {market}: {volume:.4f} @ {current_price} (Total {amount_krw} KRW)")
        return True

    def sell_market(self, market, volume, current_price):
        fee = 0.0005
        pos = self.get_position(market)
        if pos['volume'] < volume * 0.999: # Tolerance
            print(f"[Paper] Insufficient volume for {market}")
            return False

        amount_krw = volume * current_price * (1 - fee)
        self.balance += amount_krw
        import logging
        logging.info(f"[Paper] {market} SELL Executed. Recv: {amount_krw:,.0f} | New Balance: {self.balance:,.0f}")
        self.update_position(market, volume, current_price, 'ask')
        self.log_trade(market, 'ask', current_price, volume, amount_krw)
        self.save()  # Ensure state is saved
        print(f"[Paper] SELL MARKET {market}: {volume:.4f} @ {current_price} (Total {amount_krw} KRW)")
        return True

    def place_limit_order(self, market, side, price, volume):
        order = {
            'uuid': str(uuid.uuid4()),
            'market': market,
            'side': side,
            'price': price,
            'volume': volume,
            'created_at': str(datetime.now())
        }

        if side == 'bid':
            cost = price * volume * 1.0005
            if self.balance < cost:
                return None
            self.balance -= cost
        elif side == 'ask':
            pos = self.get_position(market)
            if pos['volume'] < volume:
                return None
            self.update_position(market, volume, price, 'ask')
            pass

        self.orders.append(order)
        self.save()
        print(f"[Paper] LIMIT ORDER PLACED {market} {side}: {volume:.4f} @ {price}")
        return order['uuid']

    def cancel_order(self, order_id):
        self.orders = [o for o in self.orders if o['uuid'] != order_id]
        self.save()

    def check_orders(self, current_prices):
        filled_orders = []
        remaining_orders = []

        for order in self.orders:
            market = order['market']
            if market not in current_prices:
                remaining_orders.append(order)
                continue

            curr = current_prices[market]
            side = order['side']
            price = order['price']

            filled = False
            if side == 'bid' and curr <= price:
                # Buy Limit Filled
                # Funds were locked (deducted) at placement, so we just add the position.
                self.update_position(market, order['volume'], price, 'bid')
                total = price * order['volume'] * 1.0005
                self.log_trade(market, 'bid', price, order['volume'], total)
                filled = True
            elif side == 'ask' and curr >= price:
                # Sell Limit Filled
                # Assets were locked (deducted) at placement.
                # We just credit the Earnings to balance. Do NOT deduct volume again.
                earn = price * order['volume'] * 0.9995
                self.balance += earn
                total = earn
                self.log_trade(market, 'ask', price, order['volume'], total)
                filled = True

            if filled:
                print(f"[Paper] ORDER FILLED {market} {side}: {order['volume']:.4f} @ {price} (Curr: {curr})")
                filled_orders.append(order)
            else:
                remaining_orders.append(order)

        self.orders = remaining_orders
        if filled_orders:
            self.save()
        return filled_orders

# --- Strategies ---

# --- Trailing Stop Manager ---
class TrailingStopManager:
    def __init__(self, filename="trailing_state.json"):
        self.filename = filename
        self.state = {} # {market: highest_price}
        self.load()

    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    self.state = json.load(f)
            except: self.state = {}

    def save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.state, f)

    def update_high(self, market, current_price):
        if market not in self.state:
            self.state[market] = current_price
            self.save()
        else:
            if current_price > self.state[market]:
                print(f"[{market}] New Trailing High: {current_price} (was {self.state[market]})")
                self.state[market] = current_price
                self.save()

    def check_stop(self, market, current_price, threshold=0.03):
        if market not in self.state: return False
        high = self.state[market]
        stop_price = high * (1 - threshold)
        if current_price < stop_price:
            print(f"[{market}] Trailing Stop Triggered! Curr: {current_price} < Stop: {stop_price:.2f} (High: {high})")
            return True
        return False

    def reset(self, market):
        if market in self.state:
            del self.state[market]
            self.save()

trailing_manager = TrailingStopManager()

class VolatilityBreakoutBot:
    def __init__(self, account, market):
        self.account = account
        self.market = market
        self.status = "IDLE"

    def check_pump_dump(self):
        # 3.4 Pump & Dump Filter
        # Current 1-min vol > 10 * Avg 10-min vol -> Skip
        try:
            candles = get_minute_candles(self.market, 1, 11)
            if not candles or len(candles) < 11: return False

            curr_vol = candles[0]['candle_acc_trade_volume']
            past_candles = candles[1:]
            avg_vol = sum(c['candle_acc_trade_volume'] for c in past_candles) / len(past_candles)

            if avg_vol > 0 and curr_vol > avg_vol * PUMP_VOL_MULTIPLIER:
                print(f"[{self.market}] Pump Detected! Vol: {curr_vol:.2f} > 10x Avg: {avg_vol:.2f}")
                return True
        except Exception as e:
            print(f"Pump Check Error: {e}")
        return False

    def run(self, current_price, candles, current_time=None, allow_buy=True):
        """
        candles: Upbit daily candles
        """
        if current_time is None: current_time = datetime.now()

        target_candles = candles
        price_for_signal = current_price

        if not target_candles or len(target_candles) < 20: return {}

        # --- Indicator Calc ---
        target_pool = target_candles[1:21] # Prev 20
        noise_ratios = []
        for c in target_pool:
            rng = c['high_price'] - c['low_price']
            if rng > 0:
                noise = 1 - (abs(c['opening_price'] - c['trade_price']) / rng)
                noise_ratios.append(noise)
        k = sum(noise_ratios) / len(noise_ratios) if noise_ratios else 0.5

        # Target Price
        today_open = target_candles[0]['opening_price']
        prev = target_candles[1]
        rng = prev['high_price'] - prev['low_price']
        target_price = today_open + (rng * k)

        # MA5
        closes = [c['trade_price'] for c in target_pool[:5]]
        ma5 = sum(closes) / len(closes) if closes else 0

        # ATR (for Sizing)
        atr = get_atr(target_candles, 14)

        pos = self.account.get_position(self.market)
        has_position = pos['volume'] * current_price > 5000

        print(f"[{self.market}] Curr: {current_price} | Target: {target_price:.2f} | MA5: {ma5:.2f} | k: {k:.2f}")

        # Status Dict
        gap_pct = ((current_price / target_price) - 1) * 100 if target_price > 0 else 0

        status = {
            'market': self.market,
            'current_price': current_price,
            'target_price': target_price,
            'ma5': ma5,
            'gap_pct': gap_pct,
            'k': k,
            'atr': atr,
            'vol_pct': (atr / current_price * 100) if current_price > 0 else 0,
            'dynamic_stop_pct': max(0.01, min(0.05, (atr * 2 / current_price))) if current_price > 0 else 0.03,
            'condition': 'WAIT'
        }

        if not has_position:
            if not allow_buy:
                status['condition'] = 'CLOSE_ONLY_MODE'
                return status

            # --- Buying Logic ---
            # Added breakout buffer to reduce churn
            if current_price > target_price * (1 + BREAKOUT_BUFFER_PCT) and current_price > ma5:
                # Check Cooldown
                last_trade = LAST_TRADE_TIME.get(self.market)
                if last_trade:
                    elapsed = (current_time - last_trade[0]).total_seconds()

                    # 1. Strict Cooldown (Force wait after ANY sell)
                    if elapsed < STRICT_COOLDOWN_MINUTES * 60:
                        print(f"[{self.market}] [VB] SKIP BUY: In Strict Cooldown ({elapsed/60:.1f}/{STRICT_COOLDOWN_MINUTES} min)")
                        return status

                    # 2. Heuristic Cooldown (e.g. 5 min between normal trades)
                    if elapsed < COOLDOWN_MINUTES * 60:
                         print(f"[{self.market}] [VB] SKIP BUY: In Normal Cooldown")
                         return status

                    # 3. Stop-Loss specific Cooldown (penalty)
                    elif last_trade[1] == 'SELL_TS' and elapsed < STOP_LOSS_COOLDOWN_MINUTES * 60:
                        # If in the extended cooldown window, check if trend has actually recovered
                        # Strict re-entry: Price must be above MA5 with some cushion
                        if current_price < ma5 * (1 + BREAKOUT_BUFFER_PCT):
                            print(f"[{self.market}] [VB] SKIP BUY: Extended SL Cooldown & Price < MA5")
                            return status

                # Sizing
                equity = self.account.get_estimated_equity({self.market: current_price})
                risk_amt = equity * RISK_TARGET
                vol_pct = (atr / current_price) if current_price > 0 else 0.01
                if vol_pct == 0: vol_pct = 0.01

                ideal_budget = risk_amt / vol_pct
                max_budget = equity * MAX_ALLOCATION_PCT
                budget = min(ideal_budget, max_budget)
                spendable = min(self.account.balance, budget)

                if spendable > 5000:
                    print(f"[{self.market}] [VB] BUY (Price: {current_price} > Tgt: {target_price} + Buffer)")
                    # distance check: Don't buy if price is > 10% from target_price
                    if current_price > target_price * 1.1:
                        print(f"[{self.market}] [VB] SKIP BUY: Too far from target (Distance: {gap_pct:.2f}%)")
                        return status

                    if self.account.buy_market(self.market, spendable, current_price):
                        LAST_TRADE_TIME[self.market] = (current_time, 'BUY')
                        trailing_manager.reset(self.market)
                        trailing_manager.update_high(self.market, current_price)
                        status['condition'] = 'BUY_SIGNAL_VB'
        else:
            # --- Selling Logic ---
            trailing_manager.update_high(self.market, current_price)
            # MA5-based exit removed for sensitivity; relying on Trailing Stop (3%)

            # --- Overbought Early Exit ---
            rsi = get_rsi(target_candles, 14)
            avg_price = pos['avg_price']
            if current_price > avg_price and rsi > RSI_OVERBOUGHT_THRESHOLD:
                print(f"[{self.market}] [Overbought Exit] Price {current_price} > Avg {avg_price} & RSI {rsi:.1f} > {RSI_OVERBOUGHT_THRESHOLD}. SELL.")
                if self.account.sell_market(self.market, pos['volume'], current_price):
                    LAST_TRADE_TIME[self.market] = (current_time, 'SELL_OVERBOUGHT')
                    trailing_manager.reset(self.market)
                    status['condition'] = 'SELL_OVERBOUGHT'
                return status

            # --- Averaging Down Logic ---
            if ENABLE_AVERAGING_DOWN and pos.get('rebuys', 0) < MAX_REBUYS:
                # RSI check for exhaustion
                rsi = get_rsi(target_candles, 14)
                avg_price = pos['avg_price']
                if current_price < avg_price * (1 - AVERAGING_DOWN_STEP_PCT) and rsi < 30:
                    print(f"[{self.market}] [Averaging Down] Price {current_price} < Avg {avg_price} + RSI {rsi:.1f} < 30. Buying more.")

                    equity = self.account.get_estimated_equity({self.market: current_price})
                    # Re-calc sizing for additional buy
                    risk_amt = equity * RISK_TARGET
                    vol_pct = (atr / current_price) if current_price > 0 else 0.01
                    ideal_budget = risk_amt / vol_pct
                    max_budget = equity * MAX_ALLOCATION_PCT

                    # Remaining allocation for this coin
                    current_val = pos['volume'] * current_price
                    allowed_extra = max_budget - current_val

                    budget = min(ideal_budget, allowed_extra)
                    spendable = min(self.account.balance, budget)

                    if spendable > 5000:
                        if self.account.buy_market(self.market, spendable, current_price):
                            print(f"[{self.market}] [Averaging Down] Executed Re-buy ({pos['rebuys']+1}/{MAX_REBUYS})")
                            # We don't reset trailing high here?
                            # Usually keep the high or update it to current if higher (handled above)
                            # But we might want to be more conservative.
                            # For now, just continue.
                            pass

            # Trailing Stop (Dynamic based on 2*ATR, capped between 1% and 5%)
            dynamic_stop = max(0.01, min(0.05, (atr * 2 / current_price))) if current_price > 0 else TRAILING_STOP_PCT
            if trailing_manager.check_stop(self.market, current_price, dynamic_stop):
                 print(f"[{self.market}] [Trailing Stop] SELL (Stop: {dynamic_stop*100:.2f}%)")
                 pos = self.account.get_position(self.market)
                 if self.account.sell_market(self.market, pos['volume'], current_price):
                     # Apply 60-minute penalty for stop-loss
                     LAST_TRADE_TIME[self.market] = (current_time, 'SELL_TS')
                     trailing_manager.reset(self.market)
                     status['condition'] = 'SELL_TS'
                 return status

        return status

class ConnorsRSIBot:
    def __init__(self, account, market):
        self.account = account
        self.market = market
        self.status = "IDLE"

    def run(self, current_price, candles, current_time=None):
        # candles: Daily candles (count=200+) needed for MA200
        if not candles or len(candles) < 200:
            return {}

        if current_time is None: current_time = datetime.now()

        # --- Indicator Calc ---
        # MA200 (Trend Filter) - Use last 200 closed candles
        closes = [c['trade_price'] for c in candles[1:]] # History (Yesterday backwards)

        if len(closes) < 200: return {}

        ma200 = sum(closes[:200]) / 200

        # MA5 (Exit) - Short term mean reversion
        ma5 = sum(closes[:5]) / 5

        # RSI(2) - We need to calculate this on the fly including TODAY's current price
        # Construct a synthetic list of closes: [Current_Price, Yesterday_Close, ...]
        live_closes = [current_price] + closes

        # RSI calc helper expects chronological (Old -> New).
        # live_closes is New -> Old. Reverse it.
        rsi_src = live_closes[:100] # Take enough for RSI calc
        rsi_src.reverse() # Now Old -> New [..., Yesterday, Today]

        # Calculate RSI(2)
        deltas = [rsi_src[i+1] - rsi_src[i] for i in range(len(rsi_src)-1)]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        period = 2

        if len(deltas) < period: rsi2 = 50
        else:
            avg_gain = sum(gains[:period]) / period
            avg_loss = sum(losses[:period]) / period

            # Smoothing (Wilder's)
            for i in range(period, len(deltas)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            rsi2 = 100 - (100 / (1 + rs))

        # ATR for Sizing (Vol Targeting still creates good hygiene, though CRSI doesn't explicitly use it)
        atr = get_atr(candles, 14)

        print(f"[{self.market}] [CRSI] Curr: {current_price} | MA200: {ma200:.0f} | MA5: {ma5:.0f} | RSI2: {rsi2:.1f}")

        status = {
            'market': self.market,
            'current_price': current_price,
            'target_price': 0, # Not applicable
            'ma5': ma5,
            'k': 0,
            'atr': atr,
            'vol_pct': (atr / current_price * 100) if current_price > 0 else 0,
            'condition': 'WAIT_CRSI'
        }

        pos = self.account.get_position(self.market)
        has_position = pos['volume'] * current_price > 5000

        if not has_position:
            # --- Buy Logic ---
            # 1. Trend: Price > MA200
            # 2. Trigger: RSI2 < 5

            if current_price > ma200:
                if rsi2 < 5:
                    # BUY
                    equity = self.account.get_estimated_equity({self.market: current_price})

                    # Position Sizing: Use Volatility Targeting for consistency
                    risk_amt = equity * RISK_TARGET
                    vol_pct = (atr / current_price) if current_price > 0 else 0.01
                    if vol_pct == 0: vol_pct = 0.01
                    ideal_budget = risk_amt / vol_pct
                    max_budget = equity * MAX_ALLOCATION_PCT
                    budget = min(ideal_budget, max_budget)

                    spendable = min(self.account.balance, budget)

                    if spendable > 5000:
                         # Check Cooldown
                         last_trade = LAST_TRADE_TIME.get(self.market)
                         if last_trade:
                             elapsed = (current_time - last_trade[0]).total_seconds()

                             # 1. Strict Cooldown
                             if elapsed < STRICT_COOLDOWN_MINUTES * 60:
                                 print(f"[{self.market}] [CRSI] SKIP BUY: In Strict Cooldown ({elapsed/60:.1f}/{STRICT_COOLDOWN_MINUTES} min)")
                                 return status

                             # 2. Heuristic Cooldown
                             if elapsed < COOLDOWN_MINUTES * 60:
                                 print(f"[{self.market}] [CRSI] SKIP BUY: In Normal Cooldown")
                                 return status

                             # 3. Stop-Loss check
                             elif last_trade[1] == 'SELL_TS' and elapsed < STOP_LOSS_COOLDOWN_MINUTES * 60:
                                 if current_price < ma5 * (1 + BREAKOUT_BUFFER_PCT):
                                     print(f"[{self.market}] [CRSI] SKIP BUY: Extended SL Cooldown & Price < MA5")
                                     return status

                         print(f"[{self.market}] [CRSI] BUY SIGNAL (RSI2: {rsi2:.1f} < 5)")
                         if self.account.buy_market(self.market, spendable, current_price):
                             LAST_TRADE_TIME[self.market] = (current_time, 'BUY')
                             trailing_manager.reset(self.market)
                             trailing_manager.update_high(self.market, current_price)
                             status['condition'] = 'BUY_CRSI'
                             status['target_price'] = current_price
                             status['mode'] = 'CRSI_BUY_SIGNAL'
                else:
                    status['condition'] = 'WAIT_RSI'
            else:
                status['condition'] = 'BEAR_TREND'

        else:
            # --- Sell Logic ---
            # 1. Exit: Price > MA5
            # 2. Stop Loss: 3% Trailing Stop (Safety)

            trailing_manager.update_high(self.market, current_price)

            if current_price > ma5 * (1 + BREAKOUT_BUFFER_PCT):
                print(f"[{self.market}] [CRSI] EXIT SIGNAL (Price {current_price} > MA5 {ma5:.0f} * Buffer)")
                pos = self.account.get_position(self.market)
                if self.account.sell_market(self.market, pos['volume'], current_price):
                    LAST_TRADE_TIME[self.market] = (current_time, 'SELL_CRSI_EXIT')
                    trailing_manager.reset(self.market)
                    status['condition'] = 'SELL_CRSI_EXIT'
                return status

            # --- Averaging Down Logic ---
            if ENABLE_AVERAGING_DOWN and pos.get('rebuys', 0) < MAX_REBUYS:
                # RSI check for exhaustion
                rsi = get_rsi(candles, 14)
                avg_price = pos['avg_price']
                if current_price < avg_price * (1 - AVERAGING_DOWN_STEP_PCT) and rsi < 30:
                    print(f"[{self.market}] [CRSI-Averaging] Price {current_price} < Avg {avg_price} + RSI {rsi:.1f} < 30. Buying more.")

                    equity = self.account.get_estimated_equity({self.market: current_price})

                    # Size check
                    if not candles or len(candles) < 20: atr = current_price * 0.01
                    else: atr = get_atr(candles, 14)

                    risk_amt = equity * RISK_TARGET
                    vol_pct = (atr / current_price) if current_price > 0 else 0.01
                    ideal_budget = risk_amt / vol_pct
                    max_budget = equity * MAX_ALLOCATION_PCT

                    current_val = pos['volume'] * current_price
                    allowed_extra = max_budget - current_val

                    budget = min(ideal_budget, allowed_extra)
                    spendable = min(self.account.balance, budget)

                    if spendable > 5000:
                        if self.account.buy_market(self.market, spendable, current_price):
                            print(f"[{self.market}] [CRSI-Averaging] Executed Re-buy ({pos['rebuys']+1}/{MAX_REBUYS})")
                            pass

            if trailing_manager.check_stop(self.market, current_price, 0.03): # 3% hard trailing stop
                 print(f"[{self.market}] [CRSI] Trailing Stop")
                 pos = self.account.get_position(self.market)
                 if self.account.sell_market(self.market, pos['volume'], current_price):
                     LAST_TRADE_TIME[self.market] = (current_time, 'SELL_TS')
                     trailing_manager.reset(self.market)
                     status['condition'] = 'SELL_TS'
                 return status

        return status
class BollingerBandBot:
    def __init__(self, account, market):
        self.account = account
        self.market = market
        self.status = "IDLE"

    def run(self, current_price, candles, current_time=None, allow_buy=True):
        """
        candles: Upbit 15m candles
        """
        if current_time is None: current_time = datetime.now()

        target_candles = candles
        price_for_signal = current_price

        if not target_candles or len(target_candles) < 20: return {}

        # --- Indicator Calc ---
        # BB(20, 2.0) on 15-minute candles
        # Note: k=2.0 standard for crypto usually, removing strict 3.0 logic unless user wants it
        # Original code had k=3.0 for non-USDT. Let's maximize sensitivity with k=2.0 for all for now, or keep logic.
        # User complained about "Leader-Follow" not working. Let's simplify to k=2.0 standard.
        upper, mid, lower = get_bollinger_bands(target_candles, period=20, k=2.0)
        rsi = get_rsi(target_candles, 14)
        atr = get_atr(target_candles, 14)

        # Calculate MA5 for 15m candles for cooldown check
        closes_15m = [c['trade_price'] for c in target_candles[:5]]
        ma5_15m = sum(closes_15m) / len(closes_15m) if closes_15m else 0

        print(f"[{self.market}] [BB] Curr: {current_price} | RSI: {rsi:.1f} | Lower: {lower:.2f} | Upper: {upper:.2f}")

        # Status Dict
        gap_pct = ((current_price / lower) - 1) * 100 if lower > 0 else 0

        status = {
            'market': self.market,
            'current_price': current_price,
            'target_price': lower,
            'gap_pct': gap_pct,
            'atr': atr,
            'vol_pct': (atr / current_price * 100) if current_price > 0 else 0,
            'condition': 'WAIT_BB',
            'mode': 'BB'
        }

        pos = self.account.get_position(self.market)
        has_position = pos['volume'] * current_price > 5000

        if not has_position:
            if not allow_buy:
                status['condition'] = 'CLOSE_ONLY_MODE'
                return status

            # --- Buy Logic ---
            # Condition: Price touched/below Lower and RSI < 30
            if current_price <= lower and rsi < 30:
                 # Check Cooldown
                 last_trade = LAST_TRADE_TIME.get(self.market)
                 if last_trade:
                     elapsed = (current_time - last_trade[0]).total_seconds()
                     if elapsed < COOLDOWN_MINUTES * 60:
                         # 1. Strict Cooldown
                         if elapsed < STRICT_COOLDOWN_MINUTES * 60:
                             print(f"[{self.market}] [BB] SKIP BUY: In Strict Cooldown ({elapsed/60:.1f}/{STRICT_COOLDOWN_MINUTES} min)")
                             return status

                         # 2. Heuristic Cooldown
                         print(f"[{self.market}] [BB] SKIP BUY: In Normal Cooldown")
                         return status
                     elif last_trade[1] == 'SELL_TS' and elapsed < STOP_LOSS_COOLDOWN_MINUTES * 60:
                         # If in the extended cooldown window, check if trend has actually recovered
                         # Strict re-entry: Price must be above MA5 with some cushion
                         if current_price < ma5_15m * (1 + BREAKOUT_BUFFER_PCT):
                             print(f"[{self.market}] [BB] SKIP BUY: Extended SL Cooldown & Price < MA5(15m)")
                             return status

                 # BUY
                 equity = self.account.get_estimated_equity({self.market: current_price})
                 risk_amt = equity * RISK_TARGET
                 vol_pct = (atr / current_price) if current_price > 0 else 0.01
                 ideal_budget = risk_amt / vol_pct
                 max_budget = equity * MAX_ALLOCATION_PCT
                 budget = min(ideal_budget, max_budget)
                 spendable = min(self.account.balance, budget)

                 if spendable > 5000:
                     print(f"[{self.market}] [BB] BUY SIGNAL (Price: {current_price} < Lower: {lower} & RSI: {rsi})")
                     if self.account.buy_market(self.market, spendable, current_price):
                         LAST_TRADE_TIME[self.market] = (current_time, 'BUY')
                         trailing_manager.reset(self.market)
                         trailing_manager.update_high(self.market, current_price)
                         status['condition'] = 'BUY_BB'
            else:
                if current_price <= lower:
                    status['condition'] = 'BB_OVERSOLD_ZONE'
                else:
                    status['condition'] = 'WAIT_BB'

        else:
            # --- Sell Logic ---
            trailing_manager.update_high(self.market, current_price)
            # Exit: Price >= Middle Band
            if current_price >= mid * (1 + BREAKOUT_BUFFER_PCT):
                print(f"[{self.market}] [BB Exit] Price {current_price} >= Mid {mid} * Buffer. SELL.")
                pos = self.account.get_position(self.market)
                if self.account.sell_market(self.market, pos['volume'], current_price):
                    LAST_TRADE_TIME[self.market] = (current_time, 'SELL_BB_EXIT')
                    trailing_manager.reset(self.market)
                    status['condition'] = 'SELL_BB_EXIT'
                return status

            # --- Averaging Down Logic ---
            if ENABLE_AVERAGING_DOWN and pos.get('rebuys', 0) < MAX_REBUYS:
                # RSI check for exhaustion
                rsi = get_rsi(candles, 14)
                avg_price = pos['avg_price']
                if current_price < avg_price * (1 - AVERAGING_DOWN_STEP_PCT) and rsi < 30:
                    print(f"[{self.market}] [BB-Averaging] Price {current_price} < Avg {avg_price} + RSI {rsi:.1f} < 30. Buying more.")

                    equity = self.account.get_estimated_equity({self.market: current_price})
                    risk_amt = equity * RISK_TARGET
                    vol_pct = (atr / current_price) if current_price > 0 else 0.01
                    ideal_budget = risk_amt / vol_pct
                    max_budget = equity * MAX_ALLOCATION_PCT

                    current_val = pos['volume'] * current_price
                    allowed_extra = max_budget - current_val

                    budget = min(ideal_budget, allowed_extra)
                    spendable = min(self.account.balance, budget)

                    if spendable > 5000:
                        if self.account.buy_market(self.market, spendable, current_price):
                            print(f"[{self.market}] [BB-Averaging] Executed Re-buy ({pos['rebuys']+1}/{MAX_REBUYS})")
                            pass

            if trailing_manager.check_stop(self.market, current_price, TRAILING_STOP_PCT):
                 if self.account.sell_market(self.market, pos['volume'], current_price):
                     LAST_TRADE_TIME[self.market] = (current_time, 'SELL_TS')
                     trailing_manager.reset(self.market)
                     status['condition'] = 'SELL_TS'
                 return status

        return status



class DynamicSwitchBot:
    def __init__(self, account, market):
        self.market = market
        self.account = account
        self.vb = VolatilityBreakoutBot(account, market)
        self.crsi = ConnorsRSIBot(account, market)
        self.bb = BollingerBandBot(account, market)
        self.status = "IDLE"

    def run(self, current_price, candles_data, current_time=None, allow_buy=True):
        """
        candles_data: {'daily': [...], '15m': [...]}
        """
        daily_for_adx = candles_data.get('daily')

        if not daily_for_adx: return {}

        # 1. Calculate ADX
        adx = get_adx(daily_for_adx, 14)

        final_status = {}

        if adx >= 22:
             # TREND MODE (VB)
             print(f"[{self.market}] [DYNAMIC] [TREND] ADX: {adx:.1f} >= 22 -> Using VB {'(CloseOnly)' if not allow_buy else ''}")
             final_status = self.vb.run(current_price, candles_data.get('daily'), current_time, allow_buy)
             final_status['mode'] = 'TREND_VB'
             final_status['adx'] = adx
        else:
             # RANGE MODE (BB)
             print(f"[{self.market}] [DYNAMIC] [RANGE] ADX: {adx:.1f} < 22 -> Using BB {'(CloseOnly)' if not allow_buy else ''}")
             # BB (Needs 15m)
             res_bb = self.bb.run(current_price, candles_data.get('15m'), current_time, allow_buy)
             final_status = res_bb
             final_status['adx'] = adx
             final_status['mode'] = 'RANGE_BB'

        return final_status

class MACDDivergenceBot:
    def __init__(self, account, market):
        self.account = account
        self.market = market
        self.status = "IDLE"

    def run(self, current_price, candles, current_time=None):
        """
        candles: Upbit 60m candles
        """
        if current_time is None: current_time = datetime.now()

        target_candles = candles
        price_for_signal = current_price

        if not target_candles or len(target_candles) < 50:
            return {}

        # --- Indicator Calc ---
        macds, signals, hists = get_macd(target_candles, 12, 26, 9)
        if not macds: return {}

        curr_macd = macds[-1]
        curr_hist = hists[-1]
        prev_hist = hists[-2]

        # --- Divergence Detection ---
        closes = [c['trade_price'] for c in target_candles] # Newest -> Oldest (wait, get_daily/minute returns newest first?)
        # Let's verify standard Upbit response. "get_minute_candles" calls send_request.
        # Upbit API returns Newest First.
        # My fetchers loop: `for candle in reversed(ohlcv)` in BinanceFetcher (removed).
        # `get_daily_ohlcv`: returns raw response.
        # `get_minute_candles`: returns raw response.
        # Raw Upbit response is Newest First (Index 0 is latest).
        # My `get_bollinger_bands` expects `candles[:period]`.
        # `get_macd` logic:
        #   closes = [c['trade_price'] for c in candles]
        #   closes.reverse() (Lines 310-311).
        #   So `closes` list inside `get_macd` is Oldest -> Newest.
        #   It returns `macds` list which is Oldest -> Newest.
        #   `macds[-1]` is the LATEST value.

        # Here: `closes` = [c['trade_price'] for c in target_candles].
        # If target_candles is Newest First (default Upbit), then `closes` is Newest First.
        # `closes_rev` = closes[::-1] -> Oldest First.

        closes = [c['trade_price'] for c in target_candles]
        closes_rev = closes[::-1] # Oldest -> Newest

        lookback = 30
        if len(closes_rev) < lookback: return {}

        # recent_window: last 'lookback' candles excluding current?
        # recent_window = closes_rev[-lookback:-1] creates window ending at -2 (Second to last).
        # We look for swing low in history.

        recent_window = closes_rev[-lookback:-1]
        if not recent_window: return {}

        min_price = min(recent_window)
        min_idx_local = recent_window.index(min_price)
        # min_idx_global is index in closes_rev
        min_idx_global = len(closes_rev) - lookback + min_idx_local

        swing_low_price = min_price
        # Align MACD index
        if min_idx_global < len(macds):
             swing_low_macd = macds[min_idx_global]
        else:
             swing_low_macd = 0 # Should not happen

        # Bullish Divergence:
        # Price made Lower Low (Current < Swing Low)
        # MACD made Higher Low (Current > Swing Low MACD)

        is_lower_low_price = price_for_signal < swing_low_price
        is_higher_low_macd = curr_macd > swing_low_macd

        atr = get_atr(target_candles, 14)

        # Calculate MA5 for 60m candles for cooldown check
        closes_60m = [c['trade_price'] for c in target_candles[:5]]
        ma5_60m = sum(closes_60m) / len(closes_60m) if closes_60m else 0

        print(f"[{self.market}] [MACD] Curr: {current_price} | Hist: {curr_hist:.2f} | Div: {'Yes' if (is_lower_low_price and is_higher_low_macd) else 'No'}")

        # Status Dict
        gap_pct = ((current_price / swing_low_price) - 1) * 100 if swing_low_price > 0 else 0

        status = {
            'market': self.market,
            'current_price': current_price,
            'target_price': swing_low_price,
            'gap_pct': gap_pct,
            'atr': atr,
            'vol_pct': (atr / current_price * 100) if current_price > 0 else 0,
            'condition': 'WAIT_MACD',
            'mode': 'MACD_DIVERGENCE'
        }

        pos = self.account.get_position(self.market)
        has_position = pos['volume'] * current_price > 5000

        if not has_position:
            # --- Buy Logic ---
            if is_lower_low_price and is_higher_low_macd:
                if curr_hist > 0:
                     # Check Cooldown
                     last_trade = LAST_TRADE_TIME.get(self.market)
                     if last_trade:
                         elapsed = (current_time - last_trade[0]).total_seconds()
                         if elapsed < COOLDOWN_MINUTES * 60:
                             # 1. Strict Cooldown
                             if elapsed < STRICT_COOLDOWN_MINUTES * 60:
                                 print(f"[{self.market}] [MACD] SKIP BUY: In Strict Cooldown ({elapsed/60:.1f}/{STRICT_COOLDOWN_MINUTES} min)")
                                 return status

                             # 2. Heuristic Cooldown
                             print(f"[{self.market}] [MACD] SKIP BUY: In Normal Cooldown")
                             return status
                         elif last_trade[1] == 'SELL_TS' and elapsed < STOP_LOSS_COOLDOWN_MINUTES * 60:
                             # Use daily MA5 as filter for MACD re-entry too?
                             # MACD candles are 60m. Let's use 60m MA5 for consistency here.
                             if current_price < ma5_60m * (1 + BREAKOUT_BUFFER_PCT):
                                 print(f"[{self.market}] [MACD] SKIP BUY: Extended SL Cooldown & Price < MA5(60m)")
                                 return status

                     equity = self.account.get_estimated_equity({self.market: current_price})
                     risk_amt = equity * RISK_TARGET
                     vol_pct = (atr / current_price) if current_price > 0 else 0.01
                     ideal_budget = risk_amt / vol_pct
                     max_budget = equity * MAX_ALLOCATION_PCT
                     budget = min(ideal_budget, max_budget)
                     spendable = min(self.account.balance, budget)

                     if spendable > 5000:
                         print(f"[{self.market}] [MACD] BUY (Divergence match)")
                         if self.account.buy_market(self.market, spendable, current_price):
                             LAST_TRADE_TIME[self.market] = (current_time, 'BUY')
                             trailing_manager.reset(self.market)
                             trailing_manager.update_high(self.market, current_price)
                             status['condition'] = 'BUY_MACD'
        else:
            # --- Sell Logic ---
            trailing_manager.update_high(self.market, current_price)
            # Exit: Dead Cross (Relatively relaxed with histogram check)
            if curr_hist < - (atr / current_price * 0.1) and prev_hist >= 0:
                print(f"[{self.market}] [MACD Exit] Robust Dead Cross. SELL.")
                pos = self.account.get_position(self.market)
                if self.account.sell_market(self.market, pos['volume'], current_price):
                    LAST_TRADE_TIME[self.market] = (current_time, 'SELL_MACD_CROSS')
                    trailing_manager.reset(self.market)
                    status['condition'] = 'SELL_MACD_CROSS'
                return status

            # --- Averaging Down Logic ---
            if ENABLE_AVERAGING_DOWN and pos.get('rebuys', 0) < MAX_REBUYS:
                # RSI check for exhaustion
                rsi = get_rsi(candles, 14)
                avg_price = pos['avg_price']
                if current_price < avg_price * (1 - AVERAGING_DOWN_STEP_PCT) and rsi < 30:
                    print(f"[{self.market}] [MACD-Averaging] Price {current_price} < Avg {avg_price} + RSI {rsi:.1f} < 30. Buying more.")

                    equity = self.account.get_estimated_equity({self.market: current_price})
                    risk_amt = equity * RISK_TARGET
                    vol_pct = (atr / current_price) if current_price > 0 else 0.01
                    ideal_budget = risk_amt / vol_pct
                    max_budget = equity * MAX_ALLOCATION_PCT

                    current_val = pos['volume'] * current_price
                    allowed_extra = max_budget - current_val

                    budget = min(ideal_budget, allowed_extra)
                    spendable = min(self.account.balance, budget)

                    if spendable > 5000:
                        if self.account.buy_market(self.market, spendable, current_price):
                            print(f"[{self.market}] [MACD-Averaging] Executed Re-buy ({pos['rebuys']+1}/{MAX_REBUYS})")
                            pass

            if trailing_manager.check_stop(self.market, current_price, TRAILING_STOP_PCT):
                 if self.account.sell_market(self.market, pos['volume'], current_price):
                     LAST_TRADE_TIME[self.market] = (current_time, 'SELL_TS')
                     trailing_manager.reset(self.market)
                     status['condition'] = 'SELL_TS'
                 return status

        return status

class RealAccount:
    def __init__(self, db_file=DB_FILE):
        self.db_file = db_file
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS trades
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      market TEXT,
                      side TEXT,
                      price REAL,
                      volume REAL,
                      total REAL)''')
        conn.commit()
        conn.close()

    def get_estimated_equity(self, current_prices=None):
        data = send_request('GET', '/v1/accounts', auth=True)
        total_krw = 0
        if not data: return 0

        for item in data:
            if item['currency'] == 'KRW':
                total_krw += float(item['balance'])
            else:
                curr = float(item['balance'])
                market = f"KRW-{item['currency']}"
                price = float(item['avg_buy_price'])
                if current_prices and market in current_prices:
                    price = current_prices[market]
                total_krw += curr * price
        return total_krw

    def log_trade(self, market, side, price, volume, total):
        try:
            conn = sqlite3.connect(self.db_file)
            c = conn.cursor()
            c.execute("INSERT INTO trades (timestamp, market, side, price, volume, total) VALUES (?, ?, ?, ?, ?, ?)",
                      (str(datetime.now()), market, side, price, volume, total))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"DB Log Error: {e}")

    @property
    def balance(self):
        data = send_request('GET', '/v1/accounts', auth=True)
        if not data: return 0
        for item in data:
            if item['currency'] == 'KRW':
                return float(item['balance'])
        return 0

    @property
    def positions(self):
        data = send_request('GET', '/v1/accounts', auth=True)
        pos = {}
        if not data: return pos
        for item in data:
            if item['currency'] == 'KRW': continue
            market = f"KRW-{item['currency']}"
            pos[market] = {
                'volume': float(item['balance']),
                'avg_price': float(item['avg_buy_price'])
            }
        return pos

    def get_position(self, market):
        # Inefficient for loop but consistent
        return self.positions.get(market, {'volume': 0, 'avg_price': 0})

    def get_order_status(self, uuid):
        for _ in range(5): # Try 5 times (approx 2.5 sec)
            time.sleep(0.5)
            try:
                res = send_request('GET', '/v1/order', {'uuid': uuid}, auth=True)
                if res and res.get('state') == 'done':
                    return res
            except: pass
        return None

    def buy_market(self, market, amount_krw, current_price=None):
        params = {
            'market': market,
            'side': 'bid',
            'ord_type': 'price',
            'price': str(amount_krw)
        }
        res = send_request('POST', '/v1/orders', params, auth=True)
        if res and 'uuid' in res:
            uuid = res['uuid']
            print(f"[REAL] BUY ORDER PLACED {market}: {amount_krw} KRW (UUID: {uuid})")

            # Fetch Actual Execution Details
            order_info = self.get_order_status(uuid)

            if order_info:
                # Real Execution Data
                # For Market Buy: 'price' in response is total amount spent? No, 'price' is None for market?
                # executed_volume is Volume bought.
                # trades key contains details.
                # Actually 'price' field in order info is the price set? For market order it might be null.
                # Use 'trades' to calc avg or just use 'avg_price' if available (Upbit provides 'avg_price' in order detail for filled orders)

                # Upbit Order Response (done):
                # price: null (for market), cost: total spent, paid_fee, executed_volume

                real_price = float(order_info.get('avg_price', current_price or 0))
                real_vol = float(order_info.get('executed_volume', 0))
                real_total = float(order_info.get('cost', amount_krw)) # accurate cost
                fee = float(order_info.get('paid_fee', 0))

                print(f"  => FILLED: {real_vol} @ {real_price} (Cost: {real_total}, Fee: {fee})")
                self.log_trade(market, 'bid', real_price, real_vol, real_total)
            else:
                print(f"  => TIMEOUT WAITING FILL. Logging Estimate.")
                est_vol = amount_krw / current_price if current_price else 0
                self.log_trade(market, 'bid', current_price or 0, est_vol, amount_krw)

            return True
        print(f"[REAL] BUY FAILED: {res}")
        return False

    def sell_market(self, market, volume, current_price=None):
        params = {
            'market': market,
            'side': 'ask',
            'ord_type': 'market',
            'volume': str(volume)
        }
        res = send_request('POST', '/v1/orders', params, auth=True)
        if res and 'uuid' in res:
            uuid = res['uuid']
            print(f"[REAL] SELL ORDER PLACED {market}: {volume} (UUID: {uuid})")

            # Fetch Actual Execution Details
            order_info = self.get_order_status(uuid)

            if order_info:
                real_price = float(order_info.get('avg_price', current_price or 0))
                real_vol = float(order_info.get('executed_volume', volume))
                real_total = float(order_info.get('cost', volume * (current_price or 0)))
                fee = float(order_info.get('paid_fee', 0))

                print(f"  => FILLED: {real_vol} @ {real_price} (Total: {real_total}, Fee: {fee})")
                self.log_trade(market, 'ask', real_price, real_vol, real_total)
            else:
                print(f"  => TIMEOUT WAITING FILL. Logging Estimate.")
                self.log_trade(market, 'ask', current_price or 0, volume, volume * (current_price or 0))

            return True
        print(f"[REAL] SELL FAILED: {res}")
        return False

def load_last_trades(db_file=DB_FILE):
    """
    Populate LAST_TRADE_TIME from the database on startup.
    We look for the latest trade for each market and infer the reason
    based on the side and comments if we had them, but for now we'll
    just look at the side and timestamp.
    """
    global LAST_TRADE_TIME
    try:
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        # Get the latest trade for each market
        c.execute("""
            SELECT market, side, timestamp
            FROM trades
            WHERE id IN (SELECT MAX(id) FROM trades GROUP BY market)
        """)
        rows = c.fetchall()
        for row in rows:
            market, side, ts_str = row
            try:
                # Upbit timestamp format or my log format
                # My log format: str(datetime.now()) -> '2026-01-19 08:48:36.123456'
                # Handling both with a slice
                ts = datetime.strptime(ts_str.split('.')[0], '%Y-%m-%d %H:%M:%S')

                # Heuristic: If it was a SELL, treat it as a potential stop loss for safety
                # Or we could just use a generic 'SELL' reason since we don't store
                # the specific reason in the DB yet.
                # For the cooldown logic, 'SELL_TS' triggers the longer 30m wait.
                # Let's be conservative and treat all DB sells as 'SELL_TS'
                # to ensure safety after a restart.
                reason = 'SELL_TS' if side.lower() in ['ask', 'sell'] else 'BUY'
                LAST_TRADE_TIME[market] = (ts, reason)
                print(f"[*] Loaded Last Trade for {market}: {reason} at {ts}")
            except Exception as e:
                print(f"Error parsing timestamp {ts_str}: {e}")
        conn.close()
    except Exception as e:
        print(f"Database error in load_last_trades: {e}")

# --- Main Loop ---
async def main_loop():
    if IF_PAPER_MODE:
        account = PaperAccount()
        print(f"--- Paper Trading Started (Upbit Only) ---")
    else:
        account = RealAccount()
        print(f"--- REAL TRADING Started (Upbit Only) ---")

    # Load Persistence
    load_last_trades()

    print(f"Balance: {account.balance:,.0f} KRW")

    bots = {}
    coin_selector = CoinSelector()

    # Initial Selection
    print("Selecting Best Targets...")
    target_markets = coin_selector.select_targets(top_n=20, select_count=5)
    print(f"Selected Targets: {target_markets}")

    for market in target_markets:
        bots[market] = DynamicSwitchBot(account, market)

    last_rebalance_time = datetime.now()

    try:
        while True:
            try:
                now = datetime.now()

                # --- Hourly Rebalancing ---
                if (now - last_rebalance_time).total_seconds() > 3600: # 1 Hour
                    print("\n[Rebalancing Targets...]")
                    new_targets = coin_selector.select_targets(top_n=20, select_count=5)
                    print(f"New Targets: {new_targets}")

                    # Update the Global Target List used for allow_buy check
                    target_markets = new_targets

                    # 1. Add new bots
                    for market in new_targets:
                        if market not in bots:
                            print(f"[+] Adding {market}")
                            bots[market] = DynamicSwitchBot(account, market)

                    # 2. Remove old bots (Graceful Exit)
                    current_bot_keys = list(bots.keys())
                    for market in current_bot_keys:
                        if market not in new_targets:
                            # Check position
                            pos = account.get_position(market)
                            current_price = get_current_price(market) or 0
                            if pos['volume'] * current_price > 5000:
                                print(f"[*] Keeping {market} (Has Position) for CLOSE ONLY.")
                                pass
                            else:
                                print(f"[-] Removing {market}")
                                del bots[market]

                    last_rebalance_time = now

                print(f"\n[{now.strftime('%H:%M:%S')}] --- Tick (Upbit Only | {len(bots)} Active) ---")

                # Check Global Market Condition (BTC Filter)
                market_ok = is_market_ok()

                upbit_prices = get_current_prices(list(bots.keys()))

                # 2. Sequential Bot Execution
                bot_stats = {}
                for market, bot in bots.items():
                    if market not in upbit_prices: continue
                    current_price = upbit_prices[market]

                    # Determine Status (Target or Legacy)
                    is_active_target = market in target_markets

                    # Bot can buy only if: 1. It's an active target, 2. Global market is OK
                    allow_buy = is_active_target and market_ok

                    candles_data = {}

                    # For switch bot we need Daily and 15m
                    candles_data['daily'] = get_daily_ohlcv(market, 30)
                    await asyncio.sleep(0.5) # Increased sleep for rate limit safety
                    candles_data['15m'] = get_minute_candles(market, 15, 30)
                    await asyncio.sleep(0.5)

                    # Run Bot with allow_buy flag
                    res = bot.run(current_price, candles_data, allow_buy=allow_buy)
                    if res: bot_stats[market] = res

                # Write Status
                with open("bot_status.json", "w") as f:
                    json.dump(bot_stats, f, indent=4)

                # Summary Print
                est_equity = account.get_estimated_equity(upbit_prices)
                print(f"Total Equity (Est): {est_equity:,.0f} KRW | Balance: {account.balance:,.0f} KRW")

            except Exception as e:
                print(f"Loop Error: {e}")
                import traceback
                traceback.print_exc()

            await asyncio.sleep(30) # 30s interval for better responsiveness

    finally:
        pass

if __name__ == "__main__":
    asyncio.run(main_loop())
