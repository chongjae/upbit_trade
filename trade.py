import os
import json
import time
import math
import uuid
from datetime import datetime
import urllib.request
from urllib.parse import urlencode
import sqlite3
import jwt
import hashlib
import logging
import sys
from logging.handlers import TimedRotatingFileHandler

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
STRATEGIES = {
    'KRW-HYPER': 'VB',
   'KRW-SOL': 'VB',
   'KRW-DOGE': 'VB',
   'KRW-XRP': 'VB',
   'KRW-PEPE': 'VB',
   'KRW-SUI': 'VB'
}

# --- Risk Management Config ---
RISK_TARGET = 0.02 # 2% risk per trade
MAX_ALLOCATION_PCT = 1.0 / len(STRATEGIES) if STRATEGIES else 1.0 # Dynamic allocation
TRAILING_STOP_PCT = 0.03 # 3% trailing stop
PUMP_VOL_MULTIPLIER = 10 # 10x volume spike

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
    except Exception as e:
        print(f"API Error ({endpoint}): {e}")
        return None

def get_current_price(market):
    return get_current_prices([market]).get(market)

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
    return send_request('GET', '/v1/candles/days', {'market': market, 'count': count})

def get_minute_candles(market, unit=1, count=11):
    return send_request('GET', f'/v1/candles/minutes/{unit}', {'market': market, 'count': count})

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
                self.positions = data.get('positions', {}) # {market: {'volume': 0, 'avg_price': 0}}
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
        return self.positions.get(market, {'volume': 0, 'avg_price': 0})

    def update_position(self, market, volume, price, side):
        pos = self.get_position(market)
        curr_vol = pos['volume']
        curr_avg = pos['avg_price']

        if side == 'bid': # Buy
            new_vol = curr_vol + volume
            # Avg price weighted average
            new_avg = ((curr_vol * curr_avg) + (volume * price)) / new_vol if new_vol > 0 else 0
            self.positions[market] = {'volume': new_vol, 'avg_price': new_avg}
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

    def run(self, current_price, candles, current_time=None):
        if not candles or len(candles) < 20: return {}
        if current_time is None: current_time = datetime.now()

        # --- Indicator Calc ---
        target_candles = candles[1:] # Use history for Target calc
        # Need to be careful with indexing. get_daily_ohlcv(50) -> 0 is today, 1 is yesterday.
        # Target Calc uses Yesterday's info.

        # 1. Dynamic K
        target_pool = candles[1:21] # Prev 20
        noise_ratios = []
        for c in target_pool:
            rng = c['high_price'] - c['low_price']
            if rng > 0:
                noise = 1 - (abs(c['opening_price'] - c['trade_price']) / rng)
                noise_ratios.append(noise)
        k = sum(noise_ratios) / len(noise_ratios) if noise_ratios else 0.5

        # Target Price
        today_open = candles[0]['opening_price']
        prev = candles[1]
        rng = prev['high_price'] - prev['low_price']
        target_price = today_open + (rng * k)

        # MA5
        closes = [c['trade_price'] for c in target_pool[:5]]
        ma5 = sum(closes) / len(closes) if closes else 0

        # ATR (for Sizing)
        atr = get_atr(candles, 14)

        pos = self.account.get_position(self.market)
        has_position = pos['volume'] * current_price > 5000

        print(f"[{self.market}] Curr: {current_price} | Target: {target_price:.0f} (K:{k:.2f}) | MA5: {ma5:.0f} | ATR: {atr:.0f}")

        # Status Dict
        status = {
            'market': self.market,
            'current_price': current_price,
            'target_price': target_price,
            'ma5': ma5,
            'k': k,
            'atr': atr,
            'vol_pct': (atr / current_price * 100) if current_price > 0 else 0,
            'condition': 'WAIT'
        }

        if not has_position:
            # --- Buying Logic ---

            # 1. Volatility Breakout (Main)
            if current_price > target_price and current_price > ma5:
                # Pump Check
                if self.check_pump_dump():
                     status['condition'] = 'PUMP_DETECTED'
                     return status

                # Sizing: Volatility Targeting
                # Budget = (Equity * Risk) / (ATR / Price)  <- ATR in Price terms. percent vol = ATR/Price
                # Simplified: Risk Money = Equity * 0.02.
                # Stop distance implicitly is ATR? No, VT usually equates volatility contribution.
                # Formula: Allocation = (Target Risk % * Capital) / (Asset Volatility %)
                # Asset Volatility % = ATR / Price

                equity = self.account.get_estimated_equity({self.market: current_price})
                risk_amt = equity * RISK_TARGET
                vol_pct = (atr / current_price) if current_price > 0 else 0.01
                if vol_pct == 0: vol_pct = 0.01

                ideal_budget = risk_amt / vol_pct

                # Cap allocation
                max_budget = equity * MAX_ALLOCATION_PCT
                budget = min(ideal_budget, max_budget)

                # Check Spendable Cash
                spendable = min(self.account.balance, budget)

                if spendable > 5000:
                    print(f"[{self.market}] [VB] BUY (Bud: {budget:,.0f}, Vol%: {vol_pct*100:.1f}%)")
                    if self.account.buy_market(self.market, spendable, current_price):
                        trailing_manager.reset(self.market) # Reset previous state (if any)
                        trailing_manager.update_high(self.market, current_price) # Init Trailing

            # 2. Mean Reversion (Sub)
            else:
                 if len(candles) >= 20:
                     upper, mid, lower = get_bollinger_bands(candles)
                     rsi = get_rsi(candles, 14)

                     if current_price < lower and rsi < 30:
                         # MR is high risk, use fixed small allocation or 0.5 * Normal?
                         # Use simplified fix for now or half calculated VT?
                         equity = self.account.get_estimated_equity({self.market: current_price})
                         budget = equity * 0.05 # 5% fixed for MR catch?

                         spendable = min(self.account.balance, budget)
                         if spendable > 5000:
                             print(f"[{self.market}] [MR] BUY (Dip)")
                             if self.account.buy_market(self.market, spendable, current_price):
                                 trailing_manager.reset(self.market)
                                 trailing_manager.update_high(self.market, current_price)
                                 status['condition'] = 'BUY_MR'

            if current_price > target_price and current_price > ma5:
                 status['condition'] = 'BUY_SIGNAL_VB' # Signal is active even if we didn't buy (e.g. no cash)

        else:
            # --- Selling Logic ---

            # Update Trailing High
            trailing_manager.update_high(self.market, current_price)

            avg_price = pos['avg_price']

            # 1. Check Trailing Stop
            if trailing_manager.check_stop(self.market, current_price, TRAILING_STOP_PCT):
                 print(f"[{self.market}] [Trailing Stop] SELL")
                 self.account.sell_market(self.market, pos['volume'], current_price)
                 trailing_manager.reset(self.market)
                 status['condition'] = 'SELL_TS'
                 return status

            # 2. Stop Loss (Fixed Backup - 3%)
            # Trailing stop usually covers this, but just in case
            if current_price < avg_price * 0.97:
                print(f"[{self.market}] [Hard Stop] SELL")
                self.account.sell_market(self.market, pos['volume'], current_price)
                trailing_manager.reset(self.market)
                status['condition'] = 'SELL_SL'
                return status

            # 3. MR Take Profit (Only if we are in MR trade? Hard to distinguish without tagging)
            # Mixed approach: If Price > BB Mid OR RSI > 50, and we are profitable, we can scale out?
            # Or just let Trailing Stop handle the run?
            # Report says "Take Profit" for MR.
            # Let's simple use Trailing Stop for ALL. It maximizes profit in trend and cuts loss in reversal.
            # But MR usually needs quick exit.

            # Optional: EOD Sell
            if current_time.hour == 8 and current_time.minute >= 59:
                 print(f"[{self.market}] END OF DAY SELL (09:00 KST Daily Close)")
                 self.account.sell_market(self.market, pos['volume'], current_price)
                 trailing_manager.reset(self.market)
                 status['condition'] = 'SELL_EOD'

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

# --- Main Loop ---
def main():
    if IF_PAPER_MODE:
        account = PaperAccount()
        print(f"--- Paper Trading Started (VB Unified) ---")
    else:
        account = RealAccount()
        print(f"--- REAL TRADING Started (VB Unified, Volatility Targeting) ---")

    print(f"Balance: {account.balance:,.0f} KRW")

    bots = {}
    for market in STRATEGIES.keys():
        bots[market] = VolatilityBreakoutBot(account, market)

    while True:
        try:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Tick...")

            # Batch fetch prices
            prices = get_current_prices(list(bots.keys()))

            bot_stats = {}
            for market, bot in bots.items():
                if market not in prices: continue
                # Always fetch 50 for indicators
                candles = get_daily_ohlcv(market, count=50)
                time.sleep(0.2) # Throttle candle API calls (5 req/sec is safe)

                status = bot.run(prices[market], candles)
                bot_stats[market] = status

            # Save Bot Status
            with open("bot_status.json", "w") as f:
                json.dump(bot_stats, f)

            # Print Estimated Status
            est_equity = account.get_estimated_equity(prices)
            print(f"Total Equity (Est): {est_equity:,.0f} KRW")

            print(f"Balance: {account.balance:,.0f} KRW")
            for m, pos in account.positions.items():
                if pos['volume'] > 0:
                    val = pos['volume'] * prices.get(m, 0)
                    print(f"  {m}: {pos['volume']:.4f} ({val:,.0f} KRW)")

            time.sleep(30) # Increased sleep to avoid 429

        except KeyboardInterrupt:
            print("Stopping...")
            break
        except Exception as e:
            print(f"Error in loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(10)

if __name__ == "__main__":
    main()
