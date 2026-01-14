
import os
import time
import requests
import json
from datetime import datetime, timedelta
from trade import PaperAccount, STRATEGIES, VolatilityBreakoutBot


# Limit backtest to last 30 days for minute data to avoid API overload or use fewer points?
# Upbit API restriction: /v1/candles/minutes/1 returns 200 max.
# 30 days * 24h * 60m = 43200 points.
# We need to loop.

BACKTEST_INITIAL_BALANCE = 91250


class BacktestAccount(PaperAccount):
    def __init__(self, initial_krw=1_000_000):
        # Initialize in-memory only (no file/db persistence for backtest speed)
        self.filename = "backtest_account.json"
        self.db_file = ":memory:" # Use in-memory SQLite
        self.init_db()

        self.balance = initial_krw
        self.positions = {}
        self.orders = []
        self.trades_log = [] # Keep simple list instead of DB if desired, but we reused PaperAccount structure

    def check_orders(self, current_candle):
        # Override to check against candle High/Low
        # current_candle: {market_name: {high_price, low_price, trade_price...}}

        filled_orders = []
        remaining_orders = []

        for order in self.orders:
            market = order['market']
            if market not in current_candle:
                remaining_orders.append(order)
                continue

            candle = current_candle[market]
            low = candle['low_price']
            high = candle['high_price']

            side = order['side']
            price = order['price']

            filled = False
            # Buy Limit: Filled if Price >= Low
            if side == 'bid' and low <= price:
                 self.update_position(market, order['volume'], price, 'bid')
                 total = price * order['volume'] * 1.0005
                 self.log_trade(market, 'bid', price, order['volume'], total)
                 filled = True

            # Sell Limit: Filled if Price <= High
            elif side == 'ask' and high >= price:
                 earn = price * order['volume'] * 0.9995
                 self.balance += earn
                 total = earn
                 self.log_trade(market, 'ask', price, order['volume'], total)
                 filled = True

            if filled:
                # print(f"[Backtest] ORDER FILLED {market} {side}: {order['volume']} @ {price}")
                filled_orders.append(order)
            else:
                remaining_orders.append(order)

        return filled_orders

    def log_trade(self, market, side, price, volume, total):
        self.trades_log.append({
            'timestamp': str(datetime.now()),
            'market': market,
            'side': side,
            'price': price,
            'volume': volume,
            'total': total
        })

    def save(self):
        pass # Disable file saving


import sqlite3

class BacktestDB:
    def __init__(self, db_file="backtest_cache.db"):
        self.db_file = db_file
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        # timestamp_kst string 'YYYY-MM-DD HH:mm:SS'
        # market string 'KRW-BTC'
        c.execute('''CREATE TABLE IF NOT EXISTS candles
                     (market TEXT,
                      timestamp_kst TEXT,
                      trade_price REAL,
                      high_price REAL,
                      low_price REAL,
                      opening_price REAL,
                      candle_acc_trade_volume REAL,
                      PRIMARY KEY (market, timestamp_kst))''')
        c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON candles (timestamp_kst)")
        conn.commit()
        conn.close()

    def save_bulk(self, candles):
        if not candles: return
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        data = []
        for cdl in candles:
            data.append((
                cdl['market'],
                cdl['candle_date_time_kst'],
                cdl['trade_price'],
                cdl['high_price'],
                cdl['low_price'],
                cdl['opening_price'],
                cdl['candle_acc_trade_volume']
            ))
        # REPLACE or IGNORE? REPLACE to update if needed.
        c.executemany("INSERT OR REPLACE INTO candles VALUES (?, ?, ?, ?, ?, ?, ?)", data)
        conn.commit()
        conn.close()

    def get_latest_timestamp(self, market):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        c.execute("SELECT MAX(timestamp_kst) FROM candles WHERE market=?", (market,))
        res = c.fetchone()
        conn.close()
        return res[0] if res else None

    def get_all_timestamps(self):
        # Return sorted list of distinct timestamps within DB
        # To limit scope, we could filter by date here if we only want 7 days
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        c.execute("SELECT DISTINCT timestamp_kst FROM candles ORDER BY timestamp_kst ASC")
        rows = c.fetchall()
        conn.close()
        return [r[0] for r in rows]

    def get_oldest_timestamp(self, market):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        c.execute("SELECT MIN(timestamp_kst) FROM candles WHERE market=?", (market,))
        res = c.fetchone()
        conn.close()
        return res[0] if res else None

    def get_candles_by_time(self, timestamp_kst):
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM candles WHERE timestamp_kst=?", (timestamp_kst,))
        rows = c.fetchall()
        conn.close()
        # Convert to dict format expected by BacktestAccount
        res = {}
        for r in rows:
            res[r['market']] = dict(r)
        return res

def fetch_minute_candles(market, to_datetime, count=200):
    url = "https://api.upbit.com/v1/candles/minutes/1"
    # Use 'T' separator or verify format. Upbit docs: yyyy-MM-dd HH:mm:ss
    # But let's try strict string format and ensure requests doesn't mangle it.
    to_str = to_datetime.strftime("%Y-%m-%d %H:%M:%S")
    params = {'market': market, 'to': to_str, 'count': count}

    try:
        res = requests.get(url, params=params, timeout=10)

        # Check for 429 specifically
        if res.status_code == 429:
            print(f"[429] Rate Limit on {market}. Sleeping 1s...")
            time.sleep(1)
            return [] # Retry logic handled by caller loop? Caller breaks on empty.
                      # Ideally we should retry here.

        data = res.json()

        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'error' in data:
            print(f"API Error {market}: {data['error']}")
            return []
        else:
            print(f"Unknown Response {market}: {data}")
            return []

    except Exception as e:
        print(f"Error fetching {market}: {e}")
        return []

def fetch_loop(market, start_dt, end_dt, db):
    """
    Fetches minutes backwards from end_dt down to start_dt.
    """
    curr = end_dt
    print(f"    Syncing range: {start_dt} ~ {end_dt}")

    while curr > start_dt:
        batch = fetch_minute_candles(market, curr, 200)

        if not batch:
            print("    Empty batch. Retrying in 1s...")
            time.sleep(1)
            batch = fetch_minute_candles(market, curr, 200)
            if not batch:
                print("    Still empty. Stopping this range.")
                break

        db.save_bulk(batch)

        last_timestamp = batch[-1]['candle_date_time_kst']
        last_dt = datetime.strptime(last_timestamp, "%Y-%m-%dT%H:%M:%S")

        # Check progress
        if last_dt >= curr:
             print(f"    Warning: Stuck at {last_timestamp}. Forcing jump.")
             curr = curr - timedelta(minutes=200)
        else:
             print(f"    Synced {len(batch)} candles. Last: {last_timestamp}")
             curr = last_dt - timedelta(minutes=1)

        time.sleep(0.12)

def ensure_data_in_db(days=30):
    db = BacktestDB()
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)

    print(f"Checking data for {days} days ({start_dt.strftime('%Y-%m-%d %H:%M')} ~ {end_dt.strftime('%Y-%m-%d %H:%M')})...")

    for market in STRATEGIES.keys():
        print(f"  Checking {market}...")

        latest_str = db.get_latest_timestamp(market)
        oldest_str = db.get_oldest_timestamp(market)

        if not latest_str:
            # Case 1: No data at all. Fetch all.
            print("    No cache found. Fetching all...")
            fetch_loop(market, start_dt, end_dt, db)
        else:
            latest_dt = datetime.strptime(latest_str, "%Y-%m-%dT%H:%M:%S")
            oldest_dt = datetime.strptime(oldest_str, "%Y-%m-%dT%H:%M:%S")
            print(f"    Cache: {oldest_str} ~ {latest_str}")

            # Case 2: Gap 1 (New Data: Latest -> Now)
            # Only fetch if gap is significant (> 2 mins)
            if end_dt > latest_dt + timedelta(minutes=2):
                print("    Fetching NEW data gap...")
                fetch_loop(market, latest_dt, end_dt, db)

            # Case 3: Gap 2 (Old Data: Target Start -> Oldest)
            if start_dt < oldest_dt:
                 print("    Fetching HISTORICAL data gap...")
                 # Start fetch from oldest - 1 min to avoid overlap
                 fetch_loop(market, start_dt, oldest_dt - timedelta(minutes=1), db)

            print("    Cache up to date.")

    print("Data sync complete.")
    return db

def calculate_past_month_profit():
    # 1. Sync Data
    db = ensure_data_in_db(30)

    # 2. Setup Env
    account = BacktestAccount(BACKTEST_INITIAL_BALANCE)
    bots = {}
    for market, strategy_type in STRATEGIES.items():
        # Strat is always VB now based on trade.py
        bots[market] = VolatilityBreakoutBot(account, market)

    # 3. Simulate
    # Get all timestamps from DB (sorted)
    # Filter for last 30 days only
    all_ts = db.get_all_timestamps()
    # Filter: >= start_dt
    start_str = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    sim_ts = [t for t in all_ts if t >= start_str]

    print(f"Starting Simulation over {len(sim_ts)} minutes...")
    print(f"  Range: {sim_ts[0]} ~ {sim_ts[-1]}")

    daily_cache = {}
    for m in STRATEGIES.keys():
        url = "https://api.upbit.com/v1/candles/days"
        while True:
            try:
                res = requests.get(url, params={'market': m, 'count': 40}) # Fetch enough daily candles (some buffer)
                if res.status_code == 429:
                    print(f"  [429] Rate limit for {m}, retrying...")
                    time.sleep(1)
                    continue

                data = res.json()
                if isinstance(data, list):
                    daily_cache[m] = data
                    break
                else:
                    print(f"  Error fetching daily for {m}: {data}")
                    time.sleep(1)
            except Exception as e:
                print(f"  Exception fetching daily for {m}: {e}")
                time.sleep(1)
        time.sleep(0.12)

    print("  Processing...", end='', flush=True)
    count = 0
    total_steps = len(sim_ts)

    for ts in sim_ts:
        count += 1
        if count % 2000 == 0:
             print(f" {int(count/total_steps*100)}%...", end='', flush=True)

        current_time = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")

        # Get candles for this minute from DB
        market_data = db.get_candles_by_time(ts) # {market: candle_dict}
        if not market_data: continue

        current_prices = {m: c['trade_price'] for m, c in market_data.items()}

        # 1. Check Orders
        filled = account.check_orders(market_data)

        # Process Grid Refills - REMOVED since Strategy is VB-Only now
        # But keeping loop structure just in case.

        # 2. Run Bots
        for market, bot in bots.items():
            if market not in current_prices: continue

            candles = daily_cache[market]
            cur_date_str = current_time.strftime("%Y-%m-%d")
            start_idx = -1
            for i, c in enumerate(candles):
                if c['candle_date_time_kst'].startswith(cur_date_str):
                    start_idx = i
                    break

            if start_idx != -1:
                bot.run(current_prices[market], candles[start_idx:], current_time)

    # Result
    print(f"\n--- Backtest Result ---")

    # Analyze Trades
    trade_counts = {}
    market_pnl = {} # {market: {'buys': 0, 'sells': 0}}

    for t in account.trades_log:
        m = t['market']
        if m not in trade_counts: trade_counts[m] = 0
        trade_counts[m] += 1

        if m not in market_pnl: market_pnl[m] = {'buys': 0, 'sells': 0}

        if t['side'] == 'bid':
            market_pnl[m]['buys'] += t['total']
        elif t['side'] == 'ask':
            market_pnl[m]['sells'] += t['total']

    print(f"Total Trades Executed: {len(account.trades_log)}")

    est_total = account.balance
    current_prices_last = {}

    if sim_ts:
        last_data = db.get_candles_by_time(sim_ts[-1])
        for m, c in last_data.items():
            current_prices_last[m] = c['trade_price']

    print("\n[Market Performance]")
    print(f"{'Market':<10} | {'Trades':<6} | {'Buy Total':<12} | {'Sell+Hold':<12} | {'PnL (KRW)':<10} | {'ROI':<7}")
    print("-" * 75)

    for m in STRATEGIES.keys():
        # Get Current Holding Value
        pos = account.positions.get(m, {'volume': 0})
        cur_price = current_prices_last.get(m, 0)
        hold_val = pos['volume'] * cur_price

        est_total += hold_val # Add to Total Assets

        # Stats
        stats = market_pnl.get(m, {'buys': 0, 'sells': 0})
        buys = stats['buys']
        sells_plus_hold = stats['sells'] + hold_val
        pnl = sells_plus_hold - buys

        roi = 0.0
        if buys > 0:
            roi = (pnl / buys) * 100

        t_count = trade_counts.get(m, 0)

        print(f"{m:<10} | {t_count:<6} | {buys:12,.0f} | {sells_plus_hold:12,.0f} | {pnl:10,.0f} | {roi:6.2f}%")

    initial = BACKTEST_INITIAL_BALANCE
    pnl_pct = ((est_total - initial) / initial) * 100
    print("-" * 75)
    print(f"Final Balance: {account.balance:,.0f} KRW")
    print(f"Estimated Total Assets: {est_total:,.0f} KRW")
    print(f"Total Return (30 Days): {pnl_pct:.2f}%")

    return pnl_pct

if __name__ == "__main__":
    calculate_past_month_profit()
