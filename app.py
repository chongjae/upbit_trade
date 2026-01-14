from flask import Flask, render_template, jsonify
import json
import os
import requests
import sqlite3

app = Flask(__name__)
PAPER_ACCOUNT_FILE = "paper_account.json"
DB_FILE = "trades.db"

@app.template_filter('format_price')
def format_price(value):
    if value is None:
        return "-"
    if value == 0:
        return "0"
    if value < 100:
        return "{:,.4f}".format(value)
    else:
        return "{:,.0f}".format(value)

# Helper to fetch current price (simplified, no rate limit handling in view generally)
def get_current_price(market):
    try:
        url = f"https://api.upbit.com/v1/ticker?markets={market}"
        response = requests.get(url)
        return response.json()[0]['trade_price']
    except:
        return 0

def process_trades_and_stats(page=1, per_page=10):
    trades = []
    stats = {} # {market: {'realized_pnl': 0, 'win': 0, 'loss': 0, 'cnt': 0}}

    if os.path.exists(DB_FILE):
        try:
            conn = sqlite3.connect(DB_FILE)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM trades ORDER BY id ASC")
            rows = c.fetchall()
            conn.close()

            positions = {}
            processed_trades = []

            for row in rows:
                t = dict(row)
                m = t['market']
                s = t['side']
                p = t['price']
                v = t['volume']

                if m not in stats:
                    stats[m] = {'realized_pnl': 0, 'win': 0, 'loss': 0, 'cnt': 0}
                if m not in positions:
                    positions[m] = {'vol': 0.0, 'avg': 0.0}
                pos = positions[m]

                t['pnl'] = None
                t['pnl_percent'] = 0

                stats[m]['cnt'] += 1

                if s == 'bid':
                    cost = (pos['vol'] * pos['avg']) + (v * p)
                    new_vol = pos['vol'] + v
                    pos['vol'] = new_vol
                    pos['avg'] = cost / new_vol if new_vol > 0 else 0
                elif s == 'ask':
                    pnl_amt = 0
                    if pos['avg'] > 0:
                        pnl_amt = (p - pos['avg']) * v
                        t['pnl'] = pnl_amt
                        t['pnl_percent'] = ((p - pos['avg']) / pos['avg']) * 100

                        stats[m]['realized_pnl'] += pnl_amt
                        if pnl_amt > 0: stats[m]['win'] += 1
                        else: stats[m]['loss'] += 1

                    pos['vol'] = max(0, pos['vol'] - v)

                processed_trades.append(t)

            processed_trades.reverse()
            total = len(processed_trades)
            pages = (total + per_page - 1) // per_page
            start = (page - 1) * per_page

            return processed_trades[start:start+per_page], pages, page, stats

        except Exception as e:
            print(f"DB Error: {e}")
            import traceback
            traceback.print_exc()
    return [], 0, 1, {}

# --- Auth Helper ---
def upbit_request(endpoint, params=None):
    import jwt
    import hashlib
    import uuid
    from urllib.parse import urlencode

    access_key = os.environ.get('access_key')
    secret_key = os.environ.get('secret_key')
    if not access_key or not secret_key: return None

    url = "https://api.upbit.com" + endpoint
    payload = {'access_key': access_key, 'nonce': str(uuid.uuid4())}
    if params:
        q = urlencode(params)
        m = hashlib.sha512()
        m.update(q.encode())
        payload['query_hash'] = m.hexdigest()
        payload['query_hash_alg'] = 'SHA512'
        url += '?' + q

    token = jwt.encode(payload, secret_key, algorithm='HS256')
    headers = {'Authorization': f'Bearer {token}'}
    try:
        res = requests.get(url, headers=headers)
        return res.json()
    except:
        return None

def get_real_account_info():
    # Fetch Balances
    balances = upbit_request('/v1/accounts')
    account = {'balance': 0, 'positions': {}, 'orders': []}

    if balances and isinstance(balances, list):
        for b in balances:
            if b['currency'] == 'KRW':
                account['balance'] = float(b['balance'])
            else:
                m = f"KRW-{b['currency']}"
                account['positions'][m] = {
                    'volume': float(b['balance']),
                    'avg_price': float(b['avg_buy_price'])
                }

    # Fetch Open Orders (for Locked Calc)
    markets = ['KRW-SOL', 'KRW-DOGE', 'KRW-AVAX', 'KRW-XRP', 'KRW-ETH']
    for m in markets:
        orders = upbit_request('/v1/orders', {'market': m, 'state': 'wait'})
        if orders and isinstance(orders, list):
            for o in orders:
                account['orders'].append({
                    'created_at': o['created_at'],
                    'market': o['market'],
                    'side': o['side'],
                    'price': float(o['price']),
                    'volume': float(o['volume']),
                    'uuid': o['uuid']
                })
    return account



@app.route('/')
def index():
    is_paper = os.environ.get('PAPER_MODE', 'True').lower() == 'true'

    account = {}
    trades = []

    from flask import request
    page = request.args.get('page', 1, type=int)

    if is_paper:
        if os.path.exists(PAPER_ACCOUNT_FILE):
            with open(PAPER_ACCOUNT_FILE, 'r') as f:
                account = json.load(f)
    else:
        account = get_real_account_info()

    bot_status = {}
    if os.path.exists("bot_status.json"):
        try:
            with open("bot_status.json", 'r') as f:
                bot_status = json.load(f)
        except: pass

    trades, total_pages, current_page, history_stats = process_trades_and_stats(page, 10)

    balance = account.get('balance', 0)
    positions = account.get('positions', {})

    # Calculate Total Assets
    # User Request: "Available KRW + Holdings Avg Price * Volume" + LOCKED FUNDS (Bid) + LOCKED COINS (Ask)
    total_assets = balance
    locked_funds = 0
    locked_ask_vol = {}

    orders = account.get('orders', [])
    for order in orders:
        market = order['market']
        if order['side'] == 'bid':
            cost = order['price'] * order['volume']
            locked_funds += cost
            total_assets += cost
        elif order['side'] == 'ask':
            locked_ask_vol[market] = locked_ask_vol.get(market, 0) + order['volume']

    holdings = []
    all_markets = set(positions.keys()) | set(locked_ask_vol.keys())

    for market in all_markets:
        pos = positions.get(market, {'volume': 0, 'avg_price': 0})
        avail_vol = pos['volume']
        locked_vol = locked_ask_vol.get(market, 0)
        total_vol = avail_vol + locked_vol

        if total_vol > 0:
            current_price = get_current_price(market)
            avg_price = pos['avg_price']

            invested_capital = total_vol * avg_price
            current_value = total_vol * current_price

            # Use Current Value for Estimated Total Assets
            total_assets += current_value

            pnl = ((current_price - avg_price) / avg_price) * 100 if avg_price > 0 else 0

            holdings.append({
                'market': market,
                'volume': total_vol,
                'avg_price': avg_price,
                'current_price': current_price,
                'value': current_value,
                'pnl': pnl,
                'locked': locked_vol
            })

    # Process Open Orders
    open_orders = []
    for order in orders:
        market = order['market']
        curr_p = get_current_price(market)
        locked = 0
        if order['side'] == 'bid':
            locked = order['price'] * order['volume']
        elif order['side'] == 'ask':
            locked = order['volume'] * curr_p

        open_orders.append({
            'created_at': order.get('created_at', ''),
            'market': market,
            'side': order['side'],
            'price': order['price'],
            'volume': order['volume'],
            'locked': locked,
            'current_price': curr_p
        })
    open_orders.reverse()

    # Process Performance Stats
    performance = []
    # Combine history stats and current holdings
    perf_markets = set(history_stats.keys()) | set(positions.keys()) | set(locked_ask_vol.keys())

    for m in perf_markets:
        # Get Realized PnL from History
        h_stat = history_stats.get(m, {'realized_pnl': 0, 'win': 0, 'loss': 0, 'cnt': 0})
        realized = h_stat['realized_pnl']

        # Get Unrealized PnL from Current Holdings
        unrealized = 0
        current_vol = 0

        # Find in existing holdings list
        found_h = next((h for h in holdings if h['market'] == m), None)
        if found_h:
            # Pnl amount = (Current Price - Avg Price) * Volume
            unrealized = found_h['value'] - (found_h['volume'] * found_h['avg_price'])
            current_vol = found_h['volume']

        total_pnl = realized + unrealized

        win_rate = 0
        total_finished_trades = h_stat['win'] + h_stat['loss']
        if total_finished_trades > 0:
            win_rate = (h_stat['win'] / total_finished_trades) * 100

        performance.append({
            'market': m,
            'realized_pnl': realized,
            'unrealized_pnl': unrealized,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'trade_count': h_stat['cnt'],
            'current_vol': current_vol
        })

    # Sort by Total PnL descending
    performance.sort(key=lambda x: x['total_pnl'], reverse=True)

    return render_template('index.html',
                           balance=balance,
                           locked_funds=locked_funds,
                           total_assets=total_assets,
                           holdings=holdings,
                           trades=trades,
                           open_orders=open_orders,
                           current_page=current_page,
                           total_pages=total_pages,
                           performance=performance,
                           bot_status=bot_status)

if __name__ == '__main__':
    # usage: debug=True allows auto-reload on code changes
    app.run(host='0.0.0.0', port=55555, debug=True)
