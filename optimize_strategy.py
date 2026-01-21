from backtest import run_backtest
import json

def main():
    days = 7
    selections = ['dynamic', 'bluechip']
    strategies = ['dynamic', 'vb', 'bb']

    results = []

    print(f"Starting Multi-Strategy Backtest Optimization (Last {days} days)...")

    for sel in selections:
        for strat in strategies:
            try:
                res = run_backtest(days=days, coin_selection_mode=sel, strategy_mode=strat)
                if res:
                    results.append(res)
            except Exception as e:
                print(f"Error testing {sel}/{strat}: {e}")

    # Sort by PnL
    results.sort(key=lambda x: x['pnl_pct'], reverse=True)

    print("\n" + "="*50)
    print(f"{'Selection':<15} | {'Strategy':<10} | {'PnL %':<10} | {'Trades':<8}")
    print("-" * 50)
    for r in results:
        print(f"{r['selection']:<15} | {r['strategy']:<10} | {r['pnl_pct']:>8.2f}% | {r['trades']:<8}")
    print("="*50)

    if results:
        winner = results[0]
        print(f"\nWINNING COMBINATION: {winner['selection']} + {winner['strategy']} ({winner['pnl_pct']:.2f}%)")

        # Save results to file
        with open('optimization_results.json', 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
