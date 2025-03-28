import glob
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def backtest_yyy(weights, statuses=None):
    # print(f'RECEIVED {statuses=}')

    if len(weights) != 12:
        if not statuses:
            raise AttributeError()
        new_month_indices = statuses2new_month_indices(statuses)

        weights_to_execute = [weights[i] for i in new_month_indices]
    else:
        weights_to_execute = weights

    i = 0
    initial_capital = 10000
    portfolio_value = initial_capital

    portfolio_history = [portfolio_value]
    # Create a DataFrame to track monthly PnL for each ticker
    monthly_pnl = pd.DataFrame(0.0, index=range(12), columns=tickers)

    while i < 11:
        weights = weights_to_execute[i]

        # ---- 2) Buy using these weights ----
        shares = []
        buy_prices = []
        initial_capital = portfolio_value

        for j, ticker in enumerate(tickers):
            buy_price = data_loaded[i][ticker]['price']
            buy_prices.append(buy_price)

            allocation = initial_capital * weights[j]  # portion of capital
            shares_bought = allocation / buy_price if buy_price > 0 else 0
            shares.append(shares_bought)

            # Deduct spent cash
            portfolio_value -= shares_bought * buy_price

        # ---- 3) Sell at month i+1 (end of next month), record PnL per ticker ----
        i += 1
        for j, ticker in enumerate(tickers):
            sell_price = data_loaded[i][ticker]['price']
            # PnL for this ticker in month i-1 (e.g. row 0 if i=1 now)
            pnl = shares[j] * (sell_price - buy_prices[j])
            monthly_pnl.loc[i - 1, ticker] = pnl  # store PnL

            # Update portfolio value by the proceeds of selling
            portfolio_value += shares[j] * sell_price

        portfolio_history.append(portfolio_value)

    # print("Final Portfolio Value:", portfolio_value)
    # Return both the total portfolio value history and the per-ticker monthly PnL
    return portfolio_value, portfolio_history, monthly_pnl


def statuses2new_month_indices(statuses):
    new_month_indices = {}  # Use a dictionary to store the last index for each month

    for i, status in enumerate(statuses):
        # print(f'looking at {status=}')
        # Check if it's a converged status or a regular iteration
        if "CONVERGED" in status:
            # For converged status, format is "CONVERGED month X iter Y"
            month = int(status.split()[2])
            # Always prefer converged solutions
            new_month_indices[month] = i
        else:
            # For non-converged status, format is "month X iter Y"
            month = int(status.split()[1])
            # Only add if we don't already have a converged solution for this month
            if month not in new_month_indices:
                new_month_indices[month] = i
            # If we have a regular iteration, update only if it's a later iteration
            elif "CONVERGED" not in statuses[new_month_indices[month]]:
                new_month_indices[month] = i

    # Convert the dictionary to a sorted list of indices
    return [new_month_indices[month] for month in sorted(new_month_indices.keys())]


def backtest(weights, statuses):
    # print(f'RECEIVED {statuses=}')

    if len(weights) != 12:
        new_month_indices = statuses2new_month_indices(statuses)

        weights_to_execute = [weights[i] for i in new_month_indices]
    else:
        weights_to_execute = weights

    i = 0
    initial_capital = 10000
    portfolio_value = initial_capital

    portfolio_history = [portfolio_value]
    # Create a DataFrame to track monthly PnL for each ticker
    monthly_pnl = pd.DataFrame(0.0, index=range(12), columns=tickers)

    while i < 11:
        weights = weights_to_execute[i]

        # ---- 2) Buy using these weights ----
        shares = []
        buy_prices = []
        initial_capital = portfolio_value

        for j, ticker in enumerate(tickers):
            buy_price = data_loaded[i][ticker]['price']
            buy_prices.append(buy_price)

            allocation = initial_capital * weights[j]  # portion of capital
            shares_bought = allocation / buy_price if buy_price > 0 else 0
            shares.append(shares_bought)

            # Deduct spent cash
            portfolio_value -= shares_bought * buy_price

        # ---- 3) Sell at month i+1 (end of next month), record PnL per ticker ----
        i += 1
        for j, ticker in enumerate(tickers):
            sell_price = data_loaded[i][ticker]['price']
            # PnL for this ticker in month i-1 (e.g. row 0 if i=1 now)
            pnl = shares[j] * (sell_price - buy_prices[j])
            monthly_pnl.loc[i - 1, ticker] = pnl  # store PnL

            # Update portfolio value by the proceeds of selling
            portfolio_value += shares[j] * sell_price

        portfolio_history.append(portfolio_value)

    # print("Final Portfolio Value:", portfolio_value)
    # Return both the total portfolio value history and the per-ticker monthly PnL
    return portfolio_value, portfolio_history, monthly_pnl


tickers = [
    'NVDA', 'AMD', 'MSFT', 'AAPL', 'INTC', 'PLTR',  # Technology
    'TSLA', 'AMZN', 'SBUX', 'TGT', 'NFLX', 'MCD',  # Consumer Discretionary
    'HOOD', 'BAC', 'JPM', 'MS', 'V', 'SCHW',  # Financials
    'ZG', 'PLD', 'WELL', 'SPG', 'PSA', 'EQR',  # Real Estate
    'GEV', 'XOM', 'DUK', 'NEE', 'EOG', 'SLB',  # Energy
    'TEM', 'UNH', 'PFE', 'MRNA', 'ABBV', 'MDT',  # Healthcare
    'CAT', 'BA', 'LMT', 'DE', 'GD', 'HON',  # Industrials
    'PCT', 'NEM', 'LIN', 'APD', 'FCX', 'MLM',  # Materials
    'GOOG', 'TMUS', 'META', 'DIS', 'VZ', 'CMCSA',  # Communication Services
    'COST', 'PEP', 'WMT', 'KO', 'PG', 'MO'  # Consumer Staples
]


stock_data_path = f"assets/stock_data.json"


def load_data(file_path=stock_data_path):
    """
    Loads JSON data from 'file_path' and returns it as a dictionary.
    """
    with open(file_path, 'r') as fp:
        data = json.load(fp)
    print(f"Data loaded from {file_path}")
    return data


data_loaded = load_data()


# optimizer
appendage = "2025-03-27-gpt-4o-mini"
paths = glob.glob(os.path.join(os.getcwd(), "assets",
                  f"*weights_opt_{appendage}*"))
opt_returns = []
for path in paths:
    with open(path, 'r') as f:
        weights = json.loads(f.read())

    portfolio_value, portfolio_history, monthly_pnl = backtest_yyy(weights)
    print(f"Final portfolio value: ${portfolio_value:.2f}")
    print(f"Return multiple: {portfolio_value/10000:.4f}x")
    opt_returns.append(portfolio_value)


# coord_llm25_opt75
weights_llm25_opt75_coord_files = glob.glob(os.path.join(
    os.getcwd(), "assets", "*weights_coord_llm25_opt75*"))
coord_llm25_opt75_returns = []

for weights_llm25_opt75_coord_path in weights_llm25_opt75_coord_files:
    with open(weights_llm25_opt75_coord_path, 'r') as f:
        coord_llm25_opt75_histories = json.loads(f.read())

    coord_llm25_opt75_statuses = [x[0] for x in coord_llm25_opt75_histories]
    coord_llm25_opt75_weights = [x[1:] for x in coord_llm25_opt75_histories]

    portfolio_value, portfolio_history, monthly_pnl = backtest_yyy(
        coord_llm25_opt75_weights, coord_llm25_opt75_statuses)
    coord_llm25_opt75_returns.append(portfolio_value/10000)


# coord_llm75_opt25
weights_llm75_opt25_coord_files = glob.glob(os.path.join(
    os.getcwd(), "assets", "*weights_coord_llm75_opt25*"))
coord_llm75_opt25_returns = []

for weights_llm75_opt25_coord_path in weights_llm75_opt25_coord_files:
    with open(weights_llm75_opt25_coord_path, 'r') as f:
        coord_llm75_opt25_histories = json.loads(f.read())

    coord_llm75_opt25_statuses = [x[0] for x in coord_llm75_opt25_histories]
    coord_llm75_opt25_weights = [x[1:] for x in coord_llm75_opt25_histories]

    portfolio_value, portfolio_history, monthly_pnl = backtest_yyy(
        coord_llm75_opt25_weights, coord_llm75_opt25_statuses)
    coord_llm75_opt25_returns.append(portfolio_value/10000)


# llm, coord50-50
# in the /assets folder get every file that beigns with status_ and print their filenames


def print_status_files():
    assets_path = os.path.join(os.getcwd(), "assets")

    # Check if the directory exists
    if not os.path.exists(assets_path):
        print(f"Error: Directory {assets_path} does not exist.")
        return

    # List all files in the directory
    files = os.listdir(assets_path)

    # Filter files that begin with "status_"
    status_files = [file for file in files if file.startswith(
        "status_") and "sparse" not in file]

    return status_files


status_files = print_status_files()

llm_returns = []
coord_returns = []

for i, status in enumerate(status_files):
    status_file = os.path.join(os.getcwd(), "assets", status)
    with open(status_file, 'r') as f:
        statuses = json.loads(f.read())
    identifier = '_'.join(status.split('_')[1:])[:-5]

    # llm returns
    weights_llm_path = os.path.join(
        os.getcwd(), "assets", f"weights_llm_{identifier}.json")
    with open(weights_llm_path, 'r') as f:
        weights_llm = json.loads(f.read())
    new_month_indices = statuses2new_month_indices(statuses)
    # print(f"{len(weights_llm)=}\t{len(weights_llm[0])=}\t{len(statuses)=}\t{len(new_month_indices)=}")
    portfolio_value, portfolio_history, monthly_pnl = backtest(
        weights_llm, statuses)
    llm_returns.append(portfolio_value)

    # coord 50 50 returns
    weights_coord_path = os.path.join(
        os.getcwd(), "assets", f"weights_coord_{identifier}.json")
    with open(weights_coord_path, 'r') as f:
        weights_coord = json.loads(f.read())
    weights_coord = [w[1:] for w in weights_coord]
    new_month_indices = statuses2new_month_indices(statuses)
    # print(f"{len(weights_coord)=}\t{len(weights_coord[0])=}\t{len(statuses)=}\t{len(new_month_indices)=}")
    portfolio_value, portfolio_history, monthly_pnl = backtest(
        weights_coord, statuses)
    coord_returns.append(portfolio_value)


# in the style of this code, make a graph that is all five of those things together in one bar graph, make there be no spacing between the bars in the bar graph, and add a legend to show what each color is corresponding to

# the five colors list: 5f0f40 opt, 9a031e llm, fb8b24 llm75_opt25, e36414 llm50_opt50, 0f4c5c llm25_opt75
# the five returns lists: opt_returns, llm_returns, coord_llm75_opt25_returns, coord_returns, coord_llm25_opt75_returns

returnses = [opt_returns, llm_returns, coord_llm75_opt25_returns,
             coord_returns, coord_llm25_opt75_returns]

# normalize
opt_returns = [x/10000 for x in opt_returns]
llm_returns = [x/10000 for x in llm_returns]
coord_returns = [x/10000 for x in coord_returns]

# Calculate means and standard deviations for each dataset
opt_mean = np.mean(opt_returns)
llm_mean = np.mean(llm_returns)
llm75_opt25_mean = np.mean(coord_llm75_opt25_returns)
llm50_opt50_mean = np.mean(coord_returns)
llm25_opt75_mean = np.mean(coord_llm25_opt75_returns)

opt_std = np.std(opt_returns)
llm_std = np.std(llm_returns)
llm75_opt25_std = np.std(coord_llm75_opt25_returns)
llm50_opt50_std = np.std(coord_returns)
llm25_opt75_std = np.std(coord_llm25_opt75_returns)

# Set up data and colors
labels = ['Returns']
means = [opt_mean, llm_mean, llm75_opt25_mean,
         llm50_opt50_mean, llm25_opt75_mean]
stds = [opt_std, llm_std, llm75_opt25_std, llm50_opt50_std, llm25_opt75_std]
colors = ['#5f0f40', '#9a031e', '#fb8b24', '#e36414', '#0f4c5c']
bar_names = ['OPT', 'LLM', 'LLM75_OPT25', 'LLM50_OPT50', 'LLM25_OPT75']

# Create figure
plt.figure(figsize=(10, 8))

# Create the bars with no spacing
total_width = 0.8
bar_width = total_width / len(means)
positions = np.arange(len(labels))

# Place error bars on the bars
for i in range(len(means)):
    offset = i * bar_width
    plt.bar([p + offset for p in positions],
            [means[i]],
            width=bar_width,
            color=colors[i],
            yerr=stds[i],
            capsize=5,
            label=bar_names[i])

# Customize the plot
plt.ylabel('Return Multiple')
plt.title('Average Returns with Variance')
plt.grid(axis='y', alpha=0.3)
plt.ylim(0, max(means) + 3*max(stds))
plt.ylim(0, 2)  # Keep the original y-limit

# Add text box with statistics for all datasets
stats_text = '\n'.join([
    f'{name} Mean: {mean:.3f}, Std: {std:.3f}, n: {len(x_return)}'
    for name, mean, std, x_return in zip(bar_names, means, stds, returnses)
])
plt.text(0.95, 0.95, stats_text,
         transform=plt.gca().transAxes, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add legend
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)

# Adjust xticks to center under the grouped bars
plt.xticks([p + total_width/2 - bar_width/2 for p in positions], labels)

plt.tight_layout()
plt.savefig('YYY/combined_returns_with_variance.png',
            dpi=300, bbox_inches='tight')
plt.show()
