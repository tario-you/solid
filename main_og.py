# %% [markdown]
# ### SOLID: a Framework of Synergizing Optimization and Large Language Models for Intelligent Decision-Making
# Authors: Yinsheng Wang, Tario You, Léonard Boussioux
#
# In this model, we build two agents to decide an investment strategy for a portfolio of stocks.
# For simplicity, we assume that the portfolio consists of 4 stocks: NVDA, GOOG, MSTR, and SMCI.
# The first agent is a mean-variance optimization model. It aims to minimize the portfolio variance while achieving a target return.
# The second agent is a GPT-based Language Model. Through prompt-based learning, it aims to generate a portfolio strategy that maximizes the portfolio return.
# The two agents will communicate with each other to reach a consensus on the portfolio strategy, i.e., the portfolio weights for the two stocks.

# %% [markdown]
# #### Import necessary libraries

# %%
from sklearn.model_selection import TimeSeriesSplit
from functools import partial
from pypfopt import expected_returns, risk_models, EfficientFrontier
import seaborn as sns
import matplotlib.cm as cm
import glob
from openai import OpenAI
import json
import pandas as pd
from tqdm import tqdm
import yfinance as yf
import pandas_market_calendars as mcal
import datetime
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import expected_returns
from pypfopt import plotting
from pypfopt import risk_models
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import List
from openai import OpenAI
import os
import re
import pandas as pd
from tqdm import tqdm
import yfinance as yf
from gurobipy import Model, GRB, quicksum
import warnings
from dotenv import load_dotenv
import matplotlib.colors as mcolors
from IPython.display import display

load_dotenv()

warnings.filterwarnings('ignore')

pd.set_option('display.expand_frame_repr', False)  # Prevent splitting
pd.set_option('display.max_columns', None)        # Show all columns
pd.set_option('display.max_rows', None)           # Optional: Show all rows

api_key = os.getenv("PPLX")

# tickers = ["NVDA", "GOOG", "MSTR", "SMCI", "TSLA", "WMT"] # 1
# tickers = ["AAPL", "JPM", "XOM", "JNJ", "WMT", "HD", "AMT", "BA", "NEE"] # 2
# tickers = ["NVDA", "GOOG", "MSTR", "SMCI", "TSLA", "AAPL", "JPM", "XOM", "JNJ", "WMT", "HD", "AMT", "BA", "NEE"] # 3
# tickers = ["NVDA", "GOOG", "MSTR", "SMCI", "TSLA", "AAPL", "JPM", "XOM", "JNJ", "WMT", "HD", "AMT", "BA", "NEE", "V"] # 4
# tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "JPM", "XOM", "UNH", "WMT", "HD", "CAT", "PLD", "NEE", "V", "AMD"] # 6% better
# tickers = ["MS", "BAC", "CVX", "PFE", "PEP", "COST", "NFLX", "INTC", "LMT", "CSCO", "AXP", "AMZN", "TMUS", "TM", "DUK"]
# tickers = ["CVX", "PFE", "NFLX", "LMT", "TM", "PLTR", "OKTA", "MAR", "MCD", "SBUX", "EBAY", "MRNA", "BHP", "TGT", "EOG"]
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

stock_categories = [
    "Technology",
    "Consumer Discretionary",
    "Finance",
    "Real Estate",
    "Energy",
    "Healthcare",
    "Industrial",
    "Material",
    "Communication",
    "Consumer Staples"
]

# %%
# CHANGE THESE!!
idxdidxd = 1
appendage = "nvda60"
llm_model = "gpt-4o-mini"
iteration = f"2025-03-29_{llm_model}_{idxdidxd}"

# always just appends
init_news_path = f"assets/init_news_reports.json"
stock_data_path = f"assets/stock_data.json"

pft_path = f"assets/portfolio_{appendage}.csv"
stock_price_history_image_path = f'figures/stock_price_history_{appendage}.png'

grid_image_path = f'assets/output_{appendage}_{iteration}.png'
grid_image_sparse_path = f'assets/output_sparse_{appendage}_{iteration}.png'
weights_coord_path = f"assets/weights_coord_{appendage}_{iteration}.json"
weights_coord_sparse_path = f"assets/weights_coord_sparse_{appendage}_{iteration}.json"
weights_llm_path = f"assets/weights_llm_{appendage}_{iteration}.json"
weights_llm_sparse_path = f"assets/weights_llm_sparse_{appendage}_{iteration}.json"
weights_opt_path = f"assets/weights_opt_{appendage}_{iteration}.json"
pft_value_over_time_path = f'figures/pft_value_over_time_{appendage}_{iteration}.png'
risk_path = f'figures/risk_{appendage}_{iteration}.png'
heatmap_path = f"figures/heatmap_{appendage}_{iteration}.png"
heatmap_all_path = f"figures/heatmap_all_{appendage}_{iteration}.png"
directory_path = f"assets/indiv/{appendage}_{iteration}"
directory_path_sparse = f"assets/indiv/{appendage}_{iteration}_sparse"
pnl_path = f"figures/pnl_{appendage}_{iteration}.png"
status_path = f"assets/status_{appendage}_{iteration}.json"
status_sparse_path = f"assets/status_sparse_{appendage}_{iteration}.json"

run_it_weighted = True
rerun_llm, rerun_opt, rerun_coord, rerun_llm_sparse, rerun_coord_sparse = False, False, False, False, False
graph_indiv = False

# %% [markdown]
# #### Functions to save data locally

# %%


def save_data(data, file_path=stock_data_path):
    """
    Saves the dictionary 'data' to a JSON file at 'file_path'.
    """
    with open(file_path, 'w') as fp:
        json.dump(data, fp, indent=4)
    print(f"Data saved to {file_path}")


def load_data(file_path=stock_data_path):
    """
    Loads JSON data from 'file_path' and returns it as a dictionary.
    """
    with open(file_path, 'r') as fp:
        data = json.load(fp)
    print(f"Data loaded from {file_path}")
    return data


def patch_data(
    tickers,
    file_path=stock_data_path
):
    """
    Load existing data from file_path, then patch each month's dictionary
    to include 'MSTR' and 'SMCI' using Perplexity.ai's OpenAI-like client calls.
    Finally, save the patched data back to file_path.
    """

    # 1) Load the existing data
    data = load_data(file_path)

    # 2) Set up your client, months, system prompt, etc.
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

    months = [
        "January", "Febuary", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    system_prompt = {
        "role": "system",
        "content": (
            "Show me key news on [Company X] (I'll provide you the stock ticker) from [Date Range]. This list isn't exhaustive—earnings, leadership changes, regulatory updates, major headlines, M&A, industry trends, product launches, analyst opinions, investor activism, competitor moves—but only pick what's most representative for [Company X]'s stock performance. If other items seem more important, include them. Summarize in bullet points; avoid complete sentences; aim for maximum information. You don't have to include everything, just the key pieces. Keep final summary around 400 words."
        )
    }

    # 3) Patch each month's dictionary if tickers dont exist
    for i, monthly_data in enumerate(data):
        for j in tqdm(range(len(tickers))):
            ticker = tickers[j]
            if ticker not in monthly_data:
                # Build the request messages for the missing ticker
                messages = [
                    system_prompt,
                    {
                        "role": "user",
                        "content": f"What happened to {ticker} in {months[i]} 2024?"
                    }
                ]

                # Make the API call
                response = client.chat.completions.create(
                    model="llama-3.1-sonar-large-128k-online",
                    messages=messages,
                )

                # Save the result in the monthly dictionary
                monthly_data[ticker] = {
                    "news": response.choices[0].message.content}
                # print(f'patched: month {months[i]}\t{ticker}')
            # else:
                # print(f'skipping: {ticker}')

            # save_data(data, file_path)

    # 4) Save the patched data back to the JSON file
    save_data(data, file_path)

# %% [markdown]
# #### Using Perplexity to gather news about the tickers


# %%
client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
months = ["January", "Febuary", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
data = [
    # Jan
    # {
    #     "MSFT":{
    #         "news": "yay", # news for all of Jan
    #         "price": 20 # last day's closing price - Jan 31
    #     }
    # }
]

# %% [markdown]
# #### Loading the S&P 500 tickers

# %%
file_path = 'assets/constituents.csv'
data = pd.read_csv(file_path)
constituents = data['Symbol'].tolist()

# constituents

# %% [markdown]
# #### Patching data for tickers

# %%
# patch_data(tickers)
# S&P 500: patch_data(constituents)

# %% [markdown]
# #### Initial stock introductions

# %%


def load_init_news_reports():
    with open("assets/init_news_reports.json", "r") as f:
        stock_reports = json.load(f)
    return stock_reports


def save_init_news_reports(init_news_reports):
    with open("assets/init_news_reports.json", "w") as f:
        json.dump(init_news_reports, f, indent=4)


def patch_init_news_reports():
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

    init_news_reports = load_init_news_reports()
    for ticker in tickers:
        if ticker not in init_news_reports:

            messages = [
                {
                    "role": "system",
                    "content": "You analyze and summarize companies."
                },
                {
                    "role": "user",
                    "content": f"Give me a 100 word summary about the stock ticker {ticker}"
                }
            ]

            response = client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=messages,
            )

            init_news_reports[ticker] = response.choices[0].message.content

    save_init_news_reports(init_news_reports)

# patch_init_news_reports()

# %% [markdown]
# #### Get ticker prices

# %%


def get_last_trading_day_of_month(year, month, exchange='NYSE'):
    # Create a calendar for the specified exchange
    calendar = mcal.get_calendar(exchange)

    # Get the last day of the specified month
    if month == 12:
        last_day = datetime.datetime(
            year + 1, 1, 1) - datetime.timedelta(days=1)
    else:
        last_day = datetime.datetime(
            year, month + 1, 1) - datetime.timedelta(days=1)

    # Get the schedule for the month
    schedule = calendar.schedule(
        start_date=f"{year}-{month:02d}-01", end_date=last_day)

    # If the schedule is empty, there were no trading days this month
    if schedule.empty:
        return None

    # Return the last trading day
    return schedule.index[-1].date().day


def get_stock_price(tickers, date):
    prices = yf.download(tickers, start=date, end=date +
                         datetime.timedelta(days=1))
    prices = prices["Adj Close"].dropna(how="all")
    prices = prices.values.tolist()
    return prices[0]


def get_closing_prices(data_loaded):
    year = 2024
    for month in range(1, 13):
        date = get_last_trading_day_of_month(year, month)
        datetime_obj = datetime.datetime(year, month, date)

        missing_tickers = []
        missing_indices = []
        for i, ticker in enumerate(tickers):
            ticker_data = data_loaded[month-1][ticker]
            if "price" not in ticker_data:
                missing_tickers.append(ticker)
                missing_indices.append(i)
            else:
                if np.isnan(ticker_data['price']):
                    missing_tickers.append(ticker)
                    missing_indices.append(i)

        if missing_indices:
            prices = get_stock_price(missing_tickers, datetime_obj)
            if type(prices) != type([]):
                prices = [prices]

            for i, (ticker, j) in enumerate(zip(missing_tickers, missing_indices)):
                data_loaded[month-1][ticker]['price'] = prices[i]

    save_data(data_loaded)

# get_stock_price('ZG', datetime.datetime(2024, 1, get_last_trading_day_of_month(2024, 1)))


data_loaded = load_data()
get_closing_prices(data_loaded)

# %% [markdown]
# #### Getting the tickers' historical prices for the optimizer
# This is not just closing price at the end of each month like above, it's the entire data for a year

# %%
leo_key = os.getenv("CHAT")

client = OpenAI(
    api_key=leo_key,
)

# def get_stock_price(tickers, start_date, end_date):
#     prices = yf.download(tickers, start=start_date, end=end_date)
#     prices = prices["Adj Close"].dropna(how="all")
#     return prices

# start_date = datetime.datetime(2024, 1, 1)
# end_date = datetime.datetime.today().date()

# portfolio = get_stock_price(tickers, start_date, end_date)
# portfolio.to_csv(pft_path, index=True)
portfolio = pd.read_csv(pft_path, parse_dates=True, index_col="Date")

# %%
portfolio

# %% [markdown]
# #### Historical ticker prices

# %%

df = portfolio

# Function to generate a list of colors by iterating through RGB values


def generate_colors(n_colors):
    colors = []
    for i in range(n_colors):
        r = (i * 37) % 256 / 255.0  # Example formula for varying red
        g = (i * 59) % 256 / 255.0  # Example formula for varying green
        b = (i * 83) % 256 / 255.0  # Example formula for varying blue
        colors.append((r, g, b))
    return colors


# Generate unique colors based on the number of columns
colors = generate_colors(len(df.columns))

# Plot
plt.figure(figsize=(8, 5))

# Plot each column with a unique color
for idx, column in enumerate(df.columns):
    plt.plot(df.index, df[column], label=column, color=colors[idx])

# Formatting the x-axis
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Price History")
plt.legend()

tick_indices = df.index[::40]  # Select every 60th index
plt.xticks(tick_indices, [date.strftime('%Y-%m-%d')
           for date in tick_indices], rotation=45)

plt.tight_layout()

# Save the plot
plt.savefig(stock_price_history_image_path, dpi=500, bbox_inches='tight')

# Show the plot
plt.show()


# %%
sample_cov = risk_models.sample_cov(portfolio, frequency=252)

S = risk_models.CovarianceShrinkage(portfolio).ledoit_wolf()
mu = expected_returns.capm_return(portfolio)

mu.plot.barh(figsize=(5, 3))

ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()

cleaned_weights = ef.clean_weights()
ef.portfolio_performance(verbose=True)


latest_prices = get_latest_prices(portfolio)

da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=100000)

# Number of shares of each stock to purchase
allocation, leftover = da.greedy_portfolio()

n_samples = 10000
w = np.random.dirichlet(np.ones(len(mu)), n_samples)
rets = w.dot(mu)
stds = np.sqrt((w.T * (S @ w.T)).sum(axis=0))
sharpes = rets / stds

ef = EfficientFrontier(mu, S)

fig, ax = plt.subplots(figsize=(6, 4))
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

# Find and plot the tangency portfolio
ef2 = EfficientFrontier(mu, S)
ef2.max_sharpe()
ret_tangent, std_tangent, _ = ef2.portfolio_performance()

# Plot random portfolios
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
ax.scatter(std_tangent, ret_tangent, c='red',
           marker='X', s=150, label='Max Sharpe')

# Format
ax.set_title("Efficient Frontier with random portfolios")
ax.legend()
plt.tight_layout()

mu

# %% [markdown]
# #### Generate initial stock info

# %%
with open(init_news_path, "r") as f:
    stock_reports = json.load(f)
stock_reports

# %%
# brief stock introductions obtained from Perplexity.ai


def generate_data_summary(reports):
    summary = f"Recent reports indicate:\n"
    for ticker in tickers:
        report = reports[ticker]
        summary += f"For {ticker}:\n{report}\n\n"
    return summary


initial_stock_info = generate_data_summary(stock_reports)

# %% [markdown]
# #### Dynamic YF stock data storage
# pre download the data

# %%
# get the data
# start_date = datetime.datetime(2023, 1, 1)
# end_date = datetime.datetime(2025, 1, 1)
# prices = yf.download(tickers, start=start_date, end=end_date)
# portfolio = prices["Adj Close"].dropna(how="all")
# portfolio.to_csv(pft_path, index=True)

# %%
# demonstration of the data being got
# portfolio = pd.read_csv(pft_path, parse_dates=["Date"], index_col="Date")
start_date = datetime.datetime(2023, 12, 1)
end_date = datetime.datetime(2023, 12, get_last_trading_day_of_month(2023, 12))
df_subset = portfolio.loc[start_date:end_date]
df_subset

# %% [markdown]
# #### Main class to integrate the LLM with the Portfolio Optimization model

# %%


class bcolors:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'

# %%


class CoordinationFramework():
    def __init__(self, mu, Q, target_return, penalty=1, iteration=2, verbose=False):
        """
        Initialization of the class for coordination framework
        :param np.array mu: expected return of the stocks
        :param np.array Q: covariance matrix of the stocks
        :param float target_return: target return of the portfolio
        :param float penalty: penalty term in coordination algorithm
        :param float iteration: number of iteration of coordination algorithm
        """
        # Store the init params so we can restore them later
        self._init_mu = mu
        self._init_Q = Q
        self._init_target_return = target_return
        self._init_penalty = penalty
        self._init_iteration = iteration
        self._init_verbose = verbose

        # Now do the usual initialization
        self.mu = mu  # expected return
        self.Q = Q.to_numpy()  # covariance matrix
        self.n = len(mu)  # number of stocks
        self.target_return = target_return

        self.penalty = penalty
        self.iteration = iteration
        self.verbose = verbose

        # initialization
        self.current_plan = [0.0] * self.n
        self.optimization_plan = [0.0] * self.n
        self.LLM_plan = [0.0] * self.n
        self.optimization_price = [0.0] * self.n
        self.LLM_price = [0.0] * self.n

        self.feedback_factor = 0.1

        self.plan_histories = []
        self.conversation_history = []
        self.conversation_summaries = []

    def reset_variables(self):
        """
        Reset all variables to their initial values, exactly as they were in __init__.
        """
        # Restore parameters
        self.mu = self._init_mu
        self.Q = self._init_Q.to_numpy()
        self.n = len(self._init_mu)
        self.target_return = self._init_target_return

        self.penalty = self._init_penalty
        self.iteration = self._init_iteration
        self.verbose = self._init_verbose

        # Reinitialize mutable state variables
        self.current_plan = [0.0] * self.n
        self.optimization_plan = [0.0] * self.n
        self.LLM_plan = [0.0] * self.n
        self.optimization_price = [0.0] * self.n
        self.LLM_price = [0.0] * self.n

        self.feedback_factor = 0.1

        self.plan_histories = []
        self.conversation_history = []
        self.conversation_summaries = []

    # In this function, the optimization model will update their preferred portfolio weights.
    def PortfolioOptimization_Agent(self, current_plan, verbose=False):
        # Initialize model
        self.model = Model("mean_variance_optimization")
        self.model.setParam('OutputFlag', 0)

        # Add variables (portfolio weights)
        x = {}
        for i in range(self.n):
            x[i] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"x_{i}")

        # Define terms for objective
        # ------------------------------------------------
        # Risk term = sum_{i,j} Q[i,j] * x[i] * x[j]
        risk_expr = quicksum(self.Q[i, j] * x[i] * x[j]
                             for i in range(self.n)
                             for j in range(self.n))

        # Dual (price) term = sum_{i} optimization_price[i] * x[i]
        dual_expr = quicksum(
            self.optimization_price[i] * x[i] for i in range(self.n))

        # Penalty term = penalty * sum_{i} (x[i] - current_plan[i])^2
        penalty_expr = quicksum((x[i] - current_plan[i]) * (x[i] - current_plan[i])
                                for i in range(self.n))

        # Set the model objective as:
        #   0.5 * (risk_expr) + dual_expr + 0.5 * penalty * (penalty_expr)
        self.model.setObjective(
            0.5 * risk_expr + dual_expr + 0.5 * self.penalty * penalty_expr,
            GRB.MINIMIZE
        )
        # ------------------------------------------------

        # Constraints
        # ------------------------------------------------
        # 1) Sum of weights = 1
        self.model.addConstr(quicksum(x[i]
                             for i in range(self.n)) == 1, "budget")

        # 2) Enforce minimum target return
        self.model.addConstr(
            quicksum(self.mu[i] * x[i]
                     for i in range(self.n)) >= self.target_return,
            "target_return"
        )
        # ------------------------------------------------

        # Optimize
        self.model.optimize()

        if verbose:
            print("\n[DEBUG] Building model with:")
            print(f"[DEBUG]   mu: {self.mu}")
            print(f"[DEBUG]   Q: {self.Q}")
            print(f"[DEBUG]   target_return: {self.target_return}")
            print("[DEBUG]   current_plan:", current_plan)
            print("[DEBUG]   optimization_price:", self.optimization_price)
            print("[DEBUG]   penalty:", self.penalty)
            print("[DEBUG]   n:", self.n)
            self.model.write("debug/debug_model.lp")

            status = self.model.status
            print(f"[DEBUG] Gurobi optimization status: {status}")

        # Extract optimized weights
        weights = [x[i].x for i in range(self.n)]

        return weights

    def response_format(self):
        prompt = "please write your final recommendation in a single line, in the format below:\n\n"

        # Construct the label format dynamically based on the number of tickers.
        # Example of final output:  "AAPL: X1, MSFT: X2, AMZN: X3. Replace X1, X2, X3..."
        prompt += ", ".join([f"{ticker}: X{i+1}" for i,
                            ticker in enumerate(tickers)])
        prompt += (
            ".\n\nReplace "
        )

        # Remind the model to replace placeholders with the actual confidence levels.
        prompt += ", ".join([f"X{i+1}" for i in range(len(tickers))])
        prompt += (
            " with the confidence level for each stock."
        )

        return prompt

    # In this function, the language model take the prompt as input and generate the preferred portfolio weights.
    def LLM_Agent(self, month_data, consensus_plan, month, iter, sparse, pure=False, verbose=False):
        system_prompt = """You are collaborating with an optimization model to develop an optimal portfolio strategy. The optimization model is an expert in numerical calculations and historical trend analysis, ensuring precision in minimizing risk and achieving target returns. Your role is to bring contextual understanding, strategic insights, and adaptability to the process.\nWhile you are encouraged to bring your unique perspective, remember:\n1. The optimization model has consistently outperformed by a significant margin (10-20%). Therefore, where substantial differences arise, you must carefully review the optimizer's proposal and adjust your recommendations unless you have a strong, data-backed reason to maintain your position.\n2. In this collaboration, flexibility and compromise are critical. Adapt your strategy when the optimizer provides strong evidence, as rigid decision-making has been shown to lead to suboptimal results.\n3. Your goal is to work synergistically with the optimizer to maximize overall portfolio performance. This requires understanding and integrating the optimizer's proposals to improve alignment.\n4. If you disagree, explicitly identify the key aspects of the optimizer's proposal that you are willing to accept or adjust. Never reject the optimizer's strategy outright—focus on making incremental refinements."""
        if pure:
            system_prompt = "You're a very advanced stock trading expert with a deep understanding in the politics, economics, and business of companies in relation to their stock performance. When you see a news, you will know whether or not it will have a strong impact on your trader planning the next move of investment decisions. You always maximize the profit through your stock investments."

        if "o1-mini" in llm_model:
            messages = []
            current_prompt = system_prompt + "\n"
        elif "o1" in llm_model:
            messages = [
                {"role": "developer", "content": system_prompt}
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
            ]
        messages.extend(self.conversation_history)

        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        current_prompt = ""

        if iter == 0:
            # Optionally include any initial stock info if this is the very first iteration.
            current_prompt += (
                f"{initial_stock_info}\n\n"
            )

        # summarize conversations for previous months
        # for 60 stocks, exceeds 128k tokens
        # if month != 0 and len(self.conversation_summaries) != 0:
        #     current_prompt += "Here is what happened in the last few months for you to gain a background understanding of what happened:\n"
        #     current_prompt += "\n".join(self.conversation_summaries)
        #     current_prompt += "\n\n"

        if iter == 0:
            stock_prices = f"The stock prices today are:\n"
            for i, ticker in enumerate(tickers):
                ticker_close = month_data[ticker]['price']
                stock_prices += f"{ticker} = {ticker_close}"
                if i != len(tickers) - 1:
                    stock_prices += ", "
            stock_prices += "\n"

            stock_news = ""
            for ticker in tickers:
                ticker_news = month_data[ticker]['news']
                stock_news += f"news for {ticker}:\n{ticker_news}\n\n"

            # If this is the first iteration in a given month, include relevant news and price info.
            current_prompt += (
                "Please read the following information carefully.\n\n"
                f"---\n**Stock News**\n\n{stock_news}\n\n"
                f"---\n**Recent Stock Prices**\n\n{stock_prices}\n\n"
            )

        # Begin the main decision instructions.
        current_prompt += (
            "You are a trader responsible for making portfolio allocation decisions. "
            # "Use all relevant information provided (such as any past decisions, news, or stock data) to "
            "Use all relevant information provided (such as news and stock data) to "
            "decide how much to invest in each stock.\n\n"
            "Think about:\n"
            "1. Any news articles and how they might affect each stock.\n"
            # "2. Any patterns in recent price movements.\n"
            "2. Previous decisions you have made regarding portfolio weights.\n"
        )

        # If we're past the first iteration, include guidance about consensus plans.
        if iter != 0 and self.current_plan != [0.0] * self.n and not pure:
            current_prompt += (
                "Also, you are working with a optimization model that is very proficient in numerical calculations, and here is the current plan (portfolio allocation) you guys have worked" "out. Decided if you agree with this plan, then make the necessary adjustments to your own plan: "
                f"{self.current_plan}\n\n"
            )

        current_prompt += (
            f"Also, here is the decision-price of your plan thus far: {self.LLM_price}."
            "A higher decision-price means you should adjust your plan to be higher. And a negative decision-price means you should adjust your plan to be smaller."
        )

        # Ask the model for a recommendation. Emphasize the requirement to explain reasoning first, then provide the format.
        current_prompt += (
            "### Task\n"
            "1. Carefully evaluate the optimizer's proposed portfolio weights and explain your reasoning for agreement or disagreement. When in doubt, lean towards collaboration by adjusting your recommendations closer to the optimizer's.\n"
            "2. Finalize your recommendation in the following format: [Ticker: Confidence Level]:\n"
            "   - Very Low Confidence\n"
            "   - Low Confidence\n"
            "   - Somewhat Low Confidence\n"
            "   - Neutral\n"
            "   - Somewhat High Confidence\n"
            "   - High Confidence\n"
            "   - Very High Confidence\n\n"
            "3. Conclude by summarizing how your proposal aligns with the optimizer's and why it contributes to achieving the collective goals.\n"
            "Even if you are unsure, you **must** provide the best decision you can based on the available information.\n\n"
            "Take a deep breath and work on this problem step-by-step.\n")

        if sparse:
            current_prompt += (
                "IMPORTANT: Aim for *sparsity* in your final allocation. "
                "Ideally select **only 5 to 10 stocks** to invest in (with confidence above 'Very Low'). "
                "Assign Very Low confidence to the rest (effectively zero). "
                "If you exceed 10 or go below 5 stocks rated above 'Very Low,' your proposal is invalid. "
                "Choose carefully.\n\n"
            )

        current_prompt += ("### Response Format\n"
                           "After your explanation, " + self.response_format() + ""
                           "\nExplicitly end your response in that format. "
                           "So make sure you have these stocks and confidence levels clearly written out to be parsed by a regex function."
                           "\nRemember, collaboration, adaptability, and performance are key to success."
                           )

        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------

        if verbose:
            print(f"\n# month {month} iter {iter}")
            print(f"\nprompt: \n{current_prompt}\n")

        messages.append({"role": "user", "content": current_prompt})

        result_dict = []
        retry_messages = [m for m in messages]
        attempt = 0
        retry = True
        retry_reason = ""
        missing_tickers = set(tickers)
        result_dict = {}
        while retry:
            if attempt != 0:
                if retry_reason == "INVALID FORMAT":
                    new_message = "Sorry, I could not parse your response. Please try again with the correct specified formatting: " + self.response_format()

                    missing_tickers = set(tickers)
                elif retry_reason == "ZERO SUM":
                    new_message = "The sum of the weights for each stock cannot be 0. Please try again: " + \
                        self.response_format()

                    missing_tickers = set(tickers)
                else:  # "MISSING TICKER"
                    new_message = (
                        f"You missed the following tickers: {missing_tickers}.\n"
                        "**IMPORTANT**: If you are seeing this message, it means your response did not match "
                        "the required format and our system's regex could not interpret it.\n"
                        "Please carefully review and correct your response so it follows the exact format "
                        "below. Otherwise, the regex will continue to fail, and you will keep receiving this notice.\n\n"
                        "Provide the confidence levels for these tickers in the following format:\n"
                        + self.response_format()
                    )

                retry_messages.append({
                    "role": "user",
                    "content": new_message
                })

                if verbose:
                    # PROMPT: \n{new_message}\n
                    print(
                        f"\n# month {month} iter {iter} {bcolors.RED}RETRY{bcolors.ENDC} because {retry_reason}")

            attempt += 1
            # Make an API call to ChatGPT with the prompt
            if "o1" in llm_model or "o3" in llm_model:
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=retry_messages
                )
            else:
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=retry_messages,
                    temperature=0
                )

            # Parse the decision from the response
            text = response.choices[0].message.content
            retry_messages.append({
                "role": "assistant",
                "content": text
            })

            if verbose:
                print(
                    f"{bcolors.PURPLE}[DEBUG]{bcolors.ENDC}\tChat reponse: {text}")

            CONFIDENCE_LEVELS = {
                "Very High": 0.6,
                "High": 0.5,
                "Somewhat High": 0.4,
                "Neutral": 0.3,
                "Somewhat Low": 0.2,
                "Low": 0.1,
                "Very Low": 0.0
            }

            # Create regex pattern from confidence levels
            pattern = "|".join(CONFIDENCE_LEVELS.keys())

            original_missing_tickers = missing_tickers.copy()
            for stock in original_missing_tickers:
                # Use word boundary \b to ensure exact stock matches
                match = re.search(
                    fr'\b{stock}\b.*?:\s*({pattern})',
                    text,
                    re.IGNORECASE
                )

                if match:
                    confidence = match.group(1).title()
                    result_dict[stock] = CONFIDENCE_LEVELS[confidence]
                    missing_tickers.remove(stock)
                else:
                    retry = True
                    if verbose:
                        print(f"[DEBUG]\tCouldn't fetch {stock}.")

            retry = False

            if result_dict == {}:
                if verbose:
                    print(
                        f"{bcolors.RED}[DEBUG]{bcolors.ENDC}\tInvalid format: could not find tickers, retrying.")
                retry = True
                retry_reason = "INVALID FORMAT"
                continue

            if sum(result_dict.values()) == 0:
                if verbose:
                    print(
                        f"{bcolors.RED}[DEBUG]{bcolors.ENDC}\tInvalid output: sum = 0")
                retry = True
                retry_reason = "ZERO SUM"
                continue

            if len(missing_tickers) != 0:
                print(
                    f"{bcolors.RED}[DEBUG]{bcolors.ENDC}\tmissing {missing_tickers = }")
                retry = True
                retry_reason = "MISSING TICKER"
                continue

            if verbose:
                print(
                    f"{bcolors.GREEN}[DEBUG]{bcolors.ENDC}\tfetched weights: {result_dict = }")
                retry = False

        # normalize sum to 1
        norm_factor = 1/sum(result_dict.values())
        result_dict = {k: v * norm_factor for k, v in result_dict.items()}
        result_dict = [r for r in result_dict.values()]

        self.conversation_history.append(
            {"role": "user", "content": current_prompt})
        self.conversation_history.append(
            {"role": "assistant", "content": text})

        norm_factor = 1/sum(result_dict)
        normalized_weights = [norm_factor * w for w in result_dict]

        return normalized_weights

    # update the consensus plan and activity price for the next iteration
    def update_plan(self, plan1, price1, plan2, price2):
        result = []
        # plan1 = [p for p in plan1.values()] # if using dict with tickers
        # plan2 = [p for p in plan2.values()]

        for p1, p2, pr1, pr2 in zip(plan1, plan2, price1, price2):
            average_plan = (p1 + p2) / 2
            average_price = (pr1 + pr2) / 2
            result.append(max(0, average_price / self.penalty + average_plan))
        return result

    def update_activity_price(self, current_activity_price, current_plan, new_plan):
        updated_prices = []

        for curr_price, curr_plan, new_plan_val in zip(current_activity_price, current_plan, new_plan):
            adjustment = self.penalty * (new_plan_val - curr_plan)
            updated_price = curr_price - adjustment
            updated_prices.append(updated_price)

        return updated_prices

    def test_convergence(self, all_llm_opt_plans):
        # if llm's plan and opt's plan do not differ by more than 5%
        try:
            plan_convergence = True
            for l in range(len(tickers)):
                if abs(all_llm_opt_plans[-1][l+len(tickers)+1] - all_llm_opt_plans[-1][l+2*len(tickers)+1]) > 1/20:
                    plan_convergence = False
                    break
            if plan_convergence:
                return True
        except Exception as e:
            print(f"Error occurred: {str(e)}")

        # if this iter's plan and last iter's plan do not differ by more than 0.2%
        # - weights have basically not changed in the past two iterations
        try:
            iter_convergence = True
            for i in range(1, len(all_llm_opt_plans[0])):
                if abs(all_llm_opt_plans[-1][i] - all_llm_opt_plans[-2][i]) > 1/500:
                    iter_convergence = False
                    break
            if iter_convergence:
                return True
        except Exception as e:
            print(f"Error occurred: {str(e)}")

        return False

    def test_convergence_aux(self):
        max_diff = 0
        max_diff_ticker = None

        try:
            for l, ticker in enumerate(tickers):
                diff = abs(
                    self.plan_histories[-1][l+len(tickers)+1] - self.plan_histories[-1][l+2*len(tickers)+1])
                if diff > max_diff:
                    max_diff = diff
                    max_diff_ticker = self.plan_histories[-1][0]

            # print(f"[DEBUG]\tThe max diff ticker is {max_diff_ticker} with a diff of {max_diff}")
        except Exception as e:
            print(f"[DEBUG] Error occurred: {str(e)}")
            pass

    def OptAlgorithm(self, data, verbose=False):
        self.reset_variables()

        for month, month_data in enumerate(data):
            prev_year = 2024 if month != 0 else 2023
            prev_month = month if month != 0 else 12

            prev_prev_year = prev_year if prev_month != 1 else 2023
            prev_prev_month = prev_month - 1 if prev_month != 1 else 12

            start_date = datetime.datetime(
                prev_prev_year, prev_prev_month, get_last_trading_day_of_month(prev_prev_year, prev_prev_month))
            end_date = datetime.datetime(
                prev_year, prev_month, get_last_trading_day_of_month(prev_year, prev_month))

            portfolio = pd.read_csv(pft_path, parse_dates=[
                                    "Date"], index_col="Date")

            last_month_data = portfolio.loc[start_date].fillna(0)
            month_data = portfolio.loc[end_date].fillna(0)

            self.mu = expected_returns.mean_historical_return(
                portfolio.loc[start_date:end_date])  # capm_return / mean_historical_return
            self.Q = risk_models.CovarianceShrinkage(
                portfolio.loc[start_date:end_date]).ledoit_wolf().to_numpy()

            self.optimization_plan = self.PortfolioOptimization_Agent(
                self.current_plan, verbose)

            self.current_plan = self.optimization_plan
            self.optimization_price = self.update_activity_price(
                self.optimization_price, self.optimization_plan, self.current_plan)

            self.plan_histories.append(self.optimization_plan)

        return self.plan_histories

    def OptAlgorithmParams(self, portfolio_path, best_params, data_loaded=None, verbose=False):
        """
        Generate optimal portfolio weights for each month using the specified parameters.

        Parameters:
        -----------
        portfolio_path : str
            Path to CSV file containing portfolio data
        best_params : dict
            Parameters to use for optimization (from hyperparameter tuning)
        data_loaded : list, optional
            Monthly data for additional information (same as used in OptAlgorithm)
        verbose : bool
            Whether to print detailed progress information

        Returns:
        --------
        list
            List of optimized weight lists for each month
        """
        import datetime
        import pandas as pd
        import numpy as np
        from pypfopt import expected_returns, risk_models, EfficientFrontier

        # Extract parameters
        risk_aversion = best_params.get('risk_aversion', 1.0)
        target_return = best_params.get('target_return', None)
        shrinkage_method = best_params.get('shrinkage_method', 'ledoit_wolf')
        risk_free_rate = best_params.get('risk_free_rate', 0.02)

        # List to store optimized weights for each month
        plan_histories = []

        # For each month
        for month in range(12):
            if verbose:
                print(f"Optimizing for month {month + 1}")

            # Determine the date range for this month
            prev_year = 2024 if month != 0 else 2023
            prev_month = month + 1 if month != 0 else 12

            prev_prev_year = prev_year if prev_month != 1 else 2023
            prev_prev_month = prev_month - 1 if prev_month != 1 else 12

            start_date = datetime.datetime(
                prev_prev_year, prev_prev_month,
                get_last_trading_day_of_month(prev_prev_year, prev_prev_month))

            end_date = datetime.datetime(
                prev_year, prev_month,
                get_last_trading_day_of_month(prev_year, prev_month))

            # Load portfolio data
            portfolio = pd.read_csv(portfolio_path, parse_dates=[
                                    "Date"], index_col="Date")

            # Get relevant data for the period
            try:
                last_month_data = portfolio.loc[start_date].fillna(0)
                month_data = portfolio.loc[end_date].fillna(0)
                period_data = portfolio.loc[start_date:end_date]
            except KeyError:
                if verbose:
                    print(
                        f"  Warning: Some dates not found in data, using available date range")
                period_data = portfolio[portfolio.index.to_series().between(
                    start_date, end_date)]

            # Calculate expected returns
            mu = expected_returns.mean_historical_return(period_data)

            # Calculate covariance matrix
            if shrinkage_method == 'ledoit_wolf':
                S = risk_models.CovarianceShrinkage(period_data).ledoit_wolf()
            elif shrinkage_method == 'sample':
                S = risk_models.sample_cov(period_data)
            else:
                S = risk_models.CovarianceShrinkage(
                    period_data).oracle_approximating()

            # Initialize the efficient frontier optimizer
            ef = EfficientFrontier(mu, S)

            # Optimize based on parameters
            try:
                if target_return is not None:
                    try:
                        ef.efficient_return(target_return)
                    except:
                        # Fallback to max sharpe if target return is unreachable
                        ef.max_sharpe(risk_free_rate=risk_free_rate)
                else:
                    # Use quadratic utility with risk_aversion parameter
                    ef.max_quadratic_utility(risk_aversion=risk_aversion)

                # Get the cleaned weights
                weights_dict = ef.clean_weights()

                # Convert to list in the same order as columns
                weights_list = [weights_dict[asset]
                                for asset in period_data.columns]

                if verbose:
                    # Print performance metrics for this month
                    expected_return, volatility, sharpe = ef.portfolio_performance(
                        risk_free_rate=risk_free_rate)
                    print(f"  Expected annual return: {expected_return:.4f}")
                    print(f"  Annual volatility: {volatility:.4f}")
                    print(f"  Sharpe ratio: {sharpe:.4f}")

            except Exception as e:
                if verbose:
                    print(f"  Optimization failed: {str(e)}")
                    print(f"  Using equal weights as fallback")

                # Fallback to equal weights
                n_assets = len(period_data.columns)
                weights_list = [1.0/n_assets] * n_assets

            # Add to plan histories
            plan_histories.append(weights_list)

        return plan_histories

    def extract_optimizer_performance(self, optimizer_histories):
        """Extract mean and variance from a single optimizer run"""
        # Convert to numpy array for easier calculation
        returns = []

        # Loop through each month's weights
        for month_weights in optimizer_histories:
            # Calculate the return for this month's allocation
            month_return = sum(weight * mu[i]
                               for i, weight in enumerate(month_weights))
            returns.append(month_return)

        return np.array(returns)

    def LLMAlgorithm(self, data, sparse, verbose=False):
        self.reset_variables()

        for month, month_data in enumerate(data):

            # TODO
            # self.summarize()

            self.conversation_history = []

            self.LLM_plan = self.LLM_Agent(
                month_data, self.current_plan, month, 0, sparse, pure=True, verbose=verbose)

            self.current_plan = self.LLM_plan
            self.LLM_Price = self.update_plan(
                self.optimization_plan, self.optimization_price, self.LLM_plan, self.LLM_price)

            self.plan_histories.append(self.LLM_plan)

        return self.plan_histories

    def summarize(self):
        sys_prompt = ""
        if "o1-mini" in llm_model:
            sys_prompt = "System Prompt: You're a trader planning the next move of investment decisions. You always maximize the profit through your stock investments."
            messages = []
        elif "o1" in llm_model:
            messages = [
                {
                    "role": "developer", "content": "You're a trader planning the next move of investment decisions. You always maximize the profit through your stock investments."
                }
            ]
        else:
            messages = [
                {
                    "role": "system", "content": "You're a trader planning the next move of investment decisions. You always maximize the profit through your stock investments."
                },
            ]
        messages.extend(self.conversation_history)
        messages.append({
            "role": "user",
            "content": sys_prompt+"Please summarize everything that happened in this conversation very succinctly, extracting the key pieces of information relevant to future stock assessments, as it will be used for another intelligent agent to overview what happened this month."
        })

        if "o1" in llm_model or "o3" in llm_model:
            response = client.chat.completions.create(
                model=llm_model,
                messages=messages
            )
        else:
            response = client.chat.completions.create(
                model=llm_model,
                messages=messages,
                temperature=0
            )

        text = response.choices[0].message.content
        self.conversation_summaries.append(text)

    def calculate_risk(self, weights):
        risks = []

        for w in weights:
            risk = 0.0
            for i in range(self.n):
                for j in range(self.n):
                    risk += self.Q[i, j] * w[i] * w[j]
            risks.append(risk)

        return risks

    # In this function, the coordination algorithm will update the preferred portfolio weights of the two agents.
    def CoordinationAlgorithm(self, data, sparse, verbose=False):
        self.reset_variables()

        for month, month_data in enumerate(data):

            prev_year = 2024 if month != 0 else 2023
            prev_month = month if month != 0 else 12

            start_date = datetime.datetime(prev_year, prev_month, 1)
            end_date = datetime.datetime(
                prev_year, prev_month, get_last_trading_day_of_month(prev_year, prev_month))

            portfolio = pd.read_csv(pft_path, parse_dates=[
                                    "Date"], index_col="Date")
            portfolio = portfolio.loc[start_date:end_date]

            self.mu = expected_returns.mean_historical_return(
                portfolio)  # capm_return / mean_historical_return
            self.Q = risk_models.CovarianceShrinkage(
                portfolio).ledoit_wolf().to_numpy()

            # self.summarize()

            self.conversation_history = []

            for i in range(self.iteration):
                # optimization agent get the preferred portfolio weights

                self.optimization_plan = self.PortfolioOptimization_Agent(
                    self.current_plan)
                # LLM agent get the preferred portfolio weights
                self.LLM_plan = self.LLM_Agent(
                    month_data, self.current_plan, month, i, sparse, verbose=verbose)

                # update the plan for the next iteration
                self.current_plan = self.update_plan(
                    self.optimization_plan, self.optimization_price, self.LLM_plan, self.LLM_price)

                # Coordinator update dual variable/prices
                self.optimization_price = self.update_activity_price(
                    self.optimization_price, self.optimization_plan, self.current_plan)
                self.LLM_price = self.update_activity_price(
                    self.LLM_price, self.LLM_plan, self.current_plan)

                self.penalty = 1 - i/self.iteration

                all_llm_opt = [f"month {month} iter {i}"] + self.current_plan + \
                    self.LLM_plan + self.optimization_plan

                self.plan_histories.append(all_llm_opt)

                columns = ['status']
                categories = ['all', 'llm', 'opt']
                for c in categories:
                    for ticker in tickers:
                        columns.append(f'{c} {ticker}')

                if self.test_convergence(self.plan_histories):

                    all_llm_opt = [f"CONVERGED month {month} iter {i}"] + self.current_plan + \
                        self.LLM_plan + self.optimization_plan
                    self.plan_histories[-1] = all_llm_opt
                    df = pd.DataFrame(self.plan_histories, columns=columns)
                    if verbose:
                        print("[DEBUG]\tConverged because {} < ")
                        print(df)  # display(df)
                    break

                else:
                    # get the largest gap
                    self.test_convergence_aux()

                df = pd.DataFrame(self.plan_histories, columns=columns)
                if verbose:
                    print("## updated weights\n```")
                    print(df)  # display(df)
                    print("\n```\n")

        return self.plan_histories

    def HyperparameterTuneOptimizationAlgorithm(self, param_grid=None, n_splits=5, objective="balanced", balance_weight=0.5, verbose=False):
        """
        Perform hyperparameter tuning for the Markowitz portfolio optimization.

        Parameters:
        -----------
        param_grid : dict
            Parameters to tune (e.g., {'risk_aversion': [1.0, 2.0, 5.0], 'target_return': [None, 0.15, 0.2]})
        n_splits : int
            Number of time series splits for cross-validation
        objective : str
            Metric to optimize - "sharpe", "return", "risk", or "balanced"
        balance_weight : float
            Weight for balancing return vs risk (used when objective="balanced")
            Higher values (>0.5) favor return, lower values (<0.5) favor lower risk
        verbose : bool
            Whether to print detailed progress information

        Returns:
        --------
        dict
            Best parameters and optimization results
        """
        import datetime
        from sklearn.model_selection import TimeSeriesSplit

        # Default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                'risk_aversion': [1.0, 2.0, 5.0, 10.0],
                'target_return': [None, 0.15, 0.20, 0.25],
                'shrinkage_method': ['ledoit_wolf', 'oracle_approximating'],
                'risk_free_rate': [0.01, 0.02, 0.03]
            }

        # Generate all parameter combinations
        param_combinations = []

        # Helper function to generate all combinations (recursive)
        def generate_combinations(keys, current_dict, index):
            if index == len(keys):
                param_combinations.append(current_dict.copy())
                return

            key = keys[index]
            for value in param_grid[key]:
                current_dict[key] = value
                generate_combinations(keys, current_dict, index + 1)

        # Start the recursive generation
        generate_combinations(list(param_grid.keys()), {}, 0)

        # Load portfolio data
        portfolio = pd.read_csv(pft_path, parse_dates=True, index_col="Date")

        # Create time series cross-validation splits
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Track best parameters and performance
        best_params = None
        best_score = -float('inf')  # Negative infinity for maximization
        all_results = []

        # For progress tracking
        total_combinations = len(param_combinations)
        if verbose:
            print(f"Evaluating {total_combinations} parameter combinations...")

        # For each parameter combination
        for i, params in enumerate(param_combinations):
            if verbose:
                print(
                    f"Testing combination {i+1}/{total_combinations}: {params}")

            # Store scores across CV splits
            cv_returns = []
            cv_risks = []
            cv_sharpes = []

            # For each train/test split
            for train_idx, test_idx in tscv.split(portfolio):
                train_data = portfolio.iloc[train_idx]
                test_data = portfolio.iloc[test_idx]

                # Calculate expected returns and covariance matrix for this data split
                mu = expected_returns.mean_historical_return(train_data)

                if params.get('shrinkage_method') == 'ledoit_wolf':
                    Q = risk_models.CovarianceShrinkage(
                        train_data).ledoit_wolf().to_numpy()
                elif params.get('shrinkage_method') == 'sample':
                    Q = risk_models.sample_cov(train_data).to_numpy()
                else:
                    Q = risk_models.CovarianceShrinkage(
                        train_data).oracle_approximating().to_numpy()

                # Create a temporary model with these parameters
                tmp_model = Model("mean_variance_optimization")
                tmp_model.setParam('OutputFlag', 0)

                # Number of stocks
                n = len(train_data.columns)

                # Add variables (portfolio weights)
                x = {}
                for i in range(n):
                    x[i] = tmp_model.addVar(
                        vtype=GRB.CONTINUOUS, name=f"x_{i}")

                # Define terms for objective
                # Risk term = sum_{i,j} Q[i,j] * x[i] * x[j]
                risk_expr = quicksum(Q[i, j] * x[i] * x[j]
                                     for i in range(n)
                                     for j in range(n))

                # Set objective based on risk_aversion parameter
                risk_aversion = params.get('risk_aversion', 1.0)
                tmp_model.setObjective(
                    0.5 * risk_aversion * risk_expr, GRB.MINIMIZE)

                # Constraints
                # 1) Sum of weights = 1
                tmp_model.addConstr(quicksum(x[i]
                                    for i in range(n)) == 1, "budget")

                # 2) Enforce minimum target return if specified
                target_return = params.get('target_return')
                if target_return is not None:
                    tmp_model.addConstr(
                        quicksum(mu[i] * x[i]
                                 for i in range(n)) >= target_return,
                        "target_return"
                    )

                # Optimize the model
                tmp_model.optimize()

                # Check if optimization was successful
                if tmp_model.status == GRB.OPTIMAL:
                    # Extract optimized weights
                    weights = [x[i].x for i in range(n)]

                    # Calculate expected return
                    expected_return = sum(
                        w * mu_val for w, mu_val in zip(weights, mu))

                    # Calculate risk (variance)
                    risk = sum(Q[i, j] * weights[i] * weights[j]
                               for i in range(n)
                               for j in range(n))

                    # Calculate Sharpe ratio
                    risk_free_rate = params.get('risk_free_rate', 0.02)
                    sharpe = (expected_return - risk_free_rate) / \
                        (risk**0.5) if risk > 0 else 0

                    # Store results
                    cv_returns.append(expected_return)
                    cv_risks.append(risk)
                    cv_sharpes.append(sharpe)
                else:
                    # If optimization failed, assign poor performance
                    if verbose:
                        print(
                            f"  Optimization failed for params: {params}, status: {tmp_model.status}")
                    cv_returns.append(0)
                    cv_risks.append(float('inf'))
                    cv_sharpes.append(0)

            # Calculate average metrics across CV splits
            avg_return = np.mean(cv_returns)
            avg_risk = np.mean(cv_risks)
            avg_sharpe = np.mean(cv_sharpes)

            # Determine score based on objective
            if objective == "sharpe":
                score = avg_sharpe
            elif objective == "return":
                score = avg_return
            elif objective == "risk":
                score = -avg_risk  # Negative because we want to minimize risk
            elif objective == "balanced":
                # Higher balance_weight means more emphasis on return
                score = balance_weight * avg_return - \
                    (1 - balance_weight) * avg_risk
            else:
                raise ValueError(f"Unknown objective: {objective}")

            # Track results
            result = {
                "params": params,
                "avg_return": avg_return,
                "avg_risk": avg_risk,
                "avg_sharpe": avg_sharpe,
                "score": score
            }
            all_results.append(result)

            # Update best parameters if better score found
            if score > best_score:
                best_score = score
                best_params = params
                if verbose:
                    print(
                        f"New best score: {best_score:.6f} with params: {best_params}")

        # Sort results by score (descending)
        all_results.sort(key=lambda x: x["score"], reverse=True)

        # Create summary of top parameters
        top_results = all_results[:5]  # Top 5 parameter sets
        if verbose:
            print("\nTop 5 Parameter Sets:")
            for i, result in enumerate(top_results):
                print(f"\nRank {i+1}:")
                for param, value in result["params"].items():
                    print(f"  {param}: {value}")
                print(f"  Return: {result['avg_return']:.6f}")
                print(f"  Risk: {result['avg_risk']:.6f}")
                print(f"  Sharpe: {result['avg_sharpe']:.6f}")
                print(f"  Score: {result['score']:.6f}")

        # Return best parameters and all results
        return {
            "best_params": best_params,
            "best_score": best_score,
            "objective": objective,
            "all_results": all_results,
            "top_results": top_results
        }

    def CoordinationAlgorithmWeighted(self, data, sparse, llm_decision_weighting=0.5, opt_decision_weighting=0.5, verbose=False):
        """
        Weighted coordination algorithm that allows for different weightings between LLM and optimization decisions.

        :param data: Input data for the algorithm
        :param sparse: Boolean flag for sparse portfolio allocation
        :param llm_decision_weighting: Weight for LLM decisions (between 0 and 1)
        :param opt_decision_weighting: Weight for optimization decisions (between 0 and 1)
        :param verbose: Flag for verbose output
        :return: Plan histories
        """
        # Validate weights sum to 1
        if abs(llm_decision_weighting + opt_decision_weighting - 1.0) > 1e-6:
            raise ValueError("Decision weightings must sum to 1.0")

        self.reset_variables()

        for month, month_data in enumerate(data):
            # Get the date range for this month
            prev_year = 2024 if month != 0 else 2023
            prev_month = month if month != 0 else 12

            start_date = datetime.datetime(prev_year, prev_month, 1)
            end_date = datetime.datetime(
                prev_year, prev_month, get_last_trading_day_of_month(prev_year, prev_month))

            # Load portfolio data
            portfolio = pd.read_csv(pft_path, parse_dates=[
                                    "Date"], index_col="Date")
            portfolio = portfolio.loc[start_date:end_date]

            # Calculate expected returns and covariance
            self.mu = expected_returns.mean_historical_return(portfolio)
            self.Q = risk_models.CovarianceShrinkage(
                portfolio).ledoit_wolf().to_numpy()

            self.conversation_history = []

            for i in range(self.iteration):
                # Optimization agent gets preferred portfolio weights
                self.optimization_plan = self.PortfolioOptimization_Agent(
                    self.current_plan)

                # LLM agent gets preferred portfolio weights
                self.LLM_plan = self.LLM_Agent(
                    month_data, self.current_plan, month, i, sparse, verbose=verbose)

                # Update the plan for the next iteration using weighted average
                weighted_plan = []
                for opt_weight, llm_weight, opt_price, llm_price in zip(
                        self.optimization_plan, self.LLM_plan,
                        self.optimization_price, self.LLM_price):

                    # Calculate weighted average of plans
                    weighted_plan_value = (opt_decision_weighting * opt_weight +
                                           llm_decision_weighting * llm_weight)

                    # Calculate weighted average of prices
                    weighted_price = (opt_decision_weighting * opt_price +
                                      llm_decision_weighting * llm_price)

                    # Apply the same formula as before but with weighted values
                    weighted_plan.append(
                        max(0, weighted_price / self.penalty + weighted_plan_value))

                self.current_plan = weighted_plan

                # Coordinator updates dual variables/prices
                self.optimization_price = self.update_activity_price(
                    self.optimization_price, self.optimization_plan, self.current_plan)
                self.LLM_price = self.update_activity_price(
                    self.LLM_price, self.LLM_plan, self.current_plan)

                # Reduce penalty as iterations progress
                self.penalty = 1 - i/self.iteration

                # Store iteration results
                all_llm_opt = [f"month {month} iter {i}"] + self.current_plan + \
                    self.LLM_plan + self.optimization_plan

                self.plan_histories.append(all_llm_opt)

                # Set up column names for dataframe
                columns = ['status']
                categories = ['all', 'llm', 'opt']
                for c in categories:
                    for ticker in tickers:
                        columns.append(f'{c} {ticker}')

                # Check for convergence
                if self.test_convergence(self.plan_histories):
                    all_llm_opt = [f"CONVERGED month {month} iter {i}"] + self.current_plan + \
                        self.LLM_plan + self.optimization_plan
                    self.plan_histories[-1] = all_llm_opt

                    if verbose:
                        df = pd.DataFrame(self.plan_histories, columns=columns)
                        print("[DEBUG]\tConverged")
                        print(df)
                    break
                else:
                    # Get the largest gap
                    self.test_convergence_aux()

                if verbose:
                    df = pd.DataFrame(self.plan_histories, columns=columns)
                    print("## updated weights\n```")
                    print(df)
                    print("\n```\n")

        return self.plan_histories


# %%
CoordFW = CoordinationFramework(
    mu, S, 2.7, penalty=1, iteration=10, verbose=False)
# appl15 0.93703
# morgan15 0.449055
# chev15 0.688154
# nvda60 3.2194595336914062 for opt, 2.8 for coord

# %%


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

# %%


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

# %% [markdown]
# #### Run the optimizer only, as baseline


# %%
if rerun_opt:
    opt_histories = CoordFW.OptAlgorithm(data_loaded)
    with open(weights_opt_path, "w") as f:
        json.dump(opt_histories, f, indent=4)

# -- OR --
load_opt = False
if load_opt:
    with open(weights_opt_path, "r") as f:
        opt_histories = json.load(f)

# %%
if load_opt:
    portfolio_value, portfolio_history, monthly_pnl = backtest_yyy(
        opt_histories)

# %%
if load_opt:
    # Get the optimizer returns
    optimizer_returns = CoordFW.extract_optimizer_performance(opt_histories)

    # Create the bar plot
    plt.figure(figsize=(5, 6))

    # Calculate mean and standard deviation
    mean_return = portfolio_value / 10000
    std_return = np.std(optimizer_returns)

    # Create a single skinnier bar for the mean
    plt.bar(['Average Return'], [mean_return], color='#5f0f40',
            yerr=std_return, capsize=10, width=0.3)

    # Add labels and title
    plt.ylabel('Return Multiple')
    plt.title('Average Optimizer Return with Variance')
    # Set y-limit to show error bar clearly
    plt.ylim(0, mean_return + 3*std_return)
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 2)
    # Add text annotation in the upper right corner of the plot
    plt.text(0.95, 0.95, f'Mean: {mean_return:.3f}\nStd Dev: {std_return:.3f}',
             transform=plt.gca().transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Save or show plot
    plt.tight_layout()
    plt.savefig("YYY/average_optimizer_return_with_variance.png",
                dpi=300, bbox_inches='tight')

# %%
# Generate optimized weights for each month
rerun_optimized_opt = False
appendage = "2025-03-28-gpt-4o-mini"
if rerun_optimized_opt:
    # HMMMMGE
    # Run hyperparameter tuning
    for idxd in range(10):

        results = CoordFW.HyperparameterTuneOptimizationAlgorithm(
            param_grid={
                'risk_aversion': [1.0],  # risk aversion: do not tune, set as 1
                'target_return': [2.125, 2.2625, 2.275, 2.2875],
                # shrinkage_method: use ledoit_wolf() is fine
                'shrinkage_method': ['ledoit_wolf'],
                # risk free rate: set as 0.0438 (10 Year Treasury Rate (I:10YTCMR) 4.38% for Mar 27 2025)
                'risk_free_rate': [0.0438]
            },
            objective="balanced",
            balance_weight=0.6,  # Slightly favor return over risk
            verbose=False
        )

        # Get the best parameters
        best_params = results["best_params"]
        print("Best parameters:", best_params)

        optimized_weights = CoordFW.OptAlgorithmParams(
            portfolio_path=pft_path,
            best_params=best_params,
            verbose=False
        )

        weights_opt_path = f"assets/weights_opt_{appendage}_{idxd}.json"
        with open(weights_opt_path, 'w') as f:
            json.dump(optimized_weights, f, indent=4)

# %%
if rerun_optimized_opt:
    lo = 2.2
    hi = 2.4
    new_mid = None
    while (hi-lo) > 0.0001:
        mid = (lo+hi)/2
        new_diff = (mid-lo)/2

        results = CoordFW.HyperparameterTuneOptimizationAlgorithm(
            param_grid={
                'risk_aversion': [1.0],  # risk aversion: do not tune, set as 1
                'target_return': [lo, mid, hi],
                # shrinkage_method: use ledoit_wolf() is fine
                'shrinkage_method': ['ledoit_wolf'],
                # risk free rate: set as 0.0438 (10 Year Treasury Rate (I:10YTCMR) 4.38% for Mar 27 2025)
                'risk_free_rate': [0.0438]
            },
            objective="balanced",
            balance_weight=0.6,  # Slightly favor return over risk
            verbose=False
        )

        # Get the best parameters
        best_params = results["best_params"]
        print("Best parameters:", best_params)

        new_mid = best_params['target_return']

        # Center the new search range around new_mid
        # Original code had sign errors that would make lo > hi
        lo = new_mid - new_diff
        hi = new_mid + new_diff

    print(new_mid)

# %%
if rerun_optimized_opt:
    optimized_weights = CoordFW.OptAlgorithmParams(
        portfolio_path=pft_path,
        best_params=best_params,
        verbose=False
    )

    weights_opt_path = f"assets/weights_opt_optimized_{appendage}_{idxd}.json"
    with open(weights_opt_path, 'w') as f:
        json.dump(optimized_weights, f, indent=4)

# %%

# %%
if rerun_optimized_opt:
    paths = glob.glob(os.path.join(os.getcwd(), "assets",
                      f"*weights_opt_optimized_{appendage}*"))
    opt_returns = []
    for path in paths:
        with open(path, 'r') as f:
            weights = json.loads(f.read())

        portfolio_value, portfolio_history, monthly_pnl = backtest_yyy(weights)
        print(f"Final portfolio value: ${portfolio_value:.2f}")
        print(f"Return multiple: {portfolio_value/10000:.4f}x")
        opt_returns.append(portfolio_value)

# %%
if rerun_optimized_opt:
    # Create the bar plot
    plt.figure(figsize=(5, 6))

    # Calculate mean and standard deviation
    mean_return = np.mean(optimizer_returns) / 10000
    std_return = np.std(optimizer_returns)

    # Create a single skinnier bar for the mean
    plt.bar(['Average Return'], [mean_return], color='#5f0f40',
            yerr=std_return, capsize=10, width=0.3)

    # Add labels and title
    plt.ylabel('Return Multiple')
    plt.title('Average Optimizer Return with Variance')
    # Set y-limit to show error bar clearly
    plt.ylim(0, mean_return + 3*std_return)
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 2)
    # Add text annotation in the upper right corner of the plot
    plt.text(0.95, 0.95, f'Mean: {mean_return:.3f}\nStd Dev: {std_return:.3f}',
             transform=plt.gca().transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Save or show plot
    plt.tight_layout()
    plt.savefig("YYY/average_optimizer_return_with_variance.png",
                dpi=300, bbox_inches='tight')

# %%


# %% [markdown]
# #### Run the LLM only, as baseline

# %%
if rerun_llm:
    llm_histories = CoordFW.LLMAlgorithm(data_loaded, False, True)
    with open(weights_llm_path, "w") as f:
        json.dump(llm_histories, f, indent=4)

# -- OR --

load_llm = False
if load_llm:
    with open(weights_llm_path, "r") as f:
        llm_histories = json.load(f)

# %%
if rerun_llm_sparse:
    llm_histories_sparse = CoordFW.LLMAlgorithm(data_loaded, True, True)
    with open(weights_llm_sparse_path, "w") as f:
        json.dump(llm_histories_sparse, f, indent=4)

# -- OR --
load_llm_sparse = False
if load_llm_sparse:
    with open(weights_llm_sparse_path, "r") as f:
        llm_histories_sparse = json.load(f)

# %% [markdown]
# #### Run the coordinator algorithm

# %%
# RUN IT WEIGHTED

run_it_sparse = True
run_it_weighted = False

for idxd in range(99, 110):
    weights_llm25_opt75_coord_path = f"assets/weights_coord_llm25_opt75_{appendage}_{iteration}{idxd}.json"
    weights_llm75_opt25_coord_path = f"assets/weights_coord_llm75_opt25_{appendage}_{iteration}{idxd}.json"

    weights_llmsparse25_opt75_coord_path = f"assets/weights_coord_llmsparse25_opt75_{appendage}_{iteration}{idxd}.json"
    weights_llmsparse75_opt25_coord_path = f"assets/weights_coord_llmsparse75_opt25_{appendage}_{iteration}{idxd}.json"

    if run_it_weighted:
        coord_llm25_opt75_histories = CoordFW.CoordinationAlgorithmWeighted(
            data_loaded, False, 0.25, 0.75, True)
        with open(weights_llm25_opt75_coord_path, "w") as f:
            json.dump(coord_llm25_opt75_histories, f, indent=4)

        coord_llm75_opt25_histories = CoordFW.CoordinationAlgorithmWeighted(
            data_loaded, False, 0.75, 0.25, True)
        with open(weights_llm75_opt25_coord_path, "w") as f:
            json.dump(coord_llm75_opt25_histories, f, indent=4)

    if run_it_sparse:
        coord_llmsparse25_opt75_histories = CoordFW.CoordinationAlgorithmWeighted(
            data_loaded, True, 0.25, 0.75, True)
        with open(weights_llmsparse25_opt75_coord_path, "w") as f:
            json.dump(coord_llmsparse25_opt75_histories, f, indent=4)

        coord_llmsparse75_opt25_histories = CoordFW.CoordinationAlgorithmWeighted(
            data_loaded, True, 0.75, 0.25, True)
        with open(weights_llmsparse75_opt25_coord_path, "w") as f:
            json.dump(coord_llmsparse75_opt25_histories, f, indent=4)


# %%

# %%
# get the status column
# coord_llm25_opt75_statuses = [x[0] for x in coord_llm25_opt75_histories]
# coord_llm25_opt75_weights = [x[1:] for x in coord_llm25_opt75_histories]

# coord_llm75_opt25_statuses = [x[0] for x in coord_llm75_opt25_histories]
# coord_llm75_opt25_weights = [x[1:] for x in coord_llm75_opt25_histories]

# %%
weights_llmsparse25_opt75_coord_files = glob.glob(os.path.join(
    os.getcwd(), "assets", "*weights_coord_llmsparse25_opt75*"))
coord_llmsparse25_opt75_returns = []

for weights_llmsparse25_opt75_coord_path in weights_llmsparse25_opt75_coord_files:
    with open(weights_llmsparse25_opt75_coord_path, 'r') as f:
        coord_llmsparse25_opt75_histories = json.loads(f.read())

    coord_llmsparse25_opt75_statuses = [x[0]
                                        for x in coord_llmsparse25_opt75_histories]
    coord_llmsparse25_opt75_weights = [x[1:]
                                       for x in coord_llmsparse25_opt75_histories]

    portfolio_value, portfolio_history, monthly_pnl = backtest_yyy(
        coord_llmsparse25_opt75_weights, coord_llmsparse25_opt75_statuses)
    coord_llmsparse25_opt75_returns.append(portfolio_value/10000)

# %%
weights_llmsparse75_opt25_coord_files = glob.glob(os.path.join(
    os.getcwd(), "assets", "*weights_coord_llmsparse75_opt25*"))
coord_llmsparse75_opt25_returns = []

for weights_llmsparse75_opt25_coord_path in weights_llmsparse75_opt25_coord_files:
    with open(weights_llmsparse75_opt25_coord_path, 'r') as f:
        coord_llmsparse75_opt25_histories = json.loads(f.read())

    coord_llmsparse75_opt25_statuses = [x[0]
                                        for x in coord_llmsparse75_opt25_histories]
    coord_llmsparse75_opt25_weights = [x[1:]
                                       for x in coord_llmsparse75_opt25_histories]

    portfolio_value, portfolio_history, monthly_pnl = backtest_yyy(
        coord_llmsparse75_opt25_weights, coord_llmsparse75_opt25_statuses)
    coord_llmsparse75_opt25_returns.append(portfolio_value/10000)

# %%
# fetch all weights_coord_llm25_opt75 files


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

# %%
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

# %%
# coord_llm25_opt75_statuses, coord_llm25_opt75_weights
# coord_llm75_opt25_statuses, coord_llm75_opt25_weights

# Calculate mean and standard deviation
mean_return = np.mean(coord_llm25_opt75_returns)
std_return = np.std(coord_llm25_opt75_returns)

# Create figure and axis
plt.figure(figsize=(5, 6))

# Create a single skinnier bar for the mean
plt.bar(['Average Return'], [mean_return], color='#0f4c5c',
        yerr=std_return, capsize=10, width=0.3)

# Add labels and title
plt.ylabel('Return Multiple')
plt.title('coord_llm25_opt75_returns with Variance')
# Set y-limit to show error bar clearly
plt.ylim(0, mean_return + 3*std_return)
plt.ylim(0, 2)
plt.grid(axis='y', alpha=0.3)

# Add text annotation in the upper right corner of the plot
plt.text(0.95, 0.95, f'Mean: {mean_return:.3f}\nStd Dev: {std_return:.3f}',
         transform=plt.gca().transAxes, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Show plot
plt.tight_layout()
plt.savefig('YYY/average_coord_llm25_opt75_return_with_variance.png',
            dpi=300, bbox_inches='tight')
plt.show()

# %%
# coord_llm25_opt75_statuses, coord_llm25_opt75_weights
# coord_llm75_opt25_statuses, coord_llm75_opt25_weights

# print(f'{statuses=}')

# Calculate mean and standard deviation
mean_return = np.mean(coord_llm75_opt25_returns)
std_return = np.std(coord_llm75_opt25_returns)

# Create figure and axis
plt.figure(figsize=(5, 6))

# Create a single skinnier bar for the mean
plt.bar(['Average Return'], [mean_return], color='#fb8b24',
        yerr=std_return, capsize=10, width=0.3)

# Add labels and title
plt.ylabel('Return Multiple')
plt.title('coord_llm75_opt25_returns with Variance')
# Set y-limit to show error bar clearly
plt.ylim(0, mean_return + 3*std_return)
plt.ylim(0, 2)
plt.grid(axis='y', alpha=0.3)

# Add text annotation in the upper right corner of the plot
plt.text(0.95, 0.95, f'Mean: {mean_return:.3f}\nStd Dev: {std_return:.3f}',
         transform=plt.gca().transAxes, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('YYY/average_coord_llm75_opt25_return_with_variance.png',
            dpi=300, bbox_inches='tight')
# Show plot
plt.show()


# %%

weights_llm25_opt75_coord_path = f"assets/weights_coord_llm25_opt75_{appendage}_{iteration}.json"
weights_llm75_opt25_coord_path = f"assets/weights_coord_llm75_opt25_{appendage}_{iteration}.json"

with open(weights_llm25_opt75_coord_path, 'r') as f:
    coord_llm25_opt75_histories = json.loads(f.read())

with open(weights_llm75_opt25_coord_path, 'r') as f:
    coord_llm75_opt25_histories = json.loads(f.read())

# %%
if rerun_coord:
    coord_histories = CoordFW.CoordinationAlgorithm(data_loaded, False, True)
    with open(weights_coord_path, "w") as f:
        json.dump(coord_histories, f, indent=4)

# -- OR --

with open(weights_coord_path, "r") as f:
    coord_histories = json.load(f)
weights_coord = [h[1:1+len(tickers)] for h in coord_histories]

# %%
if rerun_coord_sparse:
    coord_histories_sparse = CoordFW.CoordinationAlgorithm(
        data_loaded, True, True)
    with open(weights_coord_sparse_path, "w") as f:
        json.dump(coord_histories_sparse, f, indent=4)

# -- OR --

with open(weights_coord_sparse_path, "r") as f:
    coord_histories_sparse = json.load(f)
weights_coord_sparse = [h[1:1+len(tickers)] for h in coord_histories_sparse]


# %% [markdown]
# #### Display results

# %%
columns = ['status']
categories = ['all', 'llm', 'opt']
for c in categories:
    for ticker in tickers:
        columns.append(f'{c} {ticker}')

# %%
df = pd.DataFrame(coord_histories, columns=columns)

df

# %%
with open(status_path, "w") as f:
    json.dump(df.status.tolist(), f, indent=4)

# %%
df_sparse = pd.DataFrame(coord_histories_sparse, columns=columns)

df_sparse

# %%
with open(status_sparse_path, "w") as f:
    json.dump(df_sparse.status.tolist(), f, indent=4)

# %%
pure_llm_history = [[] for _ in range(len(tickers))]
j = -1
pattern = r"month (\d+)"
prev = -1

for iter in df['status']:
    cur = int(re.search(pattern, iter).group(1))
    if cur != prev:
        j += 1
        prev = cur
    for i in range(len(tickers)):
        pure_llm_history[i].append(llm_histories[j][i])

# pure_llm_history

# %%
pure_llm_history_sparse = [[] for _ in range(len(tickers))]
j = -1
pattern = r"month (\d+)"
prev = -1

for iter in df['status']:
    cur = int(re.search(pattern, iter).group(1))
    if cur != prev:
        j += 1
        prev = cur
    for i in range(len(tickers)):
        pure_llm_history_sparse[i].append(llm_histories_sparse[j][i])

# pure_llm_history_sparse

# %%
pure_opt_history = [[] for _ in range(len(tickers))]
j = -1
pattern = r"month (\d+)"
prev = -1

for iter in df['status']:
    cur = int(re.search(pattern, iter).group(1))
    if cur != prev:
        j += 1
        prev = cur
    for i in range(len(tickers)):
        pure_opt_history[i].append(opt_histories[j][i])

# pure_opt_history

# %%


def extract_month(x):
    m = re.search(r"month\s+(\d+)", x)
    return int(m.group(1)) if m else None


# %%
all_columns = [col for col in df.columns if "all" in col]

df_filtered = df.copy(deep=True)
df_filtered['month_num'] = df_filtered['status'].apply(extract_month)
df_filtered['next_month_num'] = df_filtered['month_num'].shift(-1)
mask = (df_filtered['month_num'] != df_filtered['next_month_num']
        ) | df_filtered['next_month_num'].isna()
df_filtered = df_filtered[mask]

df_filtered.set_index('status', inplace=True)

# %% [markdown]
# #### Graph all results

# %%

rows = 10
columns = 6

fig, axes = plt.subplots(rows, columns, figsize=(45, 30), sharey=True)
fig.suptitle("Portfolio Weights Over Time\n", fontsize=65)
plt.subplots_adjust(top=0.6)

# Flatten the 2D array of axes to make it easier to iterate
axes = axes.flatten()

# Define regions and colors
blue_region = (round(108/255, 2), round(143/255, 2), round(191/255, 2), 0.3)
green_region = (round(201/255, 2), round(230/255, 2), round(219/255, 2), 0.3)

days = []
prev = 0

for i, log in enumerate(df['status']):
    log = log.split()
    j = log.index("month")
    cur = log[j+1]
    if cur != prev:
        days.append(i-1)
        prev = cur

days.append(i+1)

regions = []

if False:
    for i in range(len(days)-1):
        region_color = blue_region if i % 2 == 0 else green_region
        regions.append((days[i], days[i+1], region_color))

    # Generate x-tick labels, hiding those without "CONVERGED"
    xtick_labels = []
    for status in df['status']:
        if "CONVERGED" in status:
            xtick_labels.append(status)
        else:
            xtick_labels.append("")  # Blank label for non-CONVERGED iterations

    for i in range(rows * columns):

        isEmpty = False

        try:
            ticker = tickers[i]
        except:
            ticker = None
            isEmpty = True

        # Fill regions
        for start, end, color in regions:
            axes[i].axvspan(start, end, color=color, alpha=0.3)

        if ticker:
            # Plot lines
            if i == 0:
                axes[i].plot(
                    df['status'], df[f'all {ticker}'], label='Coordinator', linewidth=2.5)
                axes[i].plot(
                    df['status'], df[f'llm {ticker}'], label='LLM in Coordinator', linewidth=2.5)
                axes[i].plot(
                    df['status'], df[f'opt {ticker}'], label='Opt in Coordinator', linewidth=2.5)
                axes[i].plot(df['status'], pure_llm_history[i],
                             label='Pure LLM', linewidth=2.5, alpha=0.5)
                axes[i].plot(df['status'], pure_opt_history[i],
                             label='Pure Opt', linewidth=2.5, alpha=0.5)
            else:
                axes[i].plot(df['status'], df[f'all {ticker}'], linewidth=2.5)
                axes[i].plot(df['status'], df[f'llm {ticker}'], linewidth=2.5)
                axes[i].plot(df['status'], df[f'opt {ticker}'], linewidth=2.5)
                axes[i].plot(df['status'], pure_llm_history[i],
                             linewidth=2.5, alpha=0.5)
                axes[i].plot(df['status'], pure_opt_history[i],
                             linewidth=2.5, alpha=0.5)

            # Title and tick parameters
            axes[i].set_title(f'{ticker}', fontsize=40)

        axes[i].tick_params(axis='x', rotation=90, labelsize=15)
        axes[i].tick_params(axis='y', labelsize=15)

        # Set x-tick labels with filtered labels
        axes[i].set_xticks(range(len(df['status'])))
        axes[i].set_xticklabels(xtick_labels, fontsize=15, ha='right')

        # X-axis limit
        axes[i].set_xlim(0, len(df['status']) - 1)

        # Y-label on first column
        if i in [i for i in range(0, rows*columns, columns)]:
            cat = '\n'.join(stock_categories[i//columns].split())
            axes[i].set_ylabel(f"{cat}\n\nWeights", fontsize=25)

        # X-label only on last row
        if i in [i for i in range(rows*columns-columns, rows*columns)]:
            axes[i].set_xlabel("Iteration", fontsize=25)
        else:
            # Hide x-tick labels for the first two rows
            axes[i].tick_params(axis='x', labelbottom=False)

    # Add common legend (adjust location as you like)
    # fig.legend(loc=(0.00, 0.95), ncol=2, fontsize=20)
    fig.legend(loc=(0.00, 0.965), ncol=5, fontsize=20)

    plt.tight_layout()

    plt.savefig(grid_image_path, dpi=300)  # , bbox_inches='tight'
    plt.show()

# %% [markdown]
# #### Graph individual results

# %%
# create directory
os.makedirs(directory_path, exist_ok=True)

if graph_indiv:
    for i, ticker in enumerate(tickers):
        fig, ax = plt.subplots(figsize=(12, 8))  # Adjust size as desired

        # Fill background regions
        for start, end, color in regions:
            ax.axvspan(start, end, color=color, alpha=0.3)

        # Plot lines for this ticker
        ax.plot(df['status'], df[f'all {ticker}'],
                label='Coordinator', linewidth=2.5)
        ax.plot(df['status'], df[f'llm {ticker}'],
                label='LLM in Coordinator', linewidth=2.5)
        ax.plot(df['status'], df[f'opt {ticker}'],
                label='Opt in Coordinator', linewidth=2.5)
        # ax.plot(df['status'], pure_llm_history[i], label='Pure LLM', linewidth=2.5, alpha=0.5)
        # ax.plot(df['status'], pure_opt_history[i], label='Pure Opt', linewidth=2.5, alpha=0.5)

        # Title and style
        ax.set_title(f"{ticker}", fontsize=18)
        ax.tick_params(axis='x', rotation=90, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

        # X-tick labels (with blanks for non-CONVERGED, as in your original logic)
        ax.set_xticks(range(len(df['status'])))
        ax.set_xticklabels(xtick_labels, fontsize=10, ha='right')

        # Set x-limits
        ax.set_xlim(0, len(df['status']) - 1)

        # Optionally add a legend on each chart (or remove if you prefer no legend)
        ax.legend(fontsize=10, loc='upper left')

        # Save to file; use ticker name in the filename
        plt.tight_layout()
        # or any naming scheme you like
        plt.savefig(f"{directory_path}/{ticker}.png", dpi=450)
        plt.close(fig)  # Close the figure to free memory

# %% [markdown]
# #### Sparse

# %%
# my status column is like this: "month 0 iter 0", "month 1 iter 1", "CONVERGED month 1 iter 2", "month 1 iter 0", ...
# change this to be if the month _ value of two succesive months are different, then you take the previous row. im just trying to make this system more robust because it has failed in the past to properly extract these rows


def extract_month(x):
    m = re.search(r"month\s+(\d+)", x)
    return int(m.group(1)) if m else None


df_filtered_sparse = df_sparse.copy(deep=True)

df_filtered_sparse['month_num'] = df_filtered_sparse['status'].apply(
    extract_month)
df_filtered_sparse['next_month_num'] = df_filtered_sparse['month_num'].shift(
    -1)
mask = (df_filtered_sparse['month_num'] != df_filtered_sparse['next_month_num']
        ) | df_filtered_sparse['next_month_num'].isna()
df_filtered_sparse = df_filtered_sparse[mask]
df_filtered_sparse.set_index('status', inplace=True)

# %%
rows = 10
columns = 6

fig, axes = plt.subplots(rows, columns, figsize=(45, 30), sharey=True)
fig.suptitle("(Sparse) Portfolio Weights Over Time\n", fontsize=65)
plt.subplots_adjust(top=0.6)

# Flatten the 2D array of axes to make it easier to iterate
axes = axes.flatten()

# Define regions and colors
days = []
prev = 0

for i, log in enumerate(df_sparse['status']):
    log = log.split()
    j = log.index("month")
    cur = log[j+1]
    if cur != prev:
        days.append(i-1)
        prev = cur

days.append(i+1)

regions = []
for i in range(len(days)-1):
    region_color = blue_region if i % 2 == 0 else green_region
    regions.append((days[i], days[i+1], region_color))

# Generate x-tick labels, hiding those without "CONVERGED"
xtick_labels = []
if False:
    for status in df_sparse['status']:
        if "CONVERGED" in status:
            xtick_labels.append(status)
        else:
            xtick_labels.append("")  # Blank label for non-CONVERGED iterations

    for i in range(rows * columns):

        try:
            ticker = tickers[i]
        except:
            ticker = None

        # Fill regions
        for start, end, color in regions:
            axes[i].axvspan(start, end, color=color, alpha=0.3)

        if ticker:
            # Plot lines
            if i == 0:
                axes[i].plot(
                    df_sparse['status'], df_sparse[f'all {ticker}'], label='Coordinator', linewidth=2.5)
                axes[i].plot(
                    df_sparse['status'], df_sparse[f'llm {ticker}'], label='LLM in Coordinator', linewidth=2.5)
                axes[i].plot(
                    df_sparse['status'], df_sparse[f'opt {ticker}'], label='Opt in Coordinator', linewidth=2.5)
            else:
                axes[i].plot(df_sparse['status'],
                             df_sparse[f'all {ticker}'], linewidth=2.5)
                axes[i].plot(df_sparse['status'],
                             df_sparse[f'llm {ticker}'], linewidth=2.5)
                axes[i].plot(df_sparse['status'],
                             df_sparse[f'opt {ticker}'], linewidth=2.5)

            # Title and tick parameters
            axes[i].set_title(f'{ticker}', fontsize=40)

        axes[i].tick_params(axis='x', rotation=90, labelsize=15)
        axes[i].tick_params(axis='y', labelsize=15)

        # Set x-tick labels with filtered labels
        axes[i].set_xticks(range(len(df_sparse['status'])))
        axes[i].set_xticklabels(xtick_labels, fontsize=15, ha='right')

        # X-axis limit
        axes[i].set_xlim(0, len(df_sparse['status']) - 1)

        # Y-label on first column
        if i in [i for i in range(0, rows*columns, columns)]:
            cat = '\n'.join(stock_categories[i//columns].split())
            axes[i].set_ylabel(f"{cat}\n\nWeights", fontsize=25)

        # X-label only on last row
        if i in [i for i in range(rows*columns-columns, rows*columns)]:
            axes[i].set_xlabel("Iteration", fontsize=25)
        else:
            # Hide x-tick labels for the first two rows
            axes[i].tick_params(axis='x', labelbottom=False)

    # Add common legend (adjust location as you like)
    # fig.legend(loc=(0.00, 0.95), ncol=2, fontsize=20)
    fig.legend(loc=(0.00, 0.965), ncol=5, fontsize=20)

    plt.tight_layout()

    plt.savefig(grid_image_sparse_path, dpi=300)  # , bbox_inches='tight'
    plt.show()

# %%
if True and False:
    # create directory
    os.makedirs(directory_path_sparse, exist_ok=True)

    for i, ticker in enumerate(tickers):
        fig, ax = plt.subplots(figsize=(12, 8))  # Adjust size as desired

        # Fill background regions
        for start, end, color in regions:
            ax.axvspan(start, end, color=color, alpha=0.3)

        # Plot lines for this ticker
        ax.plot(
            df_sparse['status'], df_sparse[f'all {ticker}'], label='Coordinator', linewidth=2.5)
        ax.plot(
            df_sparse['status'], df_sparse[f'llm {ticker}'], label='LLM in Coordinator', linewidth=2.5)
        ax.plot(
            df_sparse['status'], df_sparse[f'opt {ticker}'], label='Opt in Coordinator', linewidth=2.5)
        # ax.plot(df_sparse['status'], pure_llm_history[i], label='Pure LLM', linewidth=2.5, alpha=0.5)
        # ax.plot(df_sparse['status'], pure_opt_history[i], label='Pure Opt', linewidth=2.5, alpha=0.5)

        # Title and style
        ax.set_title(f"{ticker}", fontsize=18)
        ax.tick_params(axis='x', rotation=90, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

        # X-tick labels (with blanks for non-CONVERGED, as in your original logic)
        ax.set_xticks(range(len(df_sparse['status'])))
        ax.set_xticklabels(xtick_labels, fontsize=10, ha='right')

        # Set x-limits
        ax.set_xlim(0, len(df_sparse['status']) - 1)

        # Optionally add a legend on each chart (or remove if you prefer no legend)
        ax.legend(fontsize=10, loc='upper left')

        # Save to file; use ticker name in the filename
        plt.tight_layout()
        plt.savefig(f"{directory_path_sparse}/{ticker}.png",
                    dpi=450)  # or any naming scheme you like
        plt.close(fig)  # Close the figure to free memory

# %% [markdown]
# #### Backtesting
# Seeing how the strategies perform based on historical data

# %%
days = []
prev = 0

for i, log in enumerate(df['status']):
    log = log.split()
    j = log.index("month")
    cur = log[j+1]
    if cur != prev:
        days.append(i-1)
        prev = cur

days.append(i+1)

# %%
# get the beginning price for each month
df_init = df[df['status'].str.contains('iter 0')].reset_index(drop=True)
df_end = df.iloc[[d - 1 for d in days[1:]]].reset_index(drop=True)

df_init_sparse = df_sparse[df_sparse['status'].str.contains(
    'iter 0')].reset_index(drop=True)
df_end_sparse = df_sparse.iloc[[
    d - 1 for d in days[1:]]].reset_index(drop=True)

all_weights = ["all " + ticker for ticker in tickers]

# %%
# avg of pure_opt and pure_llm plan
avg_opt_llm_histories = (np.array(opt_histories) + np.array(llm_histories)) / 2
row_sums = avg_opt_llm_histories.sum(axis=1, keepdims=True)  # Sum of each row
normalized_array = avg_opt_llm_histories / \
    row_sums  # Divide each element by its row sum
avg_opt_llm_histories = normalized_array.tolist()

avg_opt_llm_sparse_histories = (
    np.array(opt_histories) + np.array(llm_histories_sparse)) / 2
row_sums = avg_opt_llm_sparse_histories.sum(axis=1, keepdims=True)
normalized_array = avg_opt_llm_sparse_histories / row_sums
avg_opt_llm_sparse_histories = normalized_array.tolist()

# %%


def backtest(df, columns=None, weights_=None):
    i = 0
    initial_capital = 10000
    portfolio_value = initial_capital

    portfolio_history = [portfolio_value]
    # Create a DataFrame to track monthly PnL for each ticker
    monthly_pnl = pd.DataFrame(0.0, index=range(12), columns=tickers)

    while i < 11:
        # ---- 1) Get the weights for this month (end of month i) ----
        if columns:
            weights = df.loc[i, columns].tolist()
        else:
            weights = weights_[i]

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

    print("Final Portfolio Value:", portfolio_value)
    # Return both the total portfolio value history and the per-ticker monthly PnL
    return portfolio_history, monthly_pnl


# For example, the coordinated strategy:
portfolio_history_coordinated, pnl_coordinated = backtest(
    df_end, columns=all_weights)
portfolio_history_coordinated_sparse, pnl_coordinated_sparse = backtest(
    df_end_sparse, columns=all_weights)

# Or the pure OPT strategy:
portfolio_history_opt, pnl_opt = backtest(df_init, weights_=opt_histories)

# Or the pure LLM strategy:
portfolio_history_llm, pnl_llm = backtest(df_init, weights_=llm_histories)
portfolio_history_llm_sparse, pnl_llm_sparse = backtest(
    df_init_sparse, weights_=llm_histories_sparse)

# Averaged LLM + OPT
portfolio_history_avg, pnl_avg = backtest(
    df_init, weights_=avg_opt_llm_histories)
portfolio_history_avg_sparse, pnl_avg_sparse = backtest(
    df_init_sparse, weights_=avg_opt_llm_sparse_histories)

# %% [markdown]
# #### Plot the backtesting

# %%

# Create labels for months 0 through 11
months = [f"{i}" for i in range(12)]

plt.figure(figsize=(10, 6))

num_plots = 7
colors = cm.get_cmap('tab10', num_plots).colors

# Plot as a line chart with markers
plt.plot(months, portfolio_history_coordinated, linestyle='-',
         linewidth=1, label='LLM+OPT', color=colors[0])
plt.plot(months, portfolio_history_coordinated_sparse, linestyle='-',
         linewidth=1, label='LLM_sparse+OPT', color=colors[1])
plt.plot(months, portfolio_history_opt, linestyle='-',
         linewidth=1, label='OPT', color=colors[2])
plt.plot(months, portfolio_history_llm, linestyle='-',
         linewidth=1, label='LLM', color=colors[3])
plt.plot(months, portfolio_history_llm_sparse, linestyle='-',
         linewidth=1, label='LLM_sparse', color=colors[4])
plt.plot(months, portfolio_history_avg, linestyle='-',
         linewidth=1, label='AVG', color=colors[5])
plt.plot(months, portfolio_history_avg_sparse, linestyle='-',
         linewidth=1, label='AVG_sparse', color=colors[6])

# Add a title, axis labels, and grid
plt.title("Portfolio Value Over Time", fontsize=25)
plt.xlabel("Month", fontsize=16)
plt.ylabel("Portfolio Value ($)", fontsize=16)

# Improve grid styling
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Add a legend with larger font size
plt.legend(fontsize=14, loc='upper left')

# Adjust layout for better spacing
plt.tight_layout()

# Display the plot
plt.savefig(pft_value_over_time_path, dpi=500, bbox_inches='tight')
plt.show()

# %% [markdown]
# #### Risk

# %%
df_filtered

# %%
df_filtered_all = df_filtered.loc[:, df_filtered.columns.str.contains(
    r'\ball\b', case=False)]
coord_end_weights = df_filtered_all.values.tolist()

df_filtered_all_sparse = df_filtered_sparse.loc[:, df_filtered_sparse.columns.str.contains(
    r'\ball\b', case=False)]
coord_end_weights_sparse = df_filtered_all_sparse.values.tolist()

# %%
df_filtered_all

# %%
coord_risks = CoordFW.calculate_risk(coord_end_weights)
coord_sparse_risks = CoordFW.calculate_risk(coord_end_weights_sparse)
llm_risks = CoordFW.calculate_risk(llm_histories)
llm_sparse_risks = CoordFW.calculate_risk(llm_histories_sparse)
opt_risks = CoordFW.calculate_risk(opt_histories)
avg_risks = CoordFW.calculate_risk(avg_opt_llm_histories)
avg_sparse_risks = CoordFW.calculate_risk(avg_opt_llm_sparse_histories)

# %%
# Create labels for months 0 through 11
months = [f"{i}" for i in range(12)]

plt.figure(figsize=(10, 6))
num_plots = 7
colors = cm.get_cmap('tab10', num_plots).colors
# Plot as a line chart with markers
plt.plot(months, coord_risks, linestyle='-',
         linewidth=1, label='LLM+OPT', color=colors[0])
plt.plot(months, coord_sparse_risks, linestyle='-',
         linewidth=1, label='LLM_sparse+OPT', color=colors[1])
plt.plot(months, opt_risks, linestyle='-',
         linewidth=1, label='OPT', color=colors[2])
plt.plot(months, llm_risks, linestyle='-',
         linewidth=1, label='LLM', color=colors[3])
plt.plot(months, llm_sparse_risks, linestyle='-',
         linewidth=1, label='LLM_sparse', color=colors[4])
plt.plot(months, avg_risks, linestyle='-',
         linewidth=1, label='AVG', color=colors[5])
plt.plot(months, avg_sparse_risks, linestyle='-',
         linewidth=1, label='AVG_sparse', color=colors[6])

# Add a title, axis labels, and grid
plt.title("Risk Over Time", fontsize=25)
plt.xlabel("Month", fontsize=16)
plt.ylabel("Risk", fontsize=16)

# Improve grid styling
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Add a legend with larger font size
plt.legend(fontsize=14, loc='upper left')

# Adjust layout for better spacing
plt.tight_layout()

# Display the plot
plt.savefig(risk_path, dpi=500, bbox_inches='tight')
plt.show()

# %% [markdown]
# #### Heat Map

# %%
# only get the coordination results
df_filtered_all = df_filtered.loc[:, df_filtered.columns.str.contains(
    r'\ball\b', case=False)]
df_filtered_all.columns = df_filtered_all.columns.str.replace(
    r'^all ', '', regex=True)

# get average
mean_values = df_filtered_all.mean()
df_filtered_ticker_name = pd.DataFrame([mean_values])
df_filtered_ticker_name.reset_index(drop=True, inplace=True)

df_filtered_ticker_name

# %%
# only get the coordination results
df_filtered_all_sparse = df_filtered_sparse.loc[:, df_filtered_sparse.columns.str.contains(
    r'\ball\b', case=False)]
df_filtered_all_sparse.columns = df_filtered_all_sparse.columns.str.replace(
    r'^all ', '', regex=True)

# get average
mean_values_sparse = df_filtered_all_sparse.mean()
df_filtered_ticker_name_sparse = pd.DataFrame([mean_values_sparse])
df_filtered_ticker_name_sparse.reset_index(drop=True, inplace=True)

df_filtered_ticker_name_sparse

# %%
llm_histories_np_array = np.array(llm_histories)
df_llm_histories = pd.DataFrame(llm_histories_np_array, columns=tickers)
mean_llm_values = df_llm_histories.mean()
df_llm_histories_mean = pd.DataFrame([mean_llm_values])
df_llm_histories_mean.reset_index(drop=True, inplace=True)

df_llm_histories_mean

# %%
llm_histories_np_array_sparse = np.array(llm_histories_sparse)
df_llm_histories_sparse = pd.DataFrame(
    llm_histories_np_array_sparse, columns=tickers)
mean_llm_values_sparse = df_llm_histories_sparse.mean()
df_llm_histories_mean_sparse = pd.DataFrame([mean_llm_values_sparse])
df_llm_histories_mean_sparse.reset_index(drop=True, inplace=True)

df_llm_histories_mean_sparse

# %%
opt_histories_np_array = np.array(opt_histories)
df_opt_histories = pd.DataFrame(opt_histories_np_array, columns=tickers)
mean_opt_values = df_opt_histories.mean()
df_opt_histories_mean = pd.DataFrame([mean_opt_values])
df_opt_histories_mean.reset_index(drop=True, inplace=True)

df_opt_histories_mean

# %%
df_mean_total = pd.concat([
    df_filtered_ticker_name,
    df_filtered_ticker_name_sparse,
    df_llm_histories_mean,
    df_llm_histories_mean_sparse,
    df_opt_histories_mean], axis=0, ignore_index=True)

# %%

vmin = 1e-3  # Set a minimum value for log scaling to avoid issues with log(0)
vmax = max([n for n in df_mean_total.values.flatten().tolist()
           if isinstance(n, float)])  # Maximum value in the data

# Adjust the figure size for better visualization
plt.figure(figsize=(20, 2.5))
ax = sns.heatmap(
    df_mean_total,
    cmap="Reds",
    linewidths=0.5,
    linecolor="black",
    annot=False,
    cbar=True,
    cbar_kws={"aspect": 5},
    square=True,
    xticklabels=True,
    yticklabels=["LLM+OPT", "LLM_sparse+OPT", "LLM", "LLM_sparse", "OPT"],
    vmin=0,
    vmax=max([n for n in df_mean_total.values.flatten().tolist() if isinstance(
        n, float)]),  # Scale from 0 to max value in the data
    norm=mcolors.LogNorm(vmin=vmin, vmax=vmax)
)

cbar = ax.collections[0].colorbar
cbar.set_ticks([1e-3, 1e-2, 1e-1, 1e0])  # Include 10^0
cbar.set_ticklabels([r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"])

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

ax.tick_params(axis="both", length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tick_params(axis="x", top=True, labeltop=True, labelbottom=False)
plt.xlabel(None)
plt.ylabel(None)

categories = [
    'Technology',
    'Consumer Discretionary',
    'Financials',
    'Real Estate',
    'Energy',
    'Healthcare',
    'Industrials',
    'Materials',
    'Communication Services',
    'Consumer Staples'
]

# Vertical/horizontal offsets for the bracket
y_bottom = 1.48
y_top = y_bottom + 0.05
margin = 0.3  # how much to pull in from each side so brackets don't overlap
linewidth = 0.75

for i, cat in enumerate(categories):
    x_left = i * 6 + margin
    x_right = (i + 1) * 6 - margin

    # Left vertical line
    ax.plot([x_left, x_left], [y_bottom, y_top],
            color="black", lw=linewidth, transform=ax.get_xaxis_transform(), clip_on=False)
    # Right vertical line
    ax.plot([x_right, x_right], [y_bottom, y_top],
            color="black", lw=linewidth, transform=ax.get_xaxis_transform(), clip_on=False)
    # Horizontal top line
    ax.plot([x_left, x_right], [y_top, y_top],
            color="black", lw=linewidth, transform=ax.get_xaxis_transform(), clip_on=False)
    # Category label
    ax.text((x_left + x_right) / 2, y_top + 0.05, '\n'.join(cat.split())+'',
            ha="center", va="bottom", transform=ax.get_xaxis_transform(), fontsize=10)

plt.title("Stock Weights", fontsize=16, pad=35)
plt.tight_layout()

plt.savefig(heatmap_path, dpi=500, bbox_inches='tight')
plt.show()

# %% [markdown]
# #### Stock Weights per Month

# %%
df_total = pd.concat([
    df_filtered_all,
    df_filtered_all_sparse,
    df_llm_histories,
    df_llm_histories_sparse,
    df_opt_histories])

# %%
# Example data: Suppose df_total has 60 rows (5 sets of 12).
# Adapt to match your real data shape.
n_rows = 60
n_cols = 10

# -- FIGURE AND AXES --
fig, axes = plt.subplots(nrows=5, figsize=(20, 23), sharex=True)

# -- SETUP LOG NORM --
vmin = 1e-3
vmax = df_total.values.max()
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

# We'll create a single colorbar at the end, so set cbar=False for each subplot
# The data range is the same for all subplots, so we can just pick the last heatmap's
# "mappable" to feed into fig.colorbar() later.
mappable = None

labels = ["LLM+OPT", "LLM_sparse+OPT", "LLM", "LLM_sparse", "OPT"]

for i in range(5):
    ax = axes[i]

    sub_df = df_total.iloc[i*12: (i+1)*12, :]

    # Create the heatmap with no colorbar
    hmap = sns.heatmap(
        sub_df,
        cmap="Reds",
        linewidths=0.5,
        linecolor="black",
        annot=False,
        square=True,
        cbar=False,      # No inline colorbar
        vmin=vmin,
        vmax=vmax,
        norm=norm,  # apply log scale  to the cbar coloring
        ax=ax
    )

    # Save the "mappable" from the last heatmap in the loop.
    # We can use any subplot's "mappable" for the colorbar,
    # but just store one (e.g. from the last iteration).
    mappable = hmap.collections[0]

    # Turn off bottom tickers; optionally place them on top.
    ax.tick_params(axis='x',
                   bottom=False, labelbottom=False,   # Turn off bottom
                   top=True, labeltop=True,           # Put ticks on top
                   length=0)

    # y-axis labels (just an example: row numbers 0..11)
    ytick_positions = np.arange(sub_df.shape[0]) + 0.5
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels([str(y) for y in range(sub_df.shape[0])], rotation=0)

    # Label each subplot on the y-axis with your desired text
    ax.set_ylabel(labels[i], fontsize=12)

    # Show only bottom & right spines; hide top & left
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(True)

    # Rotate x-tick labels on top
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# -- A SINGLE COLORBAR ON THE RIGHT, SPANNING ALL SUBPLOTS --
# 'mappable' comes from the last heatmap above.
cbar = fig.colorbar(
    mappable,
    ax=axes.ravel().tolist(),   # attach to all subplots
    orientation='vertical',
    fraction=0.02,
    pad=0.03
)
# Adjust the ticks & labels on the colorbar
cbar.set_ticks([1e-3, 1e-2, 1e-1, 1e0])
cbar.set_ticklabels([r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"])

categories = [
    'Technology',
    'Consumer Discretionary',
    'Financials',
    'Real Estate',
    'Energy',
    'Healthcare',
    'Industrials',
    'Materials',
    'Communication Services',
    'Consumer Staples'
]

# Vertical/horizontal offsets for the bracket
y_top = 6.18
y_bottom = y_top-0.01
margin = 0.3  # how much to pull in from each side so brackets don't overlap
linewidth = 0.75

for i, cat in enumerate(categories):
    x_left = i * 6 + margin
    x_right = (i + 1) * 6 - margin

    # Left vertical line
    ax.plot([x_left, x_left], [y_bottom, y_top],
            color="black", lw=linewidth, transform=ax.get_xaxis_transform(), clip_on=False)
    # Right vertical line
    ax.plot([x_right, x_right], [y_bottom, y_top],
            color="black", lw=linewidth, transform=ax.get_xaxis_transform(), clip_on=False)
    # Horizontal top line
    ax.plot([x_left, x_right], [y_top, y_top],
            color="black", lw=linewidth, transform=ax.get_xaxis_transform(), clip_on=False)
    # Category label
    ax.text((x_left + x_right) / 2, y_top + 0.01, '\n'.join(cat.split())+'',
            ha="center", va="bottom", transform=ax.get_xaxis_transform(), fontsize=10)

# -- ADD AN OVERALL TITLE --
fig.suptitle("Weights per Stock", fontsize=20, y=0.94)

plt.savefig(heatmap_all_path, dpi=500, bbox_inches='tight')
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()

# %% [markdown]
# #### Profit and loss from each stock

# %%
df_pnl_total = pd.concat([
    pnl_coordinated,
    pnl_coordinated_sparse,
    pnl_llm,
    pnl_llm_sparse,
    pnl_opt], axis=0, ignore_index=True)

# %%
plt.figure(figsize=(20, 12))

fig, axes = plt.subplots(nrows=5, figsize=(20, 23), sharex=True)

min_val = df_pnl_total.values.min()
max_val = df_pnl_total.values.max()

# 1) Get the built-in RdYlGn colormap
base_cmap = plt.get_cmap("RdYlGn", 256)  # 256 discrete colors

# 2) Convert it to a list so we can modify the middle band
colors = [base_cmap(i) for i in range(base_cmap.N)]

# 3) Make the midpoint less yellow. For example:
#    - The midpoint in a 256-color map is index ~128
#    - Replace it with something lighter (blend with white).
mid_index = 128
# RGBA of the original midpoint (~ bright yellow)
original_mid = colors[mid_index]
# Let's blend that original color with white at, say, 70% original / 30% white:
blend_ratio = 0.7
new_mid = (
    original_mid[0] * blend_ratio + 1.0 * (1 - blend_ratio),
    original_mid[1] * blend_ratio + 1.0 * (1 - blend_ratio),
    original_mid[2] * blend_ratio + 1.0 * (1 - blend_ratio),
    1.0  # keep alpha=1
)
colors[mid_index] = new_mid

# You can also adjust a small band around the midpoint if you want a wider, paler zone
# For example, re-blend indices [120..135] to smoothen the transition:
for idx in range(120, 136):
    c = colors[idx]
    colors[idx] = (
        c[0] * blend_ratio + 1.0 * (1 - blend_ratio),
        c[1] * blend_ratio + 1.0 * (1 - blend_ratio),
        c[2] * blend_ratio + 1.0 * (1 - blend_ratio),
        1.0
    )

# 4) Create a new colormap from our modified colors
my_cmap = mcolors.LinearSegmentedColormap.from_list(
    'ManualCmap',
    [
        (0.0,    (1, 0, 0)),     # red
        (0.1667, (1, 1, 0.8)),   # light yellow
        (1.0,    (0, 1, 0)),     # green
    ],
    N=256
)

my_norm = mcolors.Normalize(vmin=-200, vmax=1000, clip=True)


# heatmap:
mappable = None

for i in range(5):
    ax = axes[i]

    sub_df = df_pnl_total.iloc[i*12: (i+1)*12-1, :]

    # Create the heatmap with no colorbar
    hmap = sns.heatmap(
        sub_df,
        cmap=my_cmap,
        norm=my_norm,
        annot=False,
        cbar=False,      # No inline colorbar
        square=True,
        linewidths=0.5,
        linecolor="black",
        ax=ax
    )

    # Save the "mappable" from the last heatmap in the loop.
    # We can use any subplot's "mappable" for the colorbar,
    # but just store one (e.g. from the last iteration).
    mappable = hmap.collections[0]

    # Turn off bottom tickers; optionally place them on top.
    ax.tick_params(axis='x',
                   bottom=False, labelbottom=False,   # Turn off bottom
                   top=True, labeltop=True,           # Put ticks on top
                   length=0)

    # y-axis labels (just an example: row numbers 0..11)
    ytick_positions = np.arange(sub_df.shape[0]) + 0.5
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels([i for i in range(1, 12)], rotation=0)

    # Label each subplot on the y-axis with your desired text
    ax.set_ylabel(labels[i], fontsize=12)

    # Show only bottom & right spines; hide top & left
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(True)

    # Rotate x-tick labels on top
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# -- A SINGLE COLORBAR ON THE RIGHT, SPANNING ALL SUBPLOTS --
# 'mappable' comes from the last heatmap above.
cbar = fig.colorbar(
    mappable,  # ???
    ax=axes.ravel().tolist(),   # attach to all subplots
    orientation='vertical',
    fraction=0.02,
    pad=0.03
)

# Vertical/horizontal offsets for the bracket
y_top = 6.68
y_bottom = y_top-0.01
margin = 0.3  # how much to pull in from each side so brackets don't overlap
linewidth = 0.75

for i, cat in enumerate(categories):
    x_left = i * 6 + margin
    x_right = (i + 1) * 6 - margin

    # Left vertical line
    ax.plot([x_left, x_left], [y_bottom, y_top],
            color="black", lw=linewidth, transform=ax.get_xaxis_transform(), clip_on=False)
    # Right vertical line
    ax.plot([x_right, x_right], [y_bottom, y_top],
            color="black", lw=linewidth, transform=ax.get_xaxis_transform(), clip_on=False)
    # Horizontal top line
    ax.plot([x_left, x_right], [y_top, y_top],
            color="black", lw=linewidth, transform=ax.get_xaxis_transform(), clip_on=False)
    # Category label
    ax.text((x_left + x_right) / 2, y_top + 0.01, '\n'.join(cat.split())+'',
            ha="center", va="bottom", transform=ax.get_xaxis_transform(), fontsize=10)

# -- ADD AN OVERALL TITLE --
fig.suptitle("Monthly PnL per Ticker", fontsize=20, y=0.94)

plt.savefig(pnl_path, dpi=500, bbox_inches='tight')
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()

# %%


class MarkowitzOptimizer:
    """
    Custom optimizer for Markowitz portfolio optimization
    """

    def __init__(self, risk_aversion=1.0, target_return=None, weight_bounds=(0, 1),
                 risk_free_rate=0.02, shrinkage_method='ledoit_wolf'):
        self.risk_aversion = risk_aversion
        self.target_return = target_return
        self.weight_bounds = weight_bounds
        self.risk_free_rate = risk_free_rate
        self.shrinkage_method = shrinkage_method
        self.weights_ = None
        self.performance_ = None

    def fit(self, prices):
        """
        Optimize portfolio using price data

        Parameters:
        -----------
        prices : pandas.DataFrame
            Historical price data for assets
        """
        # Calculate expected returns
        mu = expected_returns.mean_historical_return(prices, frequency=252)

        # Calculate covariance matrix with specified shrinkage method
        if self.shrinkage_method == 'ledoit_wolf':
            S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        elif self.shrinkage_method == 'sample':
            S = risk_models.sample_cov(prices)
        else:
            S = risk_models.CovarianceShrinkage(prices).oracle_approximating()

        # Set up efficient frontier optimizer
        ef = EfficientFrontier(mu, S, weight_bounds=self.weight_bounds)

        # Optimize based on provided parameters
        if self.target_return is not None:
            try:
                ef.efficient_return(self.target_return)
            except:
                # Fallback to max sharpe if target return is unreachable
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        else:
            # Use L2 regularization with risk_aversion parameter
            ef.max_quadratic_utility(risk_aversion=self.risk_aversion)

        # Get optimized weights and performance metrics
        self.weights_ = ef.clean_weights()

        # Convert weights dict to array to match DataFrame column order
        self.weight_array = np.array(
            [self.weights_[asset] for asset in prices.columns])

        # Calculate performance metrics
        self.performance_ = ef.portfolio_performance(
            risk_free_rate=self.risk_free_rate)

        # Store for convenience
        self.expected_return = self.performance_[0]
        self.volatility = self.performance_[1]
        self.sharpe = self.performance_[2]

        return self

    def evaluate(self, test_prices):
        """
        Evaluate portfolio on test data

        Parameters:
        -----------
        test_prices : pandas.DataFrame
            Test period price data

        Returns:
        --------
        dict
            Performance metrics
        """
        # Calculate returns for test period
        returns = test_prices.pct_change().dropna(how='all')

        # Apply weights
        weights_array = np.array([self.weights_[asset]
                                 for asset in test_prices.columns])
        portfolio_returns = returns.dot(weights_array)

        # Calculate metrics
        mean_return = portfolio_returns.mean() * 252  # Annualized
        volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
        sharpe = (mean_return - self.risk_free_rate) / \
            volatility if volatility > 0 else 0

        return {
            "expected_return": mean_return,
            "volatility": volatility,
            "sharpe": sharpe
        }


def evaluate_params(prices, train_indices, test_indices, params):
    """
    Evaluate a set of parameters using a train/test split

    Parameters:
    -----------
    prices : pandas.DataFrame
        Historical price data
    train_indices : array-like
        Indices for training data
    test_indices : array-like
        Indices for test data
    params : dict
        Parameters to evaluate

    Returns:
    --------
    dict
        Evaluation results
    """
    # Create train/test splits
    train_data = prices.iloc[train_indices]
    test_data = prices.iloc[test_indices]

    # Initialize and fit optimizer with given parameters
    optimizer = MarkowitzOptimizer(**params)
    optimizer.fit(train_data)

    # Evaluate on test data
    test_performance = optimizer.evaluate(test_data)

    # Return results
    return {
        "train_performance": {
            "expected_return": optimizer.expected_return,
            "volatility": optimizer.volatility,
            "sharpe": optimizer.sharpe
        },
        "test_performance": test_performance,
        "weights": optimizer.weights_
    }


def hyperparameter_tune_optimizer(prices, param_grid, n_splits=5, objective="sharpe"):
    """
    Perform hyperparameter tuning for Markowitz optimizer

    Parameters:
    -----------
    prices : pandas.DataFrame
        Historical price data
    param_grid : dict
        Parameters to tune with lists of values to try
    n_splits : int
        Number of time series splits for cross-validation
    objective : str
        Metric to optimize ('sharpe', 'return', or 'risk')

    Returns:
    --------
    dict
        Best parameters and results
    """
    # Generate all parameter combinations
    param_combinations = []

    # Helper function to generate all combinations (recursive)
    def generate_combinations(keys, current_dict, index):
        if index == len(keys):
            param_combinations.append(current_dict.copy())
            return

        key = keys[index]
        for value in param_grid[key]:
            current_dict[key] = value
            generate_combinations(keys, current_dict, index + 1)

    # Start the recursive generation
    generate_combinations(list(param_grid.keys()), {}, 0)

    # Create time series cross-validation splits
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Track best parameters and performance
    best_params = None
    best_score = -float('inf')  # For maximizing metrics
    all_results = []

    # For each parameter combination
    for params in param_combinations:
        cross_val_results = []

        # For each train/test split
        for train_index, test_index in tscv.split(prices):
            # Evaluate parameters on this split
            split_result = evaluate_params(
                prices, train_index, test_index, params)
            cross_val_results.append(split_result)

        # Calculate average performance across all splits
        avg_train_return = np.mean(
            [r["train_performance"]["expected_return"] for r in cross_val_results])
        avg_train_volatility = np.mean(
            [r["train_performance"]["volatility"] for r in cross_val_results])
        avg_train_sharpe = np.mean(
            [r["train_performance"]["sharpe"] for r in cross_val_results])

        avg_test_return = np.mean(
            [r["test_performance"]["expected_return"] for r in cross_val_results])
        avg_test_volatility = np.mean(
            [r["test_performance"]["volatility"] for r in cross_val_results])
        avg_test_sharpe = np.mean(
            [r["test_performance"]["sharpe"] for r in cross_val_results])

        # Determine score based on objective
        if objective == "sharpe":
            score = avg_test_sharpe
        elif objective == "return":
            score = avg_test_return
        elif objective == "risk":
            score = -avg_test_volatility  # Negative because we want to minimize risk

        # Track results
        result = {
            "params": params,
            "train_return": avg_train_return,
            "train_volatility": avg_train_volatility,
            "train_sharpe": avg_train_sharpe,
            "test_return": avg_test_return,
            "test_volatility": avg_test_volatility,
            "test_sharpe": avg_test_sharpe,
            "score": score
        }
        all_results.append(result)

        # Update best parameters if better score found
        if score > best_score:
            best_score = score
            best_params = params

    # Sort results by score (descending)
    all_results.sort(key=lambda x: x["score"], reverse=True)

    # Return best parameters and all results
    return {
        "best_params": best_params,
        "best_score": best_score,
        "objective": objective,
        "results": all_results
    }


def backtest_portfolio(prices, params, window_size=60, step=20):
    """
    Backtest the optimized portfolio strategy using a rolling window approach

    Parameters:
    -----------
    prices : pandas.DataFrame
        Historical price data for assets
    params : dict
        Parameters for the optimizer
    window_size : int
        Number of days to use for training (lookback period)
    step : int
        Number of days to step forward for each rebalance

    Returns:
    --------
    pandas.DataFrame
        Backtest results
    """
    dates = prices.index
    returns = pd.DataFrame(index=prices.index)
    returns['portfolio'] = 0.0

    # For benchmarking
    returns['equal_weight'] = prices.pct_change().mean(axis=1)

    weights_history = []

    for i in range(window_size, len(dates), step):
        if i + step > len(dates):
            break

        # Training data
        train_data = prices.iloc[i-window_size:i]

        # Test data for this period
        test_data = prices.iloc[i:i+step]

        # Fit optimizer on training data
        optimizer = MarkowitzOptimizer(**params)
        optimizer.fit(train_data)

        # Get weights
        weights = optimizer.weights_
        weights_history.append(weights)

        # Calculate returns for test period
        test_returns = test_data.pct_change().dropna(how='all')

        # Apply weights
        weights_array = np.array([weights[asset]
                                 for asset in test_returns.columns])
        portfolio_returns = test_returns.dot(weights_array)

        # Record returns
        common_idx = returns.index.intersection(portfolio_returns.index)
        returns.loc[common_idx, 'portfolio'] = portfolio_returns

    # Calculate cumulative returns
    returns['portfolio_cumulative'] = (1 + returns['portfolio']).cumprod()
    returns['equal_weight_cumulative'] = (
        1 + returns['equal_weight']).cumprod()

    return returns, weights_history


def visualize_results(results, top_n=5):
    """
    Visualize hyperparameter tuning results

    Parameters:
    -----------
    results : dict
        Results from hyperparameter_tune_optimizer
    top_n : int
        Number of top parameter sets to display
    """
    # Extract top parameter sets
    top_results = results["results"][:top_n]

    # Create figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Plot sharpe ratio
    sharpe_values = [r["test_sharpe"] for r in top_results]
    param_labels = [f"Set {i+1}" for i in range(len(top_results))]

    axs[0].bar(param_labels, sharpe_values)
    axs[0].set_title("Test Sharpe Ratio by Parameter Set")
    axs[0].set_xlabel("Parameter Set")
    axs[0].set_ylabel("Sharpe Ratio")

    # Plot return vs volatility
    returns = [r["test_return"] for r in top_results]
    volatilities = [r["test_volatility"] for r in top_results]

    axs[1].scatter(volatilities, returns)
    for i, label in enumerate(param_labels):
        axs[1].annotate(label, (volatilities[i], returns[i]))

    axs[1].set_title("Risk-Return Profile by Parameter Set")
    axs[1].set_xlabel("Volatility (Risk)")
    axs[1].set_ylabel("Expected Return")
    axs[1].grid(True)

    # Display parameter details
    for i, result in enumerate(top_results):
        print(f"Parameter Set {i+1}:")
        for param, value in result["params"].items():
            print(f"  {param}: {value}")
        print(f"  Test Sharpe: {result['test_sharpe']:.4f}")
        print(f"  Test Return: {result['test_return']:.4f}")
        print(f"  Test Volatility: {result['test_volatility']:.4f}")
        print()

    plt.tight_layout()
    return fig


# %%
portfolio.head()

# %%
portfolio.tail()

# %%
# Filter for just 2024 data
portfolio_2024 = portfolio[portfolio.index.year == 2024]

# Display the filtered data
# portfolio_2024

# %%
# Define hyperparameter grid
param_grid = {
    'risk_aversion': [1.0, 2.0, 5.0, 10.0],
    'target_return': [None, 0.15, 0.20, 0.25],
    'shrinkage_method': ['ledoit_wolf', 'oracle_approximating'],
    'risk_free_rate': [0.01, 0.02, 0.03]
}

# Run hyperparameter tuning
results = hyperparameter_tune_optimizer(
    portfolio_2024,
    param_grid,
    n_splits=5,
    objective="sharpe"  # Can be "sharpe", "return", or "risk"
)

# Print best parameters
print("Best Parameters:", results["best_params"])
print("Best Score:", results["best_score"])

# Visualize results
fig = visualize_results(results, top_n=5)
plt.savefig("hyperparameter_results.png")

# Run backtest with best parameters
backtest_results, weights_history = backtest_portfolio(
    portfolio_2024, results["best_params"])

# Plot backtest results
plt.figure(figsize=(12, 6))
backtest_results[['portfolio_cumulative', 'equal_weight_cumulative']].plot()
plt.title('Portfolio Performance with Optimized Parameters')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend(['Optimized Portfolio', 'Equal Weight'])
plt.grid(True)
plt.tight_layout()
plt.savefig("backtest_results.png")

# Calculate performance metrics
portfolio_returns = backtest_results['portfolio'].dropna()

annual_return = portfolio_returns.mean() * 252
annual_volatility = portfolio_returns.std() * np.sqrt(252)
sharpe_ratio = annual_return / annual_volatility

print(f"Annual Return: {annual_return:.4f}")
print(f"Annual Volatility: {annual_volatility:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

# %%


def add_variance_visualization(prices, param_sets, window_size=60, step=20):
    """
    Visualize portfolio variance across different parameter sets with error bars

    Parameters:
    -----------
    prices : pandas.DataFrame
        Historical price data
    param_sets : list
        List of parameter dictionaries to evaluate
    window_size : int
        Lookback window size
    step : int
        Rebalancing step size

    Returns:
    --------
    matplotlib.figure.Figure
        Figure with variance visualization
    """
    # Track variance for each parameter set
    param_labels = [f"Set {i+1}" for i in range(len(param_sets))]
    all_variances = []
    mean_variances = []
    std_variances = []

    # For each parameter set
    for params in param_sets:
        variances = []
        dates = prices.index

        # Perform rolling window analysis
        for i in range(window_size, len(dates), step):
            if i + step > len(dates):
                break

            # Training data
            train_data = prices.iloc[i-window_size:i]

            # Fit optimizer on training data
            optimizer = MarkowitzOptimizer(**params)
            optimizer.fit(train_data)

            # Get portfolio variance from the optimizer
            # This is the square of the volatility
            variance = optimizer.volatility ** 2
            variances.append(variance)

        # Calculate mean and std of variances across windows
        all_variances.append(variances)
        mean_variances.append(np.mean(variances))
        std_variances.append(np.std(variances))

    # Create bar plot with error bars
    fig, ax = plt.subplots(figsize=(12, 8))

    # Bar plot
    bars = ax.bar(param_labels, mean_variances, yerr=std_variances,
                  capsize=10, alpha=0.7, color='skyblue', ecolor='black')

    # Add parameter details as annotations
    for i, param_set in enumerate(param_sets):
        # Format the parameter details
        param_text = "\n".join([f"{k}: {v}" for k, v in param_set.items()])

        # Annotate each bar
        ax.annotate(param_text,
                    xy=(i, mean_variances[i]),
                    xytext=(0, 10),  # 10 points vertically above
                    textcoords="offset points",
                    ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))

    # Add labels and title
    ax.set_title('Portfolio Variance by Parameter Set', fontsize=16)
    ax.set_xlabel('Parameter Set', fontsize=14)
    ax.set_ylabel('Portfolio Variance (with Error Bars)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Create a second figure to show variance over time for each parameter set
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    # Plot variance over time for each parameter set
    x_values = list(range(len(all_variances[0])))
    for i, variances in enumerate(all_variances):
        ax2.plot(x_values, variances, marker='o', linestyle='-',
                 label=f'Set {i+1}')

    # Add labels and title
    ax2.set_title('Portfolio Variance Over Time by Parameter Set', fontsize=16)
    ax2.set_xlabel('Rebalancing Period', fontsize=14)
    ax2.set_ylabel('Portfolio Variance', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()

    return fig, fig2


class MarkowitzOptimizer:
    """
    Sample class definition for completeness (same as in previous code)
    """

    def __init__(self, risk_aversion=1.0, target_return=None, weight_bounds=(0, 1),
                 risk_free_rate=0.02, shrinkage_method='ledoit_wolf'):
        self.risk_aversion = risk_aversion
        self.target_return = target_return
        self.weight_bounds = weight_bounds
        self.risk_free_rate = risk_free_rate
        self.shrinkage_method = shrinkage_method
        self.weights_ = None
        self.performance_ = None

    def fit(self, prices):
        # Calculate expected returns
        mu = expected_returns.mean_historical_return(
            prices, frequency=252, fill_method=None)

        # Calculate covariance matrix with specified shrinkage method
        if self.shrinkage_method == 'ledoit_wolf':
            S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        elif self.shrinkage_method == 'sample':
            S = risk_models.sample_cov(prices, frequency=252)
        else:
            S = risk_models.CovarianceShrinkage(prices).oracle_approximating()

        # Set up efficient frontier optimizer
        ef = EfficientFrontier(mu, S, weight_bounds=self.weight_bounds)

        # Optimize based on provided parameters
        if self.target_return is not None:
            try:
                ef.efficient_return(self.target_return)
            except:
                # Fallback to max sharpe if target return is unreachable
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        else:
            # Use L2 regularization with risk_aversion parameter
            ef.max_quadratic_utility(risk_aversion=self.risk_aversion)

        # Get optimized weights and performance metrics
        self.weights_ = ef.clean_weights()

        # Calculate performance metrics
        self.performance_ = ef.portfolio_performance(
            risk_free_rate=self.risk_free_rate)

        # Store for convenience
        self.expected_return = self.performance_[0]
        self.volatility = self.performance_[1]
        self.sharpe = self.performance_[2]

        return self


def calculate_efficient_frontier_range(prices, n_points=50):
    """
    Calculate and visualize the efficient frontier with parameter sets

    Parameters:
    -----------
    prices : pandas.DataFrame
        Historical price data
    n_points : int
        Number of points on the efficient frontier

    Returns:
    --------
    matplotlib.figure.Figure
        Figure with efficient frontier visualization
    """
    # Calculate expected returns and covariance matrix
    mu = expected_returns.mean_historical_return(
        prices, frequency=252, fill_method=None)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    # Set up efficient frontier and calculate range of returns
    ef = EfficientFrontier(mu, S)
    ef_returns = []
    ef_volatilities = []

    # Calculate minimum variance portfolio
    ef_min_var = EfficientFrontier(mu, S)
    ef_min_var.min_volatility()
    min_var_return, min_var_volatility, _ = ef_min_var.portfolio_performance()

    # Calculate maximum return portfolio
    ef_max_ret = EfficientFrontier(mu, S)
    ef_max_ret.max_sharpe()
    max_sharpe_return, max_sharpe_volatility, _ = ef_max_ret.portfolio_performance()

    # Generate points along the efficient frontier
    target_returns = np.linspace(
        min_var_return, max_sharpe_return * 1.2, n_points)

    for target_return in target_returns:
        ef = EfficientFrontier(mu, S)
        try:
            ef.efficient_return(target_return)
            expected_return, volatility, _ = ef.portfolio_performance()
            ef_returns.append(expected_return)
            ef_volatilities.append(volatility)
        except:
            continue

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot efficient frontier
    ax.plot(ef_volatilities, ef_returns, 'b-',
            linewidth=2, label='Efficient Frontier')

    # Mark key portfolios
    ax.scatter(min_var_volatility, min_var_return, marker='*', color='g', s=200,
               label='Minimum Variance')
    ax.scatter(max_sharpe_volatility, max_sharpe_return, marker='*', color='r', s=200,
               label='Maximum Sharpe Ratio')

    # Add parameter set portfolios
    def add_portfolio_to_plot(params, label, marker='o', color='purple'):
        optimizer = MarkowitzOptimizer(**params)
        optimizer.fit(prices)
        ax.scatter(optimizer.volatility, optimizer.expected_return,
                   marker=marker, color=color, s=150, label=label)

        # Add text label for the portfolio
        ax.annotate(label,
                    xy=(optimizer.volatility, optimizer.expected_return),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))

        return optimizer.volatility, optimizer.expected_return

    # Sample parameter sets to show on frontier
    param_sets = [
        {'risk_aversion': 1.0, 'target_return': None},
        {'risk_aversion': 5.0, 'target_return': None},
        {'risk_aversion': 10.0, 'target_return': None},
        {'target_return': 0.15, 'risk_aversion': 1.0},
        {'target_return': 0.25, 'risk_aversion': 1.0},
    ]

    # Plot parameter sets
    param_colors = ['orange', 'purple', 'brown', 'cyan', 'magenta']
    for i, params in enumerate(param_sets):
        label = f"Params {i+1}"
        add_portfolio_to_plot(params, label, color=param_colors[i])

    # Add labels and title
    ax.set_title('Efficient Frontier with Parameter Sets', fontsize=16)
    ax.set_xlabel('Portfolio Volatility (Risk)', fontsize=14)
    ax.set_ylabel('Expected Return', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    plt.tight_layout()

    return fig


# %%
# Define parameter sets to evaluate
param_sets = [
    {'risk_aversion': 1.0, 'target_return': None,
        'shrinkage_method': 'ledoit_wolf', 'risk_free_rate': 0.02},
    {'risk_aversion': 2.0, 'target_return': None,
        'shrinkage_method': 'ledoit_wolf', 'risk_free_rate': 0.02},
    {'risk_aversion': 5.0, 'target_return': None,
        'shrinkage_method': 'ledoit_wolf', 'risk_free_rate': 0.02},
    {'risk_aversion': 1.0, 'target_return': 0.20,
        'shrinkage_method': 'ledoit_wolf', 'risk_free_rate': 0.02},
    {'risk_aversion': 1.0, 'target_return': 0.25,
        'shrinkage_method': 'ledoit_wolf', 'risk_free_rate': 0.02}
]

# Visualize variance
# fig1, fig2 = add_variance_visualization(portfolio_2024, param_sets)
# fig1.savefig("portfolio_variance_bars.png")
# fig2.savefig("portfolio_variance_time.png")

# # Visualize efficient frontier
# fig3 = calculate_efficient_frontier_range(portfolio_2024)
# fig3.savefig("efficient_frontier.png")

print("Visualization complete. Check the saved PNG files.")

# %%


def plot_optimizer_returns(optimizer_returns, save_path=None):
    """
    Create a bar plot showing average optimizer returns with variance.

    Parameters:
    -----------
    optimizer_returns : array-like
        List or array of optimizer returns
    save_path : str, optional
        If provided, saves the figure to this path instead of displaying
    """
    # Calculate mean and standard deviation
    mean_return = np.mean(optimizer_returns)
    std_return = np.std(optimizer_returns)

    # Create figure and axis
    plt.figure(figsize=(5, 6))

    # Create a single skinnier bar for the mean
    plt.bar(['Average Return'], [mean_return], color='#3a86ff',
            yerr=std_return, capsize=10, width=0.3)

    # Add labels and title
    plt.ylabel('Return Multiple')
    plt.title('Average Optimizer Return with Variance')
    # Set y-limit to show error bar clearly
    plt.ylim(0, mean_return + 3*std_return)
    plt.grid(axis='y', alpha=0.3)

    # Add text annotation in the upper right corner of the plot
    plt.text(0.95, 0.95, f'Mean: {mean_return:.3f}\nStd Dev: {std_return:.3f}',
             transform=plt.gca().transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Save or show plot
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return plt.gcf()


def compare_algorithm_returns(optimizer_returns, llm_returns, save_path=None):
    """
    Create a side-by-side bar plot comparing optimizer and LLM returns with variance.

    Parameters:
    -----------
    optimizer_returns : array-like
        List or array of optimizer returns
    llm_returns : array-like
        List or array of LLM returns
    save_path : str, optional
        If provided, saves the figure to this path instead of displaying
    """
    # Calculate means and standard deviations
    mean_opt = np.mean(optimizer_returns)
    std_opt = np.std(optimizer_returns)
    mean_llm = np.mean(llm_returns)
    std_llm = np.std(llm_returns)

    # Create figure and axis
    plt.figure(figsize=(8, 6))

    # Create bar positions
    positions = np.array([0, 1])
    width = 0.35

    # Create bars
    plt.bar(positions[0], mean_opt, color='#3a86ff',
            yerr=std_opt, capsize=10, width=width,
            label='Optimizer')
    plt.bar(positions[1], mean_llm, color='#9a031e',
            yerr=std_llm, capsize=10, width=width,
            label='LLM')

    # Add labels and title
    plt.ylabel('Return Multiple')
    plt.title('Algorithm Returns Comparison with Variance')
    plt.xticks(positions, ['Optimizer', 'LLM'])
    # Set y-limit to show error bars clearly
    plt.ylim(0, max(mean_opt, mean_llm) + 3*max(std_opt, std_llm))
    plt.grid(axis='y', alpha=0.3)
    plt.legend()

    # Add text annotations for each bar
    plt.text(positions[0], mean_opt + std_opt, f'Mean: {mean_opt:.3f}\nStd: {std_opt:.3f}',
             ha='center', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.text(positions[1], mean_llm + std_llm, f'Mean: {mean_llm:.3f}\nStd: {std_llm:.3f}',
             ha='center', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Save or show plot
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return plt.gcf()


def plot_return_distribution(returns, title='Return Distribution', color='#3a86ff', bins=20, save_path=None):
    """
    Create a histogram showing the distribution of returns.

    Parameters:
    -----------
    returns : array-like
        List or array of returns
    title : str
        Plot title
    color : str
        Color for histogram
    bins : int
        Number of histogram bins
    save_path : str, optional
        If provided, saves the figure to this path instead of displaying
    """
    plt.figure(figsize=(8, 6))

    # Plot histogram
    n, bins, patches = plt.hist(
        returns, bins=bins, alpha=0.7, color=color, edgecolor='black')

    # Add vertical line for mean
    mean_return = np.mean(returns)
    plt.axvline(mean_return, color='red', linestyle='dashed',
                linewidth=2, label=f'Mean: {mean_return:.3f}')

    # Add labels and title
    plt.xlabel('Return Multiple')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()

    # Add text annotation with statistics
    std_return = np.std(returns)
    min_return = np.min(returns)
    max_return = np.max(returns)
    plt.text(0.95, 0.95,
             f'Mean: {mean_return:.3f}\nStd Dev: {std_return:.3f}\nMin: {min_return:.3f}\nMax: {max_return:.3f}',
             transform=plt.gca().transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Save or show plot
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return plt.gcf()


np.random.seed(42)  # For reproducibility
optimizer_returns = np.random.normal(
    1.15, 0.30, 100)  # Mean 1.15, std 0.3, 100 samples

# Simulate some LLM returns (replace with your actual data)
# Mean 1.05, std 0.45, 100 samples
llm_returns = np.random.normal(1.05, 0.45, 100)

# Create individual plots
plot_optimizer_returns(
    optimizer_returns, save_path="YYY_optimizer_returns.png")

# Create comparison plot
compare_algorithm_returns(optimizer_returns, llm_returns,
                          save_path="YYY_returns_comparison.png")

# Create distribution plots
plot_return_distribution(optimizer_returns,
                         title='Optimizer Returns Distribution',
                         color='#3a86ff',
                         save_path="YYY_optimizer_distribution.png")

plot_return_distribution(llm_returns,
                         title='LLM Returns Distribution',
                         color='#9a031e',
                         save_path="YYY_llm_distribution.png")

# %%
