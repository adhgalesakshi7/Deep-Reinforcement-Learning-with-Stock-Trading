import argparse
import math
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
EXTRA_DEPS = Path(r"C:\Users\HP\Documents\Codex\drl_deps")
if EXTRA_DEPS.exists():
    sys.path.insert(0, str(EXTRA_DEPS))

import gymnasium as gym
import matplotlib
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import A2C, DDPG, PPO
from stable_baselines3.common.vec_env import DummyVecEnv


matplotlib.use("Agg")
import matplotlib.pyplot as plt


TICKERS = [
    "MMM",
    "AXP",
    "AAPL",
    "BA",
    "CAT",
    "CVX",
    "CSCO",
    "KO",
    "DIS",
    "DOW",
    "GS",
    "HD",
    "IBM",
    "INTC",
    "JNJ",
    "JPM",
    "MCD",
    "MRK",
    "MSFT",
    "NKE",
    "PFE",
    "PG",
    "TRV",
    "UNH",
    "UTX",
    "VZ",
    "V",
    "WBA",
    "WMT",
    "XOM",
]

TRAIN_RANGE = ("2009-01-01", "2015-12-31")
VALID_RANGE = ("2016-01-01", "2016-12-31")
TEST_RANGE = ("2017-01-01", "2020-05-08")


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma_tp = tp.rolling(window=20).mean()
    mean_dev = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df["CCI"] = (tp - sma_tp) / (0.015 * mean_dev)

    high_diff = df["High"].diff()
    low_diff = df["Low"].diff()
    df["+DM"] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    df["-DM"] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            np.abs(df["High"] - df["Close"].shift(1)),
            np.abs(df["Low"] - df["Close"].shift(1)),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    df["+DI"] = 100 * (df["+DM"].ewm(span=14, adjust=False).mean() / atr)
    df["-DI"] = 100 * (df["-DM"].ewm(span=14, adjust=False).mean() / atr)
    dx = 100 * np.abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"])
    df["ADX"] = dx.ewm(span=14, adjust=False).mean()

    df = df.dropna()
    return df[["Open", "High", "Low", "Close", "Volume", "MACD", "Signal", "RSI", "CCI", "ADX"]]


def load_split_data(base_dir: Path):
    stock_data = {}
    skipped = []

    for ticker in TICKERS:
        csv_path = base_dir / f"{ticker}.csv"
        try:
            df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        except pd.errors.EmptyDataError:
            skipped.append(ticker)
            continue

        if df.empty:
            skipped.append(ticker)
            continue

        stock_data[ticker] = df

    raw_training_data = {}
    raw_validation_data = {}
    raw_test_data = {}

    for ticker, df in stock_data.items():
        raw_training_data[ticker] = add_technical_indicators(df.loc[TRAIN_RANGE[0] : TRAIN_RANGE[1]])
        raw_validation_data[ticker] = add_technical_indicators(df.loc[VALID_RANGE[0] : VALID_RANGE[1]])
        raw_test_data[ticker] = add_technical_indicators(df.loc[TEST_RANGE[0] : TEST_RANGE[1]])

    consistent_tickers = [
        ticker
        for ticker in stock_data
        if not raw_training_data[ticker].empty
        and not raw_validation_data[ticker].empty
        and not raw_test_data[ticker].empty
    ]
    dropped_after_split = sorted(set(stock_data) - set(consistent_tickers))

    training_data = {ticker: raw_training_data[ticker] for ticker in consistent_tickers}
    validation_data = {ticker: raw_validation_data[ticker] for ticker in consistent_tickers}
    test_data = {ticker: raw_test_data[ticker] for ticker in consistent_tickers}

    return stock_data, training_data, validation_data, test_data, skipped, dropped_after_split


class StockTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, stock_data):
        super().__init__()
        self.stock_data = {ticker: df for ticker, df in stock_data.items() if not df.empty}
        self.tickers = list(self.stock_data.keys())
        if not self.tickers:
            raise ValueError("All provided stock data is empty")

        sample_df = next(iter(self.stock_data.values()))
        self.n_features = len(sample_df.columns)

        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.tickers),), dtype=np.float32)
        self.obs_shape = self.n_features * len(self.tickers) + 2 + len(self.tickers) + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_shape,),
            dtype=np.float32,
        )

        self.initial_balance = 1000
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.total_shares_sold = {ticker: 0 for ticker in self.tickers}
        self.total_sales_value = {ticker: 0 for ticker in self.tickers}
        self.current_step = 0
        self.max_steps = max(0, min(len(df) for df in self.stock_data.values()) - 1)

    def reset(self, seed=None, options=None):
        del options
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.total_shares_sold = {ticker: 0 for ticker in self.tickers}
        self.total_sales_value = {ticker: 0 for ticker in self.tickers}
        self.current_step = 0
        return self._next_observation(), {}

    def _next_observation(self):
        frame = np.zeros(self.obs_shape, dtype=np.float32)
        idx = 0
        for ticker in self.tickers:
            df = self.stock_data[ticker]
            if self.current_step < len(df):
                frame[idx : idx + self.n_features] = df.iloc[self.current_step].values
            else:
                frame[idx : idx + self.n_features] = df.iloc[-1].values
            idx += self.n_features

        frame[-4 - len(self.tickers)] = self.balance
        frame[-3 - len(self.tickers) : -3] = [self.shares_held[ticker] for ticker in self.tickers]
        frame[-3] = self.net_worth
        frame[-2] = self.max_net_worth
        frame[-1] = self.current_step
        return frame

    def step(self, actions):
        self.current_step += 1
        if self.current_step > self.max_steps:
            return self._next_observation(), 0.0, True, False, {}

        current_prices = {}
        for i, ticker in enumerate(self.tickers):
            current_prices[ticker] = self.stock_data[ticker].iloc[self.current_step]["Close"]
            action = float(actions[i])

            if action > 0:
                shares_to_buy = int(self.balance * action / current_prices[ticker])
                cost = shares_to_buy * current_prices[ticker]
                self.balance -= cost
                self.shares_held[ticker] += shares_to_buy
            elif action < 0:
                shares_to_sell = int(self.shares_held[ticker] * abs(action))
                sale = shares_to_sell * current_prices[ticker]
                self.balance += sale
                self.shares_held[ticker] -= shares_to_sell
                self.total_shares_sold[ticker] += shares_to_sell
                self.total_sales_value[ticker] += sale

        self.net_worth = self.balance + sum(
            self.shares_held[ticker] * current_prices[ticker] for ticker in self.tickers
        )
        self.max_net_worth = max(self.net_worth, self.max_net_worth)

        reward = self.net_worth - self.initial_balance
        done = self.net_worth <= 0 or self.current_step >= self.max_steps
        return self._next_observation(), reward, done, False, {}


class EnsembleAgent:
    def __init__(self, ppo_model, a2c_model, ddpg_model):
        self.ppo_model = ppo_model
        self.a2c_model = a2c_model
        self.ddpg_model = ddpg_model

    def predict(self, obs):
        ppo_action, _ = self.ppo_model.predict(obs, deterministic=True)
        a2c_action, _ = self.a2c_model.predict(obs, deterministic=True)
        ddpg_action, _ = self.ddpg_model.predict(obs, deterministic=True)
        return np.mean([ppo_action, a2c_action, ddpg_action], axis=0)


class PPOAgent:
    def __init__(self, env, total_timesteps):
        self.model = PPO("MlpPolicy", env, verbose=0)
        self.model.learn(total_timesteps=total_timesteps, progress_bar=False)

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


class A2CAgent:
    def __init__(self, env, total_timesteps):
        self.model = A2C("MlpPolicy", env, verbose=0)
        self.model.learn(total_timesteps=total_timesteps, progress_bar=False)

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


class DDPGAgent:
    def __init__(self, env, total_timesteps):
        self.model = DDPG("MlpPolicy", env, verbose=0)
        self.model.learn(total_timesteps=total_timesteps, progress_bar=False)

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


def train_agents(training_data, total_timesteps):
    env = DummyVecEnv([lambda: StockTradingEnv(training_data)])

    started = time.time()
    ppo_agent = PPOAgent(env, total_timesteps)
    ppo_seconds = time.time() - started

    started = time.time()
    a2c_agent = A2CAgent(env, total_timesteps)
    a2c_seconds = time.time() - started

    started = time.time()
    ddpg_agent = DDPGAgent(env, total_timesteps)
    ddpg_seconds = time.time() - started

    ensemble_agent = EnsembleAgent(ppo_agent.model, a2c_agent.model, ddpg_agent.model)
    timing = {
        "PPO Agent": ppo_seconds,
        "A2C Agent": a2c_seconds,
        "DDPG Agent": ddpg_seconds,
    }
    return env, ppo_agent, a2c_agent, ddpg_agent, ensemble_agent, timing


def test_agent(env, agent, stock_data, n_tests=1000):
    metrics = {
        "steps": [],
        "balances": [],
        "net_worths": [],
        "initial_balance": env.get_attr("initial_balance")[0],
        "shares_held": {ticker: [] for ticker in stock_data.keys()},
    }

    obs = env.reset()
    for i in range(n_tests):
        metrics["steps"].append(i)
        action = agent.predict(obs)
        obs, _, dones, _ = env.step(action)

        metrics["balances"].append(env.get_attr("balance")[0])
        metrics["net_worths"].append(env.get_attr("net_worth")[0])
        env_shares_held = env.get_attr("shares_held")[0]
        for ticker in stock_data.keys():
            metrics["shares_held"][ticker].append(env_shares_held.get(ticker, 0))

        if bool(dones[0]):
            obs = env.reset()

    return metrics


def summarize_metrics(metrics, label):
    net_worths = np.asarray(metrics["net_worths"], dtype=float)
    initial_balance = float(metrics.get("initial_balance", 1000.0))
    average_net_worth = float(np.mean(net_worths)) if len(net_worths) else float("nan")
    std = float(np.std(net_worths)) if len(net_worths) else float("nan")
    sharpe = average_net_worth / std if std and not math.isclose(std, 0.0) else float("nan")
    final_net_worth = float(net_worths[-1]) if len(net_worths) else float("nan")
    total_return_pct = ((final_net_worth / initial_balance) - 1) * 100 if initial_balance else float("nan")
    running_peak = np.maximum.accumulate(net_worths) if len(net_worths) else np.array([])
    max_drawdown_pct = (
        float(np.min((net_worths / running_peak - 1) * 100))
        if len(net_worths) and np.all(running_peak)
        else float("nan")
    )
    return {
        "Agent": label,
        "Average Net Worth": average_net_worth,
        "Standard Deviation": std,
        "Sharpe Ratio": sharpe,
        "Final Net Worth": final_net_worth,
        "Total Return %": total_return_pct,
        "Max Net Worth": float(np.max(net_worths)) if len(net_worths) else float("nan"),
        "Min Net Worth": float(np.min(net_worths)) if len(net_worths) else float("nan"),
        "Max Drawdown %": max_drawdown_pct,
    }


def save_net_worth_plot(metrics_map, output_path, title):
    plt.figure(figsize=(12, 6))
    for label, metrics in metrics_map.items():
        plt.plot(metrics["steps"], metrics["net_worths"], label=label)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Net Worth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_sharpe_plot(summary_df, output_path, title):
    ordered = summary_df.sort_values(by="Sharpe Ratio", ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(ordered["Agent"], ordered["Sharpe Ratio"])
    plt.title(title)
    plt.xlabel("Agent")
    plt.ylabel("Sharpe Ratio")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate_split(name, dataset, agents, output_dir, n_tests):
    env = DummyVecEnv([lambda: StockTradingEnv(dataset)])
    metrics_map = {}
    rows = []

    for label, agent in agents.items():
        metrics = test_agent(env, agent, dataset, n_tests=n_tests)
        metrics_map[label] = metrics
        rows.append(summarize_metrics(metrics, label))

    summary_df = pd.DataFrame(rows).sort_values(by="Sharpe Ratio", ascending=False)
    summary_df.to_csv(output_dir / f"{name}_metrics.csv", index=False)
    save_net_worth_plot(metrics_map, output_dir / f"{name}_net_worth.png", f"{name.title()} Net Worth")
    save_sharpe_plot(summary_df, output_dir / f"{name}_sharpe.png", f"{name.title()} Sharpe Ratio")
    return summary_df


def write_project_evidence(
    output_dir: Path,
    stock_data: dict[str, pd.DataFrame],
    training_data: dict[str, pd.DataFrame],
    validation_data: dict[str, pd.DataFrame],
    test_data: dict[str, pd.DataFrame],
    skipped: list[str],
    dropped_after_split: list[str],
    split_summaries: dict[str, pd.DataFrame],
) -> Path:
    total_rows = sum(len(df) for df in stock_data.values())
    lines = [
        "# Trading Project Evidence",
        "",
        "## Data Coverage",
        f"- Tickers with usable CSVs: {len(stock_data)}",
        f"- Total rows across raw CSVs: {total_rows}",
        f"- Training tickers with aligned train/validation/test windows: {len(training_data)}",
        f"- Validation tickers with aligned train/validation/test windows: {len(validation_data)}",
        f"- Test tickers with aligned train/validation/test windows: {len(test_data)}",
    ]

    if skipped:
        lines.extend(["", "## Skipped Symbols", f"- {', '.join(skipped)}"])

    if dropped_after_split:
        lines.extend(
            [
                "",
                "## Dropped After Split",
                f"- {', '.join(dropped_after_split)}",
            ]
        )

    lines.extend(["", "## Evaluation Highlights"])
    for split_name, summary_df in split_summaries.items():
        best_row = summary_df.iloc[0]
        lines.extend(
            [
                f"- {split_name.title()}: {best_row['Agent']} led with Sharpe {best_row['Sharpe Ratio']:.2f}, final net worth {best_row['Final Net Worth']:.2f}, and total return {best_row['Total Return %']:.2f}%.",
            ]
        )

    lines.extend(
        [
            "",
            "## What The Project Shows",
            "- Multi-asset reinforcement learning on 30 large-cap U.S. equities.",
            "- Technical indicators: RSI, MACD, signal line, CCI, and ADX.",
            "- Comparison of PPO, A2C, DDPG, and an ensemble agent across train, validation, and test splits.",
        ]
    )

    evidence_path = output_dir / "project_evidence.md"
    evidence_path.write_text("\n".join(lines), encoding="utf-8")
    return evidence_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--tests", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "run_outputs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    stock_data, training_data, validation_data, test_data, skipped, dropped_after_split = load_split_data(PROJECT_DIR)
    print(f"Loaded {len(stock_data)} tickers from CSV files.")
    if skipped:
        print(f"Skipped empty CSVs: {', '.join(skipped)}")
    if dropped_after_split:
        print(
            "Dropped tickers with empty train/validation/test windows after indicator calculation: "
            + ", ".join(dropped_after_split)
        )
    print(f"Using {len(training_data)} consistent tickers for all environments.")
    print(f"Training sample shape for AAPL: {training_data['AAPL'].shape}")
    print(f"Validation sample shape for AAPL: {validation_data['AAPL'].shape}")
    print(f"Test sample shape for AAPL: {test_data['AAPL'].shape}")

    _, ppo_agent, a2c_agent, ddpg_agent, ensemble_agent, timing = train_agents(
        training_data,
        args.timesteps,
    )

    timing_df = pd.DataFrame(
        [{"Agent": agent, "Train Seconds": seconds} for agent, seconds in timing.items()]
    ).sort_values(by="Train Seconds")
    timing_df.to_csv(args.output_dir / "training_times.csv", index=False)
    print("Training completed.")
    print(timing_df.to_string(index=False))

    agents = {
        "PPO Agent": ppo_agent,
        "A2C Agent": a2c_agent,
        "DDPG Agent": ddpg_agent,
        "Ensemble Agent": ensemble_agent,
    }

    split_summaries = {}

    for split_name, dataset in [
        ("training", training_data),
        ("validation", validation_data),
        ("test", test_data),
    ]:
        summary_df = evaluate_split(split_name, dataset, agents, args.output_dir, args.tests)
        split_summaries[split_name] = summary_df
        print("")
        print(f"{split_name.title()} metrics:")
        print(summary_df.to_string(index=False))

    evidence_path = write_project_evidence(
        args.output_dir,
        stock_data,
        training_data,
        validation_data,
        test_data,
        skipped,
        dropped_after_split,
        split_summaries,
    )

    print("")
    print(f"Saved outputs to: {args.output_dir}")
    print(f"Saved project evidence to: {evidence_path}")


if __name__ == "__main__":
    main()
