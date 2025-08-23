import os
import random
import string
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt


@dataclass
class DemoConfig:
	seed: int = 42
	num_days: int = 90
	tweets_per_day: Tuple[int, int] = (50, 120)
	initial_price: float = 30000.0
	volatility: float = 0.02
	artifact_dir: str = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)


def generate_synthetic_tweets(start_date: datetime, num_days: int, tweets_per_day: Tuple[int, int]) -> pd.DataFrame:
	positive_phrases = [
		"bullish breakout", "to the moon", "strong support", "buying the dip",
		"institutional adoption", "ETF approval", "on-chain strength", "higher highs",
	]
	negative_phrases = [
		"bearish divergence", "rug pull", "sell-off", "weak hands",
		"regulatory FUD", "resistance rejected", "lower lows", "capitulation",
	]
	neutral_phrases = [
		"sideways action", "range-bound", "consolidation", "neutral momentum",
		"waiting for confirmation", "no clear trend",
	]

	rows: List[dict] = []
	for d in range(num_days):
		current_date = start_date + timedelta(days=d)
		num_tweets = random.randint(*tweets_per_day)
		for _ in range(num_tweets):
			mood = random.choices(["pos", "neg", "neu"], weights=[0.4, 0.35, 0.25])[0]
			if mood == "pos":
				phrase = random.choice(positive_phrases)
			elif mood == "neg":
				phrase = random.choice(negative_phrases)
			else:
				phrase = random.choice(neutral_phrases)
			hashtag = random.choice(["#BTC", "#Bitcoin", "#crypto", "#alts"])
			noise = "".join(random.choices(string.ascii_lowercase + " ", k=random.randint(5, 20))).strip()
			text = f"{phrase} {hashtag} {noise}"
			rows.append({"date": current_date.date(), "text": text})
	return pd.DataFrame(rows)


def score_sentiment(tweets: pd.DataFrame) -> pd.DataFrame:
	analyzer = SentimentIntensityAnalyzer()
	tweets = tweets.copy()
	tweets["compound"] = tweets["text"].apply(lambda t: analyzer.polarity_scores(t)["compound"])
	# Daily aggregates
	daily = tweets.groupby("date").agg(
		avg_sentiment=("compound", "mean"),
		tweet_volume=("compound", "size"),
		sentiment_volatility=("compound", "std"),
	).reset_index()
	daily["sentiment_volatility"].fillna(0.0, inplace=True)
	return daily


def simulate_prices(start_date: datetime, num_days: int, initial_price: float, volatility: float, drift: float = 0.0) -> pd.DataFrame:
	dates = [start_date.date() + timedelta(days=i) for i in range(num_days)]
	prices = [initial_price]
	for _ in range(1, num_days):
		# Geometric random walk
		r = np.random.normal(loc=drift, scale=volatility)
		prices.append(prices[-1] * (1.0 + r))
	return pd.DataFrame({"date": dates, "close": prices})


def build_strategy_features(daily_sentiment: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
	data = prices.merge(daily_sentiment, on="date", how="left").sort_values("date")
	# Fill any missing sentiment (should be minimal with synthetic generation)
	data[["avg_sentiment", "tweet_volume", "sentiment_volatility"]] = data[["avg_sentiment", "tweet_volume", "sentiment_volatility"]].fillna(method="ffill").fillna(0.0)
	# Rolling signals
	data["sentiment_ma_3"] = data["avg_sentiment"].rolling(3, min_periods=1).mean()
	data["sentiment_z"] = (data["avg_sentiment"] - data["avg_sentiment"].rolling(7, min_periods=2).mean()) / (data["avg_sentiment"].rolling(7, min_periods=2).std().replace(0, np.nan))
	data["sentiment_z"].fillna(0.0, inplace=True)
	# Simple rule: long when sentiment_z > 0, flat otherwise
	data["position"] = (data["sentiment_z"] > 0).astype(int)
	# Compute returns
	data["ret"] = data["close"].pct_change().fillna(0.0)
	data["strategy_ret"] = data["position"].shift(1).fillna(0) * data["ret"]
	data["equity_curve"] = (1.0 + data["strategy_ret"]).cumprod()
	data["buy_hold_curve"] = (1.0 + data["ret"]).cumprod()
	return data


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def save_artifacts(data: pd.DataFrame, cfg: DemoConfig) -> None:
	ensure_dir(cfg.artifact_dir)
	csv_path = os.path.join(cfg.artifact_dir, "demo_results.csv")
	plot_path = os.path.join(cfg.artifact_dir, "equity_curves.png")
	data.to_csv(csv_path, index=False)
	plt.figure(figsize=(10, 5))
	plt.plot(data["date"], data["equity_curve"], label="Strategy")
	plt.plot(data["date"], data["buy_hold_curve"], label="Buy & Hold")
	plt.title("Equity Curves: Sentiment-Z Long vs Buy & Hold")
	plt.xlabel("Date")
	plt.ylabel("Cumulative Return (x)")
	plt.legend()
	plt.tight_layout()
	plt.savefig(plot_path, dpi=150)
	plt.close()


def summarize_performance(data: pd.DataFrame) -> pd.DataFrame:
	def sharpe(returns: pd.Series, risk_free_daily: float = 0.0) -> float:
		if returns.std() == 0:
			return 0.0
		return np.sqrt(252) * (returns.mean() - risk_free_daily) / returns.std()

	summary = pd.DataFrame(
		{
			"metric": [
				"Total Return (Strategy)",
				"Total Return (Buy&Hold)",
				"Ann. Sharpe (Strategy)",
				"Ann. Sharpe (Buy&Hold)",
			],
			"value": [
				data["equity_curve"].iloc[-1] - 1.0,
				data["buy_hold_curve"].iloc[-1] - 1.0,
				sharpe(data["strategy_ret"]),
				sharpe(data["ret"]),
			],
		}
	)
	return summary


def main() -> None:
	cfg = DemoConfig()
	set_seed(cfg.seed)
	start_date = datetime.utcnow().date() - timedelta(days=cfg.num_days)
	start_dt = datetime.combine(start_date, datetime.min.time())

	tweets = generate_synthetic_tweets(start_dt, cfg.num_days, cfg.tweets_per_day)
	daily_sentiment = score_sentiment(tweets)
	prices = simulate_prices(start_dt, cfg.num_days, cfg.initial_price, cfg.volatility)
	data = build_strategy_features(daily_sentiment, prices)
	save_artifacts(data, cfg)
	summary = summarize_performance(data)
	print("=== Demo Summary ===")
	print(summary.to_string(index=False))
	print(f"Artifacts saved to: {cfg.artifact_dir}")


if __name__ == "__main__":
	main()