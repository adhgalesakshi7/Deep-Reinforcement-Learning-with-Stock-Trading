# Trading Project Evidence

## Data Coverage
- Tickers with usable CSVs: 29
- Total raw stock rows: 80,255
- Empty ticker file: UTX.csv
- Train/validation/test splits are aligned across the 29 usable tickers

## Evaluation Highlights
- Training: DDPG Agent led with Sharpe 6.60, final net worth 2023.73, and total return 102.37%
- Validation: DDPG Agent led with Sharpe 22.66, final net worth 1136.79, and total return 13.68%
- Test: DDPG Agent led with Sharpe 10.21, final net worth 1236.62, and total return 23.66%

## Training Times
- PPO Agent: 64.38s
- A2C Agent: 84.95s
- DDPG Agent: 332.21s

## What The Project Shows
- Multi-asset reinforcement learning on 30 large-cap U.S. equities.
- Technical indicators: RSI, MACD, signal line, CCI, and ADX.
- Comparison of PPO, A2C, DDPG, and an ensemble agent across train, validation, and test splits.
