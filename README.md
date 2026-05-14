# Indian Stock Trading Dashboard + DRL Experiments

## What this project now includes

This repository started as a notebook-based deep reinforcement learning stock trading experiment.
It now includes two runnable parts:

1. `main.ipynb`
   Historical DRL experimentation with PPO, A2C, DDPG, and an ensemble agent.

2. `server.py` + `web/`
   A browser dashboard focused on the Indian stock market with:
   - live quote tracking through Yahoo Finance
   - interactive price and volume charts
   - Indian stock watchlist support
   - buy, sell, or hold suggestions
   - short-term price forecasting based on recent price behaviour and technical indicators

## Indian market features

The dashboard is preconfigured with popular Indian stocks such as:

- `RELIANCE.NS`
- `TCS.NS`
- `INFY.NS`
- `HDFCBANK.NS`
- `ICICIBANK.NS`
- `SBIN.NS`
- `ITC.NS`
- `LT.NS`
- `BHARTIARTL.NS`
- `ASIANPAINT.NS`

It also shows market index snapshots for:

- Nifty 50
- Sensex
- Bank Nifty

## Files to run

- [dashboard_app.py](C:/Users/HP/Documents/Codex/2026-02-files-mentioned-by-the-user-deep/Deep-Reinforcement-Learning-with-Stock-Trading-main/dashboard_app.py)
- [server.py](C:/Users/HP/Documents/Codex/2026-02-files-mentioned-by-the-user-deep/Deep-Reinforcement-Learning-with-Stock-Trading-main/server.py)
- [run_dashboard.bat](C:/Users/HP/Documents/Codex/2026-02-files-mentioned-by-the-user-deep/Deep-Reinforcement-Learning-with-Stock-Trading-main/run_dashboard.bat)
- [run_dashboard.ps1](C:/Users/HP/Documents/Codex/2026-02-files-mentioned-by-the-user-deep/Deep-Reinforcement-Learning-with-Stock-Trading-main/run_dashboard.ps1)
- [run_project.py](C:/Users/HP/Documents/Codex/2026-02-files-mentioned-by-the-user-deep/Deep-Reinforcement-Learning-with-Stock-Trading-main/run_project.py)

## How to run the dashboard

### Option 1: Windows batch file

From the project folder:

```bat
run_dashboard.bat
```

### Option 2: PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File .\run_dashboard.ps1
```

### Option 3: Python directly

```powershell
C:\Users\HP\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe server.py
```

When the app starts, open:

- `http://127.0.0.1:8765`

## How to run the DRL backtest pipeline

```powershell
C:\Users\HP\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe run_project.py --timesteps 10000 --tests 1000
```

The DRL run writes output files under:

- [run_outputs](C:/Users/HP/Documents/Codex/2026-02-files-mentioned-by-the-user-deep/Deep-Reinforcement-Learning-with-Stock-Trading-main/run_outputs)

## Notes

- The dashboard uses live Yahoo Finance data through a lightweight Python proxy, so it works best with an internet connection.
- If a live fetch fails, the app will try to reuse the latest cached data in `live_cache`.
- The buy, sell, and hold recommendation is a model-assisted signal based on recent price fluctuation, RSI, MACD, EMA trend, and a lightweight short-term forecast.
- These suggestions are for educational purposes and are not financial advice.

## Research reference

The original notebook work was inspired by the paper:

- [Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996)

