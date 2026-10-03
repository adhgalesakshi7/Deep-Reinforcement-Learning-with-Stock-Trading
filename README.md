# Indian Stock Trading Dashboard + DRL Experiments

Indian stock trading project with two parts:

- `server.py` provides a lightweight dashboard for live quote lookup, charts, and model-assisted buy/sell/hold signals.
- `run_project.py` runs the reinforcement-learning backtest across PPO, A2C, DDPG, and an ensemble agent.

## Project Files

- [server.py](server.py)
- [run_project.py](run_project.py)
- [run_dashboard.bat](run_dashboard.bat)
- [run_dashboard.ps1](run_dashboard.ps1)
- [project_evidence.md](project_evidence.md)

## Data-backed Evidence

- 29 usable ticker CSVs
- 80,255 raw stock rows
- 1 empty ticker file: `UTX.csv`
- Training, validation, and test runs all rank DDPG first by Sharpe ratio in the saved metrics files
- Training times: PPO 64.38s, A2C 84.95s, DDPG 332.21s

See [project_evidence.md](project_evidence.md) for the full summary.

## Run

Dashboard:

```powershell
run_dashboard.bat
```

Backtest pipeline:

```powershell
python run_project.py --timesteps 10000 --tests 1000
```

## Notes

- The dashboard uses Yahoo Finance data and works best with an internet connection.
- If live fetches fail, the app reuses cached data from `live_cache`.
- The signals are for educational use only and are not financial advice.

