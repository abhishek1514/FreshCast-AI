# Run FreshCast with real-world data on Windows

## 1. Requirements

- Python 3.11 recommended (Python 3.12 is also allowed by `pyproject.toml`)
- Internet connection for the first dataset download
- Several GB of free disk space for the environment and dataset cache

## 2. Open PowerShell in this project folder

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

If PowerShell blocks virtual-environment activation, run this for the current terminal only and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Download and prepare real data

```powershell
python scripts\prepare_real_data.py
```

This downloads the public FreshRetailNet-50K dataset from Hugging Face, then writes a small selected panel to `data\raw\foodservice_daily_sales.csv`. The raw source is not embedded in this ZIP because it is third-party data; the script downloads it on your machine.

## 4. Train and evaluate

```powershell
python scripts\train_real.py
```

Wait for it to finish. It writes the trained model to `models\champion_model.joblib`, evaluation results to `data\processed\`, and logs the temporal holdout metrics in the terminal.

## 5. Launch the dashboard

```powershell
streamlit run app\streamlit_app.py
```

For the API, open another PowerShell terminal, activate `.venv`, then run:

```powershell
uvicorn freshcast.api.app:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8501` for the dashboard and `http://localhost:8000/docs` for the API docs.

## 6. Important data interpretation

FreshRetailNet-50K `sale_amount` is a normalized sales target, not physical cases. Actual supplier costs, selling prices, shelf lives, and lead times are not supplied; the demo uses documented illustrative assumptions for inventory calculations. Do not present those estimated dollar losses as actual company savings. The real-data mode does not promise the synthetic README's WAPE or cost-reduction figures.
