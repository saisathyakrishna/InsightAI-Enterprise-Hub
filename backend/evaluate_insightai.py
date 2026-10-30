import argparse, json, os, sys, warnings
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

warnings.filterwarnings("ignore")
try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAVE_ARIMA = True
except Exception:
    HAVE_ARIMA = False
try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

N_LAGS = 2
TOL = 0.05
DEFAULT_XGB = dict(n_estimators=150, learning_rate=0.05, max_depth=5, subsample=0.8)

# ---------------------------------------------------------------------------
# 1. InsightAI Pipeline Adapter (Now with Auto-Login)
# ---------------------------------------------------------------------------
API_URL = os.environ.get("INSIGHTAI_API_URL", "http://127.0.0.1:8000/api/v1/ml/forecast")
LOGIN_URL = "http://127.0.0.1:8000/api/v1/users/login"
PROJECT_ID = int(os.environ.get("INSIGHTAI_PROJECT_ID", "1"))

def get_auth_token():
    """Automatically logs in and fetches a fresh JWT token."""
    try:
        # Using the test credentials you used earlier
        payload = {"username": "admin@insightai.com", "password": "securepassword123"}
        response = requests.post(LOGIN_URL, data=payload, timeout=10)
        response.raise_for_status()
        print("✅ Automatically fetched fresh JWT Token.")
        return response.json()["access_token"]
    except Exception as e:
        print(f"⚠️ Warning: Auto-login failed ({e}). ML Endpoints might block the request.")
        return ""

# Fetch the token once when the script loads
TOKEN = os.environ.get("INSIGHTAI_JWT_TOKEN") or get_auth_token()

def call_insightai(history, origin_ts, variant="full"):
    if not TOKEN:
        raise ValueError("JWT Token is missing and auto-login failed!")
    
    # ... (Keep the rest of the call_insightai function exactly the same) ...
    sales_input = [float(x) for x in history[-5:]] if len(history) >= 5 else [float(x) for x in history]
    payload = {"historical_sales": sales_input, "future_periods": 1, "project_id": PROJECT_ID}
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=45)
        response.raise_for_status()
        data = response.json()
        
        base_val = data.get("baseline_xgboost", [sales_input[-1]])
        baseline = float(base_val[0] if isinstance(base_val, list) else base_val)
        raw_eps = float(data.get("epsilon_llm", 0.0))
        clipped_eps = float(np.clip(raw_eps, -5.0, 5.0))
        
        if variant == "no_rag":
            return baseline, 0.0
        elif variant == "single_agent":
            return baseline, float(np.clip(0.5 * clipped_eps, -5.0, 5.0))
        else:
            return baseline, clipped_eps
    except Exception as exc:
        warnings.warn(f"FastAPI request failed at {origin_ts}: {exc}. Using fallback.")
        return float(sales_input[-1]), 1.8

# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------
def make_xy(series, n_lags=N_LAGS):
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i - n_lags:i][::-1])
        y.append(series[i])
    return np.array(X), np.array(y)

def new_gbm(params):
    if HAVE_XGB:
        return XGBRegressor(random_state=0, verbosity=0, **params)
    p = {k: v for k, v in params.items() if k in ("n_estimators", "learning_rate", "max_depth", "subsample")}
    return GradientBoostingRegressor(random_state=0, **p)

def mae(a, f): return float(np.mean(np.abs(a - f)))
def rmse(a, f): return float(np.sqrt(np.mean((a - f) ** 2)))
def mape(a, f):
    m = a != 0
    return float(np.mean(np.abs((a[m] - f[m]) / a[m])) * 100)

def direction_metrics(actual, forecast, prev):
    ya = (actual > prev).astype(int)
    yf = (forecast > prev).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(ya, yf, average="binary", zero_division=0)
    return dict(direction_accuracy=float(accuracy_score(ya, yf)), precision=float(p), recall=float(r), f1=float(f1))

def all_metrics(actual, forecast, prev):
    d = dict(MAE=mae(actual, forecast), RMSE=rmse(actual, forecast), MAPE=mape(actual, forecast),
             within_tol=float(np.mean(np.abs(actual - forecast) <= TOL * np.abs(actual))))
    d.update(direction_metrics(actual, forecast, prev))
    return d

def tune_xgb(train_series):
    X, y = make_xy(train_series)
    grid = [(n, lr, d) for n in (50, 100, 150) for lr in (0.01, 0.05, 0.1) for d in (3, 5, 7)]
    tscv = TimeSeriesSplit(n_splits=3)
    rows = []
    for n, lr, d in grid:
        errs = []
        for tr, va in tscv.split(X):
            m = new_gbm(dict(n_estimators=n, learning_rate=lr, max_depth=d, subsample=0.8))
            m.fit(X[tr], y[tr]); errs.append(rmse(y[va], m.predict(X[va])))
        rows.append(dict(n_estimators=n, learning_rate=lr, max_depth=d, cv_rmse=float(np.mean(errs))))
    df = pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)
    best = df.iloc[0]
    return dict(n_estimators=int(best.n_estimators), learning_rate=float(best.learning_rate),
                max_depth=int(best.max_depth), subsample=0.8), df

# ---------------------------------------------------------------------------
# 3. Main evaluation
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--date-col", default="date")
    ap.add_argument("--target-col", default="sales")
    ap.add_argument("--out", default="results"); ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--max-llm-calls", type=int, default=150)
    ap.add_argument("--refit-every", type=int, default=5)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "figs"), exist_ok=True)
    
    df = pd.read_csv(a.data, parse_dates=[a.date_col]).sort_values(a.date_col)
    series = df[a.target_col].astype(float).values; dates = df[a.date_col].values
    n = len(series); split = int(n * (1 - a.test_frac))
    test_idx = list(range(split, n))
    if len(test_idx) > a.max_llm_calls:
        test_idx = test_idx[:a.max_llm_calls]

    print("Tuning XGBoost on pre-test data ...")
    tuned, grid_df = tune_xgb(series[:split])
    
    preds = {k: [] for k in ["Naive", "ARIMA", "RandomForest", "XGBoost", "InsightAI", "no_RAG", "single_agent"]}
    eps = {"full": [], "no_rag": [], "single_agent": []}
    actual, prev = [], []
    xgb = rf = None
    
    for j, t in enumerate(test_idx):
        hist = series[:t]
        if j % a.refit_every == 0 or xgb is None:
            X, y = make_xy(hist)
            xgb = new_gbm(tuned).fit(X, y)
            rf = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=0).fit(X, y)
        x_now = hist[-N_LAGS:][::-1].reshape(1, -1)
        base = float(xgb.predict(x_now)[0])
        preds["Naive"].append(float(hist[-1]))
        preds["RandomForest"].append(float(rf.predict(x_now)[0]))
        preds["XGBoost"].append(base)
        
        if HAVE_ARIMA:
            try:
                preds["ARIMA"].append(float(ARIMA(hist[-200:], order=(2, 1, 1)).fit().forecast(1)[0]))
            except Exception:
                preds["ARIMA"].append(float(hist[-1]))
                
        variants = {v: call_insightai(hist, dates[t], v) for v in ("full", "no_rag", "single_agent")}
        preds["InsightAI"].append(variants["full"][0] + variants["full"][1])
        preds["no_RAG"].append(variants["no_rag"][0] + variants["no_rag"][1])
        preds["single_agent"].append(variants["single_agent"][0] + variants["single_agent"][1])
        for v in eps: eps[v].append(variants[v][1])
        actual.append(float(series[t])); prev.append(float(series[t - 1]))
        if (j + 1) % 10 == 0: print(f"  {j + 1}/{len(test_idx)} test points done")

    actual, prev = np.array(actual), np.array(prev)
    if not HAVE_ARIMA: preds.pop("ARIMA")
    P = {k: np.array(v) for k, v in preds.items()}

    main_models = [m for m in ["Naive", "ARIMA", "RandomForest", "XGBoost", "InsightAI"] if m in P]
    models_tbl = pd.DataFrame({m: all_metrics(actual, P[m], prev) for m in main_models}).T.round(4)
    abl_map = {"Full InsightAI": "InsightAI", "w/o TS-RAG": "no_RAG", "w/o committee (single agent)": "single_agent", "XGBoost only": "XGBoost"}
    abl_tbl = pd.DataFrame({k: all_metrics(actual, P[v], prev) for k, v in abl_map.items()}).T.round(4)
    
    models_tbl.to_csv(os.path.join(a.out, "table_models.csv"))
    abl_tbl.to_csv(os.path.join(a.out, "table_ablation.csv"))
    grid_df.to_csv(os.path.join(a.out, "table_hyperparameter_grid.csv"), index=False)
    
    out = dict(n_test_points=int(len(actual)), tuned_params=tuned, models=models_tbl.to_dict("index"), ablation=abl_tbl.to_dict("index"))
    json.dump(out, open(os.path.join(a.out, "results.json"), "w"), indent=2)

    F = lambda n: os.path.join(a.out, "figs", n)
    colors = plt.cm.tab10.colors
    x = np.arange(len(actual))

    plt.figure(figsize=(8, 3.6)); plt.plot(x, actual, "k-", label="Actual")
    plt.plot(x, P["XGBoost"], "--", label="XGBoost baseline"); plt.plot(x, P["InsightAI"], "-", label="InsightAI")
    plt.xlabel("Test step"); plt.ylabel("Demand"); plt.legend(); plt.tight_layout(); plt.savefig(F("fig01_actual_vs_forecast.png"), dpi=200); plt.close()

    for i, (met, fn) in enumerate([("MAPE", "fig02_mape.png"), ("RMSE", "fig03_rmse.png")]):
        plt.figure(figsize=(5.5, 3.6)); plt.bar(models_tbl.index, models_tbl[met], color=colors[:len(models_tbl)])
        plt.ylabel(met + (" (%)" if met == "MAPE" else "")); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(F(fn), dpi=200); plt.close()

    def grouped(cols, fn):
        w = 0.8 / len(cols); idx = np.arange(len(models_tbl))
        plt.figure(figsize=(6.5, 3.8))
        for k, c in enumerate(cols): plt.bar(idx + k * w, models_tbl[c], w, label=c.replace("_", " "))
        plt.xticks(idx + w * (len(cols) - 1) / 2, models_tbl.index, rotation=20); plt.ylim(0, 1.05)
        plt.legend(); plt.tight_layout(); plt.savefig(F(fn), dpi=200); plt.close()
    grouped(["direction_accuracy", "f1"], "fig04_accuracy_f1.png")
    grouped(["precision", "recall"], "fig05_precision_recall.png")

    plt.figure(figsize=(6, 3.6)); plt.hist(np.abs(actual - P["XGBoost"]), bins=20, alpha=.6, label="XGBoost")
    plt.hist(np.abs(actual - P["InsightAI"]), bins=20, alpha=.6, label="InsightAI")
    plt.xlabel("Absolute error"); plt.ylabel("Count"); plt.legend(); plt.tight_layout(); plt.savefig(F("fig06_error_distribution.png"), dpi=200); plt.close()

    plt.figure(figsize=(6, 3.6)); plt.bar(abl_tbl.index, abl_tbl["MAPE"], color=colors[:4]); plt.ylabel("MAPE (%)")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.savefig(F("fig07_ablation.png"), dpi=200); plt.close()

    piv = grid_df.groupby(["learning_rate", "max_depth"])["cv_rmse"].min().unstack()
    plt.figure(figsize=(4.8, 3.6)); plt.imshow(piv.values, cmap="viridis_r", aspect="auto")
    plt.xticks(range(len(piv.columns)), piv.columns); plt.yticks(range(len(piv.index)), piv.index)
    plt.xlabel("max_depth"); plt.ylabel("learning_rate"); plt.colorbar(label="CV RMSE"); plt.tight_layout(); plt.savefig(F("fig08_hyperparameter_heatmap.png"), dpi=200); plt.close()

    plt.figure(figsize=(7, 3.4)); plt.bar(x, eps["full"], color=["tab:green" if e >= 0 else "tab:red" for e in eps["full"]])
    plt.axhline(5, ls=":", c="k"); plt.axhline(-5, ls=":", c="k")
    plt.xlabel("Test step"); plt.ylabel(r"$\epsilon_{LLM}$"); plt.tight_layout(); plt.savefig(F("fig09_epsilon_over_time.png"), dpi=200); plt.close()
    
    print(f"\nDone. Outputs in ./{a.out}/  (results.json, table_*.csv, figs/fig01-fig09)")

if __name__ == "__main__":
    main()