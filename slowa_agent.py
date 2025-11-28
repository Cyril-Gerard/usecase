#!/usr/bin/env python3
"""
SLO Degradation Early-Warning Agent (SLOWA)

Single-file implementation that:
 - Ingests metrics CSVs from metrics/*.csv
 - Loads SLO targets from slo_targets.yml and topology.json
 - Detects drift (simple z-score + percent change)
 - Forecasts short-term trend using linear regression
 - Estimates risk_score and time-to-breach
 - Produces mitigation plan (rule-based + optional Ollama LLM)
 - Writes outputs: risk JSON, mitigation text, and forecast chart PNG

Usage:
  python slowa_agent.py --metrics-dir ./metrics --slo-file ./slo_targets.yml --topology ./topology.json --out-dir ./out

Guardrails:
 - No remote actions taken. Mitigation suggestions are informational only.
 - Human approval required before any operational change.
"""
import os
import glob
import json
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from pydantic import BaseModel, Field, ValidationError
from sklearn.linear_model import LinearRegression
import requests  # optional: for local Ollama LLM (if available)

# -----------------------------
# Pydantic schemas for configs
# -----------------------------
class SLOTarget(BaseModel):
    name: str
    metric: str
    threshold: float
    direction: str = Field(..., regex="^(<=|>=)$")  # <= for latency, >= for availability/cpu? clarify accordingly
    window_minutes: int = 5  # sliding window to check SLO

class SLOTargetsFile(BaseModel):
    slos: List[SLOTarget]

class TopologyNode(BaseModel):
    id: str
    role: str
    tags: Optional[Dict[str, str]] = {}

class TopologyFile(BaseModel):
    nodes: List[TopologyNode]

# -----------------------------
# Helper utilities
# -----------------------------
def safe_load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def read_metrics_csv(path: str) -> pd.DataFrame:
    """
    Expect CSV have at least: timestamp, metric_name, value, host (optional)
    timestamp expected ISO or epoch ms
    """
    df = pd.read_csv(path)
    # normalize timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    elif "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    else:
        raise ValueError(f"No timestamp column in {path}")
    return df

# -----------------------------
# Ingestor
# -----------------------------
class MetricIngestor:
    def __init__(self, metrics_dir: str):
        self.metrics_dir = metrics_dir

    def load_all(self) -> pd.DataFrame:
        files = sorted(glob.glob(os.path.join(self.metrics_dir, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSVs found in {self.metrics_dir}")
        frames = []
        for f in files:
            df = read_metrics_csv(f)
            frames.append(df)
        all_df = pd.concat(frames, ignore_index=True)
        # ensure correct types
        all_df = all_df.sort_values("timestamp")
        return all_df

# -----------------------------
# Drift Detector
# -----------------------------
@dataclass
class DriftResult:
    metric: str
    baseline_mean: float
    recent_mean: float
    z_score: float
    pct_change: float
    drift_detected: bool

class DriftDetector:
    """
    Simple drift using percent change + z-score on recent window vs baseline.
    baseline_window: hours of historical baseline
    recent_window: minutes of recent data to check
    """
    def __init__(self, baseline_hours: int = 24, recent_minutes: int = 5, z_thresh: float = 2.0, pct_thresh: float = 0.15):
        self.baseline_hours = baseline_hours
        self.recent_minutes = recent_minutes
        self.z_thresh = z_thresh
        self.pct_thresh = pct_thresh

    def detect(self, df: pd.DataFrame, metric: str, now: Optional[pd.Timestamp] = None) -> DriftResult:
        if now is None:
            now = pd.Timestamp.utcnow()

        metric_df = df[df["metric_name"] == metric] if "metric_name" in df.columns else df[df["metric"] == metric]
        if metric_df.empty:
            raise ValueError(f"No data for metric {metric}")

        # baseline: older than baseline_hours
        cutoff_baseline = now - pd.Timedelta(hours=self.baseline_hours)
        baseline = metric_df[metric_df["timestamp"] < cutoff_baseline]["value"].dropna()
        recent_cut = now - pd.Timedelta(minutes=self.recent_minutes)
        recent = metric_df[metric_df["timestamp"] >= recent_cut]["value"].dropna()

        # fallback if baseline empty: use earlier segment
        if baseline.empty:
            baseline = metric_df[metric_df["timestamp"] < now - pd.Timedelta(minutes=self.recent_minutes * 2)]["value"].dropna()

        baseline_mean = float(baseline.mean()) if not baseline.empty else float(metric_df["value"].mean())
        recent_mean = float(recent.mean()) if not recent.empty else baseline_mean
        baseline_std = float(baseline.std(ddof=0)) if len(baseline) > 0 else 1.0
        z = (recent_mean - baseline_mean) / (baseline_std if baseline_std != 0 else 1.0)
        pct_change = (recent_mean - baseline_mean) / (baseline_mean if baseline_mean != 0 else 1.0)

        drifted = (abs(z) >= self.z_thresh) or (abs(pct_change) >= self.pct_thresh)

        return DriftResult(
            metric=metric,
            baseline_mean=baseline_mean,
            recent_mean=recent_mean,
            z_score=z,
            pct_change=pct_change,
            drift_detected=drifted
        )

# -----------------------------
# Forecaster
# -----------------------------
@dataclass
class ForecastResult:
    metric: str
    forecast_times: List[pd.Timestamp]
    forecast_values: List[float]
    predicted_breach_time: Optional[pd.Timestamp]
    breach_confidence: float  # 0-1

class Forecaster:
    """
    Simple linear regression forecast on recent window.
    horizon_minutes: how many minutes ahead to project
    """
    def __init__(self, lookback_minutes: int = 60, horizon_minutes: int = 30):
        self.lookback_minutes = lookback_minutes
        self.horizon_minutes = horizon_minutes
        self.model = LinearRegression()

    def forecast(self, df: pd.DataFrame, metric: str, slo_threshold: float, direction: str) -> ForecastResult:
        # select recent window
        now = pd.Timestamp.utcnow()
        window_start = now - pd.Timedelta(minutes=self.lookback_minutes)
        metric_df = df[(df["metric_name"] == metric) & (df["timestamp"] >= window_start)].copy()
        if metric_df.empty:
            # fallback to entire df for that metric
            metric_df = df[df["metric_name"] == metric].copy()

        metric_df = metric_df.dropna(subset=["value"])
        # create numeric time (minutes since epoch)
        metric_df["tmin"] = metric_df["timestamp"].astype("int64") // 10**9 / 60.0
        X = metric_df[["tmin"]].values
        y = metric_df["value"].values

        if len(y) < 3:
            # not enough data -> flat forecast
            forecast_times = [now + pd.Timedelta(minutes=i) for i in range(1, self.horizon_minutes + 1)]
            forecast_values = [float(y[-1]) if len(y) > 0 else 0.0] * len(forecast_times)
            return ForecastResult(metric, forecast_times, forecast_values, None, 0.0)

        # fit linear regression
        self.model.fit(X, y)
        # prepare future times
        future_times = [now + pd.Timedelta(minutes=i) for i in range(1, self.horizon_minutes + 1)]
        future_tmin = np.array([[t.astype("int64") // 10**9 / 60.0] for t in pd.to_datetime(future_times)])
        preds = self.model.predict(future_tmin).tolist()

        # estimate if/when breach occurs depending on direction
        predicted_breach_time = None
        breach_confidence = 0.0
        for t, val in zip(future_times, preds):
            breach = False
            if direction == "<=":
                # metric must be <= threshold; breach when value > threshold
                if val > slo_threshold:
                    breach = True
            else:
                # direction == ">=" -> breach when val < threshold
                if val < slo_threshold:
                    breach = True
            if breach:
                predicted_breach_time = t
                break

        # simple confidence: absolute slope magnitude normalized by recent std
        slope = float(self.model.coef_[0])
        recent_std = float(np.std(y)) if np.std(y) > 0 else 1.0
        conf = min(1.0, min(abs(slope) / (recent_std + 1e-6), 1.0))

        return ForecastResult(metric, future_times, preds, predicted_breach_time, conf)

# -----------------------------
# Risk Scorer
# -----------------------------
def compute_risk_score(drift: DriftResult, forecast: ForecastResult) -> float:
    """
    Composite risk in [0,1]. Higher => more urgent.
    Components:
     - drift contribution (abs(z) normalized)
     - pct change contribution
     - forecast breach proximity and confidence
    """
    z_factor = min(1.0, abs(drift.z_score) / 5.0)  # z around 5 is saturating
    pct_factor = min(1.0, abs(drift.pct_change) / 1.0)  # 100% change saturates
    forecast_factor = 0.0
    if forecast.predicted_breach_time:
        minutes_to_breach = max(0.0, (forecast.predicted_breach_time - pd.Timestamp.utcnow()).total_seconds() / 60.0)
        proximity = 1.0 / (1.0 + minutes_to_breach)  # nearer -> higher
        forecast_factor = proximity * forecast.breach_confidence
    risk = 0.5 * z_factor + 0.2 * pct_factor + 0.3 * forecast_factor
    return float(max(0.0, min(1.0, risk)))

# -----------------------------
# Mitigation Planner
# -----------------------------
class MitigationPlanner:
    def __init__(self, topology: TopologyFile, use_ollama: bool = False, ollama_url: str = "http://localhost:11434/api/generate"):
        self.topology = topology
        self.use_ollama = use_ollama
        self.ollama_url = ollama_url

    def rule_based(self, metric: str, drift: DriftResult, forecast: ForecastResult, slo: SLOTarget) -> List[str]:
        """
        Return a prioritized list of suggested mitigations (text).
        Rules are conservative: suggestions, no automatic actions.
        """
        suggestions = []
        # Generic suggestions based on metric pattern
        m = metric.lower()
        if "cpu" in m:
            suggestions += [
                "Investigate top CPU-consuming processes on affected hosts (ps/top).",
                "Check recent deploys or cron jobs; consider rolling back if correlated.",
                "Consider temporarily scaling out (add replicas) after manual approval.",
                "If safe, move batch jobs to off-peak window or throttle them."
            ]
        if "memory" in m or "mem" in m:
            suggestions += [
                "Check for memory leaks or OOM events; inspect application logs.",
                "Restart services gracefully after approval; prefer process-level restarts.",
                "Increase swap only as temporary mitigation; prefer adding memory or scaling."
            ]
        if "latency" in m or "response" in m or "p999" in m:
            suggestions += [
                "Identify slow endpoints using traces; increase request sampling.",
                "Scale application horizontally if autoscaling is possible (manual approval).",
                "Throttle incoming traffic, apply circuit-breaker, or route to healthy regions."
            ]
        if "error" in m or "5xx" in m:
            suggestions += [
                "Roll back recent releases that increased error rates.",
                "Redirect traffic to healthy replicas; remove suspect hosts from LB after FRR.",
                "Inspect logs and alerting traces for stack traces/root causes."
            ]
        # Add SLO-centric suggestions
        oc = f"SLO threshold {slo.threshold} ({slo.direction})"
        suggestions.append(f"Review SLO: {slo.name} ({oc}). Prepare on-call runbook and notify owners.")
        # Add forecast-specific
        if forecast.predicted_breach_time:
            suggestions.insert(0, f"Predicted breach at approx {forecast.predicted_breach_time.isoformat()}; escalate to on-call.")
        else:
            suggestions.insert(0, "No immediate timestamped breach predicted within horizon; continue monitoring.")
        # Deduplicate
        deduped = []
        for s in suggestions:
            if s not in deduped:
                deduped.append(s)
        return deduped

    def ollama_mitigations(self, metric: str, drift: DriftResult, forecast: ForecastResult, slo: SLOTarget) -> Optional[str]:
        """
        Call local Ollama (if available) with a prompt to generate a mitigation plan.
        Ollama must be running locally and accessible. This function is optional and will fail gracefully.
        """
        if not self.use_ollama:
            return None

        prompt = (
            f"You are a SRE assistant. Metric: {metric}\n"
            f"SLO: {slo.name} threshold={slo.threshold} direction={slo.direction}\n"
            f"Drift: baseline_mean={drift.baseline_mean:.3f}, recent_mean={drift.recent_mean:.3f}, "
            f"z={drift.z_score:.2f}, pct_change={drift.pct_change:.2%}\n"
            f"Forecast breach_time={forecast.predicted_breach_time}, breach_confidence={forecast.breach_confidence:.2f}\n\n"
            "Provide a short, prioritized mitigation plan (3 items), each with rationale and risk. "
            "Do NOT take any action; only suggest human-reviewed steps."
        )
        payload = {"model": "illama3.1", "prompt": prompt, "max_tokens": 400}

        try:
            resp = requests.post(self.ollama_url, json=payload, timeout=5.0)
            if resp.status_code == 200:
                return resp.text if resp.text else resp.json().get("text", None)
            else:
                return None
        except Exception as e:
            # Ollama not reachable or failed — return None so we fall back to rule_based
            return None

    def plan(self, metric: str, drift: DriftResult, forecast: ForecastResult, slo: SLOTarget) -> Dict:
        rb = self.rule_based(metric, drift, forecast, slo)
        ollama_text = None
        if self.use_ollama:
            ollama_text = self.ollama_mitigations(metric, drift, forecast, slo)
        return {"rule_based": rb, "ollama": ollama_text}

# -----------------------------
# Reporter (chart + JSON)
# -----------------------------
def plot_forecast(metric: str, df: pd.DataFrame, forecast: ForecastResult, out_path: str):
    plt.figure(figsize=(10, 4))
    # historical (last lookback)
    history = df[df["metric_name"] == metric].copy()
    if not history.empty:
        plt.plot(history["timestamp"], history["value"], label="history")
    plt.plot(forecast.forecast_times, forecast.forecast_values, label="forecast", linestyle="--")
    if forecast.predicted_breach_time:
        plt.axvline(forecast.predicted_breach_time, color="red", linestyle=":", label="predicted breach")
    plt.title(f"Forecast for {metric}")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# -----------------------------
# Main agent runner
# -----------------------------
def run_slowa(metrics_dir: str, slo_file: str, topology_file: str, out_dir: str, use_ollama: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    # load configs
    slo_raw = safe_load_yaml(slo_file)
    try:
        slo_cfg = SLOTargetsFile(**slo_raw)
    except ValidationError as e:
        raise RuntimeError(f"SLO config validation failed:\n{e}")

    topo_raw = safe_load_yaml(topology_file)
    try:
        topo = TopologyFile(**topo_raw)
    except ValidationError as e:
        raise RuntimeError(f"Topology config validation failed:\n{e}")

    ingestor = MetricIngestor(metrics_dir)
    df = ingestor.load_all()

    drift_detector = DriftDetector()
    forecaster = Forecaster()
    planner = MitigationPlanner(topo, use_ollama=use_ollama)

    now = pd.Timestamp.utcnow()
    results = []
    for slo in slo_cfg.slos:
        metric_name = slo.metric
        # compute drift
        drift = drift_detector.detect(df, metric_name, now=now)
        # forecast
        forecast = forecaster.forecast(df, metric_name, slo.threshold, slo.direction)
        # risk
        risk = compute_risk_score(drift, forecast)
        # mitigation plan
        plan = planner.plan(metric_name, drift, forecast, slo)

        # output artifacts
        safe_metric = metric_name.replace("/", "_").replace(" ", "_")
        chart_path = os.path.join(out_dir, f"forecast_{safe_metric}.png")
        plot_forecast(metric_name, df, forecast, chart_path)

        result = {
            "metric": metric_name,
            "slo": slo.dict(),
            "drift": {
                "baseline_mean": drift.baseline_mean,
                "recent_mean": drift.recent_mean,
                "z_score": drift.z_score,
                "pct_change": drift.pct_change,
                "drift_detected": drift.drift_detected,
            },
            "forecast": {
                "predicted_breach_time": forecast.predicted_breach_time.isoformat() if forecast.predicted_breach_time else None,
                "breach_confidence": forecast.breach_confidence,
                "horizon_minutes": forecaster.horizon_minutes,
            },
            "risk_score": risk,
            "mitigation_plan": plan,
            "chart": chart_path,
            "timestamp": now.isoformat(),
        }
        results.append(result)

    # write outputs
    out_json = os.path.join(out_dir, "slowa_report.json")
    with open(out_json, "w") as f:
        json.dump({"generated_at": now.isoformat(), "results": results}, f, indent=2, default=str)

    print(f"SLOWA run complete. Outputs written to {out_dir}")
    print(f"- Report: {out_json}")
    for r in results:
        print(f"- Metric {r['metric']}: risk={r['risk_score']:.3f}; chart={r['chart']}")

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="SLO Degradation Early-Warning Agent (SLOWA)")
    parser.add_argument("--metrics-dir", required=True, help="Directory containing metrics CSV files")
    parser.add_argument("--slo-file", required=True, help="Path to slo_targets.yml")
    parser.add_argument("--topology", required=True, help="Path to topology.json")
    parser.add_argument("--out-dir", default="./out", help="Output directory")
    parser.add_argument("--use-ollama", action="store_true", help="Attempt to query local Ollama for mitigation text (optional)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_slowa(args.metrics_dir, args.slo_file, args.topology, args.out_dir, use_ollama=args.use_ollama)
