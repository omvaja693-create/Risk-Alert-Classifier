"""Risk Alert Classifier — enter customer details, get Low Risk (0) / High Risk (1)."""
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Risk Alert Classifier", page_icon="🚨", layout="centered")

MODEL_PATH = Path(__file__).resolve().parent / "model" / "risk_alert_pipeline.pkl"

# Same order as the training data's X columns
FIELDS = [
    # (column, label, min, max, default, step, is_int)
    ("age", "Age", 18, 100, 36, 1, True),
    ("gender", "Gender", ["Female", "Male", "Other"]),
    ("region", "Region", ["Central", "East", "North", "South", "West"]),
    ("employment_type", "Employment type", ["Salaried", "Self-Employed", "Unemployed", "Retired", "Student"]),
    ("annual_income_inr", "Annual income (INR)", 0, 1_000_000, 38_932, 1000, True),
    ("credit_score", "Credit score", 300, 900, 682, 1, True),
    ("credit_utilization_ratio", "Credit utilization ratio (0–1)", 0.0, 1.0, 0.37, 0.01, False),
    ("missed_payments_12m", "Missed payments (12 months)", 0, 50, 1, 1, True),
    ("avg_late_payment_days", "Avg. late payment days", 0.0, 365.0, 3.9, 0.1, False),
    ("monthly_transaction_count", "Monthly transaction count", 0, 1000, 65, 1, True),
    ("monthly_spend_inr", "Monthly spend (INR)", 0, 1_000_000, 19_317, 500, True),
    ("cash_advance_count_6m", "Cash advances (6 months)", 0, 50, 0, 1, True),
    ("complaints_last_6m", "Complaints (6 months)", 0, 50, 0, 1, True),
    ("failed_login_attempts_3m", "Failed logins (3 months)", 0, 100, 2, 1, True),
    ("account_tenure_months", "Account tenure (months)", 0, 600, 48, 1, True),
    ("debt_balance_inr", "Debt balance (INR)", 0, 5_000_000, 35_638, 1000, True),
]

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 820px; }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #334155 100%);
        border-radius: 16px; padding: 1.6rem 1.8rem; margin-bottom: 1.2rem;
        color: #f8fafc; box-shadow: 0 8px 24px rgba(15, 23, 42, .18);
    }
    .hero h1 { margin: 0; font-size: 1.85rem; font-weight: 700; letter-spacing: -.01em; color: #f8fafc; padding: 0; }
    .hero p  { margin: .35rem 0 0; color: #cbd5e1; font-size: .95rem; }
    .badge-row { margin-top: .9rem; display: flex; gap: .5rem; flex-wrap: wrap; }
    .badge {
        background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.18);
        border-radius: 999px; padding: .2rem .75rem; font-size: .75rem; color: #e2e8f0;
    }

    /* Form card */
    div[data-testid="stForm"] {
        border: 1px solid #e2e8f0; border-radius: 16px; padding: 1.4rem 1.4rem .6rem;
        box-shadow: 0 2px 12px rgba(15, 23, 42, .05);
    }

    /* Full-width, prominent submit button */
    div[data-testid="stForm"] button[kind="primaryFormSubmit"],
    div[data-testid="stForm"] button[kind="primary"] {
        width: 100%; height: 3.1rem; font-size: 1.05rem; font-weight: 600;
        letter-spacing: .02em; border-radius: 12px; border: none;
        background: linear-gradient(90deg, #ef4444, #dc2626);
        box-shadow: 0 6px 16px rgba(220, 38, 38, .30);
        transition: transform .12s ease, box-shadow .12s ease;
    }
    div[data-testid="stForm"] button[kind="primaryFormSubmit"]:hover,
    div[data-testid="stForm"] button[kind="primary"]:hover {
        transform: translateY(-1px); box-shadow: 0 10px 22px rgba(220, 38, 38, .38);
    }

    /* Result card */
    .result {
        border-radius: 16px; padding: 1.4rem 1.6rem; margin: 1.2rem 0 .8rem;
        display: flex; align-items: center; justify-content: space-between; gap: 1rem;
        border: 1px solid; animation: pop .35s ease;
    }
    .result.high { background: #fef2f2; border-color: #fecaca; color: #991b1b; }
    .result.low  { background: #f0fdf4; border-color: #bbf7d0; color: #166534; }
    .result .label { font-size: .75rem; text-transform: uppercase; letter-spacing: .1em; opacity: .75; }
    .result .verdict { font-size: 1.9rem; font-weight: 700; line-height: 1.15; }
    .result .icon { font-size: 2.6rem; }
    @keyframes pop { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

    /* Gauge */
    .gauge-wrap { margin: .4rem 0 1rem; }
    .gauge-head { display: flex; justify-content: space-between; font-size: .85rem; color: #475569; margin-bottom: .35rem; }
    .gauge-head b { color: #0f172a; font-size: 1rem; }
    .gauge {
        position: relative; height: 14px; border-radius: 999px;
        background: linear-gradient(90deg, #22c55e 0%, #eab308 50%, #ef4444 100%);
    }
    .gauge .marker {
        position: absolute; top: -5px; width: 4px; height: 24px; border-radius: 2px;
        background: #0f172a; box-shadow: 0 0 0 3px rgba(255,255,255,.9);
        transform: translateX(-50%); transition: left .4s ease;
    }
    .gauge-scale { display: flex; justify-content: space-between; font-size: .7rem; color: #94a3b8; margin-top: .3rem; }

    /* Signal chips */
    .chip { display: inline-block; margin: .15rem .3rem .15rem 0; padding: .25rem .7rem; border-radius: 999px; font-size: .8rem; border: 1px solid; }
    .chip.warn { background: #fffbeb; border-color: #fde68a; color: #92400e; }
    .chip.ok   { background: #f0fdf4; border-color: #bbf7d0; color: #166534; }

    .footnote { color: #94a3b8; font-size: .75rem; text-align: center; margin-top: 1.5rem; }
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline(path: Path):
    return joblib.load(path)


def indicative_signals(v: dict) -> list[tuple[str, bool]]:
    """Simple rule-of-thumb checks on the inputs (NOT the model's own explanation)."""
    income = max(v["annual_income_inr"], 1)
    return [
        ("Low credit score (<600)", v["credit_score"] < 600),
        ("High credit utilization (>70%)", v["credit_utilization_ratio"] > 0.70),
        ("Multiple missed payments (≥3)", v["missed_payments_12m"] >= 3),
        ("Long late-payment delays (>15 days)", v["avg_late_payment_days"] > 15),
        ("Cash advances taken (≥2)", v["cash_advance_count_6m"] >= 2),
        ("Repeated complaints (≥3)", v["complaints_last_6m"] >= 3),
        ("Many failed logins (≥5)", v["failed_login_attempts_3m"] >= 5),
        ("Debt above annual income", v["debt_balance_inr"] > income),
    ]


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🚨 Risk Alert Classifier</h1>
        <p>Enter customer details to classify the account as <b>Low Risk (0)</b> or <b>High Risk (1)</b>.</p>
        <div class="badge-row">
            <span class="badge">⚡ Instant prediction</span>
            <span class="badge">📊 Probability score</span>
            <span class="badge">🧾 Downloadable report</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()
try:
    pipeline = load_pipeline(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

columns = list(getattr(pipeline, "feature_names_in_", [f[0] for f in FIELDS]))

# ----------------------------------------------------------------------------
# Form
# ----------------------------------------------------------------------------
with st.form("customer"):
    values = {}
    left, right = st.columns(2)
    for i, f in enumerate(FIELDS):
        with (left if i % 2 == 0 else right):
            if isinstance(f[2], list):
                values[f[0]] = st.selectbox(f[1], f[2])
            else:
                _, label, lo, hi, default, step, is_int = f
                cast = int if is_int else float
                values[f[0]] = st.number_input(label, cast(lo), cast(hi), cast(default), cast(step))
    submitted = st.form_submit_button("🔍  Predict Risk", type="primary", use_container_width=True)

# ----------------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------------
if submitted:
    try:
        row = pd.DataFrame([values])[columns]
        pred = int(pipeline.predict(row)[0])
        proba = float(pipeline.predict_proba(row)[0, 1]) if hasattr(pipeline, "predict_proba") else None

        # Verdict card
        if pred == 1:
            st.markdown(
                '<div class="result high"><div><div class="label">Model verdict</div>'
                '<div class="verdict">High Risk (1)</div></div><div class="icon">⚠️</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result low"><div><div class="label">Model verdict</div>'
                '<div class="verdict">Low Risk (0)</div></div><div class="icon">✅</div></div>',
                unsafe_allow_html=True,
            )

        # Probability gauge
        if proba is not None:
            pct = proba * 100
            st.markdown(
                f"""
                <div class="gauge-wrap">
                    <div class="gauge-head"><span>High-risk probability</span><b>{proba:.1%}</b></div>
                    <div class="gauge"><div class="marker" style="left:{min(max(pct, 1), 99):.1f}%"></div></div>
                    <div class="gauge-scale"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Key metrics
        income = max(values["annual_income_inr"], 1)
        m1, m2, m3 = st.columns(3)
        m1.metric("Credit score", f"{values['credit_score']}")
        m2.metric("Utilization", f"{values['credit_utilization_ratio']:.0%}")
        m3.metric("Debt / income", f"{values['debt_balance_inr'] / income:.2f}x")

        # Indicative signals
        signals = indicative_signals(values)
        flagged = [s for s, hit in signals if hit]
        with st.expander(f"Input risk signals ({len(flagged)} of {len(signals)} flagged)", expanded=bool(flagged)):
            chips = "".join(
                f'<span class="chip {"warn" if hit else "ok"}">{"⚠" if hit else "✓"} {text}</span>'
                for text, hit in signals
            )
            st.markdown(chips, unsafe_allow_html=True)
            st.caption("Rule-of-thumb checks on the entered values. They are not the model's own explanation.")

        # Report download
        report = row.copy()
        report["prediction"] = "High Risk (1)" if pred == 1 else "Low Risk (0)"
        if proba is not None:
            report["high_risk_probability"] = round(proba, 4)
        report["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.download_button(
            "⬇️  Download report (CSV)",
            report.to_csv(index=False).encode("utf-8"),
            file_name="risk_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.markdown(
    '<div class="footnote">Predictions are model-generated and should support, not replace, human review.</div>',
    unsafe_allow_html=True,
)
