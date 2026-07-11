"""Pure financial-model functions used by the Streamlit app (app.py).

Kept free of UI imports so they are unit-testable. Fixes over the original
in-app implementations:
- real calendar months (dateutil) instead of fixed 30-day steps
- every CapEx staging option is handled (unknown options fall back to
  upfront instead of crashing)
- XIRR reports failure as None instead of silently returning 0
- Monte Carlo factors are floored above zero so simulated CapEx/revenue/OpEx
  can't go negative
- summary totals separate gross revenue, OpEx, and CapEx instead of netting
  them per-month and mislabeling the sums
"""

from datetime import date, datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.optimize import fsolve

# fraction of initial CapEx deployed at each month offset
CAPEX_STAGING = {
    "Upfront (100% at start)": {0: 1.0},
    "Staged (50% + 50% at 6mo)": {0: 0.5, 6: 0.5},
    "Delayed (100% at 3mo)": {3: 1.0},
    # "Custom Staging" and anything unrecognized fall back to upfront; the
    # UI labels this explicitly so it never silently pretends to be custom.
    "Custom Staging": {0: 1.0},
}


def create_cash_flow_schedule(start_date, duration_years: int,
                              revenue_delay_months: int, initial_capex_mm: float,
                              monthly_rev_k: float, monthly_opex_k: float,
                              staging: str) -> pd.DataFrame:
    """Monthly cash-flow schedule. CapEx in $M, revenue/opex in $K/month."""
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    months = duration_years * 12
    dates = [start_date + relativedelta(months=m) for m in range(months)]

    schedule = CAPEX_STAGING.get(staging, CAPEX_STAGING["Upfront (100% at start)"])
    cash_flows = []
    for m in range(months):
        cf = -initial_capex_mm * 1_000_000 * schedule.get(m, 0.0)
        cf -= monthly_opex_k * 1000
        if m >= revenue_delay_months:
            cf += monthly_rev_k * 1000
        cash_flows.append(cf)

    return pd.DataFrame({
        "Date": dates,
        "CashFlow": cash_flows,
        "CumulativeCF": np.cumsum(cash_flows),
    })


def calculate_xnpv(cash_flows, dates, discount_rate: float) -> float:
    """Date-weighted NPV."""
    dates = list(dates)
    start = dates[0]
    total = 0.0
    for cf, d in zip(cash_flows, dates):
        years = (d - start).days / 365.25
        total += cf / (1 + discount_rate) ** years
    return total


def calculate_xirr(cash_flows, dates, guess: float = 0.1) -> float | None:
    """Date-weighted IRR. Returns None when no rate can be found (instead of
    a misleading 0): all-positive/all-negative flows, or solver failure."""
    flows = list(cash_flows)
    if not flows or all(f >= 0 for f in flows) or all(f <= 0 for f in flows):
        return None
    dates = list(dates)

    def xnpv_at(rate):
        return calculate_xnpv(flows, dates, rate)

    try:
        result, _, ier, _ = fsolve(lambda r: xnpv_at(r[0]), [guess], full_output=True)
    except Exception:
        return None
    if ier != 1:
        return None
    irr = float(result[0])
    if not np.isfinite(irr) or irr <= -0.99 or irr > 10:
        return None
    return irr


def summarize_project(cf_df: pd.DataFrame, initial_capex_mm: float,
                      monthly_rev_k: float, monthly_opex_k: float,
                      duration_years: int, revenue_delay_months: int) -> dict:
    """Honest, separately-stated totals (all in dollars)."""
    months = duration_years * 12
    revenue_months = max(0, months - revenue_delay_months)
    gross_revenue = monthly_rev_k * 1000 * revenue_months
    total_opex = monthly_opex_k * 1000 * months
    total_capex = initial_capex_mm * 1_000_000
    net = gross_revenue - total_opex - total_capex
    return {
        "gross_revenue": gross_revenue,
        "total_opex": total_opex,
        "total_capex": total_capex,
        "net_cash_flow": net,
        # undiscounted multiple on invested capital; None when nothing invested
        "roi": (net / total_capex) if total_capex > 0 else None,
    }


def payback_months(cumulative_cf) -> int | None:
    """First month the cumulative cash flow turns positive, else None."""
    for i, v in enumerate(cumulative_cf):
        if v > 0:
            return i
    return None


def monte_carlo_simulation(base_params: dict, num_runs: int = 5000,
                           seed: int | None = None) -> pd.DataFrame:
    """Parameter-uncertainty simulation. Variation factors are floored at
    0.05 so CapEx/revenue/OpEx can never go negative."""
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(num_runs):
        capex_f = max(0.05, rng.normal(1.0, 0.15))
        rev_f = max(0.05, rng.normal(1.0, 0.20))
        opex_f = max(0.05, rng.normal(1.0, 0.10))

        cf = create_cash_flow_schedule(
            base_params["start_date"], base_params["duration"],
            base_params["revenue_delay"], base_params["capex"] * capex_f,
            base_params["revenue"] * rev_f, base_params["opex"] * opex_f,
            base_params["staging"],
        )
        xnpv = calculate_xnpv(cf["CashFlow"], cf["Date"], base_params["discount_rate"])
        xirr = calculate_xirr(cf["CashFlow"], cf["Date"])
        results.append({
            "XNPV": xnpv,
            "XIRR": xirr if xirr is not None else np.nan,
            "CapEx_Factor": capex_f,
            "Revenue_Factor": rev_f,
            "OpEx_Factor": opex_f,
        })
    return pd.DataFrame(results)
