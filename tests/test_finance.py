from datetime import date

import numpy as np
import pytest

from wtrmln.finance import (
    calculate_xirr,
    calculate_xnpv,
    create_cash_flow_schedule,
    monte_carlo_simulation,
    payback_months,
    summarize_project,
)

START = date(2026, 1, 1)


def schedule(**over):
    args = dict(start_date=START, duration_years=3, revenue_delay_months=6,
                initial_capex_mm=2.0, monthly_rev_k=650, monthly_opex_k=50,
                staging="Upfront (100% at start)")
    args.update(over)
    return create_cash_flow_schedule(**args)


def test_custom_staging_does_not_crash():
    df = schedule(staging="Custom Staging")
    assert len(df) == 36
    # falls back to upfront: full capex in month 0
    assert df["CashFlow"].iloc[0] == pytest.approx(-2_000_000 - 50_000)


def test_unknown_staging_falls_back_to_upfront():
    df = schedule(staging="Something Nonexistent")
    assert df["CashFlow"].iloc[0] == pytest.approx(-2_050_000)


def test_staged_capex_splits_50_50():
    df = schedule(staging="Staged (50% + 50% at 6mo)")
    assert df["CashFlow"].iloc[0] == pytest.approx(-1_000_000 - 50_000)
    # month 6: 50% capex + opex - revenue (revenue starts month 6)
    assert df["CashFlow"].iloc[6] == pytest.approx(-1_000_000 - 50_000 + 650_000)


def test_dates_are_calendar_months():
    df = schedule()
    assert df["Date"].iloc[0] == date(2026, 1, 1)
    assert df["Date"].iloc[1] == date(2026, 2, 1)
    assert df["Date"].iloc[12] == date(2027, 1, 1)  # not 360-day drift


def test_xnpv_zero_rate_equals_sum():
    df = schedule()
    assert calculate_xnpv(df["CashFlow"], df["Date"], 0.0) == pytest.approx(
        df["CashFlow"].sum())


def test_xirr_converges_on_profitable_project():
    df = schedule()
    irr = calculate_xirr(df["CashFlow"], df["Date"])
    assert irr is not None and irr > 0
    # definition check: NPV at the found rate is ~0
    assert abs(calculate_xnpv(df["CashFlow"], df["Date"], irr)) < 1.0


def test_xirr_returns_none_when_undefined():
    # all-negative flows have no IRR; the old code returned 0 here
    dates = [date(2026, 1, 1), date(2026, 2, 1)]
    assert calculate_xirr([-100.0, -100.0], dates) is None
    assert calculate_xirr([100.0, 100.0], dates) is None


def test_summary_separates_revenue_opex_capex():
    df = schedule()
    s = summarize_project(df, 2.0, 650, 50, 3, 6)
    assert s["gross_revenue"] == pytest.approx(650_000 * 30)  # 36 - 6 months
    assert s["total_opex"] == pytest.approx(50_000 * 36)
    assert s["total_capex"] == pytest.approx(2_000_000)
    assert s["net_cash_flow"] == pytest.approx(
        s["gross_revenue"] - s["total_opex"] - s["total_capex"])
    # and it reconciles with the schedule itself
    assert s["net_cash_flow"] == pytest.approx(df["CashFlow"].sum())


def test_payback_months():
    df = schedule()
    m = payback_months(df["CumulativeCF"].values)
    assert m is not None and 0 < m < 36
    assert payback_months([-1, -2, -3]) is None


def test_monte_carlo_never_negative_inputs():
    base = dict(start_date=START, duration=2, revenue_delay=3, capex=1.0,
                revenue=100, opex=500, staging="Upfront (100% at start)",
                discount_rate=0.12)
    mc = monte_carlo_simulation(base, num_runs=300, seed=42)
    assert (mc["CapEx_Factor"] > 0).all()
    assert (mc["Revenue_Factor"] > 0).all()
    assert (mc["OpEx_Factor"] > 0).all()
    # unprofitable base: undefined IRRs become NaN, never a fake 0
    assert not (mc["XIRR"] == 0).any()
