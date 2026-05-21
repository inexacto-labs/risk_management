import numpy as np
import pandas as pd


def generate_bond_cashflows(
    face_value: float,
    coupon_rate: float,
    maturity_years: float,
    payments_per_year: int,
) -> pd.DataFrame:
    periods = max(int(round(maturity_years * payments_per_year)), 1)
    coupon_payment = face_value * coupon_rate / payments_per_year
    rows = []
    for period in range(1, periods + 1):
        time_years = period / payments_per_year
        principal = face_value if period == periods else 0.0
        cash_flow = coupon_payment + principal
        rows.append(
            {
                "Periodo": period,
                "Tiempo (años)": time_years,
                "Cupón": coupon_payment,
                "Amortización": principal,
                "Flujo": cash_flow,
            }
        )
    return pd.DataFrame(rows)


def price_fixed_rate_bond(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    maturity_years: float,
    payments_per_year: int,
) -> float:
    cashflows = generate_bond_cashflows(face_value, coupon_rate, maturity_years, payments_per_year)
    period_rate = ytm / payments_per_year
    discount = (1 + period_rate) ** cashflows["Periodo"]
    return float((cashflows["Flujo"] / discount).sum())


def bond_risk_measures(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    maturity_years: float,
    payments_per_year: int,
) -> dict:
    cashflows = generate_bond_cashflows(face_value, coupon_rate, maturity_years, payments_per_year)
    period_rate = ytm / payments_per_year
    discount = (1 + period_rate) ** cashflows["Periodo"]
    pv = cashflows["Flujo"] / discount
    price = float(pv.sum())
    cashflows = cashflows.copy()
    cashflows["Factor de descuento"] = discount
    cashflows["Valor presente"] = pv
    time_years = cashflows["Tiempo (años)"]

    macaulay = float((time_years * pv).sum() / price)
    modified = macaulay / (1 + period_rate)

    convexity_numerator = ((time_years**2 + time_years / payments_per_year) * pv).sum()
    convexity = float(convexity_numerator / (price * (1 + period_rate) ** 2))

    return {
        "cashflows": cashflows,
        "price": price,
        "macaulay_duration": macaulay,
        "modified_duration": modified,
        "convexity": convexity,
    }


def price_curve(
    face_value: float,
    coupon_rate: float,
    maturity_years: float,
    payments_per_year: int,
    base_ytm: float,
    shock_range_bps: int = 400,
    step_bps: int = 10,
) -> pd.DataFrame:
    shocks = np.arange(-shock_range_bps, shock_range_bps + step_bps, step_bps)
    rows = []
    for shock in shocks:
        shifted_ytm = max(base_ytm + shock / 10000, 0.0001)
        price = price_fixed_rate_bond(
            face_value=face_value,
            coupon_rate=coupon_rate,
            ytm=shifted_ytm,
            maturity_years=maturity_years,
            payments_per_year=payments_per_year,
        )
        rows.append(
            {
                "Shock (pbs)": shock,
                "Tasa": shifted_ytm,
                "Precio": price,
            }
        )
    return pd.DataFrame(rows)


def estimate_price_change(
    modified_duration: float,
    convexity: float,
    delta_rate: float,
    include_convexity: bool = True,
) -> float:
    linear = -modified_duration * delta_rate
    if include_convexity:
        return linear + 0.5 * convexity * (delta_rate**2)
    return linear


def run_bond_var_monte_carlo(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    maturity_years: float,
    payments_per_year: int,
    rate_vol_annual: float,
    horizon_days: int,
    confidence: float,
    simulations: int = 10000,
    seed: int = 7,
) -> dict:
    rng = np.random.default_rng(seed)
    base_price = price_fixed_rate_bond(face_value, coupon_rate, ytm, maturity_years, payments_per_year)
    horizon_scale = np.sqrt(horizon_days / 252)
    delta_rates = rng.normal(loc=0.0, scale=rate_vol_annual * horizon_scale, size=simulations)
    shocked_rates = np.clip(ytm + delta_rates, 0.0001, None)
    daily_scale = rate_vol_annual / np.sqrt(252)
    daily_increments = rng.normal(loc=0.0, scale=daily_scale, size=(simulations, horizon_days))
    cumulative = np.cumsum(daily_increments, axis=1)
    rate_paths = np.column_stack([np.full(simulations, ytm), ytm + cumulative])
    rate_paths = np.clip(rate_paths, 0.0001, None)
    terminal_rates = rate_paths[:, -1]

    simulated_prices = np.array(
        [
            price_fixed_rate_bond(face_value, coupon_rate, rate, maturity_years, payments_per_year)
            for rate in terminal_rates
        ]
    )
    pnl = simulated_prices - base_price
    losses = -pnl
    var_amount = float(np.quantile(losses, confidence))
    var_amount = max(var_amount, 0.0)
    tail_losses = losses[losses >= var_amount]
    expected_shortfall = float(tail_losses.mean()) if tail_losses.size else 0.0

    return {
        "base_price": base_price,
        "delta_rates": delta_rates,
        "shocked_rates": shocked_rates,
        "rate_paths": rate_paths,
        "terminal_rates": terminal_rates,
        "simulated_prices": simulated_prices,
        "pnl": pnl,
        "losses": losses,
        "var_amount": var_amount,
        "var_pct": var_amount / base_price if base_price else 0.0,
        "expected_shortfall": expected_shortfall,
        "confidence": confidence,
        "horizon_days": horizon_days,
    }
