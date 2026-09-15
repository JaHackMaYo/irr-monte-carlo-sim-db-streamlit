import math
from math import exp
from random import gauss, random

from config import *

from utils import correlate_data, get_percentile

def time_to_cod_generator(
    nominal_time_to_cod=18,
    delay_min=-3,
    delay_max=9,
    delay_ml=0,
    delay_distribution_sharpness=2,
    size=1,
):
    """
    Generate time-to-COD values by applying a bounded delay distribution to
    a nominal construction duration.
    The delay is sampled within the interval [delay_min, delay_max], with
    delay_ml representing the most likely delay. The shape of the distribution
    is controlled by the sharpness parameter:
    - sharpness = 1 generates a uniform distribution;
    - sharpness = 2 generates a triangular distribution;
    - sharpness > 2 produces a distribution increasingly concentrated around
      the most likely value;
    - sharpness < 1 produces a flatter distribution, with relatively greater
      weight near the minimum and maximum bounds.
    Parameters
    ----------
    nominal_time_to_cod : int, default=18
        Reference construction duration.
    delay_min : int, default=-3
        Minimum delay relative to the nominal duration.
    delay_max : int, default=9
        Maximum delay relative to the nominal duration.
    delay_ml : int or float, default=0
        Most likely delay value. Must lie between delay_min and delay_max.
    delay_distribution_sharpness : float, default=2
        Shape parameter controlling concentration around delay_ml.
        A value of 1 corresponds to a uniform distribution and a value of
        2 corresponds to a triangular distribution.
    size : int, default=1
        Number of samples to generate.
    Returns
    -------
    int or list[int]
        If size=1, returns a single integer time-to-COD value.
        Otherwise returns a list of integer time-to-COD values.
    Notes
    -----
    Sampled delays are rounded to the nearest integer before being added to
    nominal_time_to_cod.
    """
    output = []
    delay_min -= 0.5
    delay_max += 0.5
    for i in range(size):
        u = random()
        p = (delay_ml - delay_min) / (delay_max - delay_min)
        if u < p:
            x = delay_min + (delay_ml - delay_min) * (u / p) ** (1 / delay_distribution_sharpness)
        else:
            x = delay_max - (delay_max - delay_ml) * ((1 - u) / (1 - p)) ** (
                1 / delay_distribution_sharpness
            )
        output.append(nominal_time_to_cod + round(x))
    return output

def capex_generator(
    nominal_capex=1,
    capex_variation_min=-3,
    capex_variation_max=5,
    capex_variation_ml=0,
    capex_distribution_sharpness=2,
    size=1,
):
    """
    Generate one or more CAPEX values by applying a bounded random variation
    to a nominal CAPEX.
    The variation is sampled from a custom distribution defined by a minimum,
    maximum and most-likely percentage variation. The shape of the distribution
    is controlled by the sharpness parameter, which determines how strongly
    samples are concentrated around the most-likely value.
    Parameters
    ----------
    nominal_capex : float, default=1
        Base CAPEX value before applying random variation.
    capex_variation_min : float, default=-3
        Minimum percentage variation relative to nominal_capex.
    capex_variation_max : float, default=5
        Maximum percentage variation relative to nominal_capex.
    capex_variation_ml : float, default=0
        Most-likely percentage variation. This is the mode of the
        distribution and represents the variation value around which
        samples are concentrated.
    capex_distribution_sharpness : float, default=2
        Controls the concentration of samples around the most-likely value.
        A value of 1 produces a triangular distribution. Higher values
        increase the concentration around variation_ml, while lower values
        generate a flatter distribution.
    size : int, default=1
        Number of CAPEX samples to generate.
    Returns
    -------
    float or list[float]
        A single CAPEX value if size=1, otherwise a list containing
        size randomly generated CAPEX values.
    """
    capex_variation_min = capex_variation_min / 100.0
    capex_variation_max = capex_variation_max / 100.0
    capex_variation_ml = capex_variation_ml / 100.0
    output = []
    for i in range(size):
        u = random()
        p = (capex_variation_ml - capex_variation_min) / (capex_variation_max - capex_variation_min)
        if u < p:
            x = capex_variation_min + (capex_variation_ml - capex_variation_min) * (u / p) ** (
                1 / capex_distribution_sharpness
            )
        else:
            x = capex_variation_max - (capex_variation_max - capex_variation_ml) * (
                (1 - u) / (1 - p)
            ) ** (1 / capex_distribution_sharpness)
        output.append(nominal_capex * (1 + x))
    return output

def distribute_capex(
    actual_capex=1, actual_time_to_cod=18, capex_curve_inflection=0.5, capex_curve_steepness=1
):
    """
    Distribute total CAPEX over the construction period using a normalized
    sigmoid cumulative curve.
    Parameters
    ----------
    actual_capex : float, default=1
        Total CAPEX to distribute.
    actual_time_to_cod : int, default=18
        Number of periods between project start and COD. One CAPEX value is
        returned for each period.
    capex_curve_inflection : float, default=0.5
        Position of the sigmoid inflection point on the normalized timeline
        [0, 1]. Values below 0.5 shift CAPEX towards the beginning of the
        construction period, while values above 0.5 shift CAPEX towards the
        end. When inflection=0.5, the distribution is symmetric.
    capex_curve_steepness : float, default=1
        Controls the concentration of CAPEX around the inflection point.
        Lower values produce a smoother and more uniform distribution, while
        higher values concentrate spending over a narrower portion of the
        construction schedule.
        For a symmetric profile (inflection=0.5):
        - steepness ≈ 1 produces an almost linear cumulative profile
          (approximately uniform CAPEX allocation);
        - steepness ≈ 5 produces a profile similar to the previously used
          cosine distribution;
        - larger values increasingly concentrate CAPEX around the midpoint
          of the schedule.
    Returns
    -------
    list[float]
        CAPEX allocated to each period. The sum of all returned values equals
        actual_capex.
    """
    x = [i / actual_time_to_cod for i in range(actual_time_to_cod + 1)]
    cumulative = [1 / (1 + exp(-capex_curve_steepness * (xi - capex_curve_inflection))) for xi in x]
    c0 = cumulative[0]
    c1 = cumulative[-1]
    cumulative = [(c - c0) / (c1 - c0) for c in cumulative]
    weights = [cumulative[i + 1] - cumulative[i] for i in range(actual_time_to_cod)]
    return [actual_capex * w for w in weights]

def eoh_generator(
    start_month=1,
    actual_time_to_cod=18,
    operational_life=30,
    mean_eoh=None,
    std_eoh=None,
    stochastic_eoh=True,
    mean_eoh_variation=1.0,
):
    """
    Generate a monthly EOH (Equivalent Operating Hours) profile for the
    complete project life cycle.
    The function returns zero EOH during the construction period and monthly
    operating EOH values after COD. Operating EOH can be generated either
    stochastically, using monthly averages and standard deviations, or
    deterministically by scaling the monthly averages.
    Parameters
    ----------
    start_month : int, default=1
        Starting calendar month of the simulation
        (1=January, ..., 12=December).
    actual_time_to_cod : int, default=18
        Construction duration in months. EOH values are set to zero during
        this period.
    operational_life : int, default=360
        Operating life in years. One EOH value is generated for each month
        of operation.
    mean_eoh : list[float], default=[100.0]*12
        Average monthly EOH values for each calendar month, starting from
        January.
    std_eoh : list[float], default=[0.1]*12
        Relative standard deviation of EOH for each calendar month.
        Monthly standard deviation is calculated as:
            sigma = mean_eoh[month] * std_eoh[month]
    stochastic_eoh : bool, default=True
        Controls how operating EOH values are generated.
        - True: monthly EOH values are sampled from a normal distribution
          defined by the corresponding monthly mean and standard deviation.
        - False: monthly EOH values are set deterministically equal to
          mean_eoh[month] * mean_variation.
    mean_eoh_variation : float, default=1.0
        Scaling factor applied to monthly average EOH.
        Examples:
        - 1.00 : base case
        - 1.10 : increase average EOH by 10%
        - 0.90 : decrease average EOH by 10%
    Returns
    -------
    list[float]
        Monthly EOH profile consisting of:
        - actual_time_to_cod months with EOH = 0;
        - operational_life months with operating EOH values.
        The total length of the returned list is:
            actual_time_to_cod + operational_life
    Notes
    -----
    When stochastic=True, negative samples are clipped to zero.
    The combination of mean_eoh and std_eoh allows modelling seasonal
    operating patterns, while mean_variation provides a simple way to
    perform sensitivity analyses on average utilization levels.
    """
    output = [0.0] * actual_time_to_cod
    for i in range(operational_life * 12):
        month = (start_month - 1 + actual_time_to_cod + i) % 12
        mean = mean_eoh[month] * mean_eoh_variation
        if stochastic_eoh:
            sigma = mean * std_eoh[month]
            eoh = max(0.0, gauss(mean, sigma))
        else:
            eoh = mean
        output.append(eoh)
    return output

def _complete_operating_series(values, actual_time_to_cod, operational_life, profile_name, scale=1.0):
    """Build a complete project-life series from externally supplied operating values.

    External values start at COD. Values shorter than the operating life are
    extended using the last available value; longer values are truncated.
    Construction months are prepended as zeros.
    """
    if values is None:
        return None

    operating = [float(value) * scale for value in values]
    if not operating:
        raise ValueError(f"{profile_name}: the supplied series is empty.")
    if any(not math.isfinite(value) for value in operating):
        raise ValueError(f"{profile_name}: all values must be finite.")

    required_months = int(operational_life) * 12
    if required_months <= 0:
        raise ValueError("Operational life must be greater than zero.")
    if len(operating) < required_months:
        operating.extend([operating[-1]] * (required_months - len(operating)))
    else:
        operating = operating[:required_months]

    return [0.0] * int(actual_time_to_cod) + operating

def energy_price_generator(
    actual_time_to_cod=18,
    operational_life=30,
    energy_price_at_cod=100.0,
    annual_price_variation=0.0,
    series=None,
):
    """Generate monthly energy prices or use external operating values.

    ``series`` contains monthly values starting at COD. A short series is
    extended with its last value and a long series is truncated.
    """
    external = _complete_operating_series(
        series, actual_time_to_cod, operational_life, "Energy price series"
    )
    if external is not None:
        if any(value < 0 for value in external):
            raise ValueError("Energy price series values cannot be negative.")
        return external

    output = [0.0] * int(actual_time_to_cod)
    for i in range(int(operational_life) * 12):
        years_from_cod = i / 12
        output.append(energy_price_at_cod + annual_price_variation * years_from_cod)
    return output

def opex_generator(
    actual_time_to_cod=18,
    operational_life=30,
    opex=0.025,
    capacity=42.0,
    series=None,
):
    """Generate monthly OPEX or use external operating values.

    ``series`` contains monthly values starting at COD. A short series is
    extended with its last value and a long series is truncated.
    """
    external = _complete_operating_series(
        series, actual_time_to_cod, operational_life, "OPEX series"
    )
    if external is not None:
        if any(value < 0 for value in external):
            raise ValueError("OPEX series values cannot be negative.")
        return external

    monthly_opex = opex * capacity / 12
    return [0.0] * int(actual_time_to_cod) + [monthly_opex] * (int(operational_life) * 12)

def unavailability_generator(
    actual_time_to_cod=18,
    operational_life=30,
    initial_unavailability=15,
    steady_state_unavailability=3,
    ramp_down_months=12,
    aging_degradation=0.01,
    series=None,
):
    """Generate monthly unavailability or use external percentage values.

    External ``series`` values start at COD and are expressed as percentages,
    consistently with the GUI and Excel files. The returned values are always
    fractions. A short series is extended with its last value and a long series
    is truncated.
    """
    external = _complete_operating_series(
        series,
        actual_time_to_cod,
        operational_life,
        "Unavailability series",
        scale=0.01,
    )
    if external is not None:
        if any(value < 0 or value > 1 for value in external):
            raise ValueError("Unavailability series values must be between 0 and 100%.")
        return external

    operation_months = int(operational_life) * 12
    output = [0.0] * int(actual_time_to_cod)
    for i in range(operation_months):
        if i < ramp_down_months:
            u = (initial_unavailability / 100.0) + (
                (steady_state_unavailability / 100.0)
                - (initial_unavailability / 100.0)
            ) * i / max(1, ramp_down_months - 1)
        else:
            u = (steady_state_unavailability / 100.0) + (
                i - ramp_down_months
            ) * (aging_degradation / 100.0)
        output.append(u)
    return output

def evaluate_irr(cash_flows, irr_guess=0.01, irr_tol=1e-10, irr_max_iter=1000):
    """
    Calculate the Internal Rate of Return (IRR) of a cash flow series
    using the Newton-Raphson method.
    The function iteratively solves for the periodic discount rate that
    makes the Net Present Value (NPV) of the cash flows equal to zero.
    It returns both the periodic IRR and the corresponding effective
    annual IRR, assuming monthly cash flow periods.
    Parameters
    ----------
    cash_flows : iterable of float
        Sequence of periodic cash flows. The first value is typically
        the initial investment and subsequent values represent operating
        cash flows.
    irr_guess : float, default=0.01
        Initial estimate of the periodic IRR.
    irr_tol : float, default=1e-10
        Convergence tolerance for the Newton-Raphson iteration.
    irr_max_iter : int, default=1000
        Maximum number of iterations.
    Returns
    -------
    tuple
        (periodic_irr, annual_irr), where:
        - periodic_irr is the IRR per cash flow period.
        - annual_irr is the effective annual IRR calculated as
          (1 + periodic_irr)**12 - 1.
    Notes
    -----
    Cash flows are assumed to be equally spaced in time. Since the
    annualization formula uses a power of 12, the function assumes that
    each period corresponds to one month.
    """
    r = irr_guess
    for _ in range(irr_max_iter):
        npv = sum(cf / (1 + r) ** t for t, cf in enumerate(cash_flows))
        dnpv = sum(-t * cf / (1 + r) ** (t + 1) for t, cf in enumerate(cash_flows) if t > 0)
        r_new = r - npv / dnpv
        if abs(r_new - r) < irr_tol:
            r = r_new
            break
        r = r_new
    return r, (1 + r) ** 12 - 1

def calculate_revenues(eoh, unavailabilities, capacity, energy_prices):
    """Calculate gross monthly energy revenues before OPEX and taxes.
    Parameters
    ----------
    eoh : iterable of float
        Equivalent operating hours for each monthly period.
    unavailabilities : iterable of float
        Monthly unavailability fractions, expressed between 0 and 1.
    capacity : float
        Installed capacity in MW.
    energy_prices : iterable of float
        Monthly energy-price profile in €/MWh.
    Returns
    -------
    list of float
        Gross monthly revenues in M€, calculated as
        ``EOH * capacity * energy_price * (1 - unavailability) / 1e6``.
    Notes
    -----
    OPEX is intentionally excluded. It is generated separately by
    :func:`opex_generator`, deducted from taxable income in
    :func:`calculate_taxes`, and deducted from the project cash flow.
    All input iterables are expected to have the same length; ``zip`` limits
    the result to the shortest iterable.
    """
    return [
        i * capacity * k * (1 - j) / 10**6 for i, j, k in zip(eoh, unavailabilities, energy_prices)
    ]

def calculate_taxes(
    revenues,
    opexes,
    actual_capex,
    tax_rate=0.3,
    capex_depreciation_ratio=0.9,
    capex_depreciation_time=20,
):
    """Calculate monthly corporate taxes after OPEX and depreciation.
    Parameters
    ----------
    revenues : iterable of float
        Gross monthly operating revenues before OPEX, expressed in M€.
    opexes : iterable of float
        Monthly OPEX flows in M€.
    actual_capex : float
        Total project CAPEX in M€.
    tax_rate : float, default=0.3
        Corporate tax rate expressed as a decimal.
    capex_depreciation_ratio : float, default=0.9
        Fraction of CAPEX available as a depreciation deduction.
    capex_depreciation_time : int, default=20
        Straight-line depreciation period in years.
    Returns
    -------
    list of float
        Monthly tax payments in M€.
    Notes
    -----
    Taxable operating income is calculated as ``revenues - opexes``.
    Non-positive taxable income produces no tax credit and therefore zero
    tax. The monthly depreciation deduction is limited by taxable income,
    the straight-line monthly allowance, and the remaining depreciable
    CAPEX balance. ``revenues`` and ``opexes`` are expected to have the same
    length; ``zip`` limits the calculation to the shortest iterable.
    """
    revenues = [i - j for i, j in zip(revenues, opexes)]
    total_tax_shield = actual_capex * capex_depreciation_ratio
    monthly_max_tax_shield = total_tax_shield / (capex_depreciation_time * 12)
    taxes = []
    residual_tax_shield = total_tax_shield
    for i in revenues:
        if i <= 0:
            taxes.append(0)
            continue
        if residual_tax_shield <= 0:
            taxes.append(i * tax_rate)
            continue
        tax_shield = min(i, monthly_max_tax_shield, residual_tax_shield)
        taxes.append((i - tax_shield) * tax_rate)
        residual_tax_shield -= tax_shield
    return taxes

def generate_operating_profiles(
    actual_time_to_cod,
    operational_life,
    unavailability_parameters,
    unavailability_profile,
    energy_price_at_cod,
    annual_price_variation,
    energy_price_profile,
    opex,
    capacity,
    opex_profile,
):
    """Generate the complete unavailability, energy-price and OPEX series."""
    if unavailability_parameters:
        unavailability_args = tuple(unavailability_parameters)
    else:
        unavailability_args = (0.0, 0.0, 1, 0.0)

    unavailabilities = unavailability_generator(
        actual_time_to_cod,
        operational_life,
        *unavailability_args,
        series=unavailability_profile,
    )
    energy_prices = energy_price_generator(
        actual_time_to_cod,
        operational_life,
        energy_price_at_cod,
        annual_price_variation,
        series=energy_price_profile,
    )
    opexes = opex_generator(
        actual_time_to_cod,
        operational_life,
        opex,
        capacity,
        series=opex_profile,
    )
    return unavailabilities, energy_prices, opexes

def calculate_cash_flow_metrics(
    capex_flows,
    revenues_flows,
    taxes_flows,
    opexes,
    irr_guess,
    irr_tol,
    irr_max_iter,
):
    """Build net cash flows and calculate annual IRR and payback period."""
    cash_flows = [
        -capex + revenue - tax - opex
        for capex, revenue, tax, opex in zip(
            capex_flows, revenues_flows, taxes_flows, opexes
        )
    ]

    cumulative = 0.0
    pbp = len(cash_flows) / 12
    for month, cash_flow in enumerate(cash_flows, 1):
        cumulative += cash_flow
        if cumulative > 0:
            pbp = month / 12
            break

    _, irr = evaluate_irr(cash_flows, irr_guess, irr_tol, irr_max_iter)
    return cash_flows, irr, pbp

def calculate_project_flows(
    actual_capex,
    actual_time_to_cod,
    start_month,
    capex_curve_inflection,
    capex_curve_steepness,
    operational_life,
    mean_eoh,
    std_eoh,
    stochastic_eoh,
    mean_eoh_variation,
    unavailability_parameters,
    opex,
    capacity,
    energy_price_at_cod,
    annual_price_variation,
    energy_price_profile,
    unavailability_profile,
    opex_profile,
    tax_rate,
    capex_depreciation_ratio,
    capex_depreciation_time,
    irr_guess,
    irr_tol,
    irr_max_iter,
):
    """Calculate all technical and financial flows for one project case."""
    mean_eoh = [100.0] * 12 if mean_eoh is None else mean_eoh
    std_eoh = [0.1] * 12 if std_eoh is None else std_eoh
    if unavailability_parameters is None:
        unavailability_parameters = [15, 3, 12, 0.01]

    capex_flows = distribute_capex(
        actual_capex,
        actual_time_to_cod,
        capex_curve_inflection,
        capex_curve_steepness,
    ) + [0.0] * (operational_life * 12)
    eoh = eoh_generator(
        start_month,
        actual_time_to_cod,
        operational_life,
        mean_eoh,
        std_eoh,
        stochastic_eoh,
        mean_eoh_variation,
    )
    unavailabilities, energy_prices, opexes = generate_operating_profiles(
        actual_time_to_cod,
        operational_life,
        unavailability_parameters,
        unavailability_profile,
        energy_price_at_cod,
        annual_price_variation,
        energy_price_profile,
        opex,
        capacity,
        opex_profile,
    )
    revenues_flows = calculate_revenues(
        eoh, unavailabilities, capacity, energy_prices
    )
    taxes_flows = calculate_taxes(
        revenues_flows,
        opexes,
        actual_capex,
        tax_rate,
        capex_depreciation_ratio,
        capex_depreciation_time,
    )
    _, irr, pbp = calculate_cash_flow_metrics(
        capex_flows,
        revenues_flows,
        taxes_flows,
        opexes,
        irr_guess,
        irr_tol,
        irr_max_iter,
    )
    return (
        irr,
        sum(eoh),
        pbp,
        capex_flows,
        eoh,
        unavailabilities,
        energy_prices,
        revenues_flows,
        taxes_flows,
        opexes,
    )

def simulate_irr(
    actual_capex=62,
    actual_time_to_cod=18,
    start_month=1,
    capex_curve_inflection=0.5,
    capex_curve_steepness=5,
    operational_life=30,
    mean_eoh=None,
    std_eoh=None,
    stochastic_eoh=True,
    mean_eoh_variation=1.0,
    unavailability_parameters=None,
    opex=0.025,
    capacity=50,
    energy_price_at_cod=100.0,
    annual_price_variation=0.0,
    energy_price_profile=None,
    unavailability_profile=None,
    opex_profile=None,
    tax_rate=0.3,
    capex_depreciation_ratio=0.9,
    capex_depreciation_time=20,
    irr_guess=0.01,
    irr_tol=1e-10,
    irr_max_iter=1000,
):
    """Simulate one project case and return annual IRR, EOH and PBP."""
    result = calculate_project_flows(
        actual_capex,
        actual_time_to_cod,
        start_month,
        capex_curve_inflection,
        capex_curve_steepness,
        operational_life,
        mean_eoh,
        std_eoh,
        stochastic_eoh,
        mean_eoh_variation,
        unavailability_parameters,
        opex,
        capacity,
        energy_price_at_cod,
        annual_price_variation,
        energy_price_profile,
        unavailability_profile,
        opex_profile,
        tax_rate,
        capex_depreciation_ratio,
        capex_depreciation_time,
        irr_guess,
        irr_tol,
        irr_max_iter,
    )
    return result[:3]

def generate_flows(
    stochastic_capex=STOCHASTIC_CAPEX,
    stochastic_delay=STOCHASTIC_DELAY,
    nominal_time_to_cod=NOMINAL_TIME_TO_COD,
    delay_min=DELAY_MIN,
    delay_max=DELAY_MAX,
    delay_ml=DELAY_ML,
    delay_distribution_sharpness=DELAY_DISTRIBUTION_SHARPNESS,
    nominal_capex=NOMINAL_CAPEX,
    capex_variation_min=CAPEX_VARIATION_MIN,
    capex_variation_max=CAPEX_VARIATION_MAX,
    capex_variation_ml=CAPEX_VARIATION_ML,
    capex_distribution_sharpness=CAPEX_DISTRIBUTION_SHARPNESS,
    start_month=START_MONTH,
    capex_curve_inflection=CAPEX_CURVE_INFLECTION,
    capex_curve_steepness=CAPEX_CURVE_STEEPNESS,
    operational_life=OPERATIONAL_LIFE,
    mean_eoh=MEAN_EOH,
    std_eoh=STD_EOH,
    stochastic_eoh=STOCHASTIC_EOH,
    mean_eoh_variation=MEAN_EOH_VARIATION,
    unavailability_parameters=UNAVAILABILITY_PARAMETERS,
    opex=OPEX,
    capacity=CAPACITY,
    energy_price_at_cod=ENERGY_PRICE_AT_COD,
    annual_price_variation=ANNUAL_PRICE_VARIATION,
    energy_price_profile=None,
    unavailability_profile=None,
    opex_profile=None,
    tax_rate=TAX_RATE,
    capex_depreciation_ratio=CAPEX_DEPRECIATION_RATIO,
    capex_depreciation_time=CAPEX_DEPRECIATION_TIME,
    irr_guess=IRR_GUESS,
    irr_tol=IRR_TOL,
    irr_max_iter=IRR_MAX_ITER,
):
    """Sample one case and return its complete technical and financial flows."""
    actual_capex = (
        capex_generator(
            nominal_capex,
            capex_variation_min,
            capex_variation_max,
            capex_variation_ml,
            capex_distribution_sharpness,
            size=1,
        )[0]
        if stochastic_capex
        else nominal_capex
    )
    actual_time_to_cod = (
        time_to_cod_generator(
            nominal_time_to_cod,
            delay_min,
            delay_max,
            delay_ml,
            delay_distribution_sharpness,
            size=1,
        )[0]
        if stochastic_delay
        else nominal_time_to_cod
    )
    return calculate_project_flows(
        actual_capex,
        actual_time_to_cod,
        start_month,
        capex_curve_inflection,
        capex_curve_steepness,
        operational_life,
        mean_eoh,
        std_eoh,
        stochastic_eoh,
        mean_eoh_variation,
        unavailability_parameters,
        opex,
        capacity,
        energy_price_at_cod,
        annual_price_variation,
        energy_price_profile,
        unavailability_profile,
        opex_profile,
        tax_rate,
        capex_depreciation_ratio,
        capex_depreciation_time,
        irr_guess,
        irr_tol,
        irr_max_iter,
    )

def monte_carlo_irr(
    stochastic_capex=STOCHASTIC_CAPEX,
    stochastic_delay=STOCHASTIC_DELAY,
    capex_delay_correlation=CAPEX_DELAY_CORRELATION,
    nominal_time_to_cod=NOMINAL_TIME_TO_COD,
    delay_min=DELAY_MIN,
    delay_max=DELAY_MAX,
    delay_ml=DELAY_ML,
    delay_distribution_sharpness=DELAY_DISTRIBUTION_SHARPNESS,
    nominal_capex=NOMINAL_CAPEX,
    capex_variation_min=CAPEX_VARIATION_MIN,
    capex_variation_max=CAPEX_VARIATION_MAX,
    capex_variation_ml=CAPEX_VARIATION_ML,
    capex_distribution_sharpness=CAPEX_DISTRIBUTION_SHARPNESS,
    iterations=ITERATIONS,
    start_month=START_MONTH,
    capex_curve_inflection=CAPEX_CURVE_INFLECTION,
    capex_curve_steepness=CAPEX_CURVE_STEEPNESS,
    operational_life=OPERATIONAL_LIFE,
    mean_eoh=MEAN_EOH,
    std_eoh=STD_EOH,
    stochastic_eoh=STOCHASTIC_EOH,
    mean_eoh_variation=MEAN_EOH_VARIATION,
    unavailability_parameters=UNAVAILABILITY_PARAMETERS,
    opex=OPEX,
    capacity=CAPACITY,
    energy_price_at_cod=ENERGY_PRICE_AT_COD,
    annual_price_variation=ANNUAL_PRICE_VARIATION,
    energy_price_profile=None,
    unavailability_profile=None,
    opex_profile=None,
    tax_rate=TAX_RATE,
    capex_depreciation_ratio=CAPEX_DEPRECIATION_RATIO,
    capex_depreciation_time=CAPEX_DEPRECIATION_TIME,
    irr_guess=IRR_GUESS,
    irr_tol=IRR_TOL,
    irr_max_iter=IRR_MAX_ITER,
):
    if stochastic_delay:
        iterations_time_to_cod = time_to_cod_generator(
            nominal_time_to_cod,
            delay_min,
            delay_max,
            delay_ml,
            delay_distribution_sharpness,
            iterations,
        )
    else:
        iterations_time_to_cod = [nominal_time_to_cod] * iterations
    if stochastic_capex:
        iterations_capex = capex_generator(
            nominal_capex,
            capex_variation_min,
            capex_variation_max,
            capex_variation_ml,
            capex_distribution_sharpness,
            iterations,
        )
    else:
        iterations_capex = [nominal_capex] * iterations
    if stochastic_delay and stochastic_capex and capex_delay_correlation != 0:
        iterations_time_to_cod, iterations_capex = correlate_data(
            iterations_time_to_cod, iterations_capex, capex_delay_correlation
        )
    iterations_bpb = []
    iterations_irr = []
    iterations_eoh = []
    for actual_capex, actual_time_to_cod in zip(iterations_capex, iterations_time_to_cod):
        actual_irr, actual_eoh, actual_pbp = simulate_irr(
            actual_capex,
            actual_time_to_cod,
            start_month,
            capex_curve_inflection,
            capex_curve_steepness,
            operational_life,
            mean_eoh,
            std_eoh,
            stochastic_eoh,
            mean_eoh_variation,
            unavailability_parameters,
            opex,
            capacity,
            energy_price_at_cod,
            annual_price_variation,
            energy_price_profile,
            unavailability_profile,
            opex_profile,
            tax_rate,
            capex_depreciation_ratio,
            capex_depreciation_time,
            irr_guess,
            irr_tol,
            irr_max_iter,
        )
        iterations_irr.append(actual_irr)
        iterations_eoh.append(actual_eoh)
        iterations_bpb.append(actual_pbp)
    return iterations_irr, iterations_eoh, iterations_time_to_cod, iterations_capex, iterations_bpb

def calculate_stiffness(
    nominal_time_to_cod=NOMINAL_TIME_TO_COD,
    nominal_capex=NOMINAL_CAPEX,
    start_month=START_MONTH,
    capex_curve_inflection=CAPEX_CURVE_INFLECTION,
    capex_curve_steepness=CAPEX_CURVE_STEEPNESS,
    operational_life=OPERATIONAL_LIFE,
    mean_eoh=MEAN_EOH,
    std_eoh=STD_EOH,
    mean_eoh_variation=MEAN_EOH_VARIATION,
    unavailability_parameters=UNAVAILABILITY_PARAMETERS,
    opex=OPEX,
    capacity=CAPACITY,
    energy_price_at_cod=ENERGY_PRICE_AT_COD,
    annual_price_variation=ANNUAL_PRICE_VARIATION,
    energy_price_profile=None,
    unavailability_profile=None,
    opex_profile=None,
    tax_rate=TAX_RATE,
    capex_depreciation_ratio=CAPEX_DEPRECIATION_RATIO,
    capex_depreciation_time=CAPEX_DEPRECIATION_TIME,
    irr_guess=IRR_GUESS,
    irr_tol=IRR_TOL,
    irr_max_iter=IRR_MAX_ITER,
    capex_stress=CAPEX_STRESS,
    delay_stress=DELAY_STRESS,
    eoh_stress=EOH_STRESS,
):
    """Calculate the current reference case and the three IRR stiffnesses."""
    if capex_stress == 0 or delay_stress == 0 or eoh_stress == 0:
        raise ValueError("CAPEX, delay and EOH stresses must be different from zero")
    common = dict(
        iterations=1,
        stochastic_capex=False,
        stochastic_delay=False,
        nominal_time_to_cod=int(round(nominal_time_to_cod)),
        nominal_capex=nominal_capex,
        start_month=start_month,
        capex_curve_inflection=capex_curve_inflection,
        capex_curve_steepness=capex_curve_steepness,
        operational_life=operational_life,
        mean_eoh=mean_eoh,
        std_eoh=std_eoh,
        stochastic_eoh=False,
        mean_eoh_variation=mean_eoh_variation,
        unavailability_parameters=unavailability_parameters,
        opex=opex,
        capacity=capacity,
        energy_price_at_cod=energy_price_at_cod,
        annual_price_variation=annual_price_variation,
        energy_price_profile=energy_price_profile,
        unavailability_profile=unavailability_profile,
        opex_profile=opex_profile,
        tax_rate=tax_rate,
        capex_depreciation_ratio=capex_depreciation_ratio,
        capex_depreciation_time=capex_depreciation_time,
        irr_guess=irr_guess,
        irr_tol=irr_tol,
        irr_max_iter=irr_max_iter,
    )
    reference = monte_carlo_irr(**common)
    capex_result = monte_carlo_irr(
        **dict(common, nominal_capex=nominal_capex * (1 + capex_stress / 100.0))
    )
    delay_result = monte_carlo_irr(
        **dict(common, nominal_time_to_cod=int(round(nominal_time_to_cod + delay_stress)))
    )
    eoh_result = monte_carlo_irr(
        **dict(common, mean_eoh_variation=mean_eoh_variation * (1 + eoh_stress / 100.0))
    )
    reference_irr = reference[0][0]
    capex_stiffness = round((capex_result[0][0] - reference_irr) * 10000 / capex_stress)
    delay_stiffness = round((delay_result[0][0] - reference_irr) * 10000 / delay_stress)
    eoh_stiffness = round((eoh_result[0][0] - reference_irr) * 10000 / eoh_stress)
    return (
        delay_stiffness,
        capex_stiffness,
        eoh_stiffness,
        reference_irr,
        reference[1][0],
        reference[2][0],
        reference[3][0],
        reference[4][0],
    )

def my_irr_mc_function(
    stochastic_capex=STOCHASTIC_CAPEX,
    stochastic_delay=STOCHASTIC_DELAY,
    capex_delay_correlation=CAPEX_DELAY_CORRELATION,
    nominal_time_to_cod=NOMINAL_TIME_TO_COD,
    delay_min=DELAY_MIN,
    delay_max=DELAY_MAX,
    delay_ml=DELAY_ML,
    delay_distribution_sharpness=DELAY_DISTRIBUTION_SHARPNESS,
    nominal_capex=NOMINAL_CAPEX,
    capex_variation_min=CAPEX_VARIATION_MIN,
    capex_variation_max=CAPEX_VARIATION_MAX,
    capex_variation_ml=CAPEX_VARIATION_ML,
    capex_distribution_sharpness=CAPEX_DISTRIBUTION_SHARPNESS,
    iterations=ITERATIONS,
    start_month=START_MONTH,
    capex_curve_inflection=CAPEX_CURVE_INFLECTION,
    capex_curve_steepness=CAPEX_CURVE_STEEPNESS,
    operational_life=OPERATIONAL_LIFE,
    mean_eoh=MEAN_EOH,
    std_eoh=STD_EOH,
    stochastic_eoh=STOCHASTIC_EOH,
    mean_eoh_variation=MEAN_EOH_VARIATION,
    unavailability_parameters=UNAVAILABILITY_PARAMETERS,
    opex=OPEX,
    capacity=CAPACITY,
    energy_price_at_cod=ENERGY_PRICE_AT_COD,
    annual_price_variation=ANNUAL_PRICE_VARIATION,
    energy_price_profile=None,
    unavailability_profile=None,
    opex_profile=None,
    tax_rate=TAX_RATE,
    capex_depreciation_ratio=CAPEX_DEPRECIATION_RATIO,
    capex_depreciation_time=CAPEX_DEPRECIATION_TIME,
    irr_guess=IRR_GUESS,
    irr_tol=IRR_TOL,
    irr_max_iter=IRR_MAX_ITER,
    capex_stress=CAPEX_STRESS,
    delay_stress=DELAY_STRESS,
    eoh_stress=EOH_STRESS,
):
    iterations_irr, iterations_eoh, iterations_time_to_cod, iterations_capex, iterations_bpb = (
        monte_carlo_irr(
            stochastic_capex,
            stochastic_delay,
            capex_delay_correlation,
            nominal_time_to_cod,
            delay_min,
            delay_max,
            delay_ml,
            delay_distribution_sharpness,
            nominal_capex,
            capex_variation_min,
            capex_variation_max,
            capex_variation_ml,
            capex_distribution_sharpness,
            iterations,
            start_month,
            capex_curve_inflection,
            capex_curve_steepness,
            operational_life,
            mean_eoh,
            std_eoh,
            stochastic_eoh,
            mean_eoh_variation,
            unavailability_parameters,
            opex,
            capacity,
            energy_price_at_cod,
            annual_price_variation,
            energy_price_profile,
            unavailability_profile,
            opex_profile,
            tax_rate,
            capex_depreciation_ratio,
            capex_depreciation_time,
            irr_guess,
            irr_tol,
            irr_max_iter,
        )
    )
    (
        delay_stiffness,
        capex_stiffness,
        eoh_stiffness,
        reference_irr,
        reference_eoh,
        reference_time_to_cod,
        reference_capex,
        reference_pbp,
    ) = calculate_stiffness(
        nominal_time_to_cod=nominal_time_to_cod,
        nominal_capex=nominal_capex,
        start_month=start_month,
        capex_curve_inflection=capex_curve_inflection,
        capex_curve_steepness=capex_curve_steepness,
        operational_life=operational_life,
        mean_eoh=mean_eoh,
        std_eoh=std_eoh,
        mean_eoh_variation=mean_eoh_variation,
        unavailability_parameters=unavailability_parameters,
        opex=opex,
        capacity=capacity,
        energy_price_at_cod=energy_price_at_cod,
        annual_price_variation=annual_price_variation,
        energy_price_profile=energy_price_profile,
        unavailability_profile=unavailability_profile,
        opex_profile=opex_profile,
        tax_rate=tax_rate,
        capex_depreciation_ratio=capex_depreciation_ratio,
        capex_depreciation_time=capex_depreciation_time,
        irr_guess=irr_guess,
        irr_tol=irr_tol,
        irr_max_iter=irr_max_iter,
        capex_stress=capex_stress,
        delay_stress=delay_stress,
        eoh_stress=eoh_stress,
    )
    irr_variations = [(i - reference_irr) * 100 * 100 for i in iterations_irr]
    capex_variations = [(i - reference_capex) / reference_capex * 100 for i in iterations_capex]
    delays = [i - reference_time_to_cod for i in iterations_time_to_cod]
    eoh_variations = [(i - reference_eoh) / reference_eoh * 100 for i in iterations_eoh]
    linear_irr_variations = [
        i * capex_stiffness + j * delay_stiffness + k * eoh_stiffness
        for i, j, k in zip(capex_variations, delays, eoh_variations)
    ]
    percentiles_points = list(range(101))
    irr_percentiles = [get_percentile(irr_variations, i) for i in percentiles_points]
    linear_irr_percentiles = [get_percentile(linear_irr_variations, i) for i in percentiles_points]
    linear_irr_error = [(j - i) for i, j in zip(irr_percentiles, linear_irr_percentiles)]
    return (
        iterations_irr,
        reference_irr,
        irr_variations,
        iterations_eoh,
        reference_eoh,
        eoh_variations,
        iterations_time_to_cod,
        reference_time_to_cod,
        delays,
        iterations_capex,
        reference_capex,
        capex_variations,
        iterations_bpb,
        reference_pbp,
        delay_stiffness,
        capex_stiffness,
        eoh_stiffness,
        linear_irr_variations,
        percentiles_points,
        irr_percentiles,
        linear_irr_percentiles,
        linear_irr_error,
    )