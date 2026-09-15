
from math import sqrt
from random import gauss, random

def correlate_data(x=None, y=None, rho=0.5):
    """
    Introduce a target rank correlation between two data series while
    preserving their original marginal distributions.
    The function generates two correlated standard normal samples with
    correlation coefficient `rho`, uses their ranks to reorder the values
    of `x` and `y`, and returns two new series whose dependence structure
    approximates the requested correlation while keeping the original
    sorted values unchanged.
    Parameters
    ----------
    x : iterable
        First data series.
    y : iterable
        Second data series. Must have the same length as `x`.
    rho : float
        Target correlation coefficient in the range [-1, 1].
    Returns
    -------
    tuple
        (x_out, y_out), where:
        - x_out contains the values of `x` reordered according to the rank
        structure of a random normal sample.
        - y_out contains the values of `y` reordered according to the rank
        structure of a correlated normal sample.
    Notes
    -----
    This approach is based on a Gaussian copula rank-matching technique.
    The marginal distributions of `x` and `y` are preserved exactly, while
    their dependence structure is modified to approximate the specified
    correlation. The achieved Pearson correlation of the output series will
    generally differ from `rho`, especially for small samples or highly
    non-normal distributions.
    """
    if x is None:
        x = [random() for _ in range(1000)]
    if y is None:
        y = [random() for _ in range(1000)]
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if not -1 <= rho <= 1:
        raise ValueError("rho must be between -1 and 1")

    n = len(x)
    z1 = [gauss(0, 1) for _ in range(n)]
    z2 = [rho * z1[i] + sqrt(1 - rho**2) * gauss(0, 1) for i in range(n)]
    order_x = sorted(range(n), key=lambda i: z1[i])
    order_y = sorted(range(n), key=lambda i: z2[i])
    x_sorted = sorted(x)
    y_sorted = sorted(y)
    x_out = [None] * n
    y_out = [None] * n
    for rank in range(n):
        x_out[order_x[rank]] = x_sorted[rank]
        y_out[order_y[rank]] = y_sorted[rank]
    return x_out, y_out

def get_linear_fit_parameters(x, y, intercept=False):
    """
    Perform a simple linear regression between two data series.
    The function estimates the coefficients of the linear relationship
        y = a*x + b
    using the least-squares method. By default, both the slope (`a`) and
    intercept (`b`) are fitted. Alternatively, a fixed intercept can be
    specified, in which case only the slope is estimated.
    The function also computes the coefficient of determination (R²) and,
    when the intercept is fitted, the linear correlation coefficient (R).
    Parameters
    ----------
    x : iterable of float
        Independent variable.
    y : iterable of float
        Dependent variable.
    intercept : float or bool, default=False
        If False, both slope and intercept are estimated from the data.
        If a numeric value is provided, the intercept is fixed to that
        value and only the slope is fitted.
    Returns
    -------
    tuple
        (a, b, r2, r), where:
        - a : fitted slope.
        - b : fitted intercept.
        - r2 : coefficient of determination (R²).
        - r : correlation coefficient. Returned only when the intercept
        is fitted from the data and R² is positive; otherwise False.
    Notes
    -----
    The regression minimizes the sum of squared residuals between the
    observed values and the fitted line. When a fixed intercept is used,
    the reported R² is calculated relative to the mean of the observed
    data and may not be directly comparable to the unconstrained case.
    """
    n = len(x)
    if not intercept:
        mx = sum(x) / n
        my = sum(y) / n
        sxx = sum((xi - mx) ** 2 for xi in x)
        sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        a = sxy / sxx
        b = my - a * mx
    else:
        b = intercept
        a = sum(xi * (yi - b) for xi, yi in zip(x, y)) / sum(xi**2 for xi in x)
    y_fit = [a * xi + b for xi in x]
    y_mean = sum(y) / n
    ss_res = sum((yi - fi) ** 2 for yi, fi in zip(y, y_fit))
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    r2 = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    if r2 > 0 and not intercept:
        r = sqrt(r2)
    else:
        r = False
    return a, b, r2, r

def get_percentile(values, p):
    """
    Compute the p-th percentile of a numeric dataset using linear interpolation.
    Parameters
    ----------
    values : sequence of float
        Collection of numeric values. The input is internally sorted
        before the percentile is calculated.
    p : float
        Percentile to compute, expressed as a value between 0 and 100.
        For example, p=50 returns the median.
    Returns
    -------
    float
        The estimated percentile value. If the requested percentile
        falls between two data points, the result is obtained by
        linear interpolation between the surrounding values.
    Notes
    -----
    This implementation follows the common percentile definition based
    on the position:
        k = (n - 1) * p / 100
    where n is the number of values. When k is not an integer, the
    result is linearly interpolated between the adjacent sorted values.
    """
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])