"""Command-line interface for Inventory Analytics algorithms."""

import argparse
import importlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Parameter:
    name: str
    value_type: Callable[[str], Any]
    help: str
    required: bool = True
    default: Any = None


@dataclass(frozen=True)
class Algorithm:
    name: str
    label: str
    category: str
    parameters: Tuple[Parameter, ...]
    runner: str
    aliases: Tuple[str, ...] = ()
    note: str = ""


def _json_list(value: str) -> list:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise argparse.ArgumentTypeError("must be a JSON array")
    return parsed


def _json_object(value: str) -> dict:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return parsed


def _boolean(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise argparse.ArgumentTypeError("must be true or false")


NUMBER = float
INTEGER = int
LIST = _json_list
OBJECT = _json_object
BOOLEAN = _boolean


_REQUIRED = object()


def _p(name: str, value_type: Callable[[str], Any], help_text: str, default: Any = _REQUIRED) -> Parameter:
    return Parameter(name, value_type, help_text, default is _REQUIRED, None if default is _REQUIRED else default)


ALGORITHMS = (
    Algorithm("naive", "Naive forecast", "Forecasting", (_p("series", LIST, "observations as a JSON array"), _p("horizon", INTEGER, "number of periods to forecast")), "forecast_naive"),
    Algorithm("seasonal-naive", "Seasonal naive forecast", "Forecasting", (_p("series", LIST, "observations as a JSON array"), _p("season-length", INTEGER, "season length"), _p("horizon", INTEGER, "number of periods to forecast")), "forecast_seasonal_naive"),
    Algorithm("drift", "Drift forecast", "Forecasting", (_p("series", LIST, "observations as a JSON array"), _p("horizon", INTEGER, "number of periods to forecast")), "forecast_drift"),
    Algorithm("sma", "Simple moving-average forecast", "Forecasting", (_p("series", LIST, "observations as a JSON array"), _p("window", INTEGER, "moving-average window"), _p("horizon", INTEGER, "number of periods to forecast")), "forecast_sma", ("moving-average",)),
    Algorithm("ses", "Simple exponential smoothing", "Forecasting", (_p("series", LIST, "observations as a JSON array"), _p("horizon", INTEGER, "number of periods to forecast"), _p("smoothing-level", NUMBER, "smoothing level; optimize when omitted", None), _p("optimized", BOOLEAN, "optimize model parameters", True)), "forecast_ses", ("simple-exponential-smoothing",)),
    Algorithm("holt", "Holt linear trend forecast", "Forecasting", (_p("series", LIST, "observations as a JSON array"), _p("horizon", INTEGER, "number of periods to forecast"), _p("damped-trend", BOOLEAN, "use a damped trend", False), _p("smoothing-level", NUMBER, "smoothing level; optimize when omitted", None), _p("smoothing-trend", NUMBER, "trend smoothing level; optimize when omitted", None), _p("optimized", BOOLEAN, "optimize model parameters", True)), "forecast_holt"),
    Algorithm("holt-winters", "Holt-Winters seasonal forecast", "Forecasting", (_p("series", LIST, "observations as a JSON array"), _p("horizon", INTEGER, "number of periods to forecast"), _p("season-length", INTEGER, "season length"), _p("trend", str, "trend type: none, add, or mul", "add"), _p("seasonal", str, "seasonal type: add or mul", "add"), _p("damped-trend", BOOLEAN, "use a damped trend", False), _p("optimized", BOOLEAN, "optimize model parameters", True)), "forecast_holt_winters", ("hotl-winters",)),
    Algorithm("ar", "Autoregressive forecast", "Forecasting", (_p("series", LIST, "observations as a JSON array"), _p("horizon", INTEGER, "number of periods to forecast"), _p("lags", INTEGER, "number of autoregressive lags", 1), _p("trend", str, "trend: n, c, ct, or ctt", "c")), "forecast_ar"),
    Algorithm("ma", "Moving-average process forecast", "Forecasting", (_p("series", LIST, "observations as a JSON array"), _p("horizon", INTEGER, "number of periods to forecast"), _p("q", INTEGER, "moving-average order", 1), _p("confidence-level", NUMBER, "prediction interval confidence level", 0.95)), "forecast_ma"),
    Algorithm("arima", "ARIMA forecast", "Forecasting", (_p("series", LIST, "observations as a JSON array"), _p("horizon", INTEGER, "number of periods to forecast"), _p("order", LIST, "ARIMA order [p,d,q]"), _p("seasonal-order", LIST, "seasonal order [P,D,Q,s]", [0, 0, 0, 0]), _p("confidence-level", NUMBER, "prediction interval confidence level", 0.95)), "forecast_arima"),
    Algorithm("box-cox", "Box-Cox transformation", "Forecasting utilities", (_p("series", LIST, "positive observations as a JSON array"), _p("lambda", NUMBER, "transformation lambda; estimate when omitted", None)), "box_cox"),
    Algorithm("inverse-box-cox", "Inverse Box-Cox transformation", "Forecasting utilities", (_p("series", LIST, "transformed observations as a JSON array"), _p("lambda", NUMBER, "transformation lambda")), "inverse_box_cox"),
    Algorithm("difference", "Ordinary and seasonal differencing", "Forecasting utilities", (_p("series", LIST, "observations as a JSON array"), _p("differences", INTEGER, "number of ordinary differences", 1), _p("seasonal-differences", INTEGER, "number of seasonal differences", 0), _p("season-length", INTEGER, "season length", 1)), "difference"),
    Algorithm("normal-prediction-interval", "Normal prediction interval", "Forecasting utilities", (_p("mean", NUMBER, "forecast mean"), _p("std", NUMBER, "forecast standard deviation"), _p("confidence-level", NUMBER, "confidence level", 0.95)), "normal_prediction_interval"),
    Algorithm("estimated-normal-prediction-interval", "Estimated normal prediction interval", "Forecasting utilities", (_p("series", LIST, "observations as a JSON array"), _p("confidence-level", NUMBER, "confidence level", 0.95)), "estimated_prediction_interval"),
    Algorithm("naive-prediction-interval", "Naive forecast prediction interval", "Forecasting utilities", (_p("series", LIST, "observations as a JSON array"), _p("horizon", INTEGER, "number of periods to forecast"), _p("confidence-level", NUMBER, "confidence level", 0.95)), "naive_prediction_interval"),
    Algorithm("forecast-errors", "Forecast error metrics", "Forecasting utilities", (_p("actual", LIST, "actual observations"), _p("predicted", LIST, "predicted observations")), "forecast_errors"),
    Algorithm("eoq", "Economic order quantity", "Constant-demand inventory control", (_p("K", NUMBER, "fixed ordering cost"), _p("h", NUMBER, "holding cost"), _p("d", NUMBER, "demand rate"), _p("v", NUMBER, "unit purchase cost")), "eoq", ("economic-order-quantity",)),
    Algorithm("eoq-all-units-discounts", "EOQ with all-units discounts", "Constant-demand inventory control", (_p("K", NUMBER, "fixed ordering cost"), _p("h", NUMBER, "holding cost rate"), _p("d", NUMBER, "demand rate"), _p("b", LIST, "quantity breakpoints"), _p("v", LIST, "unit prices")), "eoq_all_units"),
    Algorithm("eoq-incremental-discounts", "EOQ with incremental discounts", "Constant-demand inventory control", (_p("K", NUMBER, "fixed ordering cost"), _p("h", NUMBER, "holding cost rate"), _p("d", NUMBER, "demand rate"), _p("b", LIST, "quantity breakpoints"), _p("v", LIST, "unit prices")), "eoq_incremental"),
    Algorithm("eoq-planned-backorders", "EOQ with planned backorders", "Constant-demand inventory control", (_p("K", NUMBER, "fixed ordering cost"), _p("h", NUMBER, "holding cost"), _p("d", NUMBER, "demand rate"), _p("v", NUMBER, "unit purchase cost"), _p("p", NUMBER, "backorder penalty cost")), "eoq_backorders"),
    Algorithm("epq", "Economic production quantity", "Constant-demand inventory control", (_p("K", NUMBER, "fixed setup cost"), _p("h", NUMBER, "holding cost"), _p("d", NUMBER, "demand rate"), _p("v", NUMBER, "unit production cost"), _p("p", NUMBER, "production rate")), "epq", ("economic-production-quantity",)),
    Algorithm("els", "Economic lot scheduling", "Constant-demand inventory control", (_p("n", INTEGER, "number of items"), _p("p", LIST, "production rates"), _p("d", LIST, "demand rates"), _p("h", LIST, "holding costs"), _p("s", LIST, "setup times"), _p("K", LIST, "setup costs")), "els", ("economic-lot-scheduling",)),
    Algorithm("jrp", "Joint replenishment (power-of-two policy)", "Constant-demand inventory control", (_p("n", INTEGER, "number of items"), _p("beta", INTEGER, "power-of-two bound"), _p("h", LIST, "holding costs"), _p("d", LIST, "demand rates"), _p("K", LIST, "item setup costs"), _p("K0", NUMBER, "joint setup cost")), "jrp", ("joint-replenishment",), "Requires IBM CP Optimizer."),
    Algorithm("wagner-whitin", "Wagner-Whitin dynamic lot sizing", "Time-varying deterministic inventory control", (_p("K", NUMBER, "fixed ordering cost"), _p("h", NUMBER, "holding cost"), _p("d", LIST, "period demands"), _p("I0", NUMBER, "initial inventory", 0.0)), "wagner_whitin"),
    Algorithm("wagner-whitin-cplex", "Wagner-Whitin dynamic lot sizing (MILP)", "Time-varying deterministic inventory control", (_p("K", NUMBER, "fixed ordering cost"), _p("h", NUMBER, "holding cost"), _p("d", LIST, "period demands"), _p("I0", NUMBER, "initial inventory", 0.0)), "wagner_whitin_cplex", note="Requires CPLEX."),
    Algorithm("capacitated", "Capacitated dynamic lot sizing", "Time-varying deterministic inventory control", (_p("K", NUMBER, "fixed ordering cost"), _p("v", NUMBER, "unit ordering cost"), _p("h", NUMBER, "holding cost"), _p("d", LIST, "period demands"), _p("I0", INTEGER, "initial inventory"), _p("C", INTEGER, "period capacity")), "capacitated", ("capacitated-sdp",)),
    Algorithm("capacitated-cplex", "Capacitated dynamic lot sizing (MILP)", "Time-varying deterministic inventory control", (_p("K", NUMBER, "fixed ordering cost"), _p("v", NUMBER, "unit ordering cost"), _p("h", NUMBER, "holding cost"), _p("d", LIST, "period demands"), _p("I0", INTEGER, "initial inventory"), _p("C", INTEGER, "period capacity")), "capacitated_cplex", note="Requires CPLEX."),
    Algorithm("planned-backorders", "Dynamic lot sizing with planned backorders", "Time-varying deterministic inventory control", (_p("K", NUMBER, "fixed ordering cost"), _p("v", NUMBER, "unit ordering cost"), _p("h", NUMBER, "holding cost"), _p("p", NUMBER, "backorder cost"), _p("d", LIST, "period demands"), _p("I0", NUMBER, "initial inventory", 0.0)), "planned_backorders", note="Requires CPLEX."),
    Algorithm("newsvendor", "Gaussian newsvendor", "Stochastic inventory control", (_p("mean", NUMBER, "mean demand"), _p("std", NUMBER, "demand standard deviation"), _p("o", NUMBER, "overage cost"), _p("u", NUMBER, "underage cost")), "newsvendor"),
    Algorithm("multi-period-newsvendor", "Poisson multi-period newsvendor", "Stochastic inventory control", (_p("mean", LIST, "period mean demands"), _p("o", NUMBER, "overage cost"), _p("u", NUMBER, "underage cost")), "multi_period_newsvendor"),
    Algorithm("zheng-federgruen", "Optimal stationary (s,S) policy", "Stochastic inventory control", (_p("mu", INTEGER, "Poisson demand rate"), _p("K", NUMBER, "fixed ordering cost"), _p("h", NUMBER, "holding cost"), _p("b", NUMBER, "penalty cost")), "zheng_federgruen"),
    Algorithm("serial-base-stock", "Two-echelon serial base-stock policy", "Stochastic inventory control", (_p("h-W", NUMBER, "warehouse holding cost"), _p("h-R", NUMBER, "retailer holding cost"), _p("b", NUMBER, "retailer penalty cost"), _p("L-R", INTEGER, "retailer lead time"), _p("L-W", INTEGER, "warehouse lead time"), _p("demand-rate", NUMBER, "stationary demand rate"), _p("initial-value", INTEGER, "initial warehouse-echelon search value")), "serial_base_stock"),
    Algorithm("scarf-ss", "Finite-horizon Poisson (s,S) policy", "Stochastic inventory control", (_p("K", NUMBER, "fixed ordering cost"), _p("v", NUMBER, "unit ordering cost"), _p("h", NUMBER, "holding cost"), _p("p", NUMBER, "penalty cost"), _p("d", LIST, "period Poisson demand rates"), _p("max-inv", INTEGER, "inventory bound"), _p("q", NUMBER, "demand truncation quantile", 0.9999), _p("initial-order", BOOLEAN, "allow an order in period zero", True), _p("level", INTEGER, "initial inventory", 0)), "scarf_ss", ("scarf", "ss-policy")),
    Algorithm("capacitated-stochastic", "Capacitated stochastic dynamic program", "Stochastic inventory control", (_p("K", NUMBER, "fixed ordering cost"), _p("B", INTEGER, "order capacity"), _p("v", NUMBER, "unit ordering cost"), _p("h", NUMBER, "holding cost"), _p("p", NUMBER, "penalty cost"), _p("w", NUMBER, "discount factor"), _p("demand", OBJECT, "demand object: {type: poisson|normal|pmf, ...}"), _p("min-inv", INTEGER, "minimum inventory"), _p("max-inv", INTEGER, "maximum inventory"), _p("initial-order", BOOLEAN, "allow an order in period zero", True), _p("level", INTEGER, "initial inventory", 0)), "capacitated_stochastic"),
    Algorithm("multi-item-sdp", "Two-item stochastic dynamic program", "Stochastic inventory control", (_p("K", NUMBER, "joint ordering cost"), _p("v", NUMBER, "unit ordering cost"), _p("h", NUMBER, "holding cost"), _p("p", NUMBER, "penalty cost"), _p("d", LIST, "period Poisson demand rates"), _p("max-inv", INTEGER, "inventory bound"), _p("q", NUMBER, "demand truncation quantile", 0.999), _p("initial-order", BOOLEAN, "allow an order in period zero", True), _p("level", LIST, "initial inventory pair")), "multi_item_sdp"),
    Algorithm("rq", "Static-dynamic (R,Q) approximation", "Stochastic inventory control", (_p("K", NUMBER, "fixed ordering cost"), _p("h", NUMBER, "holding cost"), _p("p", NUMBER, "penalty cost"), _p("d", LIST, "period mean demands"), _p("std-d", LIST, "period demand standard deviations"), _p("I0", NUMBER, "initial inventory", 0.0)), "rq", ("r-q",), "Requires CPLEX."),
    Algorithm("rs", "Bookbinder-Tan (R,S) approximation", "Stochastic inventory control", (_p("K", NUMBER, "fixed ordering cost"), _p("h", NUMBER, "holding cost"), _p("p", NUMBER, "penalty cost"), _p("d", LIST, "period demand rates")), "rs", ("r-s",)),
    Algorithm("ss-shortest-path", "Shortest-path (s,S) approximation", "Stochastic inventory control", (_p("K", NUMBER, "fixed ordering cost"), _p("h", NUMBER, "holding cost"), _p("p", NUMBER, "penalty cost"), _p("d", LIST, "period demand rates"), _p("I0", NUMBER, "initial inventory", float("nan"))), "ss_shortest_path"),
)

BY_NAME: Dict[str, Algorithm] = {}
for _algorithm in ALGORITHMS:
    BY_NAME[_algorithm.name] = _algorithm
    for _alias in _algorithm.aliases:
        BY_NAME[_alias] = _algorithm


def _import(module: str, name: str) -> Any:
    return getattr(importlib.import_module(module), name)


def _class_result(module: str, class_name: str, method: str, result_name: str, values: dict) -> dict:
    instance = _import(module, class_name)(**values)
    return {result_name: getattr(instance, method)()}


def _series_and_horizon(values: dict, minimum: int = 1) -> Tuple[list, int]:
    series = values["series"]
    horizon = values["horizon"]
    if len(series) < minimum:
        raise ValueError("series must contain at least {} observations".format(minimum))
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return series, horizon


def _model_result(fit: Any, forecasts: Any, **extra: Any) -> dict:
    result = {
        "fitted_values": fit.fittedvalues,
        "forecasts": forecasts,
        "parameters": dict(fit.params) if hasattr(fit.params, "keys") else fit.params,
        "residuals": fit.resid,
        "sse": getattr(fit, "sse", None),
        "aic": getattr(fit, "aic", None),
        "bic": getattr(fit, "bic", None),
    }
    result.update(extra)
    return result


def _arima_result(series: list, horizon: int, order: tuple, seasonal_order: tuple, confidence_level: float) -> dict:
    from statsmodels.tsa.arima.model import ARIMA

    fit = ARIMA(series, order=order, seasonal_order=seasonal_order).fit()
    prediction = fit.get_forecast(horizon)
    interval = prediction.conf_int(alpha=1 - confidence_level)
    return _model_result(
        fit,
        prediction.predicted_mean,
        order=order,
        seasonal_order=seasonal_order,
        lower=interval[:, 0],
        upper=interval[:, 1],
        hqic=fit.hqic,
    )


def _run_forecasting(runner: str, values: dict) -> Optional[dict]:
    if runner == "forecast_naive":
        series, horizon = _series_and_horizon(values)
        return {"forecasts": [series[-1]] * horizon}
    if runner == "forecast_seasonal_naive":
        series, horizon = _series_and_horizon(values)
        season_length = values["season_length"]
        if season_length < 1 or len(series) < season_length:
            raise ValueError("season-length must be positive and no greater than the series length")
        season = series[-season_length:]
        return {"forecasts": [season[index % season_length] for index in range(horizon)]}
    if runner == "forecast_drift":
        series, horizon = _series_and_horizon(values, 2)
        slope = (series[-1] - series[0]) / (len(series) - 1)
        return {"forecasts": [series[-1] + slope * step for step in range(1, horizon + 1)], "drift_per_period": slope}
    if runner == "forecast_sma":
        series, horizon = _series_and_horizon(values)
        window = values["window"]
        if window < 1 or window > len(series):
            raise ValueError("window must be positive and no greater than the series length")
        mean = sum(series[-window:]) / window
        return {"forecasts": [mean] * horizon, "window_mean": mean}
    if runner == "forecast_ses":
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing

        series, horizon = _series_and_horizon(values, 2)
        smoothing_level = values["smoothing_level"]
        optimized = values["optimized"] and smoothing_level is None
        fit = SimpleExpSmoothing(series, initialization_method="estimated").fit(smoothing_level=smoothing_level, optimized=optimized)
        return _model_result(fit, fit.forecast(horizon), level=fit.level)
    if runner == "forecast_holt":
        from statsmodels.tsa.holtwinters import Holt

        series, horizon = _series_and_horizon(values, 3)
        smoothing_level = values["smoothing_level"]
        smoothing_trend = values["smoothing_trend"]
        optimized = values["optimized"] and smoothing_level is None and smoothing_trend is None
        fit = Holt(series, damped_trend=values["damped_trend"], initialization_method="estimated").fit(
            smoothing_level=smoothing_level, smoothing_trend=smoothing_trend, optimized=optimized
        )
        return _model_result(fit, fit.forecast(horizon), level=fit.level, trend=fit.trend)
    if runner == "forecast_holt_winters":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        series, horizon = _series_and_horizon(values)
        trend = None if values["trend"] == "none" else values["trend"]
        if trend not in {None, "add", "mul"} or values["seasonal"] not in {"add", "mul"}:
            raise ValueError("trend must be none, add, or mul; seasonal must be add or mul")
        if len(series) < 2 * values["season_length"]:
            raise ValueError("Holt-Winters requires at least two complete seasons")
        fit = ExponentialSmoothing(
            series,
            trend=trend,
            seasonal=values["seasonal"],
            seasonal_periods=values["season_length"],
            damped_trend=values["damped_trend"],
            initialization_method="estimated",
        ).fit(optimized=values["optimized"])
        return _model_result(fit, fit.forecast(horizon), level=fit.level, trend=fit.trend, season=fit.season)
    if runner == "forecast_ar":
        from statsmodels.tsa.ar_model import AutoReg

        series, horizon = _series_and_horizon(values)
        fit = AutoReg(series, lags=values["lags"], trend=values["trend"]).fit()
        forecasts = fit.predict(start=len(series), end=len(series) + horizon - 1)
        return _model_result(fit, forecasts, selected_lags=fit.model.ar_lags, hqic=fit.hqic, sigma2=fit.sigma2)
    if runner == "forecast_ma":
        series, horizon = _series_and_horizon(values)
        return _arima_result(series, horizon, (0, 0, values["q"]), (0, 0, 0, 0), values["confidence_level"])
    if runner == "forecast_arima":
        series, horizon = _series_and_horizon(values)
        if len(values["order"]) != 3 or len(values["seasonal_order"]) != 4:
            raise ValueError("order must have 3 values and seasonal-order must have 4")
        return _arima_result(series, horizon, tuple(values["order"]), tuple(values["seasonal_order"]), values["confidence_level"])
    if runner == "box_cox":
        from scipy.stats import boxcox

        if not values["series"] or min(values["series"]) <= 0:
            raise ValueError("Box-Cox requires a non-empty, strictly positive series")
        if values["lambda"] is None:
            transformed, fitted_lambda = boxcox(values["series"])
        else:
            transformed, fitted_lambda = boxcox(values["series"], lmbda=values["lambda"]), values["lambda"]
        return {"transformed": transformed, "lambda": fitted_lambda}
    if runner == "inverse_box_cox":
        from scipy.special import inv_boxcox

        return {"transformed": inv_boxcox(values["series"], values["lambda"])}
    if runner == "difference":
        import numpy as np
        from statsmodels.tsa.statespace.tools import diff

        result = diff(np.asarray(values["series"], dtype=float), k_diff=values["differences"], k_seasonal_diff=values["seasonal_differences"], seasonal_periods=values["season_length"])
        return {"differenced": result, "observations_dropped": len(values["series"]) - len(result)}
    if runner == "normal_prediction_interval":
        from scipy.stats import norm

        critical = norm.ppf((1 + values["confidence_level"]) / 2)
        return {"lower": values["mean"] - critical * values["std"], "upper": values["mean"] + critical * values["std"], "critical_value": critical}
    if runner == "estimated_prediction_interval":
        import statistics
        from scipy.stats import t

        series = values["series"]
        if len(series) < 2:
            raise ValueError("series must contain at least two observations")
        mean, standard_deviation, degrees = statistics.mean(series), statistics.stdev(series), len(series) - 1
        critical = t.ppf((1 + values["confidence_level"]) / 2, degrees)
        width = critical * standard_deviation * math.sqrt(1 + 1 / len(series))
        return {"mean": mean, "std": standard_deviation, "lower": mean - width, "upper": mean + width, "degrees_of_freedom": degrees, "critical_value": critical}
    if runner == "naive_prediction_interval":
        import statistics
        from scipy.stats import t

        series, horizon = _series_and_horizon(values, 3)
        residuals = [series[index] - series[index - 1] for index in range(1, len(series))]
        residual_std, degrees = statistics.stdev(residuals), len(residuals) - 1
        critical = t.ppf((1 + values["confidence_level"]) / 2, degrees)
        widths = [critical * residual_std * math.sqrt(step) for step in range(1, horizon + 1)]
        return {"forecasts": [series[-1]] * horizon, "lower": [series[-1] - width for width in widths], "upper": [series[-1] + width for width in widths], "residual_std": residual_std, "degrees_of_freedom": degrees}
    if runner == "forecast_errors":
        actual, predicted = values["actual"], values["predicted"]
        if not actual or len(actual) != len(predicted):
            raise ValueError("actual and predicted must be non-empty arrays of equal length")
        errors = [actual_value - predicted_value for actual_value, predicted_value in zip(actual, predicted)]
        absolute = [abs(error) for error in errors]
        squared = [error ** 2 for error in errors]
        percentage = [abs(error / actual_value) * 100 for error, actual_value in zip(errors, actual) if actual_value != 0]
        mse = sum(squared) / len(squared)
        return {"errors": errors, "count": len(errors), "mean_error": sum(errors) / len(errors), "mae": sum(absolute) / len(absolute), "mse": mse, "rmse": math.sqrt(mse), "mape": sum(percentage) / len(percentage) if percentage else None, "mape_excluded_zero_actuals": len(actual) - len(percentage)}
    return None


def _run(algorithm: Algorithm, values: dict) -> Any:
    forecasting_result = _run_forecasting(algorithm.runner, values)
    if forecasting_result is not None:
        return forecasting_result

    eoq_module = "inventoryanalytics.lotsizing.deterministic.constant.eoq"
    class_runners = {
        "eoq": (eoq_module, "eoq", "compute_eoq", "order_quantity"),
        "eoq_all_units": (eoq_module, "eoq_all_units_discounts", "compute_eoq", "order_quantity"),
        "eoq_incremental": (eoq_module, "eoq_incremental_discounts", "compute_eoq", "order_quantity"),
        "eoq_backorders": (eoq_module, "eoq_planned_backorders", "compute_eoq", "order_quantity"),
        "epq": (eoq_module, "epq", "compute_epq", "production_quantity"),
        "els": ("inventoryanalytics.lotsizing.deterministic.constant.els", "els", "compute_els", "cycle_length"),
        "jrp": ("inventoryanalytics.lotsizing.deterministic.constant.jrp", "jrp", "solve", "cost"),
        "zheng_federgruen": ("inventoryanalytics.lotsizing.stochastic.stationary.zhengfedergruen1991", "ZhengFedergruen", "findOptimalPolicy", "policy"),
    }
    if algorithm.runner in class_runners:
        return _class_result(*class_runners[algorithm.runner], values)

    if algorithm.runner == "wagner_whitin":
        instance = _import("inventoryanalytics.lotsizing.deterministic.time_varying.wagnerwhitin1958", "WagnerWhitinDP")(**values)
        return {"order_quantities": instance.order_quantities(), "optimal_cost": instance.optimal_cost()}
    if algorithm.runner == "wagner_whitin_cplex":
        instance = _import("inventoryanalytics.lotsizing.deterministic.time_varying.wagnerwhitin1958", "WagnerWhitinCPLEX")(**values)
        return {"order_quantities": instance.order_quantities(), "optimal_cost": instance.optimal_cost()}
    if algorithm.runner == "capacitated":
        instance = _import("inventoryanalytics.lotsizing.deterministic.time_varying.capacitated", "CapacitatedLotSizingSDP")(**values)
        return {"order_quantities": instance.order_quantities(), "optimal_cost": instance.optimal_cost()}
    if algorithm.runner == "capacitated_cplex":
        instance = _import("inventoryanalytics.lotsizing.deterministic.time_varying.capacitated", "CapacitatedLotSizingCPLEX")(**values)
        return {"order_quantities": instance.order_quantities(), "optimal_cost": instance.optimal_cost()}
    if algorithm.runner == "planned_backorders":
        instance = _import("inventoryanalytics.lotsizing.deterministic.time_varying.planned_backorders", "WagnerWhitinPlannedBackordersCPLEX")(**values)
        instance.model()
        return {"order_quantities": instance.order_quantities(), "optimal_cost": instance.optimal_cost()}
    if algorithm.runner == "newsvendor":
        instance = _import("inventoryanalytics.lotsizing.stochastic.newsvendor", "Newsvendor")(values)
        quantity = instance.crit_frac_solution()
        return {"order_quantity": quantity, "expected_cost": instance.C(quantity)}
    if algorithm.runner == "multi_period_newsvendor":
        instance = _import("inventoryanalytics.lotsizing.stochastic.mpnewsvendor", "MultiPeriodNewsvendor")(values)
        result = instance.optC()
        return {"order_quantity": result.x[0], "expected_cost": result.fun}
    if algorithm.runner == "serial_base_stock":
        compute_y_R = _import("inventoryanalytics.lotsizing.stochastic.stationary.serial", "compute_y_R")
        compute_y_W = _import("inventoryanalytics.lotsizing.stochastic.stationary.serial", "compute_y_W")
        retailer = compute_y_R(values["h_W"], values["h_R"], values["b"], values["L_R"], values["demand_rate"])
        echelon = compute_y_W(values["h_R"] - values["h_W"], values["h_R"], values["b"], values["L_R"], values["h_W"], values["L_W"], values["demand_rate"], values["initial_value"])
        return {"retailer_echelon_level": retailer, "warehouse_echelon_level": echelon, "warehouse_level": echelon - retailer}
    if algorithm.runner == "scarf_ss":
        level = values.pop("level")
        instance = _import("inventoryanalytics.lotsizing.stochastic.nonstationary.sdp", "StochasticLotSizing")(**values)
        cost = instance.f(level)
        return {"optimal_cost": cost, "order_quantity": instance.q(0, level), "policy": instance.extract_sS_policy()}
    if algorithm.runner == "capacitated_stochastic":
        level = values.pop("level")
        demand_spec = values.pop("demand")
        module = "inventoryanalytics.lotsizing.stochastic.nonstationary.capacitated_sdp"
        demand_type = demand_spec.pop("type", None)
        demand_classes = {"poisson": "PoissonDemand", "normal": "NormalDemand", "pmf": "PmfDemand"}
        if demand_type not in demand_classes:
            raise ValueError("demand type must be poisson, normal, or pmf")
        values["d"] = _import(module, demand_classes[demand_type])(**demand_spec)
        instance = _import(module, "StochasticLotSizing")(**values)
        cost = instance.f(level)
        return {"optimal_cost": cost, "order_quantity": instance.q(0, level), "policy": instance.extract_skSk_policy()}
    if algorithm.runner == "multi_item_sdp":
        level = values.pop("level")
        instance = _import("inventoryanalytics.lotsizing.stochastic.nonstationary.sdp_multi_item", "MultiItemStochasticLotSizing")(**values)
        cost = instance.f(level)
        return {"optimal_cost": cost, "order_quantities": instance.q(0, level)}
    if algorithm.runner == "rq":
        instance = _import("inventoryanalytics.lotsizing.stochastic.nonstationary.RQ", "RQ_CPLEX")(**values)
        return {"order_quantities": instance.order_quantities(), "optimal_cost": instance.optimal_cost()}
    if algorithm.runner == "rs":
        instance = _import("inventoryanalytics.lotsizing.stochastic.nonstationary.RS", "RS_DP")(**values)
        return {"order_up_to_levels": instance.order_up_to_levels(), "optimal_cost": instance.optimal_cost()}
    if algorithm.runner == "ss_shortest_path":
        instance = _import("inventoryanalytics.lotsizing.stochastic.nonstationary.sS_shortest_path", "RS_DP")(**values)
        return {"order_up_to_levels": instance.order_up_to_level(), "optimal_cost": instance.optimal_cost(), "policy": instance.reorder_points({"K": values["K"], "h": values["h"], "p": values["p"], "d": values["d"]})}
    raise ValueError("unsupported algorithm: {}".format(algorithm.name))


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if hasattr(value, "tolist"):
        return _to_jsonable(value.tolist())
    if hasattr(value, "to_dict"):
        return _to_jsonable(value.to_dict())
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _to_jsonable(value.item())
    return repr(value)


def _print_catalog(as_json: bool = False) -> None:
    if as_json:
        print(json.dumps([{"method": item.name, "category": item.category, "description": item.label, "aliases": item.aliases, "note": item.note} for item in ALGORITHMS], indent=2))
        return
    category = None
    for item in ALGORITHMS:
        if item.category != category:
            category = item.category
            print("\n{}".format(category))
        note = " [{}]".format(item.note) if item.note else ""
        print("  {:<30} {}{}".format(item.name, item.label, note))


def _parser(algorithm: Optional[Algorithm] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run forecasting and inventory-control algorithms.",
        add_help=False,
        allow_abbrev=False,
    )
    parser.add_argument("--help", action="help", help="show this help message and exit")
    parser.add_argument("-method", choices=sorted(BY_NAME), help="algorithm to run")
    parser.add_argument("-list", action="store_true", help="list available algorithms")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    if algorithm:
        parser.description = "{}: {}".format(algorithm.name, algorithm.label)
        for parameter in algorithm.parameters:
            flag = "--{}".format(parameter.name)
            kwargs = {"type": parameter.value_type, "help": parameter.help}
            if parameter.required:
                kwargs["required"] = True
            else:
                kwargs["default"] = parameter.default
                kwargs["help"] += " (default: {})".format(parameter.default)
            parser.add_argument(flag, dest=parameter.name.replace("-", "_"), **kwargs)
    return parser


def _selection_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("-method", choices=sorted(BY_NAME))
    parser.add_argument("-list", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    preliminary, _ = _selection_parser().parse_known_args(argv)
    if preliminary.list:
        _print_catalog(preliminary.json)
        return 0
    if not preliminary.method:
        parser = _parser()
        parser.print_help()
        print("\nUse -list to see the algorithm catalog.")
        return 0

    algorithm = BY_NAME[preliminary.method]
    parser = _parser(algorithm)
    options = parser.parse_args(argv)
    values = {
        parameter.name.replace("-", "_"): getattr(options, parameter.name.replace("-", "_"))
        for parameter in algorithm.parameters
    }
    try:
        result = _to_jsonable(_run(algorithm, values))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ImportError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())