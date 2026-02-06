"""Time series forecasting views."""

import json
import logging
import numpy as np
from datetime import datetime, timedelta

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated, require_auth

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This forecast is for educational/informational purposes only. "
    "It is NOT financial advice. Past performance does not guarantee future results. "
    "Always consult a qualified financial advisor before making investment decisions."
)


def random_walk_forecast(prices, days=30, n_simulations=1000):
    """Generate random walk forecast with confidence intervals.

    Random walk assumes future changes are random and independent,
    which is a reasonable baseline for many financial time series.
    """
    if len(prices) < 2:
        return None

    # Calculate daily returns
    returns = np.diff(prices) / prices[:-1]
    mu = np.mean(returns)
    sigma = np.std(returns)

    last_price = prices[-1]

    # Simulate paths
    simulations = np.zeros((n_simulations, days))
    for i in range(n_simulations):
        path = [last_price]
        for _ in range(days):
            change = np.random.normal(mu, sigma)
            path.append(path[-1] * (1 + change))
        simulations[i] = path[1:]

    # Calculate percentiles
    forecast = {
        "days": list(range(1, days + 1)),
        "median": np.median(simulations, axis=0).tolist(),
        "lower_5": np.percentile(simulations, 5, axis=0).tolist(),
        "upper_95": np.percentile(simulations, 95, axis=0).tolist(),
        "lower_25": np.percentile(simulations, 25, axis=0).tolist(),
        "upper_75": np.percentile(simulations, 75, axis=0).tolist(),
    }

    return forecast


def simple_moving_average_forecast(prices, days=30, window=20):
    """Simple moving average based forecast."""
    if len(prices) < window:
        window = len(prices)

    sma = np.mean(prices[-window:])

    # Estimate volatility for confidence bands
    if len(prices) >= 2:
        returns = np.diff(prices) / prices[:-1]
        daily_vol = np.std(returns)
    else:
        daily_vol = 0.02  # Default 2% daily vol

    forecast = {
        "days": list(range(1, days + 1)),
        "median": [sma] * days,
        "lower_5": [sma * (1 - 1.65 * daily_vol * np.sqrt(d)) for d in range(1, days + 1)],
        "upper_95": [sma * (1 + 1.65 * daily_vol * np.sqrt(d)) for d in range(1, days + 1)],
    }

    return forecast


def exponential_smoothing_forecast(prices, days=30, alpha=0.3):
    """Simple exponential smoothing forecast."""
    if len(prices) < 1:
        return None

    # Calculate smoothed value
    smoothed = prices[0]
    for price in prices[1:]:
        smoothed = alpha * price + (1 - alpha) * smoothed

    # Estimate volatility
    if len(prices) >= 2:
        returns = np.diff(prices) / prices[:-1]
        daily_vol = np.std(returns)
    else:
        daily_vol = 0.02

    forecast = {
        "days": list(range(1, days + 1)),
        "median": [smoothed] * days,
        "lower_5": [smoothed * (1 - 1.65 * daily_vol * np.sqrt(d)) for d in range(1, days + 1)],
        "upper_95": [smoothed * (1 + 1.65 * daily_vol * np.sqrt(d)) for d in range(1, days + 1)],
    }

    return forecast


@csrf_exempt
@require_http_methods(["POST"])
@gated
def forecast(request):
    """Generate time series forecast.

    Supports:
    - Financial symbols via yfinance (stocks, crypto, commodities)
    - Custom data series

    Methods:
    - random_walk: Monte Carlo simulation based on historical volatility
    - sma: Simple moving average
    - exp_smooth: Exponential smoothing
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    symbol = data.get("symbol", "").strip().upper()
    custom_data = data.get("data", [])
    days = min(int(data.get("days", 30)), 365)  # Max 1 year
    method = data.get("method", "random_walk")

    prices = []
    symbol_info = {}

    # Get historical data
    if symbol:
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")

            if hist.empty:
                return JsonResponse({"error": f"No data found for symbol: {symbol}"}, status=400)

            prices = hist["Close"].tolist()

            # Get current info
            info = ticker.info
            symbol_info = {
                "symbol": symbol,
                "name": info.get("shortName", info.get("longName", symbol)),
                "currency": info.get("currency", "USD"),
                "current_price": prices[-1] if prices else None,
                "data_points": len(prices),
                "period": "1 year",
            }

        except ImportError:
            return JsonResponse({
                "error": "yfinance not available",
                "suggestion": "Provide custom data in the 'data' field instead"
            }, status=400)
        except Exception as e:
            logger.exception(f"Error fetching {symbol}")
            return JsonResponse({"error": f"Failed to fetch data for {symbol}: {str(e)}"}, status=400)

    elif custom_data:
        if not isinstance(custom_data, list) or len(custom_data) < 5:
            return JsonResponse({"error": "Need at least 5 data points"}, status=400)
        prices = [float(x) for x in custom_data]
        symbol_info = {
            "symbol": "CUSTOM",
            "name": "Custom Data",
            "current_price": prices[-1],
            "data_points": len(prices),
        }

    else:
        return JsonResponse({"error": "Provide either 'symbol' or 'data'"}, status=400)

    # Generate forecast
    if method == "random_walk":
        forecast_data = random_walk_forecast(prices, days)
        method_description = (
            "Random Walk (Monte Carlo): Simulates 1000 possible price paths based on "
            "historical volatility. Assumes future price changes are random and independent."
        )
    elif method == "sma":
        forecast_data = simple_moving_average_forecast(prices, days)
        method_description = (
            "Simple Moving Average: Projects the recent average price forward with "
            "confidence bands based on historical volatility."
        )
    elif method == "exp_smooth":
        forecast_data = exponential_smoothing_forecast(prices, days)
        method_description = (
            "Exponential Smoothing: Weights recent prices more heavily. "
            "Good for data with no clear trend."
        )
    else:
        return JsonResponse({"error": f"Unknown method: {method}. Use: random_walk, sma, exp_smooth"}, status=400)

    if not forecast_data:
        return JsonResponse({"error": "Insufficient data for forecast"}, status=400)

    # Calculate summary stats
    final_median = forecast_data["median"][-1]
    final_low = forecast_data["lower_5"][-1]
    final_high = forecast_data["upper_95"][-1]
    current = symbol_info.get("current_price", prices[-1])

    pct_change_median = ((final_median - current) / current) * 100

    # Track usage
    request.user.increment_queries()

    return JsonResponse({
        "disclaimer": DISCLAIMER,
        "symbol_info": symbol_info,
        "method": method,
        "method_description": method_description,
        "forecast_days": days,
        "forecast": forecast_data,
        "summary": {
            "current_price": round(current, 2),
            "forecast_median": round(final_median, 2),
            "forecast_range_90": [round(final_low, 2), round(final_high, 2)],
            "expected_change_pct": round(pct_change_median, 2),
        },
        "historical": {
            "prices": prices[-60:],  # Last 60 days for charting
            "high_52w": round(max(prices), 2),
            "low_52w": round(min(prices), 2),
        }
    })


@require_http_methods(["GET"])
@require_auth
def quote(request):
    """Get current quote for a symbol."""
    symbol = request.GET.get("symbol", "").strip().upper()

    if not symbol:
        return JsonResponse({"error": "Symbol required"}, status=400)

    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="5d")

        if hist.empty:
            return JsonResponse({"error": f"No data found for: {symbol}"}, status=400)

        current = hist["Close"].iloc[-1]
        prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else current
        change = current - prev_close
        change_pct = (change / prev_close) * 100

        return JsonResponse({
            "symbol": symbol,
            "name": info.get("shortName", info.get("longName", symbol)),
            "price": round(current, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "currency": info.get("currency", "USD"),
            "market_cap": info.get("marketCap"),
            "volume": int(hist["Volume"].iloc[-1]) if "Volume" in hist else None,
            "high_52w": info.get("fiftyTwoWeekHigh"),
            "low_52w": info.get("fiftyTwoWeekLow"),
            "disclaimer": "Data provided for informational purposes only. Not financial advice.",
        })

    except ImportError:
        return JsonResponse({"error": "yfinance not available"}, status=500)
    except Exception as e:
        logger.exception(f"Error fetching quote for {symbol}")
        return JsonResponse({"error": str(e)}, status=500)
