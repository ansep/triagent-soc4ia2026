import os
import requests


# Weather API (OpenWeatherMap)

def get_weather(location: str, days_forecast: int = 3) -> dict:
    """
    Get current weather and forecast for a location.
    Uses OpenWeatherMap API.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY not set")

    # Current weather
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": location, "appid": api_key, "units": "metric"}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    current = response.json()

    # 5-day forecast
    forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
    forecast_response = requests.get(forecast_url, params=params, timeout=10)
    forecast_response.raise_for_status()
    forecast_data = forecast_response.json()

    # Extract relevant data
    forecasts = []
    for item in forecast_data["list"][:days_forecast * 8:8]:  # daily
        forecasts.append({
            "date": item["dt_txt"],
            "temp_c": item["main"]["temp"],
            "humidity_pct": item["main"]["humidity"],
            "rain_mm": item.get("rain", {}).get("3h", 0),
            "description": item["weather"][0]["description"]
        })

    return {
        "location": location,
        "current": {
            "temp_c": current["main"]["temp"],
            "humidity_pct": current["main"]["humidity"],
            "description": current["weather"][0]["description"],
            "wind_speed_ms": current["wind"]["speed"],
            "rain_mm": current.get("rain", {}).get("1h", 0)
        },
        "forecast": forecasts,
        "disease_risk": _assess_disease_risk(
            current["main"]["humidity"],
            current["main"]["temp"]
        )
    }


def _assess_disease_risk(humidity: float, temp: float) -> str:
    """Assess disease risk based on weather conditions."""
    if humidity > 80 and 10 <= temp <= 25:
        return "HIGH — favorable conditions for fungal diseases"
    elif humidity > 65 and 15 <= temp <= 30:
        return "MEDIUM — moderate risk"
    else:
        return "LOW — unfavorable conditions for most diseases"
