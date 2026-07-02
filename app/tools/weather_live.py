"""Live weather tool over Open-Meteo (free, no API key).

Optional runtime adapter selected with WEATHER_SOURCE=live. It carries the
same tool name as the fixture-based WeatherLookupTool, so planners never know
which implementation is wired; the fixture stays the default and test target.
"""

from typing import Any

import httpx

from app.domain.tools import ToolResult, ToolSpec

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_WMO_CODES = {
    0: "clear sky",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "rime fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light showers",
    81: "showers",
    82: "heavy showers",
    85: "snow showers",
    86: "snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with hail",
}


def format_report(name: str, country: str, temperature: float, weather_code: int) -> str:
    condition = _WMO_CODES.get(weather_code, "unknown conditions")
    label = f"{name}, {country}" if country else name
    # :g renders 21.0 as 21 but keeps 21.5, matching the calculator's formatting care.
    return f"{label}: {temperature:g}°C, {condition}"


class LiveWeatherTool:
    spec = ToolSpec(
        name="weather_lookup",
        description="Looks up current weather for a city via Open-Meteo.",
    )

    def __init__(
        self,
        request_timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._timeout = request_timeout_seconds
        self._transport = transport  # injectable for offline tests

    async def run(self, tool_input: str) -> ToolResult:
        city = tool_input.strip()
        if not city:
            return ToolResult.failed("no city given")
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                place = await self._geocode(client, city)
                if place is None:
                    return ToolResult.failed(f"could not find a place called {city}")
                current = await self._current_weather(client, place["latitude"], place["longitude"])
        except httpx.HTTPError:
            return ToolResult.failed("live weather service is unreachable")
        return ToolResult.succeeded(
            format_report(
                place.get("name", city),
                place.get("country", ""),
                current["temperature_2m"],
                current["weather_code"],
            )
        )

    async def _geocode(self, client: httpx.AsyncClient, city: str) -> dict[str, Any] | None:
        response = await client.get(
            GEOCODING_URL, params={"name": city, "count": 1, "format": "json"}
        )
        response.raise_for_status()
        results = response.json().get("results") or []
        return results[0] if results else None

    async def _current_weather(
        self, client: httpx.AsyncClient, latitude: float, longitude: float
    ) -> dict[str, Any]:
        response = await client.get(
            FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,weather_code",
            },
        )
        response.raise_for_status()
        current: dict[str, Any] = response.json()["current"]
        return current
