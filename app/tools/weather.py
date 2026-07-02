"""Fake weather tool: deterministic fixture data, no network.

Exists so a planning model can demonstrate tool routing; the data is canned
on purpose and never changes between runs.
"""

import unicodedata

from app.domain.tools import ToolResult, ToolSpec

_FIXTURES = {
    "madrid": "Madrid: 31°C, sunny",
    "barcelona": "Barcelona: 27°C, clear skies",
    "lisbon": "Lisbon: 24°C, breezy",
    "sao paulo": "São Paulo: 22°C, cloudy",
    "rio de janeiro": "Rio de Janeiro: 28°C, humid",
    "buenos aires": "Buenos Aires: 14°C, windy",
    "mexico city": "Mexico City: 19°C, light rain",
    "london": "London: 16°C, drizzle",
    "new york": "New York: 26°C, partly cloudy",
    "tokyo": "Tokyo: 29°C, humid",
}


def _normalize(city: str) -> str:
    # Accent-insensitive lookup so "São Paulo" and "sao paulo" both hit.
    decomposed = unicodedata.normalize("NFKD", city.strip().lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


class WeatherLookupTool:
    spec = ToolSpec(
        name="weather_lookup",
        description="Returns fixed demo weather for a known list of cities.",
    )

    async def run(self, tool_input: str) -> ToolResult:
        report = _FIXTURES.get(_normalize(tool_input))
        if report is None:
            return ToolResult.failed(f"no weather data for {tool_input.strip() or 'that place'}")
        return ToolResult.succeeded(report)
