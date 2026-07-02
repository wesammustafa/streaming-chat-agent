import httpx
import pytest

from app.main import tools_from_env
from app.tools.weather import WeatherLookupTool
from app.tools.weather_live import LiveWeatherTool, format_report


@pytest.fixture
def tool() -> WeatherLookupTool:
    return WeatherLookupTool()


async def test_known_city_returns_fixture_report(tool):
    result = await tool.run("Madrid")
    assert result.ok
    assert result.content == "Madrid: 31°C, sunny"


async def test_lookup_is_case_and_accent_insensitive(tool):
    for spelling in ("são paulo", "SAO PAULO", "  São Paulo  "):
        result = await tool.run(spelling)
        assert result.ok, spelling
        assert "22°C" in result.content


async def test_unknown_city_fails_closed(tool):
    result = await tool.run("Atlantis")
    assert not result.ok
    assert "no weather data" in result.error


async def test_blank_input_fails_closed(tool):
    result = await tool.run("   ")
    assert not result.ok


def test_both_weather_tools_share_the_tool_name():
    # Planners emit "weather_lookup"; which implementation answers is wiring, not planning.
    assert LiveWeatherTool.spec.name == WeatherLookupTool.spec.name == "weather_lookup"


def test_weather_source_defaults_to_fixture(monkeypatch):
    monkeypatch.delenv("WEATHER_SOURCE", raising=False)
    tools = tools_from_env()
    assert any(isinstance(t, WeatherLookupTool) for t in tools)


def test_weather_source_live_selects_open_meteo_tool(monkeypatch):
    monkeypatch.setenv("WEATHER_SOURCE", "live")
    tools = tools_from_env()
    assert any(isinstance(t, LiveWeatherTool) for t in tools)


def test_unknown_weather_source_fails_fast(monkeypatch):
    monkeypatch.setenv("WEATHER_SOURCE", "psychic")
    with pytest.raises(ValueError, match="WEATHER_SOURCE"):
        tools_from_env()


def _mock_open_meteo(geocode_json):
    def handler(request):
        if request.url.host.startswith("geocoding-api"):
            return httpx.Response(200, json=geocode_json)
        return httpx.Response(200, json={"current": {"temperature_2m": 21.0, "weather_code": 0}})

    return httpx.MockTransport(handler)


async def test_live_tool_formats_geocoded_result():
    transport = _mock_open_meteo(
        {"results": [{"name": "Madrid", "country": "Spain", "latitude": 40.4, "longitude": -3.7}]}
    )
    result = await LiveWeatherTool(transport=transport).run("Madrid")
    assert result.ok
    assert result.content == "Madrid, Spain: 21°C, clear sky"


async def test_live_tool_unknown_place_fails_closed():
    result = await LiveWeatherTool(transport=_mock_open_meteo({"results": []})).run("Xyzzy")
    assert not result.ok
    assert result.error and "could not find" in result.error


async def test_live_tool_unreachable_service_fails_closed(monkeypatch):
    monkeypatch.setattr("app.tools.weather_live.GEOCODING_URL", "http://127.0.0.1:1/v1/search")
    result = await LiveWeatherTool(request_timeout_seconds=2).run("Madrid")
    assert not result.ok
    assert result.error and "unreachable" in result.error


@pytest.mark.parametrize(
    ("temperature", "code", "expected"),
    [
        (21.0, 0, "Athens, Greece: 21°C, clear sky"),
        (21.5, 999, "Athens, Greece: 21.5°C, unknown conditions"),
    ],
)
def test_live_report_formatting(temperature, code, expected):
    assert format_report("Athens", "Greece", temperature, code) == expected
