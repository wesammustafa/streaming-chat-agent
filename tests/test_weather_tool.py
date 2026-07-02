import pytest

from app.tools.weather import WeatherLookupTool


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
