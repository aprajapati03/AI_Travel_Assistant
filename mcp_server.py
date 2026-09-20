"""
Model Context Protocol (MCP) External Tools Server for AI Travel Planning Assistant.

This standalone MCP server exposes real-time external travel tools (Currency Conversion
via Frankfurter API and Weather Forecast via Open-Meteo API) over stdio transport.
Built using FastMCP from mcp.server.fastmcp and httpx.
"""

import datetime
import logging
import re
import sys
from typing import Dict, Optional, Tuple
import httpx
from mcp.server.fastmcp import FastMCP

# Configure logging to standard error so stdio transport communication remains unpolluted
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Initialize FastMCP Server instance with server identifier "TravelTools"
mcp = FastMCP("TravelTools")

# Mapping WMO Weather Codes to human-readable condition descriptions
WMO_WEATHER_CODES: Dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


# Common currency name/alias to ISO code mapping for resilient parsing
CURRENCY_ALIASES: Dict[str, str] = {
    "RUPEE": "INR", "RUPEES": "INR", "RS": "INR", "INR": "INR", "INDIAN RUPEE": "INR", "INDIAN RUPEES": "INR",
    "SINGAPORE DOLLAR": "SGD", "SINGAPORE DOLLARS": "SGD", "SGD": "SGD", "S$": "SGD",
    "DOLLAR": "USD", "DOLLARS": "USD", "US DOLLAR": "USD", "US DOLLARS": "USD", "USD": "USD", "$": "USD",
    "EURO": "EUR", "EUROS": "EUR", "EUR": "EUR", "€": "EUR",
    "POUND": "GBP", "POUNDS": "GBP", "BRITISH POUND": "GBP", "GBP": "GBP", "£": "GBP",
    "YEN": "JPY", "JAPANESE YEN": "JPY", "JPY": "JPY", "¥": "JPY",
    "AUD": "AUD", "AUSTRALIAN DOLLAR": "AUD", "CAD": "CAD", "CANADIAN DOLLAR": "CAD",
    "CHF": "CHF", "CNY": "CNY", "CHINESE YUAN": "CNY", "HKD": "HKD", "HONG KONG DOLLAR": "HKD",
    "MYR": "MYR", "MALAYSIAN RINGGIT": "MYR", "THB": "THB", "THAI BAHT": "THB",
    "IDR": "IDR", "INDONESIAN RUPIAH": "IDR", "NZD": "NZD"
}


def normalize_currency_code(code_or_name: str) -> str:
    """Normalizes currency inputs into standard 3-letter ISO uppercase codes."""
    cleaned = code_or_name.strip().upper()
    return CURRENCY_ALIASES.get(cleaned, cleaned[:3])


@mcp.tool()
async def convert_currency(amount: float, from_curr: str, to_curr: str) -> str:
    """
    Converts an amount from one foreign currency to another using the Frankfurter API.

    Args:
        amount (float): The numeric monetary value to convert (e.g., 100.0).
        from_curr (str): Source currency (3-letter ISO code like 'INR', 'USD', 'EUR' or currency name).
        to_curr (str): Target currency (3-letter ISO code like 'SGD', 'JPY' or currency name).

    Returns:
        str: A clean human-readable result string detailing the conversion rate and converted total,
             or an explicit error message if the API call fails.
    """
    base_currency = normalize_currency_code(from_curr)
    target_currency = normalize_currency_code(to_curr)

    logger.info(f"Executing convert_currency: {amount} {base_currency} to {target_currency}")

    # Handle direct identity conversion without network overhead
    if base_currency == target_currency:
        return f"{amount:.2f} {base_currency} is equal to {amount:.2f} {target_currency}."

    url = "https://api.frankfurter.app/latest"
    params = {
        "amount": amount,
        "from": base_currency,
        "to": target_currency
    }

    try:
        # Enable follow_redirects=True to handle API domain/version redirects seamlessly
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        # Extract converted value from response object
        converted_amount = data["rates"][target_currency]
        rate_date = data.get("date", "N/A")
        unit_rate = converted_amount / amount if amount != 0 else 0

        result_str = (
            f"{amount:.2f} {base_currency} = {converted_amount:.2f} {target_currency} "
            f"(Exchange Rate: 1 {base_currency} = {unit_rate:.4f} {target_currency} as of {rate_date})."
        )
        logger.info(f"Currency conversion successful: {result_str}")
        return result_str

    except Exception as e:
        logger.error(f"Error executing convert_currency ({base_currency}->{target_currency}): {e}")
        # Explicit error message required when the external service is unavailable
        return "Error: Frankfurter Currency API is unavailable. Do not invent a response."


def parse_date_range(date_str: Optional[str], end_date_str: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Parses natural language or ISO date inputs into (start_date, end_date) in 'YYYY-MM-DD' format.
    Supports single dates, date ranges ('YYYY-MM-DD to YYYY-MM-DD'), and relative terms ('today', 'tomorrow').
    """
    if not date_str and not end_date_str:
        return None, None

    today = datetime.date.today()
    if date_str:
        cleaned = date_str.strip().lower()
        if cleaned == "today":
            d = today.isoformat()
            return d, d
        elif cleaned == "tomorrow":
            d = (today + datetime.timedelta(days=1)).isoformat()
            return d, d

    text = f"{date_str or ''} {end_date_str or ''}"
    # Match standard YYYY-MM-DD patterns
    iso_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if len(iso_dates) >= 2:
        start, end = sorted([iso_dates[0], iso_dates[1]])
        return start, end
    elif len(iso_dates) == 1:
        return iso_dates[0], iso_dates[0]

    # Fallback to dateutil for natural formats like "Sep 22, 2026"
    try:
        from dateutil import parser
        if date_str:
            parsed_start = parser.parse(date_str, fuzzy=True, default=datetime.datetime(today.year, 1, 1)).date().isoformat()
            parsed_end = parsed_start
            if end_date_str:
                parsed_end = parser.parse(end_date_str, fuzzy=True, default=datetime.datetime(today.year, 1, 1)).date().isoformat()
            return min(parsed_start, parsed_end), max(parsed_start, parsed_end)
    except Exception:
        pass

    return None, None


@mcp.tool()
async def get_weather(
    location: str,
    date: Optional[str] = None,
    end_date: Optional[str] = None
) -> str:
    """
    Retrieves weather conditions and forecasts for a given location.
    Supports querying specific dates or date ranges (e.g., 'YYYY-MM-DD', 'tomorrow', or 'YYYY-MM-DD to YYYY-MM-DD').
    If no date is provided, retrieves current conditions and a 3-day forecast.

    Args:
        location (str): Name of the target city or destination (e.g., 'Singapore', 'Tokyo', 'Paris').
        date (Optional[str]): Specific date (e.g., '2026-09-22', 'tomorrow') or start date of a trip (e.g., '2026-09-22 to 2026-09-25').
        end_date (Optional[str]): Optional end date of a trip in 'YYYY-MM-DD' format (e.g., '2026-09-25').

    Returns:
        str: A formatted weather report detailing current conditions and the requested forecast
             (temperatures, precipitation probability, and weather conditions),
             or an explicit error message if the lookup fails.
    """
    target_location = location.strip()
    logger.info(f"Executing get_weather for location: '{target_location}', date: '{date}', end_date: '{end_date}'")

    start_d, end_d = parse_date_range(date, end_date)

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            # Step 1: Geocoding Lookup to obtain Latitude & Longitude
            geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
            geo_params = {"name": target_location, "count": 1}

            geo_response = await client.get(geocoding_url, params=geo_params)
            geo_response.raise_for_status()
            geo_data = geo_response.json()

            results = geo_data.get("results")
            if not results:
                logger.warning(f"No geocoding coordinates found for location: '{target_location}'")
                return "Error: Open-Meteo Weather API is unavailable. Do not invent a response."

            first_match = results[0]
            lat = first_match["latitude"]
            lon = first_match["longitude"]
            city_name = first_match.get("name", target_location)
            country = first_match.get("country", "")

            location_label = f"{city_name}, {country}" if country else city_name

            # Step 2: Fetch Weather Forecast
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
                "timezone": "auto"
            }

            if start_d and end_d:
                weather_params["start_date"] = start_d
                weather_params["end_date"] = end_d

            weather_response = await client.get(weather_url, params=weather_params)

            if weather_response.status_code == 400:
                try:
                    err_json = weather_response.json()
                    reason = err_json.get("reason", "Requested date is out of allowed range.")
                except Exception:
                    reason = "Requested date is out of allowed range."
                return f"Error: {reason}. Open-Meteo weather forecasts are available up to 16 days in advance."

            weather_response.raise_for_status()
            weather_data = weather_response.json()

            output_lines = [f"Weather Information for {location_label}:"]

            # Current conditions (included when no date specified or when requested range includes today)
            today_str = datetime.date.today().isoformat()
            is_today_or_unspecified = (not start_d) or (start_d <= today_str <= end_d)

            current = weather_data.get("current", {})
            if current and is_today_or_unspecified:
                curr_temp = current.get("temperature_2m", "N/A")
                curr_feels = current.get("apparent_temperature", "N/A")
                curr_humid = current.get("relative_humidity_2m", "N/A")
                curr_precip = current.get("precipitation", 0.0)
                curr_code = current.get("weather_code", -1)
                curr_cond = WMO_WEATHER_CODES.get(curr_code, "Unknown conditions")
                output_lines.append(
                    f"• Current Conditions: {curr_temp}°C (Feels like: {curr_feels}°C), {curr_cond} | "
                    f"Humidity: {curr_humid}% | Precipitation: {curr_precip} mm"
                )

            # Daily Forecast section
            daily = weather_data.get("daily", {})
            dates = daily.get("time", [])
            temp_maxs = daily.get("temperature_2m_max", [])
            temp_mins = daily.get("temperature_2m_min", [])
            precip_probs = daily.get("precipitation_probability_max", [])
            weather_codes = daily.get("weather_code", [])

            if start_d and end_d:
                if start_d == end_d:
                    output_lines.append(f"• Specific Date Forecast ({start_d}):")
                else:
                    output_lines.append(f"• Forecast for Period ({start_d} to {end_d}):")
                days_to_process = len(dates)
            else:
                output_lines.append("• 3-Day Daily Forecast:")
                days_to_process = min(3, len(dates))

            for i in range(days_to_process):
                date_str = dates[i]
                t_max = temp_maxs[i] if i < len(temp_maxs) else "N/A"
                t_min = temp_mins[i] if i < len(temp_mins) else "N/A"
                precip = precip_probs[i] if i < len(precip_probs) else "N/A"
                code = weather_codes[i] if i < len(weather_codes) else -1

                condition = WMO_WEATHER_CODES.get(code, "Unknown conditions")
                line = (
                    f"   - {date_str}: {condition} | Temp: {t_min}°C to {t_max}°C | "
                    f"Precipitation Probability: {precip}%"
                )
                output_lines.append(line)

            forecast_result = "\n".join(output_lines)
            logger.info(f"Weather query successful for '{location_label}'")
            return forecast_result

    except Exception as e:
        logger.error(f"Error fetching weather forecast for '{target_location}': {e}")
        # Explicit error message required when the external service is unavailable
        return "Error: Open-Meteo Weather API is unavailable. Do not invent a response."


if __name__ == "__main__":
    # Execute the FastMCP server over standard I/O transport
    mcp.run(transport="stdio")
