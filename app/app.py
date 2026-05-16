from __future__ import annotations

import base64
from collections import Counter
from contextlib import nullcontext
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from tensorflow.keras.layers import Concatenate, Dense, Embedding, Input, LSTM
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import L2

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODEL_PATH = ROOT_DIR / "model.h5"
ASSETS_DIR = APP_DIR / "assets"
CSS_PATH = APP_DIR / "style.css"
BACKGROUND_PATH = ASSETS_DIR / "background.jpg"
FAVICON_PATH = ASSETS_DIR / "background1.jpg"
FALLBACK_BACKGROUND_URL = "https://images.unsplash.com/photo-1504608524841-42fe6f032b4b?q=80&w=765&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

SEQ_LEN = 24
MIN_TEMP = 5.8
MAX_TEMP = 48.8
FORECAST_HORIZON_DAYS = 16

ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"

REGIONS = {
    "North Egypt": [
        {"name": "Alexandria", "lat": 31.2001, "lon": 29.9187, "id": 1},
        {"name": "Sharm", "lat": 27.9158, "lon": 34.3300, "id": 5},
    ],
    "Middle Egypt": [
        {"name": "Cairo", "lat": 30.0444, "lon": 31.2357, "id": 0},
        {"name": "Ismailia", "lat": 30.5965, "lon": 32.2715, "id": 4},
    ],
    "Upper Egypt": [
        {"name": "Luxor", "lat": 25.6872, "lon": 32.6396, "id": 2},
        {"name": "Aswan", "lat": 24.0889, "lon": 32.8998, "id": 3},
    ],
}

WEATHER_CODE_NAMES = {
    0: "Clear sky",
    1: "Overcast",
    2: "Mainly clear",
    3: "Partly cloudy",
    4: "Fog",
    5: "Mostly cloudy",
}


class PatchedEmbedding(Embedding):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)


class PatchedDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)


class PatchedLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)


def map_weather_code(code: int) -> int:
    if code == 0:
        return 0
    if code == 3:
        return 1
    if code == 1:
        return 2
    if code == 2:
        return 3
    if code in (45, 48):
        return 4
    return 5


def encode_cyclic(series: pd.Series, max_value: int) -> tuple[pd.Series, pd.Series]:
    radians = 2 * np.pi * series / max_value
    return np.sin(radians), np.cos(radians)


def scale_temperature(value: np.ndarray) -> np.ndarray:
    return (value - MIN_TEMP) / (MAX_TEMP - MIN_TEMP)


def unscale_temperature(value: np.ndarray) -> np.ndarray:
    return value * (MAX_TEMP - MIN_TEMP) + MIN_TEMP


def build_lstm_model() -> Model:
    time_input = Input(shape=(SEQ_LEN, 4), name="Time Columns")
    city_input = Input(shape=(SEQ_LEN,), name="Cities")
    weather_input = Input(shape=(SEQ_LEN,), name="Weather Code")

    city_embedding = Embedding(6, 4)(city_input)
    weather_embedding = Embedding(6, 8)(weather_input)

    concatenated = Concatenate(axis=-1)([time_input, city_embedding, weather_embedding])

    x = LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(concatenated)
    x = LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=False)(x)
    x = Dense(16, activation="relu", kernel_regularizer=L2(1e-4))(x)
    output = Dense(1)(x)

    model = Model(inputs=[time_input, city_input, weather_input], outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


@st.cache_resource
def get_model() -> Model:
    if MODEL_PATH.exists():
        return load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={
                "Embedding": PatchedEmbedding,
                "Dense": PatchedDense,
                "LSTM": PatchedLSTM,
            },
        )
    return build_lstm_model()


@st.cache_data(ttl=900)
def fetch_hourly(city: dict, start_date: date, end_date: date) -> pd.DataFrame:
    today = date.today()
    forecast_limit = today + timedelta(days=FORECAST_HORIZON_DAYS)

    if end_date > forecast_limit:
        end_date = forecast_limit
    if start_date > end_date:
        return pd.DataFrame()

    def fetch_from_api(base_url: str, api_start: date, api_end: date) -> pd.DataFrame:
        params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "hourly": ["temperature_2m", "weathercode"],
            "start_date": api_start.isoformat(),
            "end_date": api_end.isoformat(),
            "timezone": "Africa/Cairo",
        }
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ConnectionError:
            st.error("🌐 No internet connection detected. Please check your network and try again.")
            st.stop()
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ An error occurred while fetching weather data: {e}")
            st.stop()

        if "hourly" not in data:
            return pd.DataFrame()
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        return df

    frames = []
    archive_end = min(end_date, today - timedelta(days=1))
    if start_date <= archive_end:
        frames.append(fetch_from_api(ARCHIVE_API, start_date, archive_end))

    forecast_start = max(start_date, today)
    if forecast_start <= end_date:
        frames.append(fetch_from_api(FORECAST_API, forecast_start, end_date))

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time"])
    return combined.sort_values("time").reset_index(drop=True)


def build_sequences(df: pd.DataFrame, target_date: date) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[datetime], list[str]]:
    num_cols = ["hour_sin", "hour_cos", "dayofyear_sin", "dayofyear_cos"]
    df = df.sort_values("time").reset_index(drop=True)

    x_num, x_wcode, x_city, times, weather_names = [], [], [], [], []
    for i in range(SEQ_LEN, len(df)):
        current_time = df.loc[i, "time"]
        if current_time.date() != target_date:
            continue
        x_num.append(df.loc[i - SEQ_LEN : i - 1, num_cols].values)
        x_wcode.append(df.loc[i - SEQ_LEN : i - 1, "weathercode_class"].values)
        x_city.append(df.loc[i - SEQ_LEN : i - 1, "city_id"].values)
        times.append(current_time)
        weather_names.append(df.loc[i, "weather_name"])

    return (
        np.array(x_num, dtype=np.float32),
        np.array(x_wcode, dtype=np.int32),
        np.array(x_city, dtype=np.int32),
        times,
        weather_names,
    )


def is_daylight(hour: int) -> bool:
    return 6 <= hour < 18


def icon_svg(icon_key: str) -> str:
    icons = {
        "clear_day": """
        <svg class=\"icon\" viewBox=\"0 0 64 64\" aria-hidden=\"true\">
          <circle class=\"sun-core\" cx=\"32\" cy=\"32\" r=\"11\" />
          <g class=\"sun-rays\" stroke-width=\"3\" stroke-linecap=\"round\">
            <line x1=\"32\" y1=\"5\" x2=\"32\" y2=\"15\" />
            <line x1=\"32\" y1=\"49\" x2=\"32\" y2=\"59\" />
            <line x1=\"5\" y1=\"32\" x2=\"15\" y2=\"32\" />
            <line x1=\"49\" y1=\"32\" x2=\"59\" y2=\"32\" />
            <line x1=\"12\" y1=\"12\" x2=\"19\" y2=\"19\" />
            <line x1=\"45\" y1=\"45\" x2=\"52\" y2=\"52\" />
            <line x1=\"12\" y1=\"52\" x2=\"19\" y2=\"45\" />
            <line x1=\"45\" y1=\"19\" x2=\"52\" y2=\"12\" />
          </g>
        </svg>
        """,
        "clear_night": """
        <svg class=\"icon\" viewBox=\"0 0 64 64\" aria-hidden=\"true\">
          <circle class=\"moon-core\" cx=\"30\" cy=\"34\" r=\"14\" />
          <circle class=\"moon-cut\" cx=\"38\" cy=\"26\" r=\"14\" />
        </svg>
        """,
        "overcast_day": """
        <svg class=\"icon\" viewBox=\"0 0 64 64\" aria-hidden=\"true\">
          <g class=\"cloud-body\">
            <circle cx=\"22\" cy=\"34\" r=\"10\" />
            <circle cx=\"34\" cy=\"28\" r=\"12\" />
            <circle cx=\"48\" cy=\"34\" r=\"10\" />
            <rect x=\"16\" y=\"34\" width=\"36\" height=\"14\" rx=\"7\" />
          </g>
        </svg>
        """,
        "overcast_night": """
        <svg class=\"icon\" viewBox=\"0 0 64 64\" aria-hidden=\"true\">
          <g>
            <circle class=\"moon-core\" cx=\"40\" cy=\"20\" r=\"14\" />
            <circle class=\"moon-cut\" cx=\"48\" cy=\"14\" r=\"14\" />
            <g class=\"cloud-body\">
              <circle cx=\"18\" cy=\"40\" r=\"7\" />
              <circle cx=\"28\" cy=\"36\" r=\"9\" />
              <circle cx=\"38\" cy=\"40\" r=\"7\" />
              <rect x=\"14\" y=\"40\" width=\"28\" height=\"10\" rx=\"5\" />
            </g>
          </g>
        </svg>
        """,
        "mainly_clear_day": """
        <svg class=\"icon\" viewBox=\"0 0 64 64\" aria-hidden=\"true\">
          <circle class=\"sun-core\" cx=\"26\" cy=\"26\" r=\"9\" />
          <g class=\"sun-rays\" stroke-width=\"2.5\" stroke-linecap=\"round\">
            <line x1=\"26\" y1=\"8\" x2=\"26\" y2=\"14\" />
            <line x1=\"8\" y1=\"26\" x2=\"14\" y2=\"26\" />
            <line x1=\"38\" y1=\"26\" x2=\"44\" y2=\"26\" />
            <line x1=\"13\" y1=\"13\" x2=\"18\" y2=\"18\" />
          </g>
          <g class=\"cloud-body\">
            <circle cx=\"30\" cy=\"36\" r=\"9\" />
            <circle cx=\"42\" cy=\"34\" r=\"10\" />
            <rect x=\"22\" y=\"36\" width=\"28\" height=\"12\" rx=\"6\" />
          </g>
        </svg>
        """,
        "mainly_clear_night": """
        <svg class=\"icon\" viewBox=\"0 0 64 64\" aria-hidden=\"true\">
          <g>
            <circle class=\"moon-core\" cx=\"40\" cy=\"20\" r=\"13\" />
            <circle class=\"moon-cut\" cx=\"47\" cy=\"14\" r=\"13\" />
            <g class=\"cloud-body\">
              <circle cx=\"18\" cy=\"40\" r=\"6\" />
              <circle cx=\"27\" cy=\"37\" r=\"8\" />
              <rect x=\"14\" y=\"40\" width=\"22\" height=\"9\" rx=\"5\" />
            </g>
          </g>
        </svg>
        """,
        "partly_cloudy_day": """
        <svg class=\"icon\" viewBox=\"0 0 64 64\" aria-hidden=\"true\">
          <circle class=\"sun-core\" cx=\"24\" cy=\"26\" r=\"10\" />
          <g class=\"sun-rays\" stroke-width=\"2.5\" stroke-linecap=\"round\">
            <line x1=\"24\" y1=\"8\" x2=\"24\" y2=\"14\" />
            <line x1=\"8\" y1=\"26\" x2=\"14\" y2=\"26\" />
            <line x1=\"34\" y1=\"26\" x2=\"40\" y2=\"26\" />
            <line x1=\"14\" y1=\"14\" x2=\"18\" y2=\"18\" />
          </g>
          <g class=\"cloud-body\">
            <circle cx=\"30\" cy=\"38\" r=\"10\" />
            <circle cx=\"42\" cy=\"34\" r=\"12\" />
            <rect x=\"22\" y=\"38\" width=\"30\" height=\"14\" rx=\"7\" />
          </g>
        </svg>
        """,
        "partly_cloudy_night": """
        <svg class=\"icon\" viewBox=\"0 0 64 64\" aria-hidden=\"true\">
          <g>
            <circle class=\"moon-core\" cx=\"40\" cy=\"20\" r=\"13\" />
            <circle class=\"moon-cut\" cx=\"47\" cy=\"14\" r=\"13\" />
            <g class=\"cloud-body\">
              <circle cx=\"18\" cy=\"41\" r=\"7\" />
              <circle cx=\"29\" cy=\"37\" r=\"9\" />
              <rect x=\"14\" y=\"41\" width=\"24\" height=\"10\" rx=\"5\" />
            </g>
          </g>
        </svg>
        """,
        "fog_day": """
        <svg class=\"icon\" viewBox=\"0 0 64 64\" aria-hidden=\"true\">
          <g class=\"cloud-body\">
            <circle cx=\"22\" cy=\"28\" r=\"9\" />
            <circle cx=\"34\" cy=\"24\" r=\"11\" />
            <circle cx=\"48\" cy=\"28\" r=\"9\" />
            <rect x=\"16\" y=\"28\" width=\"36\" height=\"12\" rx=\"6\" />
          </g>
          <g class=\"fog-lines\" stroke-width=\"3\" stroke-linecap=\"round\">
            <line x1=\"14\" y1=\"46\" x2=\"50\" y2=\"46\" />
            <line x1=\"10\" y1=\"54\" x2=\"46\" y2=\"54\" />
          </g>
        </svg>
        """,
        "fog_night": """
        <svg class=\"icon\" viewBox=\"0 0 64 64\" aria-hidden=\"true\">
          <g>
            <circle class=\"moon-core\" cx=\"40\" cy=\"20\" r=\"12\" />
            <circle class=\"moon-cut\" cx=\"47\" cy=\"14\" r=\"12\" />
            <g class=\"cloud-body\">
              <circle cx=\"16\" cy=\"38\" r=\"6\" />
              <circle cx=\"26\" cy=\"35\" r=\"8\" />
              <circle cx=\"36\" cy=\"38\" r=\"6\" />
              <rect x=\"12\" y=\"38\" width=\"26\" height=\"9\" rx=\"5\" />
            </g>
            <g class=\"fog-lines\" stroke-width=\"3\" stroke-linecap=\"round\">
              <line x1=\"10\" y1=\"48\" x2=\"42\" y2=\"48\" />
              <line x1=\"8\" y1=\"55\" x2=\"38\" y2=\"55\" />
            </g>
          </g>
        </svg>
        """,
        "rain_day": """
        <svg class=\"icon\" viewBox=\"0 0 64 64\" aria-hidden=\"true\">
          <g class=\"cloud-body\">
            <circle cx=\"22\" cy=\"30\" r=\"9\" />
            <circle cx=\"34\" cy=\"26\" r=\"11\" />
            <circle cx=\"48\" cy=\"30\" r=\"9\" />
            <rect x=\"16\" y=\"30\" width=\"36\" height=\"12\" rx=\"6\" />
          </g>
          <g class=\"rain-drops\" stroke-width=\"3\" stroke-linecap=\"round\">
            <line x1=\"24\" y1=\"46\" x2=\"20\" y2=\"56\" />
            <line x1=\"34\" y1=\"46\" x2=\"30\" y2=\"56\" />
            <line x1=\"44\" y1=\"46\" x2=\"40\" y2=\"56\" />
          </g>
        </svg>
        """,
        "rain_night": """
        <svg class=\"icon\" viewBox=\"0 0 64 64\" aria-hidden=\"true\">
          <g>
            <circle class=\"moon-core\" cx=\"40\" cy=\"20\" r=\"12\" />
            <circle class=\"moon-cut\" cx=\"47\" cy=\"14\" r=\"12\" />
            <g class=\"cloud-body\">
              <circle cx=\"16\" cy=\"38\" r=\"6\" />
              <circle cx=\"26\" cy=\"35\" r=\"8\" />
              <circle cx=\"36\" cy=\"38\" r=\"6\" />
              <rect x=\"12\" y=\"38\" width=\"26\" height=\"9\" rx=\"5\" />
            </g>
            <g class=\"rain-drops\" stroke-width=\"3\" stroke-linecap=\"round\">
              <line x1=\"18\" y1=\"46\" x2=\"14\" y2=\"56\" />
              <line x1=\"28\" y1=\"46\" x2=\"24\" y2=\"56\" />
              <line x1=\"38\" y1=\"46\" x2=\"34\" y2=\"56\" />
            </g>
          </g>
        </svg>
        """,
    }
    return icons.get(icon_key, icons["overcast_day"])


def weather_icon(weather_name: str, hour: int) -> str:
    daylight = is_daylight(hour)
    if weather_name == "Clear sky":
        key = "clear_day" if daylight else "clear_night"
    elif weather_name == "Overcast":
        key = "overcast_day" if daylight else "overcast_night"
    elif weather_name == "Mainly clear":
        key = "mainly_clear_day" if daylight else "mainly_clear_night"
    elif weather_name == "Partly cloudy":
        key = "partly_cloudy_day" if daylight else "partly_cloudy_night"
    elif weather_name == "Mostly cloudy":
        key = "overcast_day" if daylight else "overcast_night"
    elif weather_name == "Fog":
        key = "fog_day" if daylight else "fog_night"
    else:
        key = "rain_day" if daylight else "rain_night"
    return icon_svg(key)


def render_city_section(
    city: dict,
    rows: list[dict],
    region: str,
    selected_date: date,
    show_hourly: bool,
) -> str:
    if not rows:
        return (
            f"<div class=\"city-card empty\"><div class=\"city-header\"><div><h3>{city['name']}</h3><p>{region}</p></div></div>"
            "<div class=\"empty-state\">Not enough data for this day.</div></div>"
        )

    morning = [row for row in rows if row["phase"] == "Morning"]
    night = [row for row in rows if row["phase"] == "Night"]

    def summarize_phase(phase_rows: list[dict], phase_label: str) -> tuple[str, str, str]:
        if not phase_rows:
            return "--", "--", ""
        avg_temp = sum(item["temperature"] for item in phase_rows) / len(phase_rows)
        counts = Counter(item["weather"] for item in phase_rows)
        top_weather = counts.most_common(1)[0][0]
        icon_hour = 9 if phase_label == "Morning" else 21
        icon = weather_icon(top_weather, icon_hour)
        return f"{avg_temp:.1f} C", top_weather, icon

    morning_temp, morning_weather, morning_icon = summarize_phase(morning, "Morning")
    night_temp, night_weather, night_icon = summarize_phase(night, "Night")

    summary_html = (
        "<div class=\"summary-row\">"
        "<div class=\"summary-card\">"
        "<div class=\"summary-title\">Morning average</div>"
        f"<div class=\"summary-icon\">{morning_icon}</div>"
        f"<div class=\"summary-temp\">{morning_temp}</div>"
        f"<div class=\"summary-meta\">{morning_weather}</div>"
        "</div>"
        "<div class=\"summary-card\">"
        "<div class=\"summary-title\">Night average</div>"
        f"<div class=\"summary-icon\">{night_icon}</div>"
        f"<div class=\"summary-temp\">{night_temp}</div>"
        f"<div class=\"summary-meta\">{night_weather}</div>"
        "</div>"
        "</div>"
    )

    hourly_html = ""
    if show_hourly:
        cards_html = ""
        for row in rows:
            time_label = row["time"].strftime("%I:%M %p").lstrip("0")
            temp_label = f"{row['temperature']:.1f} C"
            weather_label = row["weather"]
            phase_class = "day" if row["is_day"] else "night"
            cards_html += (
                f"<div class=\"hour-card {phase_class}\">"
                f"<div class=\"icon-wrap\">{row['icon']}</div>"
                f"<div class=\"hour-temp\">{temp_label}</div>"
                f"<div class=\"hour-meta\">{weather_label}</div>"
                f"<div class=\"hour-time\">{time_label}</div>"
                "</div>"
            )
        hourly_html = f"<div class=\"hourly-scroll\">{cards_html}</div>"

    header_date = selected_date.strftime("%B %d, %Y")
    return (
        f"<div class=\"city-card\">"
        f"<div class=\"city-header\">"
        f"<div><h3>{city['name']}</h3><p>{region}</p></div>"
        f"<div class=\"city-date\">{header_date}</div>"
        f"</div>"
        f"{summary_html}"
        f"{hourly_html}"
        f"</div>"
    )


def load_css() -> None:
    css = CSS_PATH.read_text(encoding="utf-8")
    if BACKGROUND_PATH.exists():
        bg_bytes = BACKGROUND_PATH.read_bytes()
        bg_b64 = base64.b64encode(bg_bytes).decode("ascii")
        bg_source = f"data:image/jpeg;base64,{bg_b64}"
    else:
        bg_source = FALLBACK_BACKGROUND_URL
    css = css.replace("{{background_image}}", bg_source)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="Egypt Weather",
        page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else "🌤️",
        layout="wide"
    )
    load_css()

    if "date_picker" not in st.session_state:
        st.session_state.date_picker = date.today()

    st.markdown(
        "<div class=\"app-shell\">"
        "<div class=\"hero\">"
        "<div class=\"hero-text\">"
        "<h1>Egypt Weather Forecasting System</h1>"
        "<p>Regional forecasts powered by sequence modeling</p>"
        "</div>"
        "<div class=\"hero-pill\">Daily averages</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    def go_prev():
        st.session_state.date_picker -= timedelta(days=1)

    def go_next():
        st.session_state.date_picker += timedelta(days=1)

    region = st.radio("Select Region", list(REGIONS.keys()), horizontal=True)

    nav_left, nav_center, nav_right = st.columns([1, 4, 1])
    with nav_left:
        st.button("Previous", key="prev_day", on_click=go_prev)
    with nav_center:
        today = date.today()
        max_date = today + timedelta(days=FORECAST_HORIZON_DAYS)
        st.date_input(
            "Date",
            min_value=date(2010, 1, 1),
            max_value=max_date,
            key="date_picker",
        )
    with nav_right:
        st.button("Next", key="next_day", on_click=go_next)

    selected_date = st.session_state.date_picker
    start_date = selected_date - timedelta(days=1)

    today = date.today()
    is_future = selected_date > today
    is_trained_date = selected_date <= today

    spinner_ctx = st.spinner("Forecasting daily averages...") if is_future else nullcontext()
    with spinner_ctx:
        model = get_model()

        city_sections = []
        for city in REGIONS[region]:
            df = fetch_hourly(city, start_date, selected_date)
            if df.empty:
                city_sections.append((city, []))
                continue

            df["city_id"] = city["id"]
            df["weathercode_class"] = df["weathercode"].apply(map_weather_code)
            df["weather_name"] = df["weathercode_class"].map(WEATHER_CODE_NAMES)

            hour_sin, hour_cos = encode_cyclic(df["time"].dt.hour, 24)
            day_sin, day_cos = encode_cyclic(df["time"].dt.dayofyear, 365)
            df["hour_sin"] = hour_sin
            df["hour_cos"] = hour_cos
            df["dayofyear_sin"] = day_sin
            df["dayofyear_cos"] = day_cos

            x_num, x_wcode, x_city, times, weather_names = build_sequences(df, selected_date)
            if len(times) == 0:
                city_sections.append((city, []))
                continue

            preds = model.predict([x_num, x_wcode, x_city], verbose=0).ravel()
            temps = unscale_temperature(preds)

            rows = []
            for time_value, temp, weather_name in zip(times, temps, weather_names):
                rows.append(
                    {
                        "time": time_value,
                        "temperature": float(temp),
                        "weather": weather_name,
                        "icon": weather_icon(weather_name, time_value.hour),
                        "is_day": is_daylight(time_value.hour),
                        "phase": "Morning" if 6 <= time_value.hour < 18 else "Night",
                    }
                )

            city_sections.append((city, rows))

    hide_hourly_on_date = selected_date.month == 5 and selected_date.day == 4
    show_hourly = is_trained_date and not hide_hourly_on_date
    all_cards_html = "".join(
        render_city_section(city, rows, region, selected_date, show_hourly=show_hourly)
        for city, rows in city_sections
    )
    st.markdown(
        f"<div class=\"city-grid\">{all_cards_html}</div></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
