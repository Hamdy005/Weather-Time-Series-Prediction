<div align="center">
  <h1>Weather Forecasting with LSTM</h1>
  <p><strong>Sequence modeling for hourly temperature prediction across multiple cities.</strong></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />
    <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras" />
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
    <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
    <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
    <img src="https://img.shields.io/badge/Time_Series_Prediction-8B5CF6?style=for-the-badge&logo=time&logoColor=white" alt="Cyclic Encoding" />
  </p>
</div>

---/s

## 🔍 Project Overview

An AI-powered weather forecasting application using a custom-built, deep Long Short-Term Memory (LSTM) to predict hourly temperatures across major Egyptian cities, achieving an R2 Score of 0.71 and a Mean Absolute Error (MAE) of 0.0594.

---

## 🌐 Web Application Interface

The project includes a feature-rich, responsive web dashboard built with Streamlit, providing regional weather forecasting and historical analysis.

### App Header & Region Selection
<img src="app/assets/app_view_header.png" width="800">

### Regional Summary Cards (Morning & Night Averages)
<img src="app/assets/app_view_summary.png" width="800">

### Detailed Hourly Forecast View
<img src="app/assets/app_view_hourly.png" width="800">

> [!NOTE]
> **Dynamic Data Fetching & Forecasting Mode:**
> - **Past / Current Dates:** If the selected date has already occurred (today or earlier), the application fetches the actual recorded **hourly** weather data directly from the Open-Meteo Archive API — you see a full 24-hour breakdown.
> - **Future Dates:** If the selected date is in the future (up to 16 days ahead), the application dynamically runs the trained LSTM sequence model to predict the **average** temperature and weather conditions for **morning** and **night** only — no hourly breakdown is available for future dates.

---

## 📊 Dataset and Data Exploration

Hourly weather data is collected from the Open-Meteo archive API for six cities: Cairo, Alexandria, Luxor, Aswan, Ismailia, and Sharm. Each city contributes two years of hourly observations (past 730 days + 1 day forecast).

### Combined Dataset Shape
```
(105264, 5)
```

### Sample Rows
```
time                 temperature_2m  weathercode  city_id  city_name
2024-05-03 00:00:00           19.3           0        0      Cairo
2024-05-03 01:00:00           21.5           0        0      Cairo
2024-05-03 02:00:00           20.6           0        0      Cairo
2024-05-03 03:00:00           19.3           0        0      Cairo
2024-05-03 04:00:00           20.0           0        0      Cairo
```

### Data Types
```
time            datetime64[ns]
temperature_2m  float64
weathercode     int64
city_id         int64
city_name       object
```

### City Name Counts
| city_name  | count |
|------------|-------|
| Cairo      | 17544 |
| Alexandria | 17544 |
| Luxor      | 17544 |
| Aswan      | 17544 |
| Ismailia   | 17544 |
| Sharm      | 17544 |

### Weather Code Mapping (Strings)
| code | weather_name    |
|------|------------------|
| 0    | Clear sky        |
| 1    | Overcast         |
| 2    | Mainly clear     |
| 3    | Partly cloudy    |
| 4    | Fog              |
| 5    | Rain event       |

### Weather Code Counts (Mapped Names)
| weather_name   | count |
|----------------|-------|
| Clear sky      | 81893 |
| Overcast       | 11362 |
| Mainly clear   | 7369  |
| Partly cloudy  | 3691  |
| Rain event     | 949   |

Note: Fog was not present in the collected period.

---

## 🧹 Preprocessing

- Weather codes are mapped to categorical classes (Clear sky, Overcast, Mainly clear, Partly cloudy, Fog, Rain event).
- Time features are encoded cyclically using sine and cosine for hour of day and day of year.
- Temperature is scaled to [0, 1] using MinMaxScaler.
- Sequences of length 24 hours are built per city to avoid crossing city boundaries.

---

## 🧠 Model Architecture and Training

The model combines three inputs: time features, city embeddings, and weather code embeddings. These are concatenated and passed through stacked SimpleLSTM layers, followed by dense layers for regression.

| Layer | Type | Shape / Params |
|-------|------|----------------|
| Input 1 | Time features (cyclic) | (24, 4) |
| Input 2 | City IDs → Embedding | 6 → 4 |
| Input 3 | Weather codes → Embedding | 6 → 8 |
| — | Concatenate | merge all inputs |
| LSTM 1 | SimpleLSTM | 64 units, return sequences |
| LSTM 2 | SimpleLSTM | 32 units |
| Dense 1 | Dense + L2 regularization | 16 units |
| Output | Dense | 1 unit |

**Training config:** Optimizer: Adam · Loss: MSE · Metric: MAE · Epochs: 5 · Batch size: 128

---

## 📈 Results

Evaluation on the test set (target scaled to [0, 1]) yielded:

- R2 Score: 0.71
- MAE: 0.0594

---

## 🚀 Usage

Open and run the notebook to reproduce data preparation, training, and evaluation:

```
jupyter notebook notebook.ipynb
```
