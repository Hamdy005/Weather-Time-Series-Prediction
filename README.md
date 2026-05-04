<div align="center">
  <h1>Weather Forecasting with RNN</h1>
  <p><strong>Sequence modeling for hourly temperature prediction across multiple cities.</strong></p>
</div>

---

## Project Overview

This project builds a Recurrent Neural Network (RNN) to forecast hourly temperature using historical weather data. The model learns temporal patterns from multiple Egyptian cities and incorporates categorical context (city and weather code) alongside cyclic time features.

---

## Dataset and Data Exploration

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

## Preprocessing

- Weather codes are mapped to categorical classes (Clear sky, Overcast, Mainly clear, Partly cloudy, Fog, Rain event).
- Time features are encoded cyclically using sine and cosine for hour of day and day of year.
- Temperature is scaled to [0, 1] using MinMaxScaler.
- Sequences of length 24 hours are built per city to avoid crossing city boundaries.

---

## Model Architecture and Training

The model combines three inputs: time features, city embeddings, and weather code embeddings. These are concatenated and passed through stacked SimpleRNN layers, followed by dense layers for regression.

- Inputs: (24, 4) time features, (24,) city ids, (24,) weather codes
- Embeddings: city (6 -> 4), weather code (6 -> 8)
- RNN stack: 64 units (return sequences), 32 units
- Dense: 16 units with L2 regularization, then 1 output
- Optimizer: Adam
- Loss: MSE, Metric: MAE
- Training: 5 epochs, batch size 128

---

## Results

Evaluation on the test set (target scaled to [0, 1]) yielded:

- R2 Score: 0.71
- MAE: 0.0594

---

## Usage

Open and run the notebook to reproduce data preparation, training, and evaluation:

```
jupyter notebook notebook.ipynb
```
