# 🌌 Aurora Dashboard

A real-time aurora monitoring dashboard that displays current space weather conditions including solar wind parameters, Kp-index, and NOAA geomagnetic storm scales.

## Features

- **Real-time Solar Wind Data**
  - Wind Speed (km/s)
  - Particle Density (p/cm³)
  - Temperature (K)

- **Interplanetary Magnetic Field (IMF)**
  - Bt (Total magnetic field strength)
  - Bz (North-South component - critical for aurora predictions)
  - Bx and By components

- **Geomagnetic Activity Indicators**
  - Planetary Kp Index (0-9 scale)
  - Visual Kp indicator bars
  - NOAA G-Scale (Geomagnetic Storm Scale)

- **Aurora Likelihood Assessment**
  - Real-time calculation based on current conditions
  - Overall condition status (Poor/Fair/Good/Excellent)

## Installation

1. Install required packages:
```powershell
pip install -r requirements.txt
```

## Usage

1. Start the dashboard:
```powershell
python aurora.py
```

2. Open your browser to:
```
http://localhost:5000
```

The dashboard will automatically refresh every 60 seconds with the latest data.

## Data Sources

All data is sourced from NOAA Space Weather Prediction Center (SWPC):
- Solar Wind: https://services.swpc.noaa.gov/products/solar-wind/
- Kp Index: https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json
- NOAA Scales: https://services.swpc.noaa.gov/products/noaa-scales.json

## Understanding the Data

### Solar Wind Speed
- **< 400 km/s**: Slow wind
- **400-500 km/s**: Normal
- **> 500 km/s**: Fast wind (favorable for auroras)

### Bz Component (IMF)
- **Negative values**: Favorable for auroras (allows solar wind to couple with Earth's magnetosphere)
- **< -5 nT**: Very favorable
- **Positive values**: Unfavorable (shields Earth)

### Kp Index
- **0-2**: Quiet
- **3-4**: Unsettled
- **5-6**: Minor to moderate storm (auroras visible at higher latitudes)
- **7-9**: Strong to extreme storm (auroras visible at mid-latitudes)

### NOAA G-Scale
- **G0**: No storm
- **G1**: Minor storm
- **G2**: Moderate storm
- **G3**: Strong storm
- **G4**: Severe storm
- **G5**: Extreme storm

## API Endpoint

The dashboard exposes a JSON API at `/api/aurora-data` that returns:
```json
{
  "solar_wind": {
    "time": "2025-11-11 23:00:00.000",
    "speed": 450.2,
    "density": 7.04,
    "temperature": 132935,
    "bt": 10.56,
    "bz": -2.5,
    "bx": 6.99,
    "by": -7.90
  },
  "kp_index": {
    "time": "2025-11-11 18:00:00.000",
    "kp": 3.33
  },
  "noaa_scales": {
    "g_scale": "0",
    "g_text": "none",
    ...
  },
  "aurora_likelihood": "Low - Visible near polar regions",
  "condition_status": "Fair",
  "timestamp": "2025-11-11T23:15:00.123456"
}
```

## License

Free to use and modify!
