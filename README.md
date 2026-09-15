# CropAI: Crop Infection Risk Prediction

 **SMART INDIA HACKATHON 2025 Project**

CropAI is a machine-learning web app that predicts **crop infection risk** from field sensor readings, then recommends treatments and calculates dosage, helping farmers act *before* disease hits yield. The demo targets **wheat fields in Punjab, India**.
---

**WORKING PROTOTYPE VIDEO**
https://drive.google.com/file/d/1xtdblWiC2YkrIsvF1Qsj_CaSLZIrFUVB/view?usp=drivesdk 
---

## Features

| | |
|---|---|
| 🔍 **Instant prediction** | Enter six field measurements and get an infection level from 1 to 10 |
| 🎯 **Risk gauge** | Needle gauge with low / medium / high risk bands |
| 📈 **History & trends** | Session history table with a Chart.js trend line |
| 🧪 **Pesticide recommendations** | Treatments matched to the risk tier |
| 🧮 **Dosage calculator** | Scales the recommended treatments to your field size in acres |
| 📊 **Performance tab** | Model metrics dashboard |

## Inputs

| Feature | Unit | Typical range (training data) |
|---|---|---|
| Soil moisture | % | 10 – 60 |
| Soil temperature | °C | 5 – 40 |
| Air temperature | °C | 5 – 42 |
| Air humidity | % | 30 – 95 |
| Soil salinity | dS/m | 0 – 3 |
| NDVI (vegetation index) | – | 0.2 – 0.9 |

## Risk Tiers

| Infection level | Risk | Recommended treatment |
|---|---|---|
| 1 – 3 | 🟢 Low | Neem Oil |
| 4 – 6 | 🟡 Medium | Neem Oil, Sulfur Dust |
| 7 – 10 | 🔴 High | Copper Fungicide, Potassium Bicarbonate, Mancozeb |

## Tech Stack

- **ML:** XGBoost, scikit-learn, pandas, NumPy
- **Backend:** Flask, Flask-CORS
- **Frontend:** HTML, CSS, vanilla JavaScript, Chart.js, Font Awesome

## How It Works

```
Field readings ─► StandardScaler ─► XGBoost classifier ─► Level 1–10 ─► Risk tier ─► Treatments & dosage
```

1. **Dataset:** `scripts/dataset_for_sih.py` generates 5,000 synthetic wheat-field readings. Infection scores come from agronomy-informed rules (extreme soil moisture, temperature stress, humidity above 80 %, high salinity and low NDVI all raise risk) plus Gaussian noise. Scores are normalised to 1–10 and the classes are balanced to 500 samples each.
2. **Training:** `scripts/sih_final_code.py` standardises the features, holds out a stratified 20 % test set and trains an `XGBClassifier` (700 trees, depth 7, learning rate 0.05). It reports exact accuracy and ±1-level accuracy, then saves `model/sih.json` and `model/scaler.pkl`.
3. **Serving:** `app.py` loads the model and scaler and serves the UI and prediction API.

`scripts/sih_classification.py` is an earlier experiment that treats the task as regression (`XGBRegressor`) and plots a confusion matrix of rounded predictions.

## API

### `POST /predict`

```json
{
  "soil_moisture": 30.0,
  "soil_temp": 22.0,
  "air_temp": 26.0,
  "air_humidity": 65.0,
  "soil_salinity": 1.2,
  "ndvi": 0.70
}
```

Response:

```json
{
  "infection_level": 4,
  "risk_level": "medium",
  "recommended_pesticides": ["Neem Oil", "Sulfur Dust"],
  "confidence": 87.42
}
```

### `POST /calculate_dosage`

`{ "field_size": 2.5 }` returns per-pesticide totals for wheat, using per-hectare base rates.

## Getting Started

```bash
git clone https://github.com/KINGofUP81/cropgridAI.git
cd cropgridAI
pip install -r requirements.txt

# optional: regenerate the dataset and retrain
python scripts/dataset_for_sih.py
python scripts/sih_final_code.py

python app.py
```

Open **http://127.0.0.1:5001**.

## Project Structure

```
cropgridAI/
├── app.py                              # Flask server: UI + /predict + /calculate_dosage
├── requirements.txt
├── data/
│   └── wheat_infection_punjab_5000.csv # synthetic training dataset
├── model/
│   ├── sih.json                        # trained XGBoost model
│   └── scaler.pkl                      # fitted StandardScaler
├── scripts/
│   ├── dataset_for_sih.py              # dataset generator
│   ├── sih_final_code.py               # final classifier training
│   └── sih_classification.py           # earlier regression experiment
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/app.js
```

## Current Limitations

This is a hackathon prototype, so a few pieces are still stand-ins:

- **Synthetic data:** the model learns hand-written rules, not field observations, so its accuracy says nothing yet about real-world performance.
- **Confidence score** is a placeholder random value, not the model's predicted probability.
- **Performance tab** shows static example metrics rather than values computed from the test set.
- **Dosage:** the UI calculates per acre in the browser, while `/calculate_dosage` uses per-hectare rates, and the two rate tables differ.
- **History** lives in browser memory and is lost on refresh.
- The frontend calls `http://127.0.0.1:5001` directly, so it only works when served locally.

## Roadmap

- [ ] Train on real sensor and satellite (Sentinel-2 NDVI) data
- [ ] Use calibrated `predict_proba` output as the confidence score
- [ ] Compute metrics from the held-out set and serve them to the dashboard
- [ ] Unify dosage logic in the backend with a unit selector
- [ ] Persist prediction history server-side
- [ ] Support crops beyond wheat


