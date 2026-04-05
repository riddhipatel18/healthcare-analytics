# Healthcare & Medical Analytics Dashboard

A GitHub-ready machine learning project that predicts patient condition categories (`Diabetic`, `Pneumonia`, `Cancer`) from structured medical data and serves an interactive Streamlit dashboard for EDA, model evaluation, single prediction, and batch scoring.

## Project Structure

```text
healthcare-medical-analytics/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   └── medical_conditions_dataset.csv
├── models/
│   ├── medical_condition_model.joblib
│   └── metrics.json
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_utils.py
    ├── model_utils.py
    └── visuals.py
```

## Features

- Exploratory data analysis dashboard
- Missing value analysis
- Multi-model comparison
- Trained model artifact included
- Single patient prediction form
- Batch CSV prediction with downloadable output
- Streamlit-ready deployment structure

## Local Setup

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Streamlit Deployment

1. Push this project to GitHub.
2. Open Streamlit Community Cloud.
3. Select your repository.
4. Set the main file to `app.py`.
5. Deploy.

## Input Features Used by the Model

- age
- bmi
- blood_pressure
- glucose_levels
- gender
- smoking_status

## Notes

- This project is intended for academic/demo use.
- It is not a substitute for clinical diagnosis.
- The included trained model can be regenerated anytime using `python train.py`.
