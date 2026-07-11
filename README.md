# Telecom Customer Churn Prediction

## Overview
End-to-end ML pipeline to predict customer churn for a telecom company, 
enabling proactive retention strategies.

## Dataset
IBM Telco Customer Churn dataset (7,043 customers, 21 features)

## Approach
- Data cleaning & preprocessing (null handling, encoding)
- Class imbalance handling with `class_weight="balanced"`
- Model comparison: Logistic Regression vs Random Forest vs XGBoost
- Evaluation focused on Recall for churners (class 1)

## Results
| Model | Recall (Churners) |
|-------|------------------|
| Logistic Regression (balanced) | 0.84 |
| Random Forest (balanced) | 0.45 |
| XGBoost (scale_pos_weight) | 0.68 |

**Best model: Logistic Regression with class_weight="balanced"**

## API Usage
Run the Flask app:
```bash
python app.py
```
Send a POST request:
```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"tenure": [12], "MonthlyCharges": [50]}'
```
Response:
```json
{"prediction": [0]}
```

## Project Structure
- `churnprediction.ipynb` — full pipeline notebook (EDA, preprocessing, modeling)
- `app.py` — Flask REST API
- `churn_model.pkl` — saved Logistic Regression model
- `model_columns.pkl` — saved feature columns for inference
