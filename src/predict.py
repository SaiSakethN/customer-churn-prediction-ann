from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import tensorflow as tf

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "churn_ann_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

with open(ARTIFACTS_DIR / "label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open(ARTIFACTS_DIR / "onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open(ARTIFACTS_DIR / "scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


def predict_churn(
    geography: str,
    gender: str,
    age: int,
    balance: float,
    credit_score: int,
    estimated_salary: float,
    tenure: int,
    num_of_products: int,
    has_cr_card: int,
    is_active_member: int,
) -> float:
    gender_encoded = label_encoder_gender.transform([gender])[0]

    input_data = pd.DataFrame(
        {
            "CreditScore": [credit_score],
            "Gender": [gender_encoded],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary],
        }
    )

    geo_input = pd.DataFrame({"Geography": [geography]})
    geo_encoded = onehot_encoder_geo.transform(geo_input)
    geo_feature_names = onehot_encoder_geo.get_feature_names_out(["Geography"])
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_feature_names)

    final_input = pd.concat(
        [input_data.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)],
        axis=1,
    )

    final_input_scaled = scaler.transform(final_input)

    prediction = model.predict(final_input_scaled, verbose=0)[0][0]
    return float(prediction)