from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

ARTIFACTS_DIR = Path("artifacts")
DATA_PATH = Path("data/Churn_Modelling.csv")


def load_and_preprocess_data(
    data_path: Path = DATA_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
):
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path)

    data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

    label_encoder_gender = LabelEncoder()
    data["Gender"] = label_encoder_gender.fit_transform(data["Gender"])

    onehot_encoder_geo = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    geo_encoded = onehot_encoder_geo.fit_transform(data[["Geography"]])

    geo_feature_names = onehot_encoder_geo.get_feature_names_out(["Geography"])
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_feature_names, index=data.index)

    data = pd.concat([data.drop("Geography", axis=1), geo_encoded_df], axis=1)

    X = data.drop("Exited", axis=1)
    y = data["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open(artifacts_dir / "label_encoder_gender.pkl", "wb") as file:
        pickle.dump(label_encoder_gender, file)

    with open(artifacts_dir / "onehot_encoder_geo.pkl", "wb") as file:
        pickle.dump(onehot_encoder_geo, file)

    with open(artifacts_dir / "scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled