# 📉 Customer Churn Prediction (ANN + Streamlit)

## 🌐 Live App

https://customer-churn-prediction-ann-01.streamlit.app/

---

## 🧠 Problem Statement

Customer churn is a critical challenge in banking — losing customers directly impacts revenue.

Instead of reacting after customers leave, businesses need a way to **predict churn early and take action**.

This project builds an end-to-end system to:

* Identify high-risk customers
* Predict churn probability
* Support data-driven retention strategies

---

## 🚀 What I Built

This is a **complete machine learning application**, not just a model.

✔ Data preprocessing pipeline (encoding + scaling)
✔ Artificial Neural Network (ANN) for classification
✔ Model training with early stopping
✔ Reusable prediction pipeline
✔ Interactive Streamlit web application
✔ Live deployment for real-time predictions

---

## 📊 Key Insights

* Customer activity and product usage strongly influence churn
* Higher balance does not always mean lower churn
* Certain customer segments show consistently higher churn risk

---

## 🛠 Tech Stack

* Python 3.11
* TensorFlow / Keras (ANN)
* Scikit-learn (preprocessing)
* Pandas / NumPy
* Streamlit (deployment)

---

## 🧩 How It Works

1. User enters customer details
2. Data is preprocessed using saved encoders & scaler
3. ANN model predicts churn probability
4. App returns:

   * Probability score
   * Churn / No churn decision

---

## 📈 Model Performance

* Validation Accuracy: ~86%
* Loss: ~0.33

---

## 🎯 Skills Demonstrated

* Machine Learning (ANN, classification)
* Feature engineering & preprocessing
* Model deployment (Streamlit)
* End-to-end ML pipeline design
* Problem-solving with business context

---

## ▶️ Run Locally

git clone https://github.com/SaiSakethN/customer-churn-prediction-ann.git

cd customer-churn-prediction-ann

py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt

streamlit run app.py

---

## 🔮 Future Improvements

* Hyperparameter tuning
* Model comparison (XGBoost, Random Forest)
* Explainability (SHAP)
* UI/UX improvements

---

## 👨‍💻 Author

Sai Saketh
