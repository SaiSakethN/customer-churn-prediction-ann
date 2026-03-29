import streamlit as st

from src.predict import label_encoder_gender, onehot_encoder_geo, predict_churn

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered",
)

st.title("📉 Customer Churn Prediction App")
st.write(
    "This app predicts the probability of customer churn using an "
    "Artificial Neural Network (ANN)."
)

st.subheader("Enter Customer Details")

geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
age = st.slider("Age", 18, 92, 35)
tenure = st.slider("Tenure", 0, 10, 5)
balance = st.number_input("Balance", min_value=0.0, value=60000.0, step=100.0)
num_of_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
is_active_member = st.selectbox("Is Active Member", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=100.0)

if st.button("Predict Churn"):
    probability = predict_churn(
        geography=geography,
        gender=gender,
        age=age,
        balance=balance,
        credit_score=credit_score,
        estimated_salary=estimated_salary,
        tenure=tenure,
        num_of_products=num_of_products,
        has_cr_card=has_cr_card,
        is_active_member=is_active_member,
    )

    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{probability:.2%}")

    if probability >= 0.5:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is likely to stay.")