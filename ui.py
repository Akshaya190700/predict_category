import streamlit as st
import requests

st.title("🛍️ Product Category Prediction")

api_url = "https://predict-category.onrender.com/predict"  # replace after deployment

with st.form("predict_form"):
    product_name = st.text_input("Product Name")
    description = st.text_area("Description")
    brand = st.text_input("Brand (optional)")
    product_specifications = st.text_area("Product Specifications (optional)")
    submit = st.form_submit_button("Predict")

if submit:
    if not product_name or not description:
        st.warning("Please fill in product name and description.")
    else:
        with st.spinner("Predicting..."):
            payload = {
                "product_name": product_name,
                "description": description,
                "brand": brand,
                "product_specifications": product_specifications
            }
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                data = response.json()
                st.success(f"Predicted Category: {data['predicted_category']}")
                st.write("Confidence:", round(data['confidence'], 3))
                st.subheader("Top 3 Predictions")
                for i, pred in enumerate(data["top_3_predictions"], 1):
                    st.write(f"{i}. {pred['category']} ({round(pred['confidence'], 3)})")
            else:
                st.error("Error: " + response.text)
