
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import json
import re
from typing import List, Optional

try:
    model = joblib.load('product_category_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    
    with open('category_mapping.json', 'r') as f:
        category_mapping = json.load(f)
    
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

app = FastAPI(
    title="Product Category Prediction API",
    description="API for predicting product categories based on product details",
    version="1.0.0"
)

class ProductInput(BaseModel):
    product_name: str
    description: str
    brand: Optional[str] = ""
    product_specifications: Optional[str] = ""

class PredictionOutput(BaseModel):
    predicted_category: str
    confidence: float
    top_3_predictions: List[dict]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_categories: int

def combine_features(product_name: str, description: str, brand: str, product_specifications: str) -> str:
    """Combine product features into single text"""
    # Clean text
    def clean_text(text):
        text = re.sub(r'[^\w\s]', ' ', str(text))
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text
    
    name_clean = clean_text(product_name)
    desc_clean = clean_text(description)
    brand_clean = clean_text(brand)
    
    # Handle product specifications
    specs_text = ""
    if product_specifications and product_specifications.strip():
        try:
            specs_clean = clean_text(product_specifications)
            specs_text = f" {specs_clean}"
        except:
            specs_text = ""
    
    combined_text = f"{name_clean} {desc_clean} {brand_clean}{specs_text}"
    return combined_text

@app.get("/")
async def root():
    return {"message": "Product Category Prediction API", "status": "active"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        num_categories=len(label_encoder.classes_)
    )

@app.get("/categories")
async def get_categories():
    """Get list of all possible categories"""
    return {"categories": label_encoder.classes_.tolist()}

@app.post("/predict", response_model=PredictionOutput)
async def predict_category(product_input: ProductInput):
    """
    Predict product category based on product details
    
    - **product_name**: Name of the product
    - **description**: Product description
    - **brand**: Product brand (optional)
    - **product_specifications**: Product specifications (optional)
    """
    try:
        # Combining features
        combined_text = combine_features(
            product_input.product_name,
            product_input.description,
            product_input.brand,
            product_input.product_specifications
        )
        
        # Check if text is not empty
        if not combined_text.strip():
            raise HTTPException(status_code=400, detail="Input text is empty after preprocessing")
        
        # Vectorize text
        text_vectorized = tfidf_vectorizer.transform([combined_text])
        
        # Predict
        probabilities = model.predict_proba(text_vectorized)[0]
        predicted_class_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_class_idx]
        predicted_category = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_categories = label_encoder.inverse_transform(top_3_indices)
        top_3_confidences = probabilities[top_3_indices]
        
        top_predictions = [
            {"category": str(cat), "confidence": float(conf)}
            for cat, conf in zip(top_3_categories, top_3_confidences)
        ]
        
        return PredictionOutput(
            predicted_category=predicted_category,
            confidence=float(confidence),
            top_3_predictions=top_predictions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)








