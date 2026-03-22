import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import google.generativeai as genai
import cv2
import numpy as np

# --- 1. Configuration & Setup ---
st.set_page_config(page_title="Skin Lesion Diagnostic AI", layout="centered")

# Configure Generative AI (Google Gemini) for Multilingual Explanations
# Replace 'YOUR_API_KEY' with your actual Google AI Studio API Key
# Read API Key from Streamlit Secrets
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except KeyError:
    st.error("API Key not found in Secrets. Please configure GEMINI_API_KEY in Streamlit settings.")

# Use the 2026-appropriate mode
model_genai = genai.GenerativeModel('gemini-3-flash')

# Classes based on common skin lesion datasets (e.g., HAM10000)
CLASS_NAMES = [
    "Actinic keratoses", "Basal cell carcinoma", "Benign keratosis-like lesions", 
    "Dermatofibroma", "Melanoma", "Melanocytic nevi", "Vascular lesions"
]

# --- 2. Image Preprocessing (Dull Razor Algorithm) ---
def preprocess_image(image):
    """Removes hair and noise from the skin lesion image."""
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Blackhat filtering to find hair artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Create mask for hair
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Inpaint the original image using the mask
    dst = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)
    return Image.fromarray(dst)

# --- 3. Vision Transformer (ViT) Inference ---
@st.cache_resource
def load_vit_model():
    """Loads a pre-trained Vision Transformer for skin classification."""
    # Note: In a real project, you would load your custom fine-tuned weights here.
    # For this template, we use a standard ViT pre-trained on ImageNet.
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    model.heads = torch.nn.Linear(model.heads[0].in_features, len(CLASS_NAMES))
    model.eval()
    return model

def predict(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, index = torch.max(probabilities, 1)
    return CLASS_NAMES[index.item()], confidence.item()

# --- 4. Web Interface ---
st.title("🔬 AI Skin Lesion Diagnostic System")
st.write("Upload a photo of the skin lesion for a preliminary AI analysis.")

# Language Selection for Multilingual Support
language = st.selectbox("Select Language / भाषा चुनें", ["English", "Hindi", "Spanish", "Telugu", "French"])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
# Mobile optimization: allow direct camera capture
camera_photo = st.camera_input("Or take a photo with your mobile camera")

input_image = uploaded_file if uploaded_file else camera_photo

if input_image:
    image = Image.open(input_image)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    with st.spinner('Analyzing...'):
        # 1. Preprocessing
        clean_image = preprocess_image(image)
        
        # 2. Vision Transformer Prediction
        vit_model = load_vit_model()
        label, confidence = predict(clean_image, vit_model)
        
        # 3. Generative AI Multilingual Explanation
        prompt = (f"The AI detected a skin lesion as '{label}' with {confidence*100:.1f}% confidence. "
                  f"Explain what this is in {language}, keep it simple, and advise to consult a doctor.")
        response = model_genai.generate_content(prompt)
        
        # --- Display Results ---
        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {confidence*100:.1f}%")
        st.subheader(f"Explanation ({language})")
        st.write(response.text)

st.warning("⚠️ Disclaimer: This is an AI-powered tool for educational purposes only. It is NOT a substitute for professional medical advice.")