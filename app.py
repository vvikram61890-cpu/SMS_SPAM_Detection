import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import numpy as np

# --- Configuration ---
# *** IMPORTANT: REPLACE THIS PLACEHOLDER with your actual Hugging Face model repository ID ***
# Example: "Vikram-P/sms-spam-detector-bert"
LOCAL_MODEL_PATH = "Pugazh24/sms-spam-detector-bert" 
id2label = {0: "HAM", 1: "SPAM"} 
MODEL_NAME = "DistilBERT (Fine-Tuned)"

# --- 1. Model Loading with Caching (CRITICAL FOR CLOUD DEPLOYMENT) ---
@st.cache_resource
def load_bert_artifacts(path):
    """
    Loads the BERT model and tokenizer directly from the Hugging Face Hub (HF).
    Caching ensures the large model files are downloaded only once.
    """
    try:
        # Loading directly from the Hugging Face Hub path
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        
        # Hardcoding the calculated F1 score for display, as metrics files aren't always available on the Hub
        best_f1 = 0.9634  
            
        return tokenizer, model, best_f1
    except Exception as e:
        # Display this error if the app cannot fetch the model from the Hub
        st.error(f"Failed to load model from Hugging Face Hub: {path}")
        st.error("Please ensure the model repository exists and is public.")
        st.exception(e)
        return None, None, 0.0

tokenizer, model, best_f1 = load_bert_artifacts(LOCAL_MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model:
    model.to(device)
    model.eval()
    
    # --- 3. Streamlit Web Interface ---
    st.set_page_config(page_title="BERT Spam Detector", layout="centered")
    st.title("🚀 BERT Enhanced SMS Spam Detector")
    st.markdown("---")
    st.info(f"Model: **{MODEL_NAME}** | Expected Performance (F1-Score): **{best_f1:.4f}**")
    

    # Text input box for the user
    sms_input = st.text_area(
        "Enter the SMS Message:", 
        height=150, 
        placeholder="Example: URGENT! You have won a £1000 prize! Txt CLAIM to 81010 now!"
    )

    if st.button('Classify SMS', type="primary"):
        if sms_input:
            with st.spinner('Classifying with DistilBERT...'):
                # 1. Tokenize the input
                inputs = tokenizer(
                    sms_input,
                    truncation=True,
                    padding="max_length",
                    max_length=512, 
                    return_tensors="pt"
                )
                
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # 2. Predict the label
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    prediction_id = torch.argmax(logits, dim=1).item()

                # 3. Get the prediction and confidence score
                prediction_label = id2label[prediction_id]
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
                confidence = probabilities[prediction_id]

                # 4. Display the result
                if prediction_label == 'SPAM':
                    st.error(f"**Classification: SPAM!** 🚨")
                    st.markdown(f"**Confidence:** {confidence:.2%}")
                    st.markdown("This message exhibits strong indicators of unsolicited or fraudulent content.")
                else:
                    st.success(f"**Classification: HAM (Legitimate)** ✅")
                    st.markdown(f"**Confidence:** {confidence:.2%}")
                    st.markdown("This message is highly likely to be legitimate.")
        else:
            st.info("Please enter a message to classify.")
            
    st.markdown("---")
    st.caption("Project Enhanced by DistilBERT Transformer. Developed by Vikram .P ")

