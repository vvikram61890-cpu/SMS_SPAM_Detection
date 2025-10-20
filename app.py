import streamlit as st
import torch
import numpy as np
import sys

# --- Configuration ---
LOCAL_MODEL_PATH = "Pugazh24/sms-spam-detector-bert" 
id2label = {0: "HAM", 1: "SPAM"} 
MODEL_NAME = "DistilBERT (Fine-Tuned)"

# Initialize the app first
st.set_page_config(page_title="BERT Spam Detector", layout="centered")
st.title("🚀 BERT Enhanced SMS Spam Detector")
st.markdown("---")

# Debug information (optional - can be removed in production)
with st.expander("Debug Info", expanded=False):
    st.write(f"Python version: {sys.version}")
    st.write(f"PyTorch available: {torch.__version__}")

# --- 1. Model Loading with Caching ---
@st.cache_resource
def load_bert_artifacts(path):
    """
    Loads the BERT model and tokenizer directly from the Hugging Face Hub.
    """
    try:
        # Import inside function to handle potential import errors
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        st.info("📥 Downloading model from Hugging Face Hub...")
        
        # Loading directly from the Hugging Face Hub path
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        
        # Hardcoded F1 score for display
        best_f1 = 0.9634  
            
        return tokenizer, model, best_f1
        
    except ImportError as e:
        st.error("❌ Transformers import failed!")
        st.error(f"Error details: {e}")
        st.info("Please ensure transformers==4.57.1 is installed in requirements.txt")
        return None, None, 0.0
        
    except Exception as e:
        st.error(f"❌ Failed to load model from: {path}")
        st.error(f"Error: {str(e)}")
        st.info("""
        Common solutions:
        1. Check if the model repository exists: https://huggingface.co/Pugazh24/sms-spam-detector-bert
        2. Ensure the repository is public
        3. Check your internet connection
        4. Try using a different model path
        """)
        return None, None, 0.0

# Load model with progress indication
with st.spinner('Loading model and dependencies...'):
    tokenizer, model, best_f1 = load_bert_artifacts(LOCAL_MODEL_PATH)

# Check if model loaded successfully
if model is not None and tokenizer is not None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    st.success("✅ Model loaded successfully!")
    st.info(f"**Model:** {MODEL_NAME} | **Expected F1-Score:** {best_f1:.4f} | **Device:** {device}")
    st.markdown("---")

    # Text input box for the user
    sms_input = st.text_area(
        "Enter the SMS Message:", 
        height=150, 
        placeholder="Example: URGENT! You have won a £1000 prize! Txt CLAIM to 81010 now!",
        help="Enter the SMS message you want to classify as spam or ham"
    )

    # Classification button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        classify_btn = st.button('🔍 Classify SMS', type="primary", use_container_width=True)

    if classify_btn:
        if sms_input.strip():
            with st.spinner('Analyzing message with DistilBERT...'):
                try:
                    # 1. Tokenize the input
                    inputs = tokenizer(
                        sms_input,
                        truncation=True,
                        padding="max_length",
                        max_length=512, 
                        return_tensors="pt"
                    )
                    
                    # Move inputs to the same device as model
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
                    st.markdown("## 📊 Classification Result")
                    
                    if prediction_label == 'SPAM':
                        st.error(f"**🚨 SPAM DETECTED!**")
                        st.metric("Confidence", f"{confidence:.2%}")
                        st.warning("⚠️ This message exhibits strong indicators of unsolicited or fraudulent content.")
                    else:
                        st.success(f"**✅ LEGITIMATE MESSAGE (HAM)**")
                        st.metric("Confidence", f"{confidence:.2%}")
                        st.info("💡 This message is highly likely to be legitimate.")
                    
                    # Show probability distribution
                    with st.expander("📈 View Detailed Probabilities"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("HAM Probability", f"{probabilities[0]:.2%}")
                        with col2:
                            st.metric("SPAM Probability", f"{probabilities[1]:.2%}")
                        
                        # Progress bars for visualization
                        st.progress(float(probabilities[0]), text=f"HAM: {probabilities[0]:.2%}")
                        st.progress(float(probabilities[1]), text=f"SPAM: {probabilities[1]:.2%}")
                        
                except Exception as e:
                    st.error(f"❌ Error during classification: {str(e)}")
                    st.info("Please try again with a different message.")
        else:
            st.warning("⚠️ Please enter a message to classify.")
            
else:
    st.error("""
    ❌ Model failed to load. Please check:
    
    1. **Model Repository**: Ensure https://huggingface.co/Pugazh24/sms-spam-detector-bert exists and is public
    2. **Requirements**: Verify all packages in requirements.txt are installed
    3. **Internet Connection**: The app needs internet to download the model
    4. **Dependencies**: Check that transformers==4.57.1 is correctly installed
    """)
    
    # Quick fix suggestion
    with st.expander("🛠️ Quick Fix - Try Alternative Model"):
        st.write("If the main model fails, you can try this alternative approach:")
        if st.button("Use Fallback Model"):
            st.info("This would load a different pre-trained model...")
            # You could implement fallback logic here

st.markdown("---")
st.caption("Project Enhanced by DistilBERT Transformer | Developed by Vikram .P")

# Sidebar with additional info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses a fine-tuned **DistilBERT** model to classify SMS messages as **SPAM** or **HAM** (legitimate).
    
    **How it works:**
    - Input SMS message is processed by the transformer model
    - Model analyzes patterns and features
    - Returns classification with confidence score
    
    **Model Info:**
    - Base: DistilBERT (distilled version of BERT)
    - Fine-tuned on SMS spam dataset
    - Expected F1-score: ~96.3%
    """)
