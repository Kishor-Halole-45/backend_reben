import io
import os
from typing import List, Tuple

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from configilm.extra.BENv2_utils import NEW_LABELS

from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    HAS_BLIP = True
except ImportError:
    HAS_BLIP = False

try:
    from transformers import ViltProcessor, ViltForQuestionAnswering
    HAS_VQA = True
except ImportError:
    HAS_VQA = False

# ==================== MODEL CONFIGS ====================
REBEN_MODEL_NAME = "hackelle/resnet18-all-v0.1.1"
REBEN_TARGET_SIZE = 120
REBEN_TARGET_CHANNELS = 12

# BLIP model configs for Captioning
BLIP_BASE_MODEL = "Salesforce/blip-image-captioning-base"
CAPTION_ADAPTER_DIR = None  # Set to local adapter path if available

# ViLBERT model for VQA (dedicated VQA model)
VQA_MODEL_NAME = "dandelin/vilt-b32-finetuned-vqa"
VQA_ADAPTER_DIR = None  # Set to local adapter path if available

SUPPORTED_TYPES = ["tif", "tiff", "jpg", "jpeg", "png"]

st.set_page_config(page_title="Satellite Image Analysis - Multi-Model Demo", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg: #0b1016;
        --panel: #111923;
        --panel-2: #161f2b;
        --border: #273342;
        --text: #edf3f8;
        --muted: #a8b4c2;
        --accent: #f8fafc;
        --error-bg: rgba(95, 16, 16, 0.6);
        --error-border: #c54f4f;
        --success-bg: rgba(16, 80, 50, 0.6);
        --success-border: #4ade80;
    }
    .stApp {
        background: var(--bg);
        color: var(--text);
    }
    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    .demo-box {
        background: rgba(17, 25, 35, 0.96);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 8px 22px rgba(0,0,0,0.2);
    }
    h1 {
        font-size: 2.2rem !important;
        line-height: 1.15 !important;
        margin-bottom: 0.35rem !important;
        color: var(--text) !important;
    }
    .subtitle {
        margin-bottom: 1rem;
        color: var(--muted);
        font-size: 0.96rem;
    }
    .upload-wrap {
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    .stFileUploader > div {
        background: rgba(18, 27, 38, 0.95);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.3rem 0.5rem;
    }
    .preview-card {
        background: rgba(16,22,31,0.95);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.8rem;
        min-height: 320px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .preview-card img {
        max-height: 300px;
        width: 100%;
        object-fit: cover;
        border-radius: 8px;
    }
    .error-box {
        background: var(--error-bg);
        border: 1px solid var(--error-border);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin-top: 1rem;
        color: #ffd7d7;
    }
    .error-box .title {
        color: #ff8787;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .success-box {
        background: var(--success-bg);
        border: 1px solid var(--success-border);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin-top: 1rem;
        color: #d1fae5;
    }
    .success-box .title {
        color: #4ade80;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .result-card {
        background: rgba(16, 22, 31, 0.95);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .primary-btn button {
        background: #dfeaf7 !important;
        color: #111923 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================== MODEL LOADING FUNCTIONS ====================

@st.cache_resource
def load_reben_model():
    """Load REBEN land cover classification model"""
    model = BigEarthNetv2_0_ImageClassifier.from_pretrained(REBEN_MODEL_NAME)
    model.eval()
    return model


@st.cache_resource
def load_vqa_model():
    """Load ViLBERT model for proper VQA tasks"""
    if not HAS_VQA:
        return None, None
    
    try:
        processor = ViltProcessor.from_pretrained(VQA_MODEL_NAME)
        model = ViltForQuestionAnswering.from_pretrained(VQA_MODEL_NAME).to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        return model, processor
    except Exception as e:
        st.error(f"Error loading VQA model: {e}")
        return None, None


@st.cache_resource
def load_caption_model():
    """Load BLIP model for image captioning"""
    if not HAS_BLIP:
        return None, None
    
    processor = BlipProcessor.from_pretrained(BLIP_BASE_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_BASE_MODEL).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load adapter if available
    if CAPTION_ADAPTER_DIR and os.path.exists(CAPTION_ADAPTER_DIR):
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, CAPTION_ADAPTER_DIR)
        except Exception as e:
            st.warning(f"Could not load Caption adapter: {e}")
    
    model.eval()
    return model, processor


# ==================== IMAGE PREPROCESSING ====================

def preprocess_image_reben(image: Image.Image, target_size: int = REBEN_TARGET_SIZE, target_channels: int = REBEN_TARGET_CHANNELS):
    """Preprocess image for REBEN model (multi-channel satellite data)"""
    rgb_image = image.convert("RGB")
    rgb_image = rgb_image.resize((target_size, target_size))
    arr = np.asarray(rgb_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))

    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif tensor.shape[0] == 2:
        tensor = torch.cat([tensor, tensor[:1]], dim=0)

    repeated = []
    for i in range(target_channels):
        repeated.append(tensor[i % tensor.shape[0]])
    tensor = torch.stack(repeated, dim=0)
    return tensor.unsqueeze(0)


def validate_reben_input(image: Image.Image):
    """Basic sanity check only. REBEN accepts valid satellite / aerial imagery in common formats."""
    rgb = image.convert("RGB")
    arr = np.asarray(rgb)

    if arr.size == 0:
        raise ValueError("The uploaded image is empty.")

    if rgb.width < 16 or rgb.height < 16:
        raise ValueError("REBEN needs a larger image. Please upload a clearer satellite patch.")

    return True


def preprocess_image_blip(image: Image.Image, processor) -> torch.Tensor:
    """Preprocess image for BLIP models (VQA, Captioning)"""
    rgb_image = image.convert("RGB")
    return processor(images=rgb_image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")


# ==================== PREDICTION FUNCTIONS ====================

def predict_reben_labels(model, image: Image.Image) -> List[Tuple[str, float]]:
    """Get top-5 land cover predictions from REBEN model."""
    validate_reben_input(image)

    tensor = preprocess_image_reben(image)
    with torch.no_grad():
        logits = model(tensor)
    probs = torch.sigmoid(logits).squeeze(0)
    top_idx = torch.argsort(probs, descending=True)[:5]
    items = []
    for idx in top_idx:
        label = NEW_LABELS[int(idx)]
        score = float(probs[int(idx)])
        items.append((label, score))
    return items


def predict_vqa(model, processor, image: Image.Image, question: str) -> str:
    """Answer a question about the satellite image using ViLBERT VQA model"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Process image and question for ViLBERT
        inputs = processor(images=image.convert("RGB"), text=question, return_tensors="pt").to(device)
        
        # Generate answer IDs using the VQA model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Get the predicted answer ID (highest probability)
        answer_idx = logits.argmax(-1).item()
        
        # Get answer from the model's vocabulary
        # ViLT uses the VQA answer vocabulary (3129 possible answers)
        answer = model.config.id2label.get(answer_idx, f"Answer ID: {answer_idx}")
        
        return answer if answer else "Unable to generate answer"
    
    except Exception as e:
        return f"Error in VQA: {str(e)}"


def predict_caption(model, processor, image: Image.Image) -> str:
    """Generate a caption for the satellite image using BLIP"""
    inputs = processor(images=image.convert("RGB"), return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        out = model.generate(**inputs, max_length=100)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


# ==================== MAIN APPLICATION ====================

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🛰️ Model Selection")
    selected_model = st.radio(
        "Choose a model:",
        ["REBEN (Land Cover)", "VQA (Question Answering)", "Image Caption"],
        index=0
    )
    
    st.divider()
    
    st.markdown("### ℹ️ Model Info")
    if selected_model == "REBEN (Land Cover)":
        st.info(
            "**REBEN** - BigEarthNet v2.0\n\nClassifies satellite imagery into land-cover types.\n\n⚠️ This model is trained for satellite / remote-sensing patches and is unreliable on ordinary camera photos.\n\n✅ Supports: TIFF, JPG, PNG"
        )
    elif selected_model == "VQA (Question Answering)":
        st.info("**Visual Question Answering**\n\nAsk questions about satellite images and get answers.\n\n✅ Supports: TIFF, JPG, PNG")
    else:
        st.info("**Image Captioning**\n\nGenerate descriptions for satellite images.\n\n✅ Supports: TIFF, JPG, PNG")

# Main content area
with st.container():
    st.markdown("<h1>🛰️ Satellite Image Analysis - Multi-Model Demo</h1>", unsafe_allow_html=True)
    
    if selected_model == "REBEN (Land Cover)":
        st.markdown(
            "<div class='subtitle'>Upload a satellite image to classify land cover types and conditions.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class='demo-box'>
                <strong>REBEN input requirement:</strong><br>
                • Use a <b>satellite / remote-sensing image</b> or land-cover patch<br>
                • Ideal: aerial or Earth observation data with ground features<br>
                • Accepts: TIFF, JPG, PNG<br>
                • Avoid: ordinary camera photos, indoor pictures, close-up objects, or random scenery<br>
                • This model is trained for <b>land-cover classification</b>, not generic image recognition
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif selected_model == "VQA (Question Answering)":
        st.markdown(
            "<div class='subtitle'>Upload a satellite image and ask questions about it.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='subtitle'>Upload a satellite image to generate a detailed description.</div>",
            unsafe_allow_html=True,
        )
    
    st.divider()
    
    # ========== IMAGE UPLOAD SECTION ==========
    uploaded = st.file_uploader("📁 Upload Image", type=SUPPORTED_TYPES, label_visibility="collapsed")
    
    if uploaded is not None:
        image_bytes = uploaded.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Display image preview
        col_preview, col_controls = st.columns([2, 1])
        
        with col_preview:
            st.markdown("<div class='preview-card'>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_controls:
            st.markdown("**Image Details:**")
            st.write(f"📏 Size: {image.size}")
            st.write(f"🎨 Mode: {image.mode}")
            
            if st.button("🗑️ Clear Image"):
                st.session_state.pop("uploaded_file", None)
                st.rerun()
        
        st.divider()
        
        # ========== MODEL-SPECIFIC INTERFACE ==========
        
        if selected_model == "REBEN (Land Cover)":
            st.markdown("### 🔍 Land Cover Classification")
            st.markdown(
                """
                <div class='demo-box'>
                    <strong>Best image types for REBEN:</strong><br>
                    - satellite scene / aerial map<br>
                    - land-cover patches from Earth observation datasets<br>
                    - images showing fields, water, roads, forests, urban areas, bare land<br><br>
                    <strong>Bad input for REBEN:</strong><br>
                    - indoor or office photos<br>
                    - portraits or people<br>
                    - random product shots or close-up objects<br>
                    - generic camera images without land surface structure
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("🚀 Predict Land Cover", key="reben_predict"):
                try:
                    model = load_reben_model()
                    with st.spinner("🔄 Running REBEN model..."):
                        preds = predict_reben_labels(model, image)

                    if preds and preds[0][1] < 0.45:
                        st.warning(
                            "⚠️ REBEN prediction is low-confidence for this input. This may happen with non-satellite imagery or weak land-cover patterns."
                        )

                    st.markdown("<div class='success-box'><div class='title'>✅ Classification Results</div></div>", unsafe_allow_html=True)

                    # Display results as a formatted table
                    results_data = []
                    for i, (label, score) in enumerate(preds, 1):
                        results_data.append({
                            "Rank": i,
                            "Land Cover Type": label,
                            "Confidence": f"{score:.2%}"
                        })

                    st.dataframe(results_data, use_container_width=True)

                    # Visual confidence bars
                    st.markdown("**Confidence Distribution:**")
                    for label, score in preds:
                        st.progress(score, text=f"{label}: {score:.2%}")

                except Exception as exc:
                    st.markdown(
                        f"""<div class='error-box'><div class='title'>❌ Error</div><code>{str(exc)}</code></div>""",
                        unsafe_allow_html=True,
                    )
        
        elif selected_model == "VQA (Question Answering)":
            if not HAS_VQA:
                st.error("❌ VQA dependencies not installed. Run: `pip install transformers`")
            else:
                st.markdown("### 🤔 Ask a Question About the Image")
                
                # Show model info
                with st.expander("ℹ️ Model Details"):
                    st.write("**Model:** ViLBERT (Vision-Language BERT) - Dedicated VQA Model")
                    st.write("**Architecture:** Multi-modal transformer for visual question answering")
                    st.write("**Training Data:** COCO-VQA dataset (100K+ images, 900K+ QA pairs)")
                    st.write("**Device:** GPU (CUDA)" if torch.cuda.is_available() else "**Device:** CPU")
                    st.write("**Answer Format:** Single predicted answer from VQA vocabulary")
                
                question = st.text_input("Type your question:", placeholder="e.g., What type of terrain is this?")
                
                if st.button("🚀 Get Answer", key="vqa_predict") and question:
                    try:
                        model, processor = load_vqa_model()
                        if model is None or processor is None:
                            st.error("❌ Could not load VQA model")
                        else:
                            with st.spinner("🔄 Analyzing image and processing your question..."):
                                answer = predict_vqa(model, processor, image, question)
                            
                            # Display results
                            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                            st.markdown("<div class='success-box'><div class='title'>✅ Model Response</div></div>", unsafe_allow_html=True)
                            
                            col_q, col_a = st.columns([1, 2])
                            with col_q:
                                st.markdown("**Your Question:**")
                                st.write(question)
                            with col_a:
                                st.markdown("**Model Answer:**")
                                st.info(f"🎯 {answer}", icon="✨")
                            
                            # Add confidence indicator
                            st.success("✓ Answer generated by ViLBERT VQA model based on image analysis", icon="✅")
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    except Exception as exc:
                        st.markdown(
                            f"""<div class='error-box'><div class='title'>❌ Error</div><code>{str(exc)}</code></div>""",
                            unsafe_allow_html=True,
                        )
        
        else:  # Image Caption
            if not HAS_BLIP:
                st.error("❌ BLIP dependencies not installed. Run: `pip install transformers`")
            else:
                st.markdown("### 📝 Generate Image Caption")
                if st.button("🚀 Generate Caption", key="caption_predict"):
                    try:
                        model, processor = load_caption_model()
                        if model is None:
                            st.error("❌ Could not load Caption model")
                        else:
                            with st.spinner("🔄 Generating caption..."):
                                caption = predict_caption(model, processor, image)
                            
                            st.markdown("<div class='result-card'><div class='success-box'><div class='title'>✅ Image Caption</div></div>", unsafe_allow_html=True)
                            st.markdown(f"**Caption:** {caption}")
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    except Exception as exc:
                        st.markdown(
                            f"""<div class='error-box'><div class='title'>❌ Error</div><code>{str(exc)}</code></div>""",
                            unsafe_allow_html=True,
                        )
    
    else:
        # No image uploaded yet
        st.markdown(
            """
            <div class='preview-card'>
                <div style='text-align:center; color:#c2ceda; opacity:0.9; font-size:1.1rem;'>
                    📤 Upload an image to get started
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
