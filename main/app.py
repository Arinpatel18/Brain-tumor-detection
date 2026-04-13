import streamlit as st
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from collections import OrderedDict
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import base64
from io import BytesIO
import random
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# --- Dice Coefficient and Loss for U-Net ---
smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# Configure dark mode and page layout
st.set_page_config(page_title="Tumor Detection", layout="wide", initial_sidebar_state="collapsed")

# Function to convert image to base64 for CSS background
def get_base64_placeholder():
    try:
        bg_path = os.path.join(current_dir, "baack.jpg")
        return "data:image/jpeg;base64," + base64.b64encode(open(bg_path, "rb").read()).decode()
    except FileNotFoundError:
        return "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4MDAiIGhlaWdodD0iNjAwIj48ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9ImciIHgxPSIwJSIgeTE9IjAlIiB4Mj0iMTAwJSIgeTI9IjEwMCUiPjxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiMxMjEyMTIiIHN0b3Atb3BhY2l0eT0iMSIvPjxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iIzFmMWYxZiIgc3RvcC1vcGFjaXR5PSIxIi8+PC9saW5lYXJHcmFkaWVudD48L2RlZnM+PHJlY3Qgd2lkdGg9IjgwMCIgaGVpZ2h0PSI2MDAiIGZpbGw9InVybCgjZykiLz48Y2lyY2xlIGN4PSI0MDAiIGN5PSIzMDAiIHI9IjIwMCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMzAzMDMwIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1kYXNoYXJyYXk9IjEwLDUiLz48L3N2Zz4="

st.markdown(f"""
    <style>
        .stApp {{
            background-image: url("{get_base64_placeholder()}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stApp:before {{
            content: "";
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background-color: rgba(18, 18, 18, 1);
            z-index: -1;
        }}
        .main .block-container {{
            padding-top: 0.5rem !important;
            margin-top: 0 !important;
            position: relative;
            z-index: 1;
        }}
        header {{ visibility: hidden !important; height: 0 !important; padding: 0 !important; margin: 0 !important; }}
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        html, body, [class*="css"] {{ color: white; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        .main {{ background-color: rgba(30, 30, 30, 0.7); padding: 1rem; border-radius: 10px; backdrop-filter: blur(5px); }}
        h1, h2, h3, h4, h5, h6, p {{ color: white; }}
        h1 {{
            font-weight: 700; margin-bottom: 1rem; text-align: center;
            background: linear-gradient(90deg, #03DAC6, #BB86FC);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            padding-top: 0.5rem;
        }}
        .stButton button {{
            background-color: #03DAC6; color: black; padding: 0.75em 2em; font-size: 16px;
            font-weight: 600; border-radius: 8px; border: none; width: 100%; transition: all 0.3s ease;
        }}
        .stButton button:hover {{ background-color: #018786; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }}
        .upload-container {{
            border: 2px dashed #6b6b6b; border-radius: 10px; padding: 1.5rem; text-align: center;
            background-color: rgba(45, 45, 45, 0.7); margin-bottom: 1rem; transition: all 0.3s ease;
        }}
        .upload-container:hover {{ border-color: #03DAC6; background-color: rgba(51, 51, 51, 0.7); }}
        .results-container {{ background-color: rgba(45, 45, 45, 0.7); border-radius: 10px; padding: 1rem; margin-top: 1rem; backdrop-filter: blur(5px); }}
        .card {{ background-color: rgba(45, 45, 45, 0.8); border-radius: 10px; padding: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 100%; backdrop-filter: blur(3px); }}
        .section-title {{ color: #BB86FC; margin-bottom: 0.75rem; font-weight: 600; }}
        .stFileUploader {{ padding-top: 0 !important; }}
        .model-card {{ background-color: rgba(45, 45, 45, 0.5); border-radius: 8px; border-left: 4px solid #03DAC6; padding: 12px; margin-bottom: 12px; }}
    </style>
""", unsafe_allow_html=True)

os.environ['ULTRALYTICS_HUB'] = '0'

MODEL_CONFIGS = {
    "YOLO-V11 Object Detection": {
        "type": "yolo",
        "path": os.path.join(parent_dir, "YOLO-V11/best.pt"),
        "description": "YOLO-V11 Model for general detection"
    },
    "YOLO-V8 Object Detection": {
        "type": "yolo",
        "path": os.path.join(parent_dir, "YOLO-V8/best.pt"),
        "description": "YOLO-V8 Model for bounding boxes"
    },
    "U-Net Segmentation": {
        "type": "unet",
        "path": os.path.join(parent_dir, "UNet/unet.h5"),
        "description": "U-Net Image Segmentation for brain tumors"
    },
    "CNN Classification": {
        "type": "cnn",
        "path": os.path.join(parent_dir, "CNN+computer_vision/segmentation-canny.h5"),
        "description": "CNN based Brain Tumor Classification"
    }
}

# Allow user to adjust confidence threshold
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.01,
    max_value=1.0,
    value=0.25,
    step=0.01,
    help="Adjust the minimum confidence required to display a detection."
)

classes_model2 = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

class_labels_model1 = { 
    0: 'Glioma', 1: 'Meningioma', 3: 'Pituitary', 2: 'Background'
}
class_colors_model1 = {
    0: 'darkmagenta', 1: 'orange', 3: 'darkcyan', 2: 'darkslategray'
}

@st.cache_resource
def load_yolo_model(model_path):
    model = YOLO(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model, device

@st.cache_resource
def load_unet_model_cached(model_path):
    model_url = "https://www.dropbox.com/scl/fi/ci9go9wim9xxf3clukagc/BHAI_MERA_model.h5?rlkey=kzk2z468492l86bx6lfzucixl&dl=1"
    if not os.path.exists(model_path):
        response = requests.get(model_url)
        with open(model_path, "wb") as f:
            f.write(response.content)
    model = load_model(model_path, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})
    model.compile(loss=dice_loss, optimizer=Adam(1e-4), metrics=[dice_coef, 'accuracy'])
    return model

@st.cache_resource
def load_cnn_model_cached(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def plot_boxes_model1(ax, image, boxes, labels, prediction_confidences=None):
    ax.imshow(image)
    for i, label in enumerate(labels):
        x_min, y_min, x_max, y_max = boxes[i]
        width, height = x_max - x_min,  y_max - y_min
        class_label = class_labels_model1[label.item()]
        class_label += ' tumor' if class_label != 'Background' else ''
        class_color = class_colors_model1[label.item()]
        legend = class_label
        bounding_box = plt.Rectangle((x_min, y_min), width, height, linewidth=3, label=legend)
        label_text = ax.text(x_min+4, y_min-5, class_label)
        label_text.set_bbox(dict(facecolor=to_rgba(class_color, alpha=0.4)))
        if prediction_confidences is not None:
            confidence_label = f' conf: {prediction_confidences[i]:.2f}'
            bounding_box.set_edgecolor(class_color)
            bounding_box.set_facecolor('none')
            label_text.set_text(class_label + confidence_label)
        ax.add_patch(bounding_box)

def add_legends_model1(fig):
    box_legends = [ax.get_legend_handles_labels() for ax in fig.axes]
    if not box_legends: return
    box_legends, labels = [sum(lol, []) for lol in zip(*box_legends)]
    if len(labels) == 0: return
    box_legends = sorted(zip(labels, box_legends), key=lambda x: x[0])
    labels, box_legends = zip(*box_legends)
    box_labeled_legends = OrderedDict(zip(labels, box_legends))
    fig.legend(box_labeled_legends.values(), box_labeled_legends.keys())

def run_yolo_v11(model, device, image_pil, image_np):
    results = model.predict(image_pil, device=device, verbose=False)[0]
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#1e1e1e')
    boxes = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    mask = confidences > confidence_threshold
    boxes, confidences, classes = boxes[mask], confidences[mask], classes[mask]
    
    plot_boxes_model1(ax, image_np, boxes, classes, confidences)
    add_legends_model1(fig)
    fig.tight_layout()
    tumor_classes = [c for c in classes if int(c) != 2]
    return fig, tumor_classes, classes, confidences

def run_yolo_v8(model, device, image_pil, image_np):
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#1e1e1e')
    ax.imshow(image_np)
    ax.axis('off')
    result = model.predict(image_pil, device=device, verbose=False)[0]
    detected_classes = []
    detection_confidences = []
    for detection in result.boxes:
        conf = detection.conf[0].cpu().numpy()
        cls = int(detection.cls[0].cpu().numpy())
        if conf > confidence_threshold:
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f"{classes_model2[cls]} {conf:.2f}", color='white', fontsize=12, backgroundcolor='red')
            detected_classes.append(cls)
            detection_confidences.append(conf)
    fig.tight_layout()
    return fig, detected_classes, detection_confidences

def run_unet(model, image_np):
    image = cv2.resize(image_np, (256, 256))
    x = image / 255.0
    x = np.expand_dims(x, axis=0)
    y_pred = model.predict(x, verbose=False)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = (y_pred >= confidence_threshold).astype(np.uint8)
    mask_rgb = np.zeros_like(image)
    mask_rgb[y_pred == 1] = [255, 0, 0]
    overlay = cv2.addWeighted(image, 1.0, mask_rgb, 0.5, 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='#1e1e1e')
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input (Resized)", color="white")
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Predicted Mask Overlay", color="white")
    axes[1].axis('off')
    fig.tight_layout()
    return fig, (y_pred == 1).sum() > 0

def run_cnn(model, image_np):
    img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    # The CNN model expects input shape (256, 128, 3)
    resize = tf.image.resize(img, (256, 128))
    resize = resize / 255.0
    image_tensor = tf.expand_dims(resize, 0)
    predictions = model.predict(image_tensor, verbose=False)
    
    if len(predictions[0]) == 1:
        prediction = predictions[0][0]
        class_name = "Tumor" if prediction > 0.5 else "No Tumor"
        confidence = prediction if prediction > 0.5 else 1 - prediction
    else:
        prediction = np.argmax(predictions[0])
        class_name = classes_model2[prediction]
        confidence = predictions[0][prediction]
        
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='#1e1e1e')
    ax.imshow(resize.numpy())
    title = f"Prediction: {class_name}\nConfidence: {confidence:.2f}"
    ax.set_title(title, color="white")
    ax.axis('off')
    fig.tight_layout()
    return fig, class_name, confidence

st.markdown('<h1 style="color: #003366; background-color: rgba(255, 255, 255, 0.7); padding: 12px; border-radius: 12px; text-align: center;">🧠 Brain Tumor Detection Hub</h1>', unsafe_allow_html=True)

with st.container():
    st.markdown('<h3 class="section-title">Model Selection</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 3])
    with col1:
        selected_model_name = st.selectbox("Choose a detection model:", options=list(MODEL_CONFIGS.keys()), index=0)
    selected_model_config = MODEL_CONFIGS[selected_model_name]
    with col2:
        st.markdown(f'<div class="model-card"><strong>{selected_model_name}</strong><br><small>{selected_model_config["description"]}</small></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)
    
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown(f'<h3 class="section-title">Analysis Results using {selected_model_name}</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h4>Original MRI Scan</h4>', unsafe_allow_html=True)
        st.image(image_np, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="card"><h4>Detection Results</h4>', unsafe_allow_html=True)
        with st.spinner(f"Analyzing with {selected_model_name}..."):
            m_type = selected_model_config["type"]
            path = selected_model_config["path"]
            
            try:
                if m_type == "yolo":
                    model, device = load_yolo_model(path)
                    if "V11" in selected_model_name:
                        fig, tumor_classes, f_classes, f_confs = run_yolo_v11(model, device, image_pil, image_np)
                        st.pyplot(fig)
                        if len(tumor_classes) > 0:
                            for t in set(tumor_classes):
                                indices = [i for i, c in enumerate(f_classes) if c == t]
                                confs = [f_confs[i] for i in indices]
                                avg_conf = sum(confs) / len(confs)
                                st.markdown(f'<div style="background-color: {class_colors_model1[int(t)]}30; padding: 10px; border-radius: 5px; margin-bottom: 10px;"><span style="color: {class_colors_model1[int(t)]}; font-weight: bold;">{class_labels_model1[int(t)]}</span> - Detected: {len(indices)}, Avg Conf: {avg_conf:.0%}</div>', unsafe_allow_html=True)
                        else:
                            st.success("No tumors detected.")
                            
                    else:
                        fig, detected_classes, detection_confidences = run_yolo_v8(model, device, image_pil, image_np)
                        st.pyplot(fig)
                        if len(detected_classes) > 0:
                            class_summary = {}
                            for i, cls in enumerate(detected_classes):
                                class_summary.setdefault(cls, []).append(detection_confidences[i])
                            for cls, confs in class_summary.items():
                                class_name = classes_model2[cls]
                                avg_conf = sum(confs) / len(confs)
                                st.markdown(f'<div style="background-color: #03DAC630; padding: 10px; border-radius: 5px; margin-bottom: 10px;"><span style="color: #03DAC6; font-weight: bold;">{class_name}</span> - Detected: {len(confs)}, Avg Conf: {avg_conf:.0%}</div>', unsafe_allow_html=True)
                        else:
                            st.info("No detections found.")
                            
                elif m_type == "unet":
                    model = load_unet_model_cached(path)
                    fig, tumor_detected = run_unet(model, image_np)
                    st.pyplot(fig)
                    if tumor_detected:
                        st.warning("U-Net Mask Highlights Potential Tumor Regions in Red.")
                    else:
                        st.success("No apparent tumor masks generated.")
                        
                elif m_type == "cnn":
                    model = load_cnn_model_cached(path)
                    fig, cls_name, conf = run_cnn(model, image_np)
                    st.pyplot(fig)
                    st.markdown(f'<div style="background-color: #BB86FC30; padding: 10px; border-radius: 5px; margin-bottom: 10px;"><span style="color: #BB86FC; font-weight: bold;">Predicted Class: {cls_name}</span><br>Confidence: {conf:.2%}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="text-align: center; padding: 2rem; background-color: rgba(45, 45, 45, 0.7); border-radius: 10px;"><p style="font-size: 1.2rem;">Upload a brain MRI scan image to detect potential tumors.</p><p style="font-size: 0.9rem; margin-top: 15px;">Choose between YOLO-V11, YOLO-V8, U-Net Segmentation, and CNN Classification.</p></div>', unsafe_allow_html=True)
