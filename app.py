# -*- coding: utf-8 -*-
"""
AI-Based Security Detection System
===================================
Modules:
1. Fire Detection - Real-time fire/smoke detection
2. Missed Kid Detection - Face verification system

Author: Updated System
Date: 2026-01-21
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import tempfile
import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import pickle


# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="🔒 AI Security System",
    page_icon="🔒",
    layout="wide"
)


# ============================================
# Custom CSS
# ============================================
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://img.freepik.com/vector-premium/concepto-cibernetico-tecnologia-abstracta-vector-fondo_115579-1416.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.75);
        z-index: -1;
    }
    .stMarkdown, .stText, p, span, label {
        color: #FFFFFF !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
    }
    h1, h2, h3 {
        color: #FFFFFF !important;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.9);
        font-weight: bold;
    }
    [data-testid="stSidebar"] {
        background: rgba(20, 20, 30, 0.95);
        backdrop-filter: blur(10px);
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stMetricValue"] {
        color: #00FF00 !important;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
    }
    [data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
    }
    .stButton > button {
        background: rgba(0, 120, 255, 0.9);
        color: white;
        border: 2px solid rgba(255, 255, 255, 0.3);
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.5);
    }
    .stButton > button:hover {
        background: rgba(0, 150, 255, 1);
        border: 2px solid rgba(255, 255, 255, 0.5);
        transform: scale(1.02);
    }
    [data-testid="stFileUploader"] {
        background: rgba(30, 30, 40, 0.8);
        border-radius: 10px;
        padding: 10px;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    .stAlert {
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(10px);
        border-left: 4px solid;
    }
    [data-testid="column"] {
        background: rgba(20, 20, 30, 0.6);
        border-radius: 15px;
        padding: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }
    [data-testid="stImage"] {
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔒 AI-Based Security Detection System")
st.markdown("**Dual-Module Security System:** 🔥 Fire Detection • 🚨 Missed Kid Detection")


# ============================================
# MODEL 1: Fire Detection Model
# ============================================
class FireDetectionModel(nn.Module):
    """Fire and smoke detection model using MobileNetV2"""
    def __init__(self, num_classes=1):
        super(FireDetectionModel, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=False)
        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.mobilenet(x)


# ============================================
# MODEL 2: Kid Face Recognition Model
# ============================================
class ImprovedFaceRecognitionCNN(nn.Module):
    """Face recognition model - same architecture as training"""
    def __init__(self, num_classes=20):
        super(ImprovedFaceRecognitionCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def extract_embedding(self, x):
        """Extract 256-dim embedding (before classifier)"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


# ============================================
# Load Kid Reference Database
# ============================================
def load_kid_reference_database():
    """Load reference embeddings from saved .pkl file"""
    try:
        embeddings_path = 'face_embeddings.pkl'
        metadata_path = 'face_metadata.pkl'
        
        if not os.path.exists(embeddings_path):
            st.sidebar.error(f"❌ Embeddings not found: {embeddings_path}")
            return {}
        
        # Load embeddings
        with open(embeddings_path, 'rb') as f:
            embeddings_dict = pickle.load(f)
        
        # Load metadata (optional)
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
        
        st.sidebar.success(f"✅ Loaded embeddings for {len(embeddings_dict)} kids")
        
        # Show sample info
        for kid_id in list(embeddings_dict.keys())[:3]:
            emb_shape = embeddings_dict[kid_id]['embedding'].shape
            st.sidebar.info(f"   👤 {kid_id}: {emb_shape}")
        
        return embeddings_dict
        
    except Exception as e:
        st.sidebar.error(f"❌ Database load error: {str(e)}")
        return {}


# ============================================
# Similarity Calculation
# ============================================
def calculate_similarity_score(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    # Ensure both are 2D arrays for sklearn
    emb1 = embedding1.reshape(1, -1) if embedding1.ndim == 1 else embedding1
    emb2 = embedding2.reshape(1, -1) if embedding2.ndim == 1 else embedding2
    
    # Cosine similarity
    similarity = cosine_similarity(emb1, emb2)[0][0]
    
    # Convert from [-1, 1] to [0, 1] range
    similarity = (similarity + 1) / 2
    
    return similarity


def find_best_match_from_embeddings(query_embedding, reference_database, threshold=0.65):
    """Find best matching kid from reference embeddings"""
    best_kid_id = None
    best_similarity = 0.0
    
    # Compare with all kids in database
    for kid_id, kid_data in reference_database.items():
        ref_embedding = kid_data['embedding']
        
        # Calculate similarity
        similarity = calculate_similarity_score(query_embedding, ref_embedding)
        
        # Track best match
        if similarity > best_similarity:
            best_similarity = similarity
            best_kid_id = kid_id
    
    # Apply threshold
    if best_similarity >= threshold:
        return best_kid_id, best_similarity
    else:
        return None, best_similarity


# ============================================
# Load Models
# ============================================
def load_kid_model_updated(device):
    """Load the trained face recognition model"""
    model_path = 'face_recognition_model.pth'
    
    try:
        if not os.path.exists(model_path):
            st.sidebar.error(f"❌ Model not found: {model_path}")
            return None
        
        # Create model instance (20 classes for 20 kids)
        model = ImprovedFaceRecognitionCNN(num_classes=20)
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        st.sidebar.success("✅ Kid Recognition Model: Loaded")
        st.sidebar.info("   📊 Embedding dim: 256")
        
        return model
        
    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {str(e)}")
        return None


@st.cache_resource
def load_all_models():
    """Load both detection models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models_dict = {}

    # Model 1: Fire Detection
    fire_model_path = 'best_mobile_accuracy_model.pth'
    if os.path.exists(fire_model_path):
        try:
            fire_model = FireDetectionModel(num_classes=1)
            checkpoint = torch.load(fire_model_path, map_location=device, weights_only=False)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    fire_model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    fire_model.load_state_dict(checkpoint['state_dict'])
                else:
                    fire_model.load_state_dict(checkpoint)
            else:
                fire_model = checkpoint
            
            fire_model.to(device).eval()
            models_dict['fire'] = fire_model
            st.sidebar.success("✅ Fire Detection Model: Loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Fire model error: {str(e)[:100]}")
            models_dict['fire'] = None
    else:
        st.sidebar.warning(f"⚠️ Fire model not found: {fire_model_path}")
        models_dict['fire'] = None

    # Model 2: Kid Recognition
    models_dict['kid'] = load_kid_model_updated(device)

    return models_dict, device


# ============================================
# Preprocessing Transforms
# ============================================
fire_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============================================
# Alarm Functions
# ============================================
def play_alarm_sound():
    """Play alarm sound for security alerts"""
    audio_html = """
        <audio autoplay loop>
            <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg">
        </audio>
    """
    return audio_html


def stop_alarm():
    """Stop alarm sound"""
    return ""


# ============================================
# DETECTION FUNCTION 1: Fire Detection
# ============================================
def detect_fire(frame, model, device, threshold, debug=False):
    """Detect fire/smoke in the frame"""
    if model is None:
        annotated = frame.copy()
        cv2.putText(annotated, "Fire Model Not Loaded", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        return annotated, False, 0.0

    # Preprocess
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = fire_transform(frame_rgb).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output)[0][0].cpu().item()
        confidence = probability
        fire_detected = probability > threshold

    # Annotate frame
    annotated_frame = frame.copy()

    if fire_detected:
        color = (0, 0, 255)  # Red
        label = '🔥 FIRE DETECTED!'
        
        # Draw alert border
        h, w = frame.shape[:2]
        cv2.rectangle(annotated_frame, (10, 10), (w-10, h-10), color, 5)
        
        # Draw text
        cv2.putText(annotated_frame, label, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(annotated_frame, f'Confidence: {confidence:.2%}',
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        cv2.putText(annotated_frame, 'Fire Status: SAFE', (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if debug:
        st.sidebar.write(f"🔥 Fire confidence: {confidence:.4f}")

    return annotated_frame, fire_detected, confidence


# ============================================
# DETECTION FUNCTION 2: Missing Kid Detection
# ============================================
def detect_missing_kid(frame, model, device, reference_database, 
                       threshold=0.65, metric='cosine', debug=False):
    """Detect missing kids using face embeddings and similarity matching"""
    annotated_frame = frame.copy()
    kid_detected = False
    best_similarity = 0.0
    best_kid_id = "Unknown"
    
    if model is None or not reference_database:
        cv2.putText(annotated_frame, "Kid Detection: Model/DB Missing", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return annotated_frame, False, 0.0, "Unknown"
    
    try:
        # Step 1: Face Detection
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if debug:
            st.sidebar.write(f"👤 Faces detected: {len(faces)}")
        
        if len(faces) == 0:
            cv2.putText(annotated_frame, "Kid Status: SAFE (No Faces)", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return annotated_frame, False, 0.0, "Unknown"
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Step 2: Extract and preprocess face
            face_img = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (96, 96))  # Model expects 96x96
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Transform to tensor
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            face_pil = Image.fromarray(face_rgb)
            face_tensor = transform(face_pil).unsqueeze(0).to(device)
            
            # Step 3: Extract embedding (256-dim)
            with torch.no_grad():
                query_embedding = model.extract_embedding(face_tensor)
                query_embedding = query_embedding.cpu().numpy().flatten()
            
            if debug:
                st.sidebar.write(f"📊 Query embedding shape: {query_embedding.shape}")
                st.sidebar.write(f"📊 Query embedding norm: {np.linalg.norm(query_embedding):.4f}")
            
            # Step 4: Find best match from reference database
            kid_id, similarity = find_best_match_from_embeddings(
                query_embedding, 
                reference_database, 
                threshold
            )
            
            if debug:
                st.sidebar.write(f"🔍 Compared with: {len(reference_database)} kids")
                st.sidebar.write(f"📊 Best match: {kid_id if kid_id else 'None'}")
                st.sidebar.write(f"📊 Similarity: {similarity:.4f} ({similarity*100:.1f}%)")
            
            # Update best match across all faces in frame
            if similarity > best_similarity:
                best_similarity = similarity
                best_kid_id = kid_id if kid_id else "Unknown"
                kid_detected = (kid_id is not None)
            
            # Step 5: Annotate frame
            if kid_id:  # Match found (similarity >= threshold)
                box_color = (0, 0, 255)  # Red for alert
                label = f"⚠️ {kid_id}"
                sublabel = f"Match: {similarity*100:.1f}%"
            else:  # Below threshold
                box_color = (0, 255, 0)  # Green for safe
                label = "Unknown"
                sublabel = f"Score: {similarity*100:.1f}%"
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), box_color, 3)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(annotated_frame, (x, y-40), 
                         (x + max(label_size[0], 150), y), box_color, -1)
            
            # Draw text labels
            cv2.putText(annotated_frame, label, (x+5, y-22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, sublabel, (x+5, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Overall status banner
        if kid_detected:
            cv2.putText(annotated_frame, f"🚨 MISSING KID: {best_kid_id}", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(annotated_frame, f"Similarity: {best_similarity*100:.1f}%", 
                       (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Draw alert border
            h, w = frame.shape[:2]
            cv2.rectangle(annotated_frame, (10, 10), (w-10, h-10), (0, 0, 255), 5)
        else:
            cv2.putText(annotated_frame, "Kid Status: SAFE", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
    except Exception as e:
        if debug:
            st.sidebar.error(f"❌ Kid detection error: {str(e)}")
        cv2.putText(annotated_frame, f"Error: {str(e)[:40]}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return annotated_frame, kid_detected, best_similarity, best_kid_id


# ============================================
# Combined Detection Pipeline
# ============================================
def run_all_detections(frame, models, device, thresholds, kid_reference_db, 
                       similarity_metric, debug=False):
    """Run both detection modules on the frame"""
    detections = {
        'fire': {'detected': False, 'confidence': 0.0},
        'kid': {'detected': False, 'similarity': 0.0, 'kid_id': 'Unknown'}
    }

    annotated_frame = frame.copy()

    # Run Fire Detection
    _, fire_detected, fire_conf = detect_fire(
        frame, models['fire'], device, thresholds['fire'], debug
    )
    detections['fire']['detected'] = fire_detected
    detections['fire']['confidence'] = fire_conf

    # Run Missed Kid Detection
    _, kid_detected, kid_similarity, kid_id = detect_missing_kid(
        frame, models['kid'], device, kid_reference_db, 
        thresholds['kid'], similarity_metric, debug
    )
    detections['kid']['detected'] = kid_detected
    detections['kid']['similarity'] = kid_similarity
    detections['kid']['kid_id'] = kid_id

    # Annotate frame with combined results
    annotated_frame = annotate_combined_detections(frame, detections)

    return annotated_frame, detections


def annotate_combined_detections(frame, detections):
    """Annotate frame with combined detection results"""
    annotated_frame = frame.copy()
    h, w = frame.shape[:2]

    # Determine overall threat status
    any_threat = (detections['fire']['detected'] or detections['kid']['detected'])

    if any_threat:
        # Draw alert border
        cv2.rectangle(annotated_frame, (10, 10), (w-10, h-10), (0, 0, 255), 5)

        y_offset = 50

        # Fire Detection Alert
        if detections['fire']['detected']:
            cv2.putText(annotated_frame, '🔥 FIRE DETECTED!', (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(annotated_frame, f"   Confidence: {detections['fire']['confidence']:.2%}",
                       (20, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 80

        # Missing Kid Alert
        if detections['kid']['detected']:
            cv2.putText(annotated_frame, '🚨 MISSING KID DETECTED!', (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
            cv2.putText(annotated_frame, f"   Identity: {detections['kid']['kid_id']}",
                       (20, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(annotated_frame, f"   Similarity: {detections['kid']['similarity']:.2%}",
                       (20, y_offset + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    else:
        # Safe status
        cv2.putText(annotated_frame, '✅ ALL SYSTEMS SAFE', (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(annotated_frame, 'No Threats Detected', (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return annotated_frame


# ============================================
# Sidebar Settings
# ============================================
st.sidebar.header("⚙️ System Configuration")

# Input source selection
input_source = st.sidebar.radio(
    "📥 Input Source",
    ["📹 Webcam", "📁 Upload Video", "🖼️ Upload Image"],
    index=0
)

# Configure input based on selection
if input_source == "📹 Webcam":
    camera_index = st.sidebar.selectbox("Camera Device", options=[0, 1, 2])
    uploaded_video = None
    uploaded_image = None
elif input_source == "📁 Upload Video":
    camera_index = None
    uploaded_video = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv', 'webm'])
    uploaded_image = None
else:
    camera_index = None
    uploaded_video = None
    uploaded_image = st.sidebar.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png', 'bmp'])

st.sidebar.markdown("---")

# Detection Module Selection
st.sidebar.subheader("🔍 Detection Modules")
enable_fire = st.sidebar.checkbox("🔥 Fire Detection", value=True)
enable_kid = st.sidebar.checkbox("🚨 Missing Kid Detection", value=True)

st.sidebar.markdown("---")

# Confidence Thresholds
st.sidebar.subheader("🎯 Detection Thresholds")
fire_threshold = st.sidebar.slider(
    "Fire Confidence", 0.01, 0.99, 0.50, 0.01,
    help="Minimum confidence to detect fire"
)
kid_threshold = st.sidebar.slider(
    "Kid Similarity", 0.50, 0.99, 0.65, 0.01,
    help="Minimum similarity to match a missing kid (65% recommended)"
)

thresholds = {
    'fire': fire_threshold if enable_fire else 1.0,
    'kid': kid_threshold if enable_kid else 1.0
}

st.sidebar.markdown("---")

# Kid Face Verification Settings
st.sidebar.subheader("🚨 Face Verification Settings")
similarity_metric = st.sidebar.radio(
    "Similarity Metric",
    ["cosine", "euclidean"],
    index=0,
    help="Cosine: Angle-based (recommended)"
)

st.sidebar.markdown("---")

# Alert Settings
st.sidebar.subheader("🔔 Alert Configuration")
enable_sound = st.sidebar.checkbox("🔊 Sound Alarm", value=True)

st.sidebar.markdown("---")

# Performance Settings
st.sidebar.subheader("⚡ Performance Tuning")
frame_skip = st.sidebar.slider(
    "Process Every N Frames", 1, 10, 3, 1,
    help="Skip frames to improve processing speed"
)
display_size = st.sidebar.slider(
    "Display Size (%)", 50, 100, 80, 10,
    help="Adjust display resolution"
)

# Debug Mode
debug_mode = st.sidebar.checkbox("🛠 Debug Mode", value=False)

st.sidebar.markdown("---")

# Diagnostic Tools
if st.sidebar.button("🔬 Run System Diagnostics"):
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 System Diagnostics")
    
    # Check reference database
    kid_reference_db = load_kid_reference_database()
    st.sidebar.write(f"📊 Database kids: {len(kid_reference_db)}")
    if kid_reference_db:
        sample_kid = list(kid_reference_db.keys())[0]
        st.sidebar.write(f"👤 Sample ID: {sample_kid}")
        st.sidebar.write(f"📊 Embedding shape: {kid_reference_db[sample_kid]['embedding'].shape}")


# ============================================
# Load Models and Reference Database
# ============================================
st.sidebar.markdown("---")
st.sidebar.subheader("📦 System Loading")

# Load reference database
kid_reference_db = load_kid_reference_database()

# Load models
models, device = load_all_models()

st.sidebar.markdown("---")
st.sidebar.info(f"🚨 Missing Kids: {len(kid_reference_db)} identities")
st.sidebar.info(f"💻 Compute Device: {device}")


# ============================================
# Main Layout
# ============================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📹 Live Input Feed")
    frame_placeholder = st.empty()

with col2:
    st.subheader("🔍 Detection Results")
    result_placeholder = st.empty()

# Alert and Audio Placeholders
alert_placeholder = st.empty()
audio_placeholder = st.empty()

# Metrics Display
met1, met2, met3, met4 = st.columns(4)
with met1:
    fps_metric = st.empty()
with met2:
    status_metric = st.empty()
with met3:
    fire_metric = st.empty()
with met4:
    kid_metric = st.empty()


# ============================================
# Control Buttons
# ============================================
if input_source == "📹 Webcam":
    start_button = st.button("🎥 Start Camera", type="primary")
    stop_button = st.button("⏹️ Stop Camera")
    process_image_button = False
elif input_source == "📁 Upload Video":
    start_button = st.button("▶️ Process Video", type="primary")
    stop_button = st.button("⏹️ Stop Processing")
    process_image_button = False
else:
    process_image_button = st.button("🔍 Analyze Image", type="primary")
    start_button = False
    stop_button = False


# ============================================
# Session State Management
# ============================================
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False

if start_button:
    st.session_state.camera_running = True

if stop_button:
    st.session_state.camera_running = False


# ============================================
# IMAGE PROCESSING MODE
# ============================================
if input_source == "🖼️ Upload Image" and uploaded_image is not None:
    if process_image_button:
        # Load image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is not None:
            # Run all detections
            annotated_image, detections = run_all_detections(
                image, models, device, thresholds, kid_reference_db, 
                similarity_metric, debug_mode
            )

            # Display results
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(image_rgb, channels="RGB", use_container_width=True)
            result_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

            # Show alerts
            any_threat = (detections['fire']['detected'] or detections['kid']['detected'])

            if any_threat:
                alert_messages = []
                if detections['fire']['detected']:
                    alert_messages.append(f"🔥 Fire ({detections['fire']['confidence']:.1%})")
                if detections['kid']['detected']:
                    alert_messages.append(
                        f"🚨 Missing Kid: {detections['kid']['kid_id']} "
                        f"({detections['kid']['similarity']:.1%})"
                    )

                alert_placeholder.error("⚠️ THREATS DETECTED: " + " | ".join(alert_messages))

                if enable_sound:
                    audio_placeholder.markdown(play_alarm_sound(), unsafe_allow_html=True)
            else:
                alert_placeholder.success("✅ Safe - No Threats Detected")

            # Update metrics
            status_metric.metric("Status", "🟢 ANALYZED")
            fire_metric.metric("🔥 Fire", "Yes" if detections['fire']['detected'] else "No")
            kid_metric.metric(
                "🚨 Missing Kid", 
                detections['kid']['kid_id'] if detections['kid']['detected'] else "No"
            )
        else:
            st.error("❌ Failed to load image")
    elif uploaded_image is not None:
        st.info("👆 Click 'Analyze Image' to detect threats")


# ============================================
# VIDEO/WEBCAM PROCESSING MODE
# ============================================
elif st.session_state.camera_running:
    if input_source == "📹 Webcam":
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        source_name = "Camera"
    else:
        if uploaded_video is None:
            st.error("❌ Please upload a video file first!")
            st.session_state.camera_running = False
            st.stop()

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        source_name = "Video"

    if not cap.isOpened():
        st.error(f"❌ Cannot access {source_name.lower()}")
        st.session_state.camera_running = False
    else:
        frame_count = 0
        detection_counts = {'fire': 0, 'kid': 0}
        start_time = time.time()
        last_detection_result = (None, None)

        stop_placeholder = st.empty()
        stop_placeholder.info(f"📹 {source_name} running... Click 'Stop' to end")

        while st.session_state.camera_running:
            ret, frame = cap.read()

            if not ret:
                st.info("📹 Video ended")
                break

            frame_count += 1

            # Process every Nth frame for performance
            if frame_count % frame_skip == 0:
                annotated_frame, detections = run_all_detections(
                    frame, models, device, thresholds, kid_reference_db, 
                    similarity_metric, debug_mode
                )
                last_detection_result = (annotated_frame, detections)
            else:
                if last_detection_result[0] is not None:
                    annotated_frame, detections = last_detection_result
                else:
                    annotated_frame = frame.copy()
                    detections = {
                        'fire': {'detected': False, 'confidence': 0.0},
                        'kid': {'detected': False, 'similarity': 0.0, 'kid_id': 'Unknown'}
                    }

            # Update detection counts
            if detections['fire']['detected']:
                detection_counts['fire'] += 1
            if detections['kid']['detected']:
                detection_counts['kid'] += 1

            # Check for threats
            any_threat = (detections['fire']['detected'] or detections['kid']['detected'])

            # Display alerts
            if any_threat:
                alert_messages = []
                if detections['fire']['detected']:
                    alert_messages.append(f"🔥 Fire ({detections['fire']['confidence']:.1%})")
                if detections['kid']['detected']:
                    alert_messages.append(
                        f"🚨 Missing Kid: {detections['kid']['kid_id']} "
                        f"({detections['kid']['similarity']:.1%})"
                    )

                alert_placeholder.error("⚠️ THREATS DETECTED: " + " | ".join(alert_messages))

                if enable_sound:
                    audio_placeholder.markdown(play_alarm_sound(), unsafe_allow_html=True)
            else:
                alert_placeholder.success("✅ Safe - No Threats Detected")
                audio_placeholder.markdown(stop_alarm(), unsafe_allow_html=True)

            # Calculate FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            # Resize frames for display
            display_width = int(frame.shape[1] * display_size / 100)
            display_height = int(frame.shape[0] * display_size / 100)

            frame_display = cv2.resize(frame, (display_width, display_height))
            annotated_display = cv2.resize(annotated_frame, (display_width, display_height))

            frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
            annotated_rgb = cv2.cvtColor(annotated_display, cv2.COLOR_BGR2RGB)

            # Update display every other frame
            if frame_count % 2 == 0:
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                result_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

                fps_metric.metric("FPS", f"{fps:.1f}")
                status_metric.metric("Status", "🟢 LIVE")
                fire_metric.metric("🔥 Fire", detection_counts['fire'])
                kid_metric.metric("🚨 Missing Kid", detection_counts['kid'])

            if not st.session_state.camera_running:
                break

        cap.release()
        stop_placeholder.success(f"✅ {source_name} stopped")
        status_metric.metric("Status", "🔴 STOPPED")

else:
    if input_source == "📹 Webcam":
        st.info("👆 Click 'Start Camera' to begin live security monitoring")
    elif input_source == "📁 Upload Video":
        st.info("👆 Upload a video file and click 'Process Video'")
    else:
        st.info("👆 Upload an image and click 'Analyze Image'")


# ============================================
# Footer
# ============================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>🔒 AI Security System v2.0 | Fire Detection + Missing Kid Detection</p>
        <p>Powered by PyTorch • OpenCV • Streamlit</p>
    </div>
""", unsafe_allow_html=True)