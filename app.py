from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io
import json
import os
from datetime import datetime
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Model configuration - MUST match training (384x384 for EfficientNetV2B0)
IMG_SIZE = 384

# Model paths (try multiple formats)
MODEL_PATHS = [
    'palm_model_best.keras',           
    'palm_disease_model.h5',           
    'palm_disease_model',              
    
]

# Class labels - MUST match training order
CLASS_LABELS = {
    "0": "Dryness",
    "1": "Fungal disease",
    "2": "Magnesium Deficiency",
    "3": "Scale insect"
}

# Disease information database
DISEASE_INFO = {
    "Dryness": {
        "description": "Characterized by dried, brown, or yellow fronds. Often caused by insufficient water or extreme heat.",
        "causes": ["Water stress", "High temperatures", "Low humidity", "Poor soil drainage"],
        "treatment": [
            "Increase irrigation frequency",
            "Apply mulch around base to retain moisture",
            "Check soil drainage",
            "Monitor weather conditions"
        ],
        "severity": "Medium",
        "color": "#f39c12"
    },
    "Fungal disease": {
        "description": "Fungal infections appear as spots, lesions, or discoloration on fronds. Can spread rapidly in humid conditions.",
        "causes": ["High humidity", "Poor air circulation", "Infected plant material", "Wounded tissue"],
        "treatment": [
            "Remove infected fronds immediately",
            "Apply fungicide (copper-based recommended)",
            "Improve air circulation",
            "Avoid overhead watering",
            "Sanitize pruning tools"
        ],
        "severity": "High",
        "color": "#e74c3c"
    },
    "Magnesium Deficiency": {
        "description": "Yellowing of older fronds while veins remain green. Common in sandy soils with high rainfall.",
        "causes": ["Low soil magnesium", "High potassium levels", "Sandy soil", "Excessive rainfall"],
        "treatment": [
            "Apply magnesium sulfate (Epsom salt): 2-4 lbs per tree",
            "Use slow-release magnesium fertilizer",
            "Test soil pH (optimal: 6.0-7.0)",
            "Apply foliar spray for quick results"
        ],
        "severity": "Medium",
        "color": "#f39c12"
    },
    "Scale insect": {
        "description": "Small, round insects that attach to fronds and suck plant sap, causing yellowing and stunted growth.",
        "causes": ["Pest infestation", "Stressed plants", "Lack of natural predators", "Poor plant hygiene"],
        "treatment": [
            "Apply horticultural oil spray",
            "Use insecticidal soap",
            "Introduce natural predators (ladybugs)",
            "Remove heavily infested fronds",
            "Regular monitoring and early detection"
        ],
        "severity": "High",
        "color": "#e74c3c"
    }
}

def load_model():
    """Load the palm disease detection model"""
    logger.info("="*70)
    logger.info("Loading Palm Disease Detection Model")
    logger.info("="*70)
    
    for model_path in MODEL_PATHS:
        if not os.path.exists(model_path):
            logger.debug(f"Model not found: {model_path}")
            continue
            
        try:
            logger.info(f"Attempting to load: {model_path}")
            
            # Load model without compilation
            model = keras.models.load_model(
                model_path, 
                compile=False
            )
            
            # Recompile with AdamW optimizer (matching training)
            model.compile(
                optimizer=keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-6),
                loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                metrics=['accuracy']
            )
            
            # Verify model can make predictions
            logger.info("Testing model inference...")
            test_input = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            test_pred = model.predict(test_input, verbose=0)
            
            logger.info("="*70)
            logger.info(f"✓ Model loaded successfully from: {model_path}")
            logger.info(f"  Input shape:  {model.input_shape}")
            logger.info(f"  Output shape: {model.output_shape}")
            logger.info(f"  Classes:      {test_pred.shape[1]}")
            logger.info(f"  Image size:   {IMG_SIZE}x{IMG_SIZE}")
            logger.info("="*70)
            
            return model, model_path
            
        except Exception as e:
            logger.warning(f"✗ Failed to load {model_path}: {str(e)}")
            continue
    
    logger.error("Could not load model from any available path")
    return None, None

# Initialize model
model, model_path = load_model()

# Validate initialization
if model is None:
    logger.critical("="*70)
    logger.critical("ERROR: Could not load model from any format")
    logger.critical("Please ensure you have one of the following files:")
    for path in MODEL_PATHS:
        logger.critical(f"  - {path}")
    logger.critical("="*70)
    exit(1)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_file):
    """
    Preprocess uploaded image for model prediction.
    Must match training preprocessing pipeline:
    1. Convert to RGB
    2. Resize to 384x384 (IMG_SIZE)
    3. Convert to float32
    4. Apply EfficientNetV2 preprocessing
    """
    try:
        # Open and convert image to RGB
        img = Image.open(image_file).convert('RGB')
        
        # Resize to model input size (384x384)
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply EfficientNetV2 preprocessing (converts to range expected by model)
        img_array = tf.keras.applications.efficientnet_v2.preprocess_input(
            img_array * 255.0  # preprocess_input expects [0, 255]
        )
        
        return img_array
        
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")

@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided', 'success': False}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type. Only PNG, JPG, JPEG allowed',
                'success': False
            }), 400
        
        # Preprocess image
        logger.info(f"Processing image: {file.filename}")
        img_array = preprocess_image(file)
        
        # Make prediction
        logger.info("Running model inference...")
        predictions = model.predict(img_array, verbose=0)
        
        # Get predicted class and confidence
        class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][class_idx])
        
        # Get disease name
        disease_name = CLASS_LABELS[str(class_idx)]
        
        # Get all predictions (for chart)
        all_predictions = {
            CLASS_LABELS[str(i)]: float(predictions[0][i]) 
            for i in range(len(CLASS_LABELS))
        }
        
        # Get disease info
        disease_details = DISEASE_INFO.get(disease_name, {
            "description": "No information available",
            "causes": [],
            "treatment": [],
            "severity": "Unknown",
            "color": "#6b7280"
        })
        
        # Log prediction
        logger.info(f"✓ Prediction: {disease_name} (confidence: {confidence:.2%})")
        
        response = {
            'success': True,
            'disease': disease_name,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'details': disease_details,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(response)
    
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error', 'success': False}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': model_path,
        'classes': list(CLASS_LABELS.values()),
        'num_classes': len(CLASS_LABELS),
        'image_size': IMG_SIZE,
        'version': '1.0.0',
        'model_architecture': 'EfficientNetV2B0'
    })

@app.route('/api/diseases', methods=['GET'])
def get_diseases():
    """Get all disease information"""
    return jsonify(DISEASE_INFO)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    logger.warning("File upload too large")
    return jsonify({
        'error': 'File too large. Maximum size is 10MB',
        'success': False
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'success': False
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error("Internal server error", exc_info=True)
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  PalmCare AI - Disease Detection API")
    print("="*70)
    print(f"  Model:        {model_path}")
    print(f"  Architecture: EfficientNetV2B0")
    print(f"  Input size:   {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Classes:      {len(CLASS_LABELS)} diseases")
    print(f"  Max upload:   10MB")
    print(f"  Formats:      {', '.join(ALLOWED_EXTENSIONS).upper()}")
    print(f"  Server:       http://localhost:5000")
    print("="*70)
    print("\n[INFO] Server starting...\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)