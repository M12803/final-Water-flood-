from flask import Flask, render_template, request, send_file
import io
import os
from PIL import Image
import numpy as np
import tifffile
import torch
from torchvision import transforms
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import UNet model
try:
    from models.unet_model import UNet
except ImportError as e:
    raise ImportError(f"Could not import UNet from models.unet_model: {str(e)}. Ensure models/unet_model.py exists and defines UNet class.")

app = Flask(__name__)

# Define model path
MODEL_PATH = r"E:\Cellula Tasks\Week 3\finall water flood\models\Moodel 1.pth"

# Load the pre-trained UNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = UNet(in_channels=12, out_channels=1)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return "No file selected", 400
    
    if not file.filename.lower().endswith('.tif'):
        return "Only TIFF files are supported", 400

    try:
        logger.debug("Reading TIFF image")
        img = tifffile.imread(file)
        
        logger.debug(f"Image shape: {img.shape}, dtype: {img.dtype}")
        if len(img.shape) == 2:  # Grayscale to 12 channels
            img = np.stack([img] * 12, axis=2)
        elif len(img.shape) == 3:
            if img.shape[2] < 12:
                img = np.pad(img, ((0, 0), (0, 0), (0, 12 - img.shape[2])), mode='constant')
            elif img.shape[2] > 12:
                img = img[:, :, :12]
            if img.shape[0] <= 12 and img.shape[0] != img.shape[1]:
                img = np.transpose(img, (1, 2, 0))
        
        if img.dtype != np.uint8:
            img = (img / img.max() * 255).astype(np.uint8) if img.max() > 0 else img.astype(np.uint8)
        
        logger.debug(f"Processed image shape: {img.shape}")
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)  # Remove extra transform

        logger.debug("Running prediction")
        with torch.no_grad():  # ✅ FIXED INDENTATION HERE
            pred = model(img_tensor)
            pred = torch.sigmoid(pred)  # Ensure values are in range (0,1)

        # Debugging: Check prediction values
        logger.debug(f"Prediction min: {pred.min().item()}, max: {pred.max().item()}")

        pred = (pred * 255).byte()  # Scale to (0,255)

        # ✅ Fix: Ensure we convert Tensor to NumPy correctly
        pred_np = pred.squeeze().cpu().detach().numpy()

        # Ensure it's a valid 2D grayscale image
        if pred_np.ndim == 3:
            pred_np = pred_np[0]  # Take the first channel if it's (1, H, W)

        if pred_np.ndim != 2:
            raise ValueError(f"Expected 2D prediction array, got shape {pred_np.shape}")

        # ✅ Fix: Convert to uint8 if needed
        pred_np = pred_np.astype(np.uint8)

        # Convert NumPy array to PIL Image
        pred_img = Image.fromarray(pred_np)
        logger.debug(f"Generated mask shape: {pred_np.shape}")

        img_byte_arr = io.BytesIO()
        pred_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return send_file(
            img_byte_arr,
            mimetype='image/png',
            as_attachment=False
        )
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return f"Error processing image: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
