import os
import datetime
from PIL import Image
from src.config import REPORTS_PATH

def generate_html_report(image_path, prediction_result, overlay_img):
    os.makedirs(REPORTS_PATH, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    orig_path = os.path.join(REPORTS_PATH, f"orig_{timestamp}.jpg")
    overlay_path = os.path.join(REPORTS_PATH, f"overlay_{timestamp}.jpg")
    
    Image.fromarray(image_path.astype('uint8')).save(orig_path)
    Image.fromarray(overlay_img.astype('uint8')).save(overlay_path)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Skin Lesion AI Report</title></head>
    <body style="font-family: Arial; padding: 30px;">
        <h2 style="color: #004d40;">Skin Lesion AI Assessment Report</h2>
        <p><b>Date:</b> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><b>Predicted Class:</b> {prediction_result['predicted_class']}</p>
        <p><b>Model Confidence:</b> {prediction_result['confidence'] * 100:.2f}%</p>
        <h3>Grad-CAM Explainable AI Overlay</h3>
        <img src="{os.path.basename(overlay_path)}" width="300" style="border: 1px solid #ccc;" />
        <p><b>Disclaimer:</b> AI research prototype. Consult a dermatologist.</p>
    </body>
    </html>
    """
    path = os.path.join(REPORTS_PATH, "report.html")
    with open(path, 'w') as f:
        f.write(html_content)
    return path