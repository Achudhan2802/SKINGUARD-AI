import os
# Suppress TensorFlow oneDNN informational logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import shap
from src.config import MODEL_PATH, CLASS_NAMES_PATH, LESION_TYPE_DICT

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_clinical_report(image_path, output_pdf_path="Clinical_Diagnostic_Report.pdf"):
    print("[INFO] Loading assets and model...")
    with open(CLASS_NAMES_PATH, 'r') as f:
        idx_to_class = json.load(f)
        
    from src.model import build_efficientnet_model
    model, _ = build_efficientnet_model(num_classes=len(idx_to_class))
    if os.path.exists(MODEL_PATH):
        model.load_weights(MODEL_PATH)
    else:
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")

    print(f"[INFO] Processing image: {image_path}")
    original_img = Image.open(image_path).convert("RGB")
    img_resized = original_img.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)

    # 1. Model Prediction
    preds = model.predict(input_tensor)[0]
    pred_idx = np.argmax(preds)
    confidence = float(preds[pred_idx] * 100)
    
    short_code = idx_to_class.get(str(pred_idx), idx_to_class.get(pred_idx, "nv"))
    full_clinical_name = LESION_TYPE_DICT.get(short_code, "Melanocytic nevi")

    # 2. Generate SHAP Explanation Heatmap
    print("[INFO] Generating SHAP explainable AI attribution map...")
    background = np.zeros((1, 224, 224, 3), dtype=np.float32)
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(input_tensor)
    
    if isinstance(shap_values, list):
        current_shap = shap_values[pred_idx]
    else:
        current_shap = shap_values

    # Plot SHAP overlay heatmap
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_array)
    shap_img = np.abs(current_shap[0]).mean(axis=-1)
    shap_img = (shap_img - shap_img.min()) / (shap_img.max() - shap_img.min() + 1e-8)
    ax.imshow(shap_img, cmap='jet', alpha=0.5)
    ax.axis('off')
    ax.set_title(f"SHAP Attribution: {full_clinical_name}", fontsize=9, fontweight='bold', color='#1b365d')
    
    os.makedirs("outputs", exist_ok=True)
    shap_img_path = os.path.join("outputs", "shap_explanation.png")
    plt.savefig(shap_img_path, bbox_inches='tight', dpi=200)
    plt.close()
    
    # Save preview image
    preview_path = os.path.join("outputs", "lesion_preview.jpg")
    original_img.resize((200, 200)).save(preview_path)

    # 3. Build ReportLab PDF
    print(f"[INFO] Compiling ReportLab PDF to {output_pdf_path}...")
    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('ClinicTitle', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#1b365d'), spaceAfter=4)
    subtitle_style = ParagraphStyle('ClinicSub', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#64748b'), spaceAfter=15)
    heading_style = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#1b365d'), spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9.5, textColor=colors.HexColor('#2c3e50'), spaceAfter=8, leading=13)
    bold_body = ParagraphStyle('BoldBody', parent=body_style, fontName='Helvetica-Bold')

    # Header block
    story.append(Paragraph("AI Dermatology &amp; Oncology Center", title_style))
    story.append(Paragraph("Advanced Computer-Aided Diagnostic Laboratory &bull; Clinical Evaluation Report", subtitle_style))
    
    # Patient Info Table
    patient_data = [
        [Paragraph("<b>Patient Name:</b> John Doe", body_style), Paragraph("<b>Report ID:</b> AI-DERM-2026-9042", body_style)],
        [Paragraph("<b>Age / Gender:</b> 35 / Male", body_style), Paragraph("<b>Date:</b> September 5, 2026", body_style)],
        [Paragraph("<b>Modality:</b> Dermoscopic Deep Learning (EfficientNetB0)", body_style), Paragraph("<b>Source:</b> HAM10000 Dataset Benchmark", body_style)]
    ]
    t_patient = Table(patient_data, colWidths=[270, 270])
    t_patient.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_patient)
    story.append(Spacer(1, 10))

    # Section 1: Images & SHAP
    story.append(Paragraph("1. Submitted Lesion &amp; Explainable AI (SHAP) Attribution", heading_style))
    story.append(Paragraph("The visual attribution map highlights localized pixel regions (pigment networks and structural asymmetry) that heavily influenced the deep learning classification decision.", body_style))
    
    img_data = [
        [Paragraph("<b>Original Lesion Input</b>", body_style), Paragraph("<b>SHAP Attribution Heatmap</b>", body_style)],
        [RLImage(preview_path, width=150, height=150), RLImage(shap_img_path, width=150, height=150)]
    ]
    t_imgs = Table(img_data, colWidths=[270, 270])
    t_imgs.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_imgs)
    story.append(Spacer(1, 10))

    # Section 2: Clinical Findings
    story.append(Paragraph("2. Diagnostic Classification &amp; Clinical Reasoning", heading_style))
    
    result_data = [
        [Paragraph("Primary Prediction Code:", bold_body), Paragraph(f"<code>{short_code}</code>", body_style)],
        [Paragraph("Full Clinical Diagnosis Name:", bold_body), Paragraph(f"<b><font color='#1b365d' size=11>{full_clinical_name}</font></b>", body_style)],
        [Paragraph("Confidence Score:", bold_body), Paragraph(f"<b>{confidence:.2f}%</b>", body_style)],
    ]
    t_result = Table(result_data, colWidths=[180, 360])
    t_result.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_result)
    story.append(Spacer(1, 8))
    
    reasoning_text = f"<b>Clinical Interpretation:</b> The EfficientNet convolutional neural network analyzed border structure, pigment dispersion, and lesion asymmetry. Based on learned pattern extraction matching benchmark dermatology repositories, the sample has been categorized as a <b>{full_clinical_name}</b> with a confidence rating of {confidence:.2f}%."
    story.append(Paragraph(reasoning_text, body_style))
    story.append(Spacer(1, 5))

    # Section 3: Disclaimer
    story.append(Paragraph("3. Clinical Disclaimer", heading_style))
    disclaimer_text = "This document is generated by an artificial intelligence diagnostic assistant for research and evaluation purposes. It does not replace professional dermatological evaluation, physical dermoscopy, or histopathological biopsy."
    story.append(Paragraph(disclaimer_text, body_style))
    
    # Build PDF
    doc.build(story)
    print(f"[SUCCESS] ReportLab Clinical PDF generated successfully at: {output_pdf_path}")

if __name__ == "__main__":
    # Make sure your test image name matches here (e.g., 'ISIC_0024306.jpg')
    sample_image = "ISIC_0024306.jpg" 
    
    if os.path.exists(sample_image):
        generate_clinical_report(sample_image)
    else:
        print(f"[ERROR] Image file '{sample_image}' not found in project directory. Please place it in the root folder or update the path.")