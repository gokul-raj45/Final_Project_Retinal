import os
import logging
import numpy as np
import tensorflow as tf
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


import urllib.request

from flask import Flask, render_template, request, send_file ,redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from datetime import datetime

from PIL import Image



# NEW: image validation layer (classical CV heuristics, no new model)
from validation import is_retinal_fundus_image

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

# Required for flash() to work — replace with a real secret in production
app.secret_key = "replace-this-with-a-secure-random-value"


# ---------------- Load Hybrid Models ----------------
os.makedirs("models", exist_ok=True)

RESNET_PATH = "models/eye_disease_resnet.h5"

if not os.path.exists(RESNET_PATH):
    logging.info("Downloading ResNet model...")
    urllib.request.urlretrieve(
        "https://huggingface.co/gokulraj-45/eye-disease-model/resolve/main/eye_disease_resnet.h5",
        RESNET_PATH
    )



logging.info("Loading ResNet model...")
resnet_model = load_model(RESNET_PATH)


logging.info("Hybrid models loaded successfully!")

# ---------------- Config ----------------
CATEGORIES = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]

UPLOAD_FOLDER = "static/uploads"
REPORT_FOLDER = "static/reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER

# ---------------- Recommendations ----------------
RECOMMENDATIONS = {
    "Normal": [
        "Follow 20-20-20 rule (every 20 min, look 20 feet away for 20 sec)",
        "Eat foods rich in Vitamin A (carrots, spinach)",
        "Stay hydrated and maintain good sleep cycle",
        "Wear UV-protected sunglasses outdoors"
    ],

    "Cataract": [
        "Use anti-glare glasses to reduce light sensitivity",
        "Wear sunglasses to protect from UV rays",
        "Increase intake of antioxidants (leafy greens, carrots, citrus fruits)",
        "Avoid smoking and excessive alcohol",
        "Consult ophthalmologist — surgery is effective in advanced stages",
        "Avoid self-medication; follow doctor-prescribed eye drops only"
    ],

    "Glaucoma": [
        "Do moderate exercises like walking or yoga (improves blood circulation)",
        "Avoid heavy weight lifting and inverted yoga positions",
        "Eat leafy greens (spinach, kale) and omega-3 foods (fish, nuts)",
        "Limit caffeine intake (can increase eye pressure)",
        "Use prescribed eye drops regularly (doctor advice required)",
        "Regular eye pressure monitoring is essential"
    ],

    "Diabetic Retinopathy": [
        "Strict blood sugar control (monitor glucose regularly)",
        "Do regular physical activity (walking, light exercise)",
        "Eat low-sugar, high-fiber diet (vegetables, whole grains)",
        "Avoid sugary foods, soft drinks, and processed snacks",
        "Control blood pressure and cholesterol levels",
        "Consult eye specialist regularly — early treatment prevents blindness"
    ]
}

def get_recommendations(disease, confidence):
    recs = RECOMMENDATIONS[disease].copy()

    if disease != "Normal":
        if confidence >= 90:
            recs.append("⚠ Immediate medical consultation is strongly recommended")
        elif confidence >= 75:
            recs.append("⚠ Monitor symptoms and schedule doctor visit soon")

    return recs

# ---------------- Hybrid Prediction ----------------
def hybrid_predict(img_array):
    pred_resnet = resnet_model.predict(img_array, verbose=0)
    final_pred = pred_resnet

    class_index = np.argmax(final_pred)
    confidence = float(np.max(final_pred))

    prob_dict = {
        CATEGORIES[i]: float(final_pred[0][i])
        for i in range(len(CATEGORIES))
    }

    return CATEGORIES[class_index], confidence, prob_dict

# ---------------- Grad-CAM ----------------
def apply_retinal_mask(img):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    radius = min(cx, cy) - 5

    Y, X = np.ogrid[:h, :w]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2

    img_masked = img.copy()
    img_masked[~mask] = 0
    return img_masked, mask

def generate_gradcam(img_array, model, last_conv_layer_name, original_img_path):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    img = cv2.imread(original_img_path)
    img = cv2.resize(img, (224, 224))

    img_masked, mask = apply_retinal_mask(img)

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap[~mask] = 0

    superimposed_img = cv2.addWeighted(img_masked, 0.6, heatmap, 0.4, 0)

    gradcam_path = original_img_path.replace(".", "_gradcam.")
    cv2.imwrite(gradcam_path, superimposed_img)

    return gradcam_path

# ---------------- Probability Bar Chart ----------------
def generate_probability_chart(prob_dict, save_path):
    labels = list(prob_dict.keys())
    values = list(prob_dict.values())

    #  Convert to percentage
    values_percent = [v * 100 for v in values]

    #  Bigger & sharper chart
    plt.figure(figsize=(8, 5), dpi=300)

    bars = plt.bar(labels, values_percent)

    #  Axis labels (clear)
    plt.ylabel("Confidence (%)", fontsize=13, fontweight='bold')
    plt.xlabel("Disease Class", fontsize=13, fontweight='bold')

    #  Title
    plt.title("Class-wise Prediction Probabilities", fontsize=16, fontweight='bold', pad=15)

    #  Y limit (proper spacing for labels)
    plt.ylim(0, max(values_percent) + 10)

    #  Tick size improve
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    #  Highlight predicted class
    max_index = values.index(max(values))
    for i, bar in enumerate(bars):
        if i == max_index:
            bar.set_color("#ff4d4d")   # red (predicted)
        else:
            bar.set_color("#4dabf7")   # blue

    #  VALUE LABEL (CLEAR & BIG)
    for i, v in enumerate(values_percent):
        plt.text(
            i,
            v + 1,
            f"{v:.1f}%",
            ha="center",
            fontsize=12,
            fontweight="bold"
        )

    #  Grid
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    #  Layout fix
    plt.tight_layout()

    #  Save HIGH QUALITY (PDF friendly)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# ---------------- Routes ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # ---- Retinal image validation (pre-prediction gate) ----
            # This runs BEFORE the image ever touches ResNet50/DenseNet121.
            # If validation fails, we stop here and never call hybrid_predict().
            valid, validation_info = is_retinal_fundus_image(filepath)
            logging.info(f"Validation result for {file.filename}: {validation_info}")

            if not valid:
                # Remove the invalid upload so it doesn't clutter the folder
                try:
                    os.remove(filepath)
                except OSError:
                    pass

                flash(
                    "This is not a retinal fundus image. "
                    "Please upload a valid retinal eye image and try again.",
                    "error"
                )
                return redirect(url_for("index"))



            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            result, confidence, prob_dict = hybrid_predict(img_array)

            gradcam_path = generate_gradcam(
                img_array,
                resnet_model,
                "block5_conv4",
                filepath
            )

            chart_path = os.path.join(app.config["REPORT_FOLDER"], "prob_chart.png")
            generate_probability_chart(prob_dict, chart_path)

            report_path = os.path.join(
                app.config["REPORT_FOLDER"],
                f"{os.path.splitext(file.filename)[0]}_report.pdf"
            )

            generate_pdf(
                report_path,
                filepath,
                gradcam_path,
                result,
                prob_dict,
                get_recommendations(result, confidence),
                chart_path
            )

            return render_template(
                "result.html",
                result=result,
                confidence=round(confidence * 100, 2),
                img_path=filepath,
                gradcam_path=gradcam_path,
                recommendations=get_recommendations(result, confidence),
                report_path=report_path,
                prob_dict=prob_dict
            )

    return render_template("index.html")

# ---------------- NEW ROUTE (VIEW REPORT) ----------------
@app.route("/view_report/<path:filename>")
def view_report(filename):
    base_name = os.path.splitext(os.path.basename(filename))[0]

    gradcam_jpg = os.path.join("static", "uploads", base_name + "_gradcam.jpg")
    gradcam_jpeg = os.path.join("static", "uploads", base_name + "_gradcam.jpeg")

    if os.path.exists(gradcam_jpg):
        gradcam_path = "uploads/" + base_name + "_gradcam.jpg"
    elif os.path.exists(gradcam_jpeg):
        gradcam_path = "uploads/" + base_name + "_gradcam.jpeg"
    else:
        gradcam_path = None

    chart_path = "reports/prob_chart.png"


    prob_dict = {
        "Cataract": float(request.args.get("cataract", 0)),
        "Diabetic Retinopathy": float(request.args.get("dr", 0)),
        "Glaucoma": float(request.args.get("glaucoma", 0)),
        "Normal": float(request.args.get("normal", 0)),
    }

    return render_template(
        "view_report.html",
        gradcam_path=gradcam_path,
        chart_path=chart_path,
        prob_dict=prob_dict   
    )

# ---------------- PDF Generator ----------------
def generate_pdf(pdf_path, img_path, gradcam_path, result, prob_dict, recommendations, chart_path):

    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # ---------------- HEADER ----------------
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 40, "AI-Based Retinal Examination Report")

    c.setFont("Helvetica", 10)
    c.drawString(50, height - 60, "Department of Ophthalmology")
    c.drawString(400, height - 60, "Report ID: AI-RET-2026")

    c.line(50, height - 70, width - 50, height - 70)

    # ---------------- CASE INFO ----------------
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 95, "Case Information")

    file_name = os.path.basename(img_path)
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 110, f"Case ID: {file_name}")
    c.drawString(50, height - 125, "Image Source: Public Dataset")
    c.drawString(50, height - 140, "Input Type: Fundus Retina Scan")
    c.drawString(50, height - 155, f"Processing Date: {datetime.now().strftime('%d-%b-%Y')}")

    # ---------------- DIAGNOSIS ----------------
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 185, "Diagnosis Summary")

    c.setFont("Helvetica", 11)
    c.drawString(50, height - 200, f"Primary Diagnosis : {result}")

    # ---------------- CONFIDENCE ----------------
    confidence = max(prob_dict.values()) * 100

    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.blue)
    c.drawString(50, height - 220, f"Prediction Confidence: {round(confidence,2)}%")
    c.setFillColor(colors.black)

    # ---------------- SEVERITY + RISK ----------------
    if result == "Normal":
        severity = "None"
        risk = "Low"
    elif confidence >= 95:
        severity = "Severe"
        risk = "High"
    elif confidence >= 75:
        severity = "Moderate"
        risk = "Medium"
    else:
        severity = "Mild"
        risk = "Low"

    c.drawString(50, height - 240, f"Severity Level: {severity}")
    c.setFillColor(colors.red)
    c.drawString(50, height - 255, f"Risk Level: {risk}")
    c.setFillColor(colors.black)

    # ---------------- IMAGES ----------------
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 285, "Model Interpretation & Probability Analysis")

    image_y = height - 580

    c.drawImage(ImageReader(gradcam_path), 50, image_y, width=240, height=240)
    c.drawImage(ImageReader(chart_path), 320, image_y, width=240, height=240)

    # ---------------- AI INTERPRETATION ----------------
    y = image_y - 40

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "AI Interpretation")
    y -= 20

    c.setFont("Helvetica", 10)

    lines = [
        f"The model identified retinal abnormalities associated with {result}.",
        "Highlighted regions indicate disease-affected retinal areas.",
        f"With a confidence of {round(confidence,2)}%, this suggests a clinically",
        "significant condition requiring further medical evaluation."
    ]

    for line in lines:
        c.drawString(50, y, line)
        y -= 15

    # ---------------- NEW PAGE ----------------
    c.showPage()
    y = height - 50

    # ---------------- RECOMMENDATIONS ----------------
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Clinical Recommendations")
    y -= 25

    c.setFont("Helvetica", 11)

    for rec in recommendations:
        c.drawString(60, y, f"- {rec}")
        y -= 15

    # ---------------- FOLLOW-UP ----------------
    y -= 10
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Follow-Up Recommendation")
    y -= 20

    followups = [
        "Immediate consultation with ophthalmologist",
        "Schedule retinal screening within 1 month",
        "Monitor relevant health parameters regularly"
    ]

    c.setFont("Helvetica", 11)

    for f in followups:
        c.drawString(60, y, f"- {f}")
        y -= 15

    # ---------------- FOOTER ----------------
    footer_y = 80
    c.line(50, footer_y + 20, width - 50, footer_y + 20)

    c.setFont("Helvetica", 9)
    c.drawString(50, footer_y, "Disclaimer: This AI-generated report is for clinical decision support only.")
    c.drawRightString(width - 50, footer_y - 10, "Authorized by AI Diagnostic System")

    c.save()
@app.route("/download/<path:filename>")
def download_report(filename):
    return send_file(filename, as_attachment=True)

# ---------------- Run ----------------
if __name__ == "__main__":
    logging.info("Starting Flask app...")
    app.run(debug=True)
