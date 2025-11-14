import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Nutrient Deficiency Detector", layout="wide")

st.title("🧬 AI-Based Nutrient Deficiency Detector (Selfie Analysis)")
st.write("Upload a selfie to detect possible nutrient deficiencies.")

uploaded_file = st.file_uploader("Upload a Clear Front-Facing Photo", type=["jpg", "jpeg", "png"])

# --------------------------
# Helper functions
# --------------------------

def analyze_paleness(face_roi):
    """Simple brightness / redness based analysis."""
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

    brightness = np.mean(hsv[:, :, 2])
    redness = np.mean(face_roi[:, :, 2])

    return brightness, redness

def detect_iron_deficiency(eye_roi):
    brightness = np.mean(eye_roi)
    if brightness > 160:  # pale threshold
        return "⚠ Possible Iron Deficiency (pale inner eyelids detected)"
    return "✔ Iron Deficiency Not Indicated"

def detect_b12_deficiency(lip_roi):
    redness = np.mean(lip_roi[:, :, 2])
    if redness < 120:
        return "⚠ Possible Vitamin B12 Deficiency (lip paleness)"
    return "✔ B12 Deficiency Not Indicated"

def detect_dehydration(face_brightness):
    if face_brightness > 170:
        return "⚠ Possible Dehydration (dry / overly bright skin)"
    return "✔ Dehydration Not Indicated"

# --------------------------
# Main Processing Section
# --------------------------

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=300)

    img = Image.open(uploaded_file)
    img = np.array(img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Load face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                         "haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(img_bgr, 1.3, 5)

    if len(faces) == 0:
        st.error("No face detected. Please upload a clear front photo.")
    else:
        for (x, y, w, h) in faces:
            face_roi = img_bgr[y:y+h, x:x+w]

            # --- Basic Analysis ---
            face_brightness, face_redness = analyze_paleness(face_roi)

            # Mock ROIs for simplicity
            eye_roi = face_roi[h//5 : h//3, w//4 : 3*w//4]
            lip_roi = face_roi[2*h//3 : 5*h//6, w//4 : 3*w//4]

            # Display results
            st.subheader("🔍 Analysis Results")

            st.write("### 🩸 Iron Deficiency Check")
            st.success(detect_iron_deficiency(eye_roi))

            st.write("### 🧬 Vitamin B12 Deficiency Check")
            st.success(detect_b12_deficiency(lip_roi))

            st.write("### 💧 Dehydration Check")
            st.success(detect_dehydration(face_brightness))

            st.write("### 🩻 Hemoglobin (via Face Paleness)")
            if face_brightness > 180 and face_redness < 130:
                st.warning("⚠ Possible Low Hemoglobin (overall facial paleness)")
            else:
                st.success("✔ Hemoglobin Levels Look Normal")

            st.markdown("---")
            st.info("⚠ This is a preliminary screening tool — not a medical diagnosis.")
