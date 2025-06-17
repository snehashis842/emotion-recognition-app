import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import re
import time
import pandas as pd
import base64

# Page config
st.set_page_config(page_title="😊 Emotion Detector", page_icon="😎", layout="centered")


# Load model
@st.cache_resource
def load_emotion_model():
    return load_model("emotion_recognition_model.keras")


model = load_emotion_model()
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Session state init
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_emotion(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    prediction = model.predict(reshaped, verbose=0)
    return emotion_labels[np.argmax(prediction)]


# Navigation bar
def side_navigation():
    return st.sidebar.selectbox(
        "📂 Navigate to",
        ["🏠 Home", "📸 Camera", "🖼️ Upload Image", "ℹ️ About", "🚪 Logout"],
    )


# Background and banner
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    login_bg_style = f"""<style>
    [data-testid="stApp"] {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover; background-repeat: no-repeat;
        background-attachment: fixed; background-position: center;
    }}
    .login-container {{
        background-color: transparent; box-shadow: none;
        padding: 40px; border-radius: 12px; width: 400px;
        margin: auto; margin-top: 0px; text-align: center;
    }}
    .login-container input, .login-container button {{ margin-top: 12px; }}
    </style>"""
    st.markdown(login_bg_style, unsafe_allow_html=True)


def render_header():
    logo_html = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap');
    .asdp-banner {
        width: 100%; background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
        padding: 15px 30px 5px 30px; display: flex; align-items: center;
        justify-content: center; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        border-bottom: 3px solid #1e90ff; border-radius: 0 0 12px 12px;
    }
    .asdp-banner h1 {
        color: #ffffff; font-size: 26px; font-family: 'Orbitron', sans-serif;
        margin: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { text-shadow: 0 0 10px #1e90ff, 0 0 20px #1e90ff; }
        to { text-shadow: 0 0 20px #00bfff, 0 0 30px #00bfff; }
    }
    </style>
    <div class="asdp-banner"><h1>ASDP AI 2025 – Face Emotion Detector</h1></div>
    """
    st.markdown(logo_html, unsafe_allow_html=True)


def login_page():
    # Check if the user is already logged in
    if "logged_in" in st.session_state and st.session_state.logged_in:
        # User is logged in, skip the login page and go to the main app
        st.write("You are already logged in!")
        # Add navigation or main app content here
        return

    set_background("assets/abc.jpg")
    render_header()

    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.header("🔐 Sign In")

    # Use a form to prevent reruns while typing
    with st.form(key="login_form"):
        email = st.text_input("📧 Email", key="email_input")
        password = st.text_input("🔐 Password", type="password", key="password_input")
        submit = st.form_submit_button("Login")

        if submit:
            if not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email):
                st.error("❌ Enter a valid email.")
            elif len(password) < 8 or not re.search(
                r"[!@#$%^&*(),.?\":{}|<>]", password
            ):
                st.error(
                    "❌ Password must be 8+ characters and include a special symbol."
                )
            else:
                st.session_state.logged_in = True
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# home page
def home_page():
    st.markdown(
        """
        <style>
            .home-section h1 {
                text-align: center;
                color: #0F62FE;
                font-size: 38px;
                font-weight: bold;
                margin-bottom: 0.5em;
            }
            .home-section h3 {
                color: #0F62FE;
                margin-top: 2em;
                font-weight: 700;
            }
            .home-section p {
                font-size: 17px;
                color: #333333;
                line-height: 1.8;
            }
            ul {
                padding-left: 1.2em;
            }
            li {
                margin-bottom: 0.5em;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='home-section'>", unsafe_allow_html=True)

    st.markdown(
        "<h1>🏠 Welcome to ASDP AI – Real-Time Emotion Intelligence</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <p>
        <strong>ASDP AI</strong> is a cutting-edge emotion recognition platform engineered for the future of intelligent interaction.
        By fusing advanced deep learning with real-time computer vision, our solution translates facial expressions into actionable emotional insights—
        empowering industries to create smarter, more empathetic technologies.
    </p>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<h3>🎯 Platform Capabilities</h3>", unsafe_allow_html=True)
    st.markdown(
        """
    <ul>
        <li>📷 <strong>Real-Time Emotion Detection</strong> – Analyze live facial emotions instantly via your webcam.</li>
        <li>🖼️ <strong>Static Image Recognition</strong> – Upload images to receive detailed emotion profiling with confidence metrics.</li>
        <li>📊 <strong>Live Probability Dashboards</strong> – Visualize emotional states across seven categories with intuitive graphs.</li>
        <li>🔐 <strong>On-Device Processing</strong> – Privacy-first architecture ensures all data remains fully local.</li>
        <li>⚙️ <strong>Optimized for Edge</strong> – Lightweight CNN models designed for speed, scalability, and offline deployment.</li>
    </ul>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<h3>💼 Industry Applications</h3>", unsafe_allow_html=True)
    st.markdown(
        """
    <ul>
        <li>🎓 <strong>Education Technology</strong> – Monitor student engagement and emotional feedback during online learning sessions.</li>
        <li>🛒 <strong>Retail & CX Analytics</strong> – Capture customer sentiment in real-time to drive adaptive shopping experiences.</li>
        <li>🧘 <strong>Mental Wellness Platforms</strong> – Enable non-intrusive emotional monitoring for therapists and coaches.</li>
        <li>🤖 <strong>Conversational Interfaces</strong> – Infuse chatbots and voice agents with emotional awareness for improved empathy and trust.</li>
    </ul>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<h3>🚀 Why ASDP AI Stands Out</h3>", unsafe_allow_html=True)
    st.markdown(
        """
    <ul>
        <li>• Developed using industry-standard frameworks: <strong>TensorFlow</strong>, <strong>Keras</strong>, and <strong>OpenCV</strong></li>
        <li>• Delivers real-time performance via <strong>Haar Cascade face detection</strong> and optimized CNNs</li>
        <li>• Deployed via <strong>Streamlit</strong> for rapid development, ease of use, and cross-platform compatibility</li>
        <li>• <strong>Cloud-independent</strong> design ensures offline use with no data leakage risks</li>
    </ul>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <h3 style="text-align:center; margin-top: 2em;">🌐 Get Started Now</h3>
    <p style="text-align:center;">
        Explore our <strong>📸 Camera</strong> module for live emotion analysis<br>
        or try the <strong>🖼️ Upload Image</strong> feature to test single image inference.
    </p>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


# Camera Page
def camera_page():
    st.subheader("📷 Real-Time Emotion Detection")
    col1, col2 = st.columns(2)
    if col1.button("▶️ Start Camera"):
        st.session_state.camera_running = True
    if col2.button("⏹️ Stop Camera"):
        st.session_state.camera_running = False

    FRAME_WINDOW = st.empty()
    camera = cv2.VideoCapture(0)

    while st.session_state.camera_running:
        start_time = time.time()
        ret, frame = camera.read()
        if not ret:
            st.error("Camera error.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for x, y, w, h in faces:
            face_img = frame[y : y + h, x : x + w]
            emotion = detect_emotion(face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                frame,
                emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2,
            )
        fps = int(1 / (time.time() - start_time))
        cv2.putText(
            frame,
            f"FPS: {fps}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    camera.release()
    cv2.destroyAllWindows()


# Upload Page
def upload_page():
    st.subheader("🖼️ Upload Image for Emotion Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        img_array = np.array(img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        result_img = img_bgr.copy()
        results = []
        if len(faces) == 0:
            st.warning("⚠️ No face detected.")
        else:
            for idx, (x, y, w, h) in enumerate(faces):
                face_img = result_img[y : y + h, x : x + w]
                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray_face, (48, 48))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 48, 48, 1))
                prediction = model.predict(reshaped, verbose=0)[0]
                emotion = emotion_labels[np.argmax(prediction)]
                confidence = {
                    emotion_labels[i]: float(prediction[i])
                    for i in range(len(emotion_labels))
                }
                cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    result_img,
                    emotion,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36, 255, 12),
                    2,
                )
                results.append(
                    {"face_idx": idx + 1, "emotion": emotion, "confidence": confidence}
                )
            st.image(
                cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                caption="Analysis Result",
                use_container_width=True,
            )
            for result in results:
                st.markdown(
                    f"**Face {result['face_idx']}: Detected Emotion - _{result['emotion']}_**"
                )
                conf_df = pd.DataFrame(
                    {
                        "Emotion": list(result["confidence"].keys()),
                        "Confidence (%)": [
                            round(c * 100, 2) for c in result["confidence"].values()
                        ],
                    }
                )
                st.bar_chart(conf_df.set_index("Emotion"))


# About + Careers + Contact
def about_page():
    st.markdown(
        "<h1 style='text-align:center; color:#4A90E2;'>About This Application</h1>",
        unsafe_allow_html=True,
    )

    st.markdown("## ℹ️ Overview")
    st.markdown(
        """
        The **Emotion Detector App** is an advanced AI solution built with **Keras**, **OpenCV**, and **Streamlit**. It empowers users to recognize human emotions from facial expressions using real-time video or uploaded images. 
        Designed with privacy in mind, all emotion recognition is performed locally on your device.
    """
    )

    st.markdown("## 😃 Supported Emotions")
    st.markdown(
        "- 😠 Angry\n- 🤢 Disgust\n- 😨 Fear\n- 😊 Happy\n- 😢 Sad\n- 😲 Surprise\n- 😐 Neutral"
    )

    st.markdown("## 🧠 How It Works")
    st.markdown(
        """
    1. Faces are detected using OpenCV's Haar Cascade classifier.
    2. Images are converted to 48x48 grayscale input format.
    3. A deep CNN model trained on the FER-2013 dataset predicts the emotion.
    """
    )

    st.markdown("## 🔐 Privacy Commitment")
    st.markdown(
        "- All processing is local.\n- No data is sent or stored on external servers."
    )

    st.markdown("## 📞 Contact Us")
    team_members = [
        {
            "name": "Snehashis Das",
            "role": "Lead Developer & Project Architect",
            "phone": "+91-9330759496",
            "email": "snehashisdas842@gmail.com ",
            "image": "assets/snehashis.jpg",
        },
    ]

    cols = st.columns(4)
    for col, member in zip(cols, team_members):
        col.image(member["image"], use_container_width=True)
        col.markdown(
            f"**{member['name']}**  \n*{member['role']}*  \n📞 {member['phone']}  \n📧 {member['email']}"
        )

    st.markdown(
        "<p style='text-align:center; color:gray;'>© 2025 Emotion Detector App — All rights reserved.</p>",
        unsafe_allow_html=True,
    )


# App Entry
if not st.session_state.logged_in:
    login_page()
else:
    render_header()
    page = side_navigation()
    if page == "🏠 Home":
        home_page()
    elif page == "📸 Camera":
        camera_page()
    elif page == "🖼️ Upload Image":
        upload_page()
    elif page == "ℹ️ About":
        about_page()
    elif page == "🚪 Logout":
        st.session_state.logged_in = False
        st.session_state.camera_running = False
        st.success("🔓 Logged out successfully!")
