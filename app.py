import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import re
import time
import pandas as pd
import base64
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASDP AI – Emotion Detector",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Orbitron:wght@600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 100%);
    }
    [data-testid="stSidebar"] * { color: #e0e6f0 !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: #8fa3bf !important; font-size: 12px; }

    /* Banner */
    .asdp-banner {
        width: 100%;
        background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
        padding: 14px 28px;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        border-bottom: 3px solid #1e90ff;
        border-radius: 0 0 14px 14px;
        margin-bottom: 24px;
    }
    .asdp-banner h1 {
        color: #ffffff;
        font-size: 24px;
        font-family: 'Orbitron', sans-serif;
        margin: 0;
        animation: glow 2.5s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { text-shadow: 0 0 8px #1e90ff, 0 0 18px #1e90ff; }
        to   { text-shadow: 0 0 18px #00bfff, 0 0 30px #00bfff; }
    }

    /* Cards */
    .card {
        background: #ffffff;
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin-bottom: 18px;
        border: 1px solid #e8edf3;
    }
    .card-dark {
        background: #1b2838;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 18px;
        border: 1px solid #2a3a4a;
    }

    /* Emotion badge */
    .emotion-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 15px;
        letter-spacing: 0.5px;
    }

    /* Confidence bar */
    .conf-row { margin-bottom: 8px; }
    .conf-label { font-size: 13px; font-weight: 600; color: #334155; margin-bottom: 2px; }
    .conf-bar-bg { background: #e8edf3; border-radius: 999px; height: 10px; overflow: hidden; }
    .conf-bar-fill { height: 10px; border-radius: 999px; transition: width 0.4s ease; }

    /* Login */
    .login-wrap {
        max-width: 420px;
        margin: 48px auto;
        background: rgba(255,255,255,0.96);
        border-radius: 18px;
        padding: 40px 36px;
        box-shadow: 0 8px 40px rgba(0,0,0,0.15);
    }
    .login-wrap h2 { text-align: center; color: #0d1b2a; font-family: 'Orbitron', sans-serif; font-size: 20px; }

    /* Stat tiles */
    .stat-tile {
        text-align: center;
        background: linear-gradient(135deg, #1e3a5f, #2c5364);
        border-radius: 12px;
        padding: 18px 10px;
        color: white;
    }
    .stat-tile .val { font-size: 32px; font-weight: 700; font-family: 'Orbitron', sans-serif; }
    .stat-tile .lbl { font-size: 12px; opacity: 0.75; margin-top: 4px; }

    /* History table */
    .hist-row {
        display: flex; align-items: center; gap: 12px;
        padding: 8px 14px;
        border-radius: 8px;
        background: #f8fafc;
        margin-bottom: 6px;
        border: 1px solid #e8edf3;
    }
    .hist-time { font-size: 11px; color: #94a3b8; min-width: 70px; }
    .hist-face { font-size: 12px; color: #64748b; min-width: 60px; }

    /* Footer */
    .footer { text-align: center; color: #94a3b8; font-size: 12px; margin-top: 40px; padding-top: 16px; border-top: 1px solid #e8edf3; }

    /* Hide streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────────
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOTION_EMOJI = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😊",
    "Sad": "😢",
    "Surprise": "😲",
    "Neutral": "😐",
}
EMOTION_COLOR = {
    "Angry": "#ef4444",
    "Disgust": "#a855f7",
    "Fear": "#f97316",
    "Happy": "#22c55e",
    "Sad": "#3b82f6",
    "Surprise": "#eab308",
    "Neutral": "#94a3b8",
}

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in {
    "logged_in": False,
    "camera_running": False,
    "history": [],  # list of {time, face, emotion, confidence}
    "username": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Model & detector ───────────────────────────────────────────────────────────
@st.cache_resource
def load_emotion_model():
    return load_model("emotion_recognition_model.keras")


@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


model = load_emotion_model()
face_cascade = load_face_cascade()


# ── Core detection helper ──────────────────────────────────────────────────────
def predict_emotion(face_bgr):
    """Return (top_emotion, confidence_dict) for a single face crop (BGR)."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    preds = model.predict(reshaped, verbose=0)[0]
    top = EMOTION_LABELS[np.argmax(preds)]
    conf = {EMOTION_LABELS[i]: float(preds[i]) for i in range(len(EMOTION_LABELS))}
    return top, conf


def detect_faces(img_bgr, scale=1.1, neighbors=4):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scale, neighbors)
    return faces


# ── UI helpers ─────────────────────────────────────────────────────────────────
def render_header():
    st.markdown(
        "<div class='asdp-banner'><h1>ASDP AI 2025 – Face Emotion Detector</h1></div>",
        unsafe_allow_html=True,
    )


def emotion_badge(emotion: str) -> str:
    color = EMOTION_COLOR.get(emotion, "#94a3b8")
    emoji = EMOTION_EMOJI.get(emotion, "")
    return (
        f"<span class='emotion-badge' "
        f"style='background:{color}22; color:{color}; border:1.5px solid {color};'>"
        f"{emoji} {emotion}</span>"
    )


def confidence_bars(conf_dict: dict):
    sorted_items = sorted(conf_dict.items(), key=lambda x: -x[1])
    html = ""
    for em, val in sorted_items:
        pct = round(val * 100, 1)
        color = EMOTION_COLOR.get(em, "#94a3b8")
        emoji = EMOTION_EMOJI.get(em, "")
        html += (
            f"<div class='conf-row'>"
            f"<div class='conf-label'>{emoji} {em} <span style='float:right;color:{color};font-weight:700'>{pct}%</span></div>"
            f"<div class='conf-bar-bg'><div class='conf-bar-fill' style='width:{pct}%;background:{color};'></div></div>"
            f"</div>"
        )
    st.markdown(html, unsafe_allow_html=True)


def set_background(image_path: str):
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""<style>
            [data-testid="stApp"] {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover; background-attachment: fixed; background-position: center;
            }}
            </style>""",
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        pass  # background image optional


# ── Pages ──────────────────────────────────────────────────────────────────────


def login_page():
    set_background("assets/abc.jpg")
    render_header()

    st.markdown("<div class='login-wrap'>", unsafe_allow_html=True)
    st.markdown("<h2>🔐 Sign In</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("login_form"):
        email = st.text_input("📧 Email address")
        password = st.text_input("🔑 Password", type="password")
        submit = st.form_submit_button("Login →", use_container_width=True)

        if submit:
            if not email or not password:
                st.error("Please fill in both fields.")
            elif not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email):
                st.error("Enter a valid email address.")
            elif len(password) < 8 or not re.search(
                r"[!@#$%^&*(),.?\":{}|<>]", password
            ):
                st.error("Password must be 8+ characters and include a special symbol.")
            else:
                st.session_state.logged_in = True
                st.session_state.username = email.split("@")[0]
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def sidebar_nav():
    with st.sidebar:
        st.markdown(
            f"<div style='text-align:center;padding:18px 0 12px'>"
            f"<div style='font-size:42px'>👤</div>"
            f"<div style='font-weight:600;font-size:15px;margin-top:6px'>{st.session_state.username}</div>"
            f"<div style='font-size:11px;opacity:0.5;margin-top:2px'>Logged in</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        page = st.selectbox(
            "Navigate",
            [
                "🏠 Home",
                "📸 Camera",
                "🖼️ Upload Image",
                "📋 History",
                "ℹ️ About",
                "🚪 Logout",
            ],
            label_visibility="collapsed",
        )

        # Quick stats in sidebar
        if st.session_state.history:
            st.divider()
            emotions_seen = [h["emotion"] for h in st.session_state.history]
            from collections import Counter

            top = Counter(emotions_seen).most_common(1)[0]
            st.markdown(
                f"<div style='font-size:11px;opacity:0.6;margin-bottom:6px'>SESSION STATS</div>"
                f"<div style='font-size:13px'>🔍 Analyses: <b>{len(st.session_state.history)}</b></div>"
                f"<div style='font-size:13px;margin-top:4px'>🏆 Top emotion: <b>{EMOTION_EMOJI.get(top[0],'')} {top[0]}</b></div>",
                unsafe_allow_html=True,
            )

    return page


def home_page():
    c1, c2, c3 = st.columns(3)
    tiles = [
        ("📸", "Live Camera", "Real-time emotion detection via webcam"),
        ("🖼️", "Image Upload", "Batch analysis of uploaded photos"),
        ("📋", "History Log", "Review all past detection results"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], tiles):
        col.markdown(
            f"<div class='stat-tile'><div style='font-size:32px'>{icon}</div>"
            f"<div style='font-weight:700;font-size:16px;margin:8px 0 4px'>{title}</div>"
            f"<div style='font-size:12px;opacity:0.7'>{desc}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Platform Capabilities")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            "<div class='card'>"
            "<b>📷 Real-Time Detection</b><br><span style='color:#64748b;font-size:14px'>"
            "Analyze live facial emotions instantly via your webcam with FPS counter.</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='card'>"
            "<b>🔐 On-Device Processing</b><br><span style='color:#64748b;font-size:14px'>"
            "Privacy-first — all inference runs locally, nothing leaves your machine.</span></div>",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            "<div class='card'>"
            "<b>📊 Confidence Breakdown</b><br><span style='color:#64748b;font-size:14px'>"
            "See probability scores across all 7 emotion categories per detected face.</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='card'>"
            "<b>📋 Session History</b><br><span style='color:#64748b;font-size:14px'>"
            "Every detection is logged with timestamp, face index, and confidence data.</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### 💼 Industry Applications")
    apps = [
        ("🎓", "EdTech", "Monitor student engagement during online learning."),
        ("🛒", "Retail", "Capture customer sentiment at point of experience."),
        ("🧘", "Wellness", "Non-intrusive monitoring for therapists & coaches."),
        ("🤖", "Chatbots", "Infuse conversational agents with emotional awareness."),
    ]
    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, apps):
        col.markdown(
            f"<div class='card' style='text-align:center'>"
            f"<div style='font-size:28px'>{icon}</div>"
            f"<div style='font-weight:700;margin:8px 0 4px'>{title}</div>"
            f"<div style='color:#64748b;font-size:13px'>{desc}</div></div>",
            unsafe_allow_html=True,
        )


def camera_page():
    st.subheader("📸 Real-Time Emotion Detection")

    info_col, ctrl_col = st.columns([3, 1])
    with info_col:
        st.markdown(
            "<div class='card' style='padding:12px 16px'>"
            "<span style='color:#64748b;font-size:13px'>"
            "Point your webcam at a face and click <b>Start</b>. "
            "Press <b>Stop</b> or navigate away to release the camera.</span></div>",
            unsafe_allow_html=True,
        )
    with ctrl_col:
        start = st.button("▶ Start", use_container_width=True, type="primary")
        stop = st.button("⏹ Stop", use_container_width=True)

    if start:
        st.session_state.camera_running = True
    if stop:
        st.session_state.camera_running = False

    status_box = st.empty()
    frame_box = st.empty()
    emotion_box = st.empty()

    if not st.session_state.camera_running:
        status_box.info("Camera is stopped. Press ▶ Start to begin.")
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        st.error(
            "❌ Could not access webcam. Check permissions or try a different device index."
        )
        st.session_state.camera_running = False
        return

    try:
        while st.session_state.camera_running:
            t0 = time.time()
            ret, frame = camera.read()
            if not ret:
                st.error("❌ Camera read failed.")
                break

            faces = detect_faces(frame)
            detected_info = []

            for x, y, w, h in faces:
                face_crop = frame[y : y + h, x : x + w]
                emotion, conf = predict_emotion(face_crop)
                color = tuple(
                    int(c * 255)
                    for c in [
                        int(EMOTION_COLOR[emotion][1:3], 16) / 255,
                        int(EMOTION_COLOR[emotion][3:5], 16) / 255,
                        int(EMOTION_COLOR[emotion][5:7], 16) / 255,
                    ]
                )[
                    ::-1
                ]  # hex → BGR
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{EMOTION_EMOJI.get(emotion, '')} {emotion}"
                cv2.putText(
                    frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                )
                detected_info.append((emotion, conf))

                # Log to history
                st.session_state.history.append(
                    {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "face": f"Face {len(st.session_state.history)+1}",
                        "emotion": emotion,
                        "confidence": conf,
                        "source": "Camera",
                    }
                )

            fps = max(1, int(1 / (time.time() - t0 + 1e-9)))
            cv2.putText(
                frame,
                f"FPS: {fps}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Faces: {len(faces)}",
                (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            frame_box.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True
            )

            if detected_info:
                em, conf = detected_info[0]
                emotion_box.markdown(
                    f"<div class='card' style='text-align:center;padding:14px'>"
                    f"<div style='font-size:13px;color:#64748b;margin-bottom:6px'>Detected emotion</div>"
                    f"{emotion_badge(em)}</div>",
                    unsafe_allow_html=True,
                )
            else:
                emotion_box.markdown(
                    "<div class='card' style='text-align:center;padding:14px;color:#94a3b8'>"
                    "No face detected</div>",
                    unsafe_allow_html=True,
                )

            time.sleep(0.03)  # ~30 fps cap to reduce CPU load

    finally:
        camera.release()
        cv2.destroyAllWindows()
        status_box.info("Camera stopped.")


def upload_page():
    st.subheader("🖼️ Upload Image for Emotion Detection")
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"]
    )

    if not uploaded_file:
        st.markdown(
            "<div class='card' style='text-align:center;padding:40px;color:#94a3b8'>"
            "<div style='font-size:48px'>🖼️</div>"
            "<div style='margin-top:12px'>Upload an image to begin analysis</div></div>",
            unsafe_allow_html=True,
        )
        return

    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    faces = detect_faces(img_bgr)

    if len(faces) == 0:
        st.warning(
            "⚠️ No faces detected in this image. Try a clearer or higher-resolution photo."
        )
        col_prev, _ = st.columns([1, 1])
        col_prev.image(img, caption="Uploaded image", use_container_width=True)
        return

    result_img = img_bgr.copy()
    results = []

    for idx, (x, y, w, h) in enumerate(faces):
        face_crop = result_img[y : y + h, x : x + w]
        emotion, conf = predict_emotion(face_crop)
        hex_col = EMOTION_COLOR.get(emotion, "#1e90ff")
        bgr_color = (
            int(hex_col[5:7], 16),
            int(hex_col[3:5], 16),
            int(hex_col[1:3], 16),
        )
        cv2.rectangle(result_img, (x, y), (x + w, y + h), bgr_color, 3)
        label = f"{EMOTION_EMOJI.get(emotion,'')} {emotion}"
        cv2.putText(
            result_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, bgr_color, 2
        )
        cv2.putText(
            result_img,
            f"#{idx+1}",
            (x + 4, y + h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            bgr_color,
            2,
        )
        results.append({"idx": idx + 1, "emotion": emotion, "confidence": conf})

        st.session_state.history.append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "face": f"Face {idx+1}",
                "emotion": emotion,
                "confidence": conf,
                "source": uploaded_file.name,
            }
        )

    # Show annotated image
    img_col, info_col = st.columns([1, 1])
    with img_col:
        st.image(
            cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
            caption=f"Detected {len(faces)} face(s)",
            use_container_width=True,
        )

    with info_col:
        st.markdown(
            f"<div class='card'>"
            f"<b style='font-size:15px'>📊 Detection Summary</b><br>"
            f"<span style='color:#64748b;font-size:13px'>{len(faces)} face(s) found</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        for r in results:
            with st.expander(
                f"Face #{r['idx']} — {EMOTION_EMOJI.get(r['emotion'],'')} {r['emotion']}",
                expanded=(len(results) == 1),
            ):
                confidence_bars(r["confidence"])

    # If multiple faces, show comparison table
    if len(results) > 1:
        st.markdown("#### 📋 Multi-Face Summary")
        df = pd.DataFrame(
            [
                {
                    "Face": f"#{r['idx']}",
                    "Emotion": r["emotion"],
                    "Confidence (%)": round(r["confidence"][r["emotion"]] * 100, 1),
                }
                for r in results
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)


def history_page():
    st.subheader("📋 Detection History")

    if not st.session_state.history:
        st.markdown(
            "<div class='card' style='text-align:center;padding:40px;color:#94a3b8'>"
            "<div style='font-size:40px'>📋</div>"
            "<div style='margin-top:12px'>No detections yet — use Camera or Upload Image to get started.</div></div>",
            unsafe_allow_html=True,
        )
        return

    col1, col2, col3 = st.columns(3)
    from collections import Counter

    emotions_seen = [h["emotion"] for h in st.session_state.history]
    top_em = Counter(emotions_seen).most_common(1)[0]

    col1.markdown(
        f"<div class='stat-tile'><div class='val'>{len(st.session_state.history)}</div><div class='lbl'>TOTAL DETECTIONS</div></div>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"<div class='stat-tile'><div class='val'>{EMOTION_EMOJI.get(top_em[0],'')}</div><div class='lbl'>TOP EMOTION: {top_em[0].upper()}</div></div>",
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"<div class='stat-tile'><div class='val'>{len(set(emotions_seen))}</div><div class='lbl'>UNIQUE EMOTIONS</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Emotion distribution chart
    dist_df = pd.DataFrame(Counter(emotions_seen).items(), columns=["Emotion", "Count"])
    dist_df = dist_df.sort_values("Count", ascending=False)
    st.bar_chart(dist_df.set_index("Emotion"), height=200)

    st.markdown("#### Recent Detections")
    for entry in reversed(st.session_state.history[-50:]):
        em = entry["emotion"]
        color = EMOTION_COLOR.get(em, "#94a3b8")
        emoji = EMOTION_EMOJI.get(em, "")
        conf = round(entry["confidence"][em] * 100, 1)
        st.markdown(
            f"<div class='hist-row'>"
            f"<span class='hist-time'>{entry['time']}</span>"
            f"<span class='hist-face'>{entry['face']}</span>"
            f"<span class='emotion-badge' style='background:{color}22;color:{color};border:1.5px solid {color};font-size:12px;padding:3px 10px'>{emoji} {em}</span>"
            f"<span style='font-size:12px;color:#94a3b8;margin-left:auto'>{conf}% · {entry['source']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.history = []
        st.rerun()


def about_page():
    st.subheader("ℹ️ About This App")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            "<div class='card'>"
            "<b style='font-size:15px'>🧠 How It Works</b><br><br>"
            "<span style='color:#64748b;font-size:14px;line-height:1.8'>"
            "1. Faces are detected using OpenCV's Haar Cascade classifier.<br>"
            "2. Each face is converted to 48×48 grayscale format.<br>"
            "3. A deep CNN trained on FER-2013 predicts across 7 emotions.<br>"
            "4. Confidence scores are shown for every detected face."
            "</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='card'>"
            "<b style='font-size:15px'>🔐 Privacy</b><br><br>"
            "<span style='color:#64748b;font-size:14px'>"
            "All processing happens entirely on your device. No images or data are "
            "transmitted to any external server."
            "</span></div>",
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            "<div class='card'>"
            "<b style='font-size:15px'>😃 Supported Emotions</b><br><br>"
            + "".join(
                f"<span style='display:inline-block;margin:3px 4px;"
                f"background:{EMOTION_COLOR[em]}22;color:{EMOTION_COLOR[em]};"
                f"border:1.5px solid {EMOTION_COLOR[em]};border-radius:999px;"
                f"padding:3px 12px;font-size:13px;font-weight:600'>"
                f"{EMOTION_EMOJI[em]} {em}</span>"
                for em in EMOTION_LABELS
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='card'>"
            "<b style='font-size:15px'>⚙️ Tech Stack</b><br><br>"
            "<span style='color:#64748b;font-size:14px;line-height:1.8'>"
            "TensorFlow · Keras · OpenCV · Streamlit · NumPy · Pillow"
            "</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### 👤 Developer")
    dev_col, _ = st.columns([1, 2])
    with dev_col:
        st.markdown(
            "<div class='card' style='text-align:center'>"
            "<div style='font-size:48px'>👨‍💻</div>"
            "<div style='font-weight:700;margin-top:8px'>Snehashis Das</div>"
            "<div style='color:#64748b;font-size:13px'>Lead Developer & Project Architect</div>"
            "<div style='margin-top:10px;font-size:13px'>📞 +91-9330759496</div>"
            "<div style='font-size:13px'>📧 snehashisdas842@gmail.com</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='footer'>© 2025 ASDP AI – Emotion Detector · All rights reserved.</div>",
        unsafe_allow_html=True,
    )


# ── App entry ──────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    login_page()
else:
    render_header()
    page = sidebar_nav()

    if page == "🏠 Home":
        home_page()
    elif page == "📸 Camera":
        camera_page()
    elif page == "🖼️ Upload Image":
        upload_page()
    elif page == "📋 History":
        history_page()
    elif page == "ℹ️ About":
        about_page()
    elif page == "🚪 Logout":
        st.session_state.logged_in = False
        st.session_state.camera_running = False
        st.session_state.history = []
        st.session_state.username = ""
        st.rerun()
