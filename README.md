# 😄 ASDP AI – Face Emotion Detector

ASDP AI is a real-time facial emotion detection web application developed using **Streamlit**, **OpenCV**, and a **custom CNN-based deep learning model** trained on a balanced **FER-2013** dataset. It provides an intuitive interface to recognize human emotions from both live webcam feed and uploaded images.

---

## 🚀 Features

- 🔐 **Secure Login System** – Basic user authentication
- 📸 **Real-Time Emotion Detection** – Detects facial emotions via webcam
- 🖼️ **Image Upload Analysis** – Analyze static photos for emotions
- 📊 **Confidence Score Visualization** – Displays probability for each emotion class
- 👥 **About Us Page** – Showcases developer profile and contact info

---

## 🧠 Model Overview

- **Architecture:** Custom 5-layer CNN
- **Dataset:** FER-2013 (Balanced: 5,000 images per class)
- **Accuracy:** ~70% on test set
- **Preprocessing:** Grayscale conversion, normalization, class rebalancing
- **Frameworks:** TensorFlow, Keras, OpenCV

---

## 📁 Project Structure

emotion-recognition-app/
│
├── app.py                         # Main Streamlit app
├── emotion_recognition_model.keras  # Trained CNN model
├── data balancing.ipynb          # Dataset preprocessing notebook
├── model Tranning.ipynb          # Model training notebook
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── assets/                       # UI images and emotion icons
└── .streamlit/                   # Streamlit configuration files


---

## ⚙️ Installation & Setup

To run this app locally:

1. **Clone the repository**

git clone git@github.com:snehashis842/emotion-recognition-app.git
cd emotion-recognition-app

2. **Install Dependencies**
pip install -r requirements.txt

3. **Launch the Application**
pip install -r requirements.txt

📸 Screenshots
🏠 Home Interface


📷 Webcam Detection


🖼️ Image Upload Result


🙋‍♂️ Developer Profile
👨‍💻 Snehashis Das
Lead Developer & Project Architect

📧 Email: snehashisdas842@gmail.com

📱 Phone: +91-9330759496

🌐 GitHub: @snehashis842

🔗 LinkedIn: linkedin.com/in/snehashisdas

🛠️ Tech Stack
Category	Tools & Technologies
Language	Python
Frontend	Streamlit, HTML, CSS
Computer Vision	OpenCV
Deep Learning	TensorFlow, Keras
Data Handling	NumPy, Pandas
Visualization	Matplotlib, Seaborn
Version Control	Git, GitHub

📈 Future Improvements
🌐 Deploy to Streamlit Cloud or Hugging Face Spaces

📊 Add session tracking & emotion analytics

📷 Expand dataset with user-submitted facial data

🧠 Improve model with advanced CNN layers & dropout

👨‍👩‍👧 Real-time multi-face detection & tracking

📄 License
This project is intended for academic and educational purposes only.
Please contact the developer for permission regarding reuse or contributions.

⭐️ Show Your Support
If you found this project helpful, please consider giving it a star ⭐ on GitHub!
