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
├── app.py # Main Streamlit application
├── emotion_recognition_model.keras # Trained CNN model
├── data balancing.ipynb # Notebook for dataset rebalancing
├── model Tranning.ipynb # CNN training process
├── requirements.txt # Project dependencies
├── README.md # Project documentation
├── assets/ # UI images and emotion icons
└── .streamlit/ # Streamlit configuration files

---

## ⚙️ Installation & Setup

To run this app locally:

1. **Clone the repository**

```bash
git clone git@github.com:snehashis842/emotion-recognition-app.git
cd emotion-recognition-app

2. Install dependencies

pip install -r requirements.txt

3. Run the Streamlit app

streamlit run app.py

📸 Screenshots


Home Interface
![Webcam Demo](assets/home.png)


Webcam Detection Demo
![Webcam Demo](assets/webcam.png)


Image Upload Result

![Image Upload](assets/upload.png)


🙋‍♂️ Developer Profile
👨‍💻 Snehashis Das
Lead Developer & Project Architect

📧 Email: snehashisdas842@gmail.com

📱 Phone: +91-9330759496

🌐 GitHub: github.com/snehashis842

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
 Deploy to Streamlit Cloud for public access

 Add user session tracking & analytics

 Expand dataset with user-submitted faces

 Improve model accuracy with more layers & dropout

 Multi-face detection & emotion tracking in real-time

📄 License
This project is for academic and educational purposes only.
Please contact the author for reuse or extension.

⭐️ If you found this project helpful, please consider giving it a ⭐ on GitHub!


---

```
