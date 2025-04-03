// Static/js/app.js

document.addEventListener('DOMContentLoaded', function () {
    // Theme toggle functionality
    const themeToggle = document.getElementById('theme-toggle');

    // Check for saved theme preference or default to light
    if (localStorage.getItem('theme') === 'dark' ||
        (!localStorage.getItem('theme') && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        document.documentElement.classList.add('dark');
    } else {
        document.documentElement.classList.remove('dark');
    }

    themeToggle.addEventListener('click', function () {
        document.documentElement.classList.toggle('dark');
        if (document.documentElement.classList.contains('dark')) {
            localStorage.setItem('theme', 'dark');
        } else {
            localStorage.setItem('theme', 'light');
        }
    });

    // File Upload Handling
    const uploadForm = document.getElementById('upload-form');
    const imageUpload = document.getElementById('image-upload');
    const uploadedImage = document.getElementById('uploaded-image');
    const uploadResult = document.getElementById('upload-result');
    const uploadEmotion = document.getElementById('upload-emotion');

    uploadForm.addEventListener('submit', function (e) {
        e.preventDefault();

        if (!imageUpload.files[0]) {
            alert('Please select an image to upload');
            return;
        }

        const formData = new FormData();
        formData.append('image', imageUpload.files[0]);

        // Show loading state
        uploadEmotion.textContent = 'Processing...';
        uploadResult.classList.remove('hidden');
        uploadResult.classList.add('animate-fade-in');

        // Display the uploaded image
        const reader = new FileReader();
        reader.onload = function (e) {
            uploadedImage.src = e.target.result;
        };
        reader.readAsDataURL(imageUpload.files[0]);

        // Send to backend for processing
        fetch('/process-image', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                uploadEmotion.textContent = data.emotion;
                updateEmotionCount(data.emotion);
            })
            .catch(error => {
                console.error('Error:', error);
                uploadEmotion.textContent = 'Error';
            });
    });

    // Webcam Functionality
    const startWebcamBtn = document.getElementById('start-webcam');
    const stopWebcamBtn = document.getElementById('stop-webcam');
    const webcam = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const webcamEmotion = document.getElementById('webcam-emotion');

    let streaming = false;
    let stream = null;
    let interval = null;

    startWebcamBtn.addEventListener('click', function () {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(function (videoStream) {
                    stream = videoStream;
                    webcam.srcObject = stream;
                    webcam.classList.remove('hidden');
                    canvas.classList.remove('hidden');
                    webcamEmotion.classList.remove('hidden');
                    webcamEmotion.textContent = 'Initializing...';

                    startWebcamBtn.disabled = true;
                    stopWebcamBtn.disabled = false;

                    streaming = true;

                    // Start real-time emotion detection
                    interval = setInterval(captureAndDetect, 1000); // Process every second
                })
                .catch(function (error) {
                    console.error('Error accessing webcam:', error);
                    alert('Could not access webcam. Please check permissions.');
                });
        } else {
            alert('Your browser does not support webcam access');
        }
    });

    stopWebcamBtn.addEventListener('click', function () {
        if (streaming && stream) {
            clearInterval(interval);

            stream.getTracks().forEach(function (track) {
                track.stop();
            });

            webcam.srcObject = null;
            webcam.classList.add('hidden');
            canvas.classList.add('hidden');
            webcamEmotion.classList.add('hidden');

            startWebcamBtn.disabled = false;
            stopWebcamBtn.disabled = true;

            streaming = false;
        }
    });

    function captureAndDetect() {
        if (streaming) {
            // Draw current webcam frame to canvas
            const context = canvas.getContext('2d');
            canvas.width = webcam.videoWidth;
            canvas.height = webcam.videoHeight;
            context.drawImage(webcam, 0, 0, canvas.width, canvas.height);

            // Convert canvas to blob and send to server
            canvas.toBlob(function (blob) {
                const formData = new FormData();
                formData.append('image', blob, 'webcam-capture.jpg');

                fetch('/process-image', {
                    method: 'POST',
                    body: formData
                })
                    .then(response => response.json())
                    .then(data => {
                        webcamEmotion.textContent = data.emotion;
                        updateEmotionCount(data.emotion);

                        // Highlight the detected emotion in the stats
                        highlightEmotion(data.emotion);
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        webcamEmotion.textContent = 'Error';
                    });
            }, 'image/jpeg');
        }
    }

    // Emotion statistics tracking
    const emotionCounts = {
        'Happy': 0,
        'Sad': 0,
        'Angry': 0,
        'Surprise': 0,
        'Fear': 0,
        'Disgust': 0,
        'Neutral': 0
    };

    function updateEmotionCount(emotion) {
        if (emotion in emotionCounts) {
            emotionCounts[emotion]++;
            document.getElementById(`${emotion.toLowerCase()}-count`).textContent = emotionCounts[emotion];
        }
    }

    function highlightEmotion(emotion) {
        // Remove previous highlights
        document.querySelectorAll('.emotion-active').forEach(el => {
            el.classList.remove('emotion-active');
        });

        // Add highlight to current emotion
        const emotionElement = document.getElementById(`${emotion.toLowerCase()}-count`).parentElement;
        if (emotionElement) {
            emotionElement.classList.add('emotion-active');
        }
    }
});