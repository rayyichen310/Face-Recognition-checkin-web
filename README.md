
# Face-Recognition-checkin-web
A web application demonstrating real-time face detection and recognition using Python, Flask, OpenCV, MediaPipe, and ONNX (ArcFace). Features include live webcam recognition, user registration with face data, check-in logging, and an admin panel

# [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/rayyichen310/Face-Recognition-checkin-web)


---

## 📦 Installation & Setup Guide

### 1️⃣ Install Required Packages

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git
```

### 2️⃣ Create and Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Clone the Repository

```bash
git clone https://github.com/rayyichen310/Face-Recognition-checkin-web.git
cd Face-Recognition-checkin-web
```

### 4️⃣ Install Git LFS and Pull Large Files

```bash
sudo apt install git-lfs
git lfs install
git lfs pull
```

### 5️⃣ Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 6️⃣ Generate HTTPS Certificate (for development)

```bash
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
```

> ⚠️ This creates a self-signed certificate. For production, use a trusted CA like Let’s Encrypt.

---



## 🚀 Start the Server

### Using Gunicorn + Gevent (Recommended for Production)

```bash
gunicorn app:app \
  --bind 0.0.0.0:5000 \
  --worker-class gevent \
  --workers 2 \
  --certfile=cert.pem \
  --keyfile=key.pem
```
