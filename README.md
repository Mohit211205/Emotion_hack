# 🤖 Emotion-Aware Human-System Interaction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?style=for-the-badge&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green?style=for-the-badge&logo=opencv)
![Flask](https://img.shields.io/badge/Flask-3.x-black?style=for-the-badge&logo=flask)
![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-teal?style=for-the-badge&logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

### 🏆 HCL IIT Hackathon 2026 — Problem Statement 9
**Real-time Multimodal Emotion Detection with Intelligent Soft Bot**

</div>

---

## 📌 Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Emotion-Bot Behavior Mapping](#emotion-bot-behavior-mapping)
- [Project Structure](#project-structure)
- [Evaluation Criteria](#evaluation-criteria)
- [Team](#team)

---

## 🎯 Overview

This project implements a **real-time emotion-aware human-system interaction** platform that:

- Detects human emotional states from **facial expressions** using Deep Learning
- Analyzes **voice/audio cues** for multimodal emotion recognition
- Controls an **animated soft bot** that adapts its behavior based on detected emotions
- Logs and visualizes **emotion timelines** during interaction
- Provides a **live web dashboard** for real-time monitoring

> Built as part of **HCL IIT Hackathon 2026** under Problem Statement 9: Emotion-Aware Human-System Interaction

---

## 🏗️ System Architecture

┌─────────────────────────────────────────────────────┐
│                    INPUT LAYER                       │
│         📷 Camera Input    🎤 Microphone Input       │
└──────────────┬──────────────────────┬───────────────┘
↓                      ↓
┌──────────────────────┐  ┌──────────────────────────┐
│  FACE DETECTION      │  │  VOICE DETECTION          │
│  DeepFace + OpenCV   │  │  Librosa + SoundDevice    │
│  Custom CNN (FER2013)│  │  Audio Feature Extraction │
└──────────┬───────────┘  └────────────┬─────────────┘
↓                           ↓
┌─────────────────────────────────────────────────────┐
│              EMOTION CLASSIFIER                      │
│     happy / sad / angry / neutral /                  │
│     surprise / fear / disgust                        │
│         + Multimodal Fusion Layer                    │
└─────────────────────┬───────────────────────────────┘
↓
┌─────────────────────────────────────────────────────┐
│                BEHAVIOR MAPPER                       │
│     Maps each emotion → unique bot behavior          │
└─────────────────────┬───────────────────────────────┘
↓
┌─────────────────────────────────────────────────────┐
│              SOFT BOT SIMULATOR                      │
│     Animated Web Bot (Flask + HTML5 Canvas)          │
│     Real-time behavior adaptation                    │
└─────────────────────┬───────────────────────────────┘
↓
┌─────────────────────────────────────────────────────┐
│         EMOTION TIMELINE LOGGER + VISUALIZER         │
│     CSV Logging + Matplotlib Dashboard               │
└─────────────────────────────────────────────────────┘


---

## ✨ Features

### Core Features
- 🎥 **Real-time Face Emotion Detection** — 30 FPS webcam processing with DeepFace
- 🧠 **Custom CNN Model** — Trained on FER-2013 dataset (28,000+ images)
- 🎤 **Voice Emotion Detection** — Real-time audio analysis using Librosa
- 🔀 **Multimodal Fusion** — Combines face + voice for accurate emotion prediction
- 🤖 **Animated Soft Bot** — Web-based humanoid bot with 7 unique behaviors
- 📊 **Live Dashboard** — Real-time emotion confidence bars and statistics
- 📈 **Emotion Timeline** — Visual graph of emotions over session
- 💾 **CSV Logging** — Every emotion logged with timestamp and confidence

### Technical Features
- ⚡ **Threaded Architecture** — Smooth performance with background processing
- 🔄 **Emotion Smoothing** — Anti-flicker using sliding window average
- 📦 **Face Bounding Box** — Visual face tracking with emotion label
- 🎯 **7 Emotion Classes** — happy, sad, angry, neutral, surprise, fear, disgust
- 📱 **Web Interface** — Accessible from any browser on local network

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Face Detection | DeepFace + OpenCV | Real-time facial emotion detection |
| Custom CNN Model | TensorFlow/Keras | FER-2013 trained emotion classifier (emotion_model.h5) |
| Transfer Learning | VGG16 + Keras | Alternative FER-2013 model (emotion_model_v2.h5) |
| Voice Detection (ML) | Librosa + scikit-learn RandomForest | Audio emotion with trained model (voice_model.pkl) |
| Voice Detection (Rule) | Librosa + SoundDevice | Rule-based audio fallback (voice_emotion2.py) |
| Voice Model Training | CREMA-D + RAVDESS + scikit-learn | Voice emotion model training pipeline |
| Web Bot UI (Flask) | Flask + HTML5 Canvas | Animated soft bot web interface (port 5000) |
| Web Bot UI (FastAPI) | FastAPI + WebSocket + HTML5 Canvas | Real-time multimodal bot via WebSocket |
| Audio Transformer | wav2vec2 (HuggingFace Transformers) | Deep learning audio emotion in server.py |
| Live Dashboard | Matplotlib | Real-time emotion visualization |
| Data Logging | Python CSV | Emotion timeline logging |
| Performance | Python Threading | Smooth real-time processing |
| Version Control | Git + GitHub | Team collaboration |

---

## 📊 Dataset

### FER-2013 (Facial Expression Recognition)
- **Source:** [Kaggle FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- **Size:** 35,887 grayscale images (48×48 pixels)
- **Classes:** 7 emotions
- **Split:** 28,709 training / 7,178 testing

### RAVDESS (Audio Emotions)
- **Source:** [Zenodo RAVDESS](https://zenodo.org/record/1188976)
- **Type:** Speech and song emotional audio recordings
- **Used for:** Voice emotion model training (`train_voice_model.py`)

### CREMA-D (Audio Emotions)
- **Source:** [GitHub CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
- **Type:** Crowd-sourced emotional multimodal actors dataset (WAV files)
- **Used for:** Voice emotion model training alongside RAVDESS

### Distribution
| Emotion | Train Images | Test Images |
|---------|-------------|-------------|
| Happy | 7,215 | 1,774 |
| Neutral | 4,965 | 1,233 |
| Sad | 4,830 | 1,247 |
| Angry | 3,995 | 958 |
| Surprise | 3,171 | 831 |
| Fear | 4,097 | 1,024 |
| Disgust | 436 | 111 |

---

## ⚙️ Installation

### Prerequisites
- Python 3.13+
- MacOS / Linux / Windows
- Webcam + Microphone
- 4GB RAM minimum (8GB+ recommended for VGG16 / wav2vec2)

### Step 1 — Clone repository
```bash
git clone https://github.com/Mohit211205/Emotion_hack.git
cd Emotion_hack
```

### Step 2 — Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux
# OR
.venv\Scripts\activate     # Windows
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Install system dependencies (Mac only)
```bash
brew install portaudio
pip install pyaudio
```

---

## ▶️ How to Run

### Option A — OpenCV Desktop App (Standalone)

#### Terminal 1 — Main Emotion Detector
```bash
source .venv/bin/activate
python3 app.py
```
> Webcam opens with real-time face + voice emotion detection and animated bot panel side-by-side

#### Terminal 2 — Live Dashboard (Optional)
```bash
source .venv/bin/activate
python3 dashboard.py
```
> Live matplotlib dashboard with emotion timeline, pie chart, and frequency graphs

---

### Option B — Flask Web Bot (Browser, port 5000)

```bash
source .venv/bin/activate
python3 bot_server.py
```
> Open browser at `http://localhost:5000`  
> Webcam frames are sent to Flask, DeepFace analyzes them, animated web bot responds

---

### Option C — FastAPI WebSocket Server (Real-time, browser)

```bash
source .venv/bin/activate
pip install fastapi uvicorn
uvicorn server:app --reload --port 8000
```
> Open browser at `http://localhost:8000`  
> Uses WebSocket for real-time face + audio emotion fusion with wav2vec2 transformer model

---

### Train Models (Optional)

#### Train face CNN (FER-2013)
```bash
python3 train_model.py
```
> Trains custom CNN on FER-2013 dataset; saves `emotion_model.h5` (~1–2 hrs on CPU)

#### Train face model v2 (VGG16 transfer learning)
```bash
python3 train_v2.py
```
> Fine-tunes VGG16 on FER-2013; saves `emotion_model_v2.h5`

#### Train voice emotion model (CREMA-D / RAVDESS)
```bash
python3 train_voice_model.py
```
> Trains Random Forest on audio features from `voice_dataset/`; saves `voice_model.pkl`

---

### Test Voice Detection (Optional)
```bash
python3 voice_test.py
```
> Starts microphone and prints live voice emotion every 3 seconds

---

## 🤖 Emotion-Bot Behavior Mapping

| Emotion | Confidence | Bot Response | Bot Action | Bot Color |
|---------|-----------|--------------|------------|-----------|
| 😊 Happy | High | "You look happy!" | Dancing 🕺 | Green |
| 😢 Sad | Medium | "Are you okay?" | Comforting 🤗 | Blue |
| 😤 Angry | High | "Take a deep breath" | Backing away 🚶 | Red |
| 😐 Neutral | — | "Ready to assist" | Standing by 🤖 | Grey |
| 😲 Surprise | High | "Wow, surprised!" | Looking around 👀 | Cyan |
| 😨 Fear | Medium | "Don't worry!" | Moving closer 🤝 | Purple |
| 🤢 Disgust | Low | "Something wrong?" | Pausing ⏸️ | Orange |

---

## 📁 Project Structure

```
Emotion_hack/
│
├── app.py                   # Main app — OpenCV webcam + face/voice emotion + animated bot panel
├── bot_server.py            # Flask web server (port 5000) — browser-based animated soft bot
├── server.py                # FastAPI WebSocket server — real-time face + audio multimodal fusion
├── index.html               # Web UI for server.py (live camera + robot response via WebSocket)
│
├── voice_emotion.py         # Voice emotion module — ML model (voice_model.pkl) + rule-based fallback
├── voice_emotion2.py        # Simpler rule-based voice emotion (no ML model required)
├── voice_test.py            # Standalone test script for voice emotion detection
│
├── dashboard.py             # Live matplotlib dashboard — timeline, pie chart, frequency bars
│
├── train_model.py           # CNN model training on FER-2013 → emotion_model.h5
├── train_v2.py              # VGG16 transfer learning on FER-2013 → emotion_model_v2.h5
├── train_voice_model.py     # Random Forest voice model on CREMA-D/RAVDESS → voice_model.pkl
│
├── _bot_simulator.py        # Bot simulator placeholder
│
├── emotion_model.h5         # Trained face CNN (generated by train_model.py)
├── emotion_model_v2.h5      # Trained VGG16 model (generated by train_v2.py)
├── voice_model.pkl          # Trained voice RandomForest (generated by train_voice_model.py)
├── emotion_log.csv          # Session emotion log (auto-generated at runtime)
├── emotion_timeline.png     # Timeline graph (auto-generated after app.py session)
├── training_results.png     # CNN training accuracy plot (generated by train_model.py)
├── training_v2_results.png  # VGG16 training plot (generated by train_v2.py)
│
├── dataset/                 # FER-2013 dataset (not in repo)
│   ├── train/
│   └── test/
│
├── voice_dataset/           # CREMA-D / RAVDESS audio dataset (not in repo)
│   └── AudioWAV/
│
├── requirements.txt         # Python dependencies
└── README.md                # This file
```


---

## 📋 Evaluation Criteria Coverage

| Criteria | Implementation | Status |
|----------|---------------|--------|
| Emotion Recognition Accuracy | Custom CNN on FER-2013 + DeepFace | ✅ |
| Behavior Adaptation Quality | 7 unique bot behaviors per emotion | ✅ |
| Real-time Performance | Threaded, 30FPS, 5-frame interval | ✅ |
| Multimodal Integration Bonus | Face + Voice fusion | ✅ Bonus |
| Demo & Presentation | Live web bot + dashboard | ✅ |
| Emotion Timeline Logging | CSV + Matplotlib graph | ✅ |
| Minimum 4 emotions | 7 emotions detected | ✅ |

---

## 👥 Team

| Name | Role | GitHub |
|------|------|--------|
| Mohit | Team Lead & Integration | [@Mohit211205](https://github.com/Mohit211205) |
| Janhavi | ML & Emotion Detection | [@janhavi-5002](https://github.com/janhavi-5002) |

> **HCL IIT Hackathon 2026** — Problem Statement 9: Emotion-Aware Human-System Interaction

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">
Built with ❤️ for HCL IIT Hackathon 2026
</div>
