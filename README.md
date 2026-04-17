# 🤖 Emotion-Aware Robot

A real-time multimodal emotion detection system that recognizes human emotions through **facial expressions** and **voice**, then brings a 2D robot avatar to life that reacts with matching animations and colors — all in the browser.

---

## ✨ Features

- 🎥 **Facial Emotion Recognition** — Analyzes live webcam frames using [DeepFace](https://github.com/serengil/deepface)
- 🎙️ **Audio Emotion Recognition** — Classifies speech emotion from the microphone using a fine-tuned [wav2vec2](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition) model
- 🔀 **Sensor Fusion** — Intelligently combines face and audio confidence scores to pick the most reliable emotion source
- 🤖 **Animated Robot Avatar** — A canvas-drawn robot character that reacts in real time with unique animations, colors, and messages per emotion
- 📈 **Emotion Timeline** — A live scrolling bar chart showing the history of detected emotions and confidence levels
- ⚡ **WebSocket Streaming** — Low-latency bidirectional communication between browser and server using FastAPI WebSockets

---

## 🧠 Supported Emotions

| Emotion   | Robot Color | Animation | Message           |
|-----------|-------------|-----------|-------------------|
| 😊 Happy   | `#00ff88`   | Wave      | Khush ho! 😊      |
| 😢 Sad     | `#4488ff`   | Droop     | Udaas ho 😢       |
| 😠 Angry   | `#ff4444`   | Retreat   | Gussa mat karo! 😠|
| 😨 Fear    | `#ff8800`   | Shake     | Daro mat! 😨      |
| 😲 Surprise| `#ffff00`   | Jump      | Surprised! 😲     |
| 🤢 Disgust | `#aa44ff`   | Turn      | Eww! 🤢           |
| 😐 Neutral | `#ffffff`   | Idle      | Neutral 😐        |

---

## 🛠️ Tech Stack

| Layer     | Technology                                              |
|-----------|---------------------------------------------------------|
| Backend   | Python, [FastAPI](https://fastapi.tiangolo.com/), WebSockets |
| Face AI   | [DeepFace](https://github.com/serengil/deepface), OpenCV, NumPy |
| Audio AI  | [HuggingFace Transformers](https://huggingface.co/), wav2vec2 |
| Frontend  | HTML5 Canvas, Vanilla JavaScript, WebSocket API         |

---

## 📁 Project Structure

```
Emotion_hack/
├── server.py      # FastAPI backend — WebSocket endpoint, emotion fusion logic
├── index.html     # Single-page frontend — robot canvas, webcam feed, timeline
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- A webcam and microphone

### 1. Clone the repository

```bash
git clone https://github.com/Mohit211205/Emotion_hack.git
cd Emotion_hack
```

### 2. Install dependencies

```bash
pip install fastapi uvicorn opencv-python-headless deepface transformers torch numpy
```

> **Note:** The audio model (`wav2vec2-lg-xlsr-en-speech-emotion-recognition`) will be downloaded automatically from HuggingFace on first run (~1 GB). Ensure you have a stable internet connection.

### 3. Start the server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 4. Open in your browser

Navigate to [http://localhost:8000](http://localhost:8000)

> ✅ Allow camera and microphone access when prompted.

---

## 🎮 How to Use

1. **Webcam feed** starts automatically — your face is analyzed every 1.5 seconds.
2. Click **🎤 Enable Microphone** to activate audio emotion detection.
3. Watch the **robot avatar** update its pose, color, and message in real time.
4. Check the **Emotion Timeline** at the bottom for a history of detected emotions.

---

## ⚙️ How It Works

```
Browser                          Server (FastAPI)
  │                                    │
  │── WebSocket frame + audio ────────►│
  │                                    ├─ DeepFace → face_emotion, face_conf
  │                                    ├─ wav2vec2 → audio_emotion, audio_conf
  │                                    ├─ fuse_emotions() → final_emotion
  │◄── emotion, behavior, log ─────────│
  │                                    │
  ├─ Robot canvas redraws
  ├─ Emotion badge updates
  └─ Timeline chart repaints
```

### Fusion Logic

| Condition                     | Result               |
|-------------------------------|----------------------|
| `face_conf > 60%`             | Use face emotion     |
| `audio_conf > 40%` (fallback) | Use audio emotion    |
| Neither threshold met         | Default to face      |

---

## 🖼️ UI Overview

```
┌─────────────────────┬──────────────────────────┐
│   Live Camera Feed  │    Robot Avatar           │
│   [video element]   │    [canvas animation]     │
│                     │    😊 happy               │
│   [🎤 Mic Button]   │    Face: 82%  Audio: 45%  │
├─────────────────────┴──────────────────────────┤
│              Emotion Timeline                   │
│   [█▓░░▓▓███░░▓▓▓▓▓▓░░░▓▓▓██████]              │
└─────────────────────────────────────────────────┘
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to open an [issue](https://github.com/Mohit211205/Emotion_hack/issues) or submit a pull request.

---

## 📄 License

This project is open source. Feel free to use, modify, and distribute it.
