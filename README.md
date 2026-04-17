# 🤖 Emotion-Aware Human-System Interaction

Real-time multimodal emotion detection system with intelligent soft bot.
Built for HCL IIT Hackathon 2026.

## Run
```bash
python3 app.py          # webcam detection
python3 bot_server.py   # web bot
python3 dashboard.py    # live dashboard
```

## Tech Stack
- DeepFace + OpenCV (face detection)
- Librosa + SoundDevice (voice detection)
- Flask + HTML5 Canvas (web bot)
- TensorFlow/Keras (custom FER-2013 model)
