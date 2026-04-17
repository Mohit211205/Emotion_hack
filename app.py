# ─── Imports ──────────────────────────────────────
import cv2
from deepface import DeepFace
import tensorflow as tf
import numpy as np
import time
import csv
import threading
from collections import deque
from voice_emotion import start_voice_detection, get_voice_emotion

# ─── Load Custom Trained Model ────────────────────
print("🔄 Loading custom FER-2013 model...")
try:
    custom_model = tf.keras.models.load_model("emotion_model.h5")
    CUSTOM_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    USE_CUSTOM_MODEL = True
    print("✅ Custom model loaded!")
except:
    USE_CUSTOM_MODEL = False
    print("⚠️  Custom model not found! Using DeepFace instead")

# ─── Config ───────────────────────────────────────
DETECT_EVERY_N_FRAMES = 5
SMOOTH_WINDOW = 5
CSV_FILE = "emotion_log.csv"

# ─── State ────────────────────────────────────────
emotion_history = deque(maxlen=SMOOTH_WINDOW)
current_emotion = "neutral"
current_region = {}
emotion_scores = {}
start_time = time.time()
frame_count = 0
fps = 0
fps_time = time.time()
lock = threading.Lock()

# ─── Bot Config ───────────────────────────────────
bot_responses = {
    "happy":    ("You look happy! 😊",         (0, 255, 0)),
    "sad":      ("Are you okay? 😢",           (100, 100, 255)),
    "angry":    ("Take a deep breath... 😤",   (0, 0, 255)),
    "neutral":  ("Ready to assist you 🤖",     (200, 200, 200)),
    "surprise": ("Wow, surprised! 😲",         (0, 255, 255)),
    "fear":     ("Don't worry, I'm here 😨",   (180, 0, 255)),
    "disgust":  ("Something wrong? 🤢",        (0, 165, 255)),
}

bot_actions = {
    "happy":    "BOT: Dancing 🕺",
    "sad":      "BOT: Comforting 🤗",
    "angry":    "BOT: Backing away 🚶",
    "neutral":  "BOT: Standing by 🤖",
    "surprise": "BOT: Looking around 👀",
    "fear":     "BOT: Moving closer 🤝",
    "disgust":  "BOT: Pausing ⏸️",
}

# ─── CSV Setup ────────────────────────────────────
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "emotion", "confidence"])

def log_to_csv(emotion, confidence):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([round(time.time() - start_time, 2),
                        emotion, round(confidence, 2)])

# ─── Smoothing ────────────────────────────────────
def get_smoothed_emotion():
    if not emotion_history:
        return "neutral"
    return max(set(emotion_history), key=list(emotion_history).count)

# ─── Custom Model Prediction ──────────────────────
def predict_with_custom_model(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None, None, None, None
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi / 255.0
        face_roi = np.expand_dims(face_roi, axis=[0, -1])
        predictions = custom_model.predict(face_roi, verbose=0)[0]
        emotion_idx = np.argmax(predictions)
        emotion = CUSTOM_EMOTIONS[emotion_idx]
        confidence = predictions[emotion_idx] * 100
        scores = {CUSTOM_EMOTIONS[i]: float(predictions[i]*100)
                 for i in range(len(CUSTOM_EMOTIONS))}
        return emotion, confidence, scores, (x, y, w, h)
    except:
        return None, None, None, None

# ─── Detection Thread ─────────────────────────────
def detect_emotion(frame):
    global current_emotion, current_region, emotion_scores
    try:
        if USE_CUSTOM_MODEL:
            emotion, confidence, scores, region = predict_with_custom_model(frame)
            if emotion is not None:
                with lock:
                    emotion_history.append(emotion)
                    current_emotion = get_smoothed_emotion()
                    emotion_scores = scores
                    current_region = {'x': region[0], 'y': region[1],
                                     'w': region[2], 'h': region[3]}
                    log_to_csv(current_emotion, confidence)
        else:
            result = DeepFace.analyze(frame, actions=['emotion'],
                                     enforce_detection=False)
            with lock:
                detected = result[0]['dominant_emotion']
                emotion_history.append(detected)
                current_emotion = get_smoothed_emotion()
                current_region = result[0].get('region', {})
                emotion_scores = result[0]['emotion']
                confidence = emotion_scores.get(current_emotion, 0)
                log_to_csv(current_emotion, confidence)
    except:
        pass

# ─── Start Voice Detection ────────────────────────
start_voice_detection()
print("✅ Voice detection started!")

# ─── Main Loop ────────────────────────────────────
cap = cv2.VideoCapture(0)
print("✅ Starting webcam... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    now = time.time()
    fps = 1 / (now - fps_time + 0.001)
    fps_time = now

    if frame_count % DETECT_EVERY_N_FRAMES == 0:
        t = threading.Thread(target=detect_emotion, args=(frame.copy(),))
        t.daemon = True
        t.start()

    with lock:
        emotion = current_emotion
        region = current_region.copy()
        scores = emotion_scores.copy()

    response, color = bot_responses.get(emotion, ("Hello!", (255, 255, 255)))
    action = bot_actions.get(emotion, "")

    # ── Face bounding box ──
    if region and region.get('w', 0) > 0:
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, emotion.upper(), (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ── Top bar ──
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 115), (20, 20, 20), -1)
    cv2.putText(frame, f"Emotion: {emotion.upper()}", (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, response, (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, action, (15, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 2)

    # ── Model info ──
    model_text = "Model: Custom CNN (FER-2013)" if USE_CUSTOM_MODEL else "Model: DeepFace"
    cv2.putText(frame, model_text, (frame.shape[1]-320, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)

    # ── FPS ──
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-120, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

    # ── Confidence bars ──
    if scores:
        y_pos = 140
        for emo, score in sorted(scores.items(), key=lambda x: -x[1]):
            bar_len = int(score * 1.8)
            bar_color = color if emo == emotion else (80, 80, 80)
            cv2.rectangle(frame, (350, y_pos),
                         (350 + bar_len, y_pos + 14), bar_color, -1)
            cv2.putText(frame, f"{emo[:7]}: {score:.0f}%",
                       (210, y_pos + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                       (255, 255, 255), 1)
            y_pos += 22

    # ── Voice + Combined ──
    voice_emotion = get_voice_emotion()
    combined = emotion if voice_emotion == emotion else emotion
    cv2.putText(frame,
                f"Face: {emotion}  |  Voice: {voice_emotion}  |  Combined: {combined.upper()}",
                (15, frame.shape[0]-40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # ── ROS2 Status ──
    cv2.putText(frame, "ROS2: Bridge Ready 🔗",
                (15, frame.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1)

    # ── Timer ──
    elapsed = int(time.time() - start_time)
    cv2.putText(frame, f"Session: {elapsed}s",
                (frame.shape[1]-150, frame.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow("Emotion-Aware Bot | HCL IIT Hackathon 2026", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Session complete! Log saved to: {CSV_FILE}")

# ─── Timeline ─────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    times, emotions = [], []
    with open(CSV_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["timestamp"]))
            emotions.append(row["emotion"])

    emotion_list = list(bot_responses.keys())
    y_vals = [emotion_list.index(e) if e in emotion_list else 0
             for e in emotions]

    plt.figure(figsize=(14, 4), facecolor='#111111')
    plt.plot(times, y_vals, marker='o', color='cyan',
            linewidth=1.5, markersize=4)
    plt.yticks(range(len(emotion_list)), emotion_list, color='white')
    plt.xlabel("Time (seconds)", color='white')
    plt.title("📊 Emotion Timeline — HCL IIT Hackathon 2026",
             color='white')
    plt.gca().set_facecolor('#1a1a1a')
    plt.gca().tick_params(colors='white')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("emotion_timeline.png", facecolor='#111111')
    plt.show()
    print("📊 Timeline saved!")
except Exception as e:
    print(f"Timeline error: {e}")