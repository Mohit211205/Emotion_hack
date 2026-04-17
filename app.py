
# ─── Imports ──────────────────────────────────────
import cv2
from deepface import DeepFace
import time
import csv
import threading
from collections import deque
from voice_emotion import start_voice_detection, get_voice_emotion

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

# ─── Bot config ───────────────────────────────────
bot_responses = {
    "happy":    ("You look happy! 😊",        (0, 255, 0)),
    "sad":      ("Are you okay? 😢",          (100, 100, 255)),
    "angry":    ("Take a deep breath... 😤",  (0, 0, 255)),
    "neutral":  ("Ready to assist you 🤖",    (200, 200, 200)),
    "surprise": ("Wow, surprised! 😲",        (0, 255, 255)),
    "fear":     ("Don't worry, I'm here 😨",  (180, 0, 255)),
    "disgust":  ("Something wrong? 🤢",       (0, 165, 255)),
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
        writer.writerow([round(time.time() - start_time, 2), emotion, round(confidence, 2)])

# ─── Smoothing ────────────────────────────────────
def get_smoothed_emotion():
    if not emotion_history:
        return "neutral"
    return max(set(emotion_history), key=list(emotion_history).count)

# ─── Detection Thread ─────────────────────────────
def detect_emotion(frame):
    global current_emotion, current_region, emotion_scores
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
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

# ─── Main Loop ────────────────────────────────────
cap = cv2.VideoCapture(0)
print("✅ Starting... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # FPS
    now = time.time()
    fps = 1 / (now - fps_time + 0.001)
    fps_time = now

    # Run detection every N frames
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
        cv2.putText(frame, emotion.upper(), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ── Top info bar ──
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 115), (20, 20, 20), -1)
    cv2.putText(frame, f"Emotion: {emotion.upper()}", (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, response, (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, action, (15, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 2)

    # ── FPS ──
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    # ── Confidence bars ──
    if scores:
        y_pos = 140
        for emo, score in sorted(scores.items(), key=lambda x: -x[1]):
            bar_len = int(score * 1.8)
            bar_color = color if emo == emotion else (80, 80, 80)
            cv2.rectangle(frame, (350, y_pos), (350 + bar_len, y_pos + 14), bar_color, -1)
            cv2.putText(frame, f"{emo[:7]}: {score:.0f}%", (210, y_pos + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            y_pos += 22

    # ── Voice + Combined ──
    voice_emotion = get_voice_emotion()
    combined = emotion if voice_emotion == emotion else emotion
    cv2.putText(frame, f"Face: {emotion}  |  Voice: {voice_emotion}  |  Combined: {combined.upper()}",
                (15, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # ── Timer ──
    elapsed = int(time.time() - start_time)
    cv2.putText(frame, f"Time: {elapsed}s", (15, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # ── Show frame ──
    cv2.imshow("Emotion-Aware Bot", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Done! Log saved to: {CSV_FILE}")

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
    y_vals = [emotion_list.index(e) if e in emotion_list else 0 for e in emotions]

    plt.figure(figsize=(14, 4))
    plt.plot(times, y_vals, marker='o', color='cyan', linewidth=1.5, markersize=4)
    plt.yticks(range(len(emotion_list)), emotion_list)
    plt.xlabel("Time (seconds)")
    plt.title("Emotion Timeline")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("emotion_timeline.png")
    plt.show()
    print("📊 Timeline saved!")
except Exception as e:
    print(f"Timeline error: {e}")