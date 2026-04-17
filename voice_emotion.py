import sounddevice as sd
import numpy as np
import librosa
import threading
import time

# ─── Config ───────────────────────────────────────
SAMPLE_RATE = 22050
DURATION = 3  # seconds per audio chunk
current_voice_emotion = "neutral"
voice_lock = threading.Lock()

# ─── Simple rule-based voice emotion from audio features ──
def analyze_voice(audio_data):
    global current_voice_emotion
    try:
        audio = audio_data.flatten().astype(np.float32)

        # Extract features
        energy = np.mean(librosa.feature.rms(y=audio))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        tempo, _ = librosa.beat.beat_track(y=audio, sr=SAMPLE_RATE)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=SAMPLE_RATE)
        pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0

        # Rule-based classification
        if energy > 0.08 and pitch > 200:
            emotion = "angry"
        elif energy > 0.06 and tempo > 120:
            emotion = "happy"
        elif energy < 0.02 and pitch < 150:
            emotion = "sad"
        elif energy < 0.01:
            emotion = "neutral"
        else:
            emotion = "neutral"

        with voice_lock:
            current_voice_emotion = emotion

        print(f"🎤 Voice: {emotion} | Energy:{energy:.3f} Pitch:{pitch:.0f} Tempo:{float(tempo):.0f}")

    except Exception as e:
        print(f"Voice analysis error: {e}")

def record_and_analyze():
    while True:
        try:
            audio = sd.rec(int(DURATION * SAMPLE_RATE),
                          samplerate=SAMPLE_RATE,
                          channels=1, dtype='float32')
            sd.wait()
            t = threading.Thread(target=analyze_voice, args=(audio,))
            t.daemon = True
            t.start()
        except Exception as e:
            print(f"Recording error: {e}")
            time.sleep(1)

def start_voice_detection():
    t = threading.Thread(target=record_and_analyze)
    t.daemon = True
    t.start()
    print("🎤 Voice detection started!")

def get_voice_emotion():
    with voice_lock:
        return current_voice_emotion