from flask import Flask, jsonify, render_template_string
import threading
import csv
import os
import time

app = Flask(__name__)
CSV_FILE = "emotion_log.csv"

def get_latest_emotion():
    if not os.path.exists(CSV_FILE):
        return "neutral", 0.0
    try:
        with open(CSV_FILE, "r") as f:
            rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                return last["emotion"], float(last["confidence"])
    except:
        pass
    return "neutral", 0.0

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Emotion-Aware Soft Bot</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #0a0a0a;
            font-family: 'Segoe UI', sans-serif;
            color: white;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        h1 {
            font-size: 1.8em;
            margin-bottom: 5px;
            color: #00ffcc;
            letter-spacing: 2px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 0.9em;
        }
        .main-container {
            display: flex;
            gap: 40px;
            align-items: flex-start;
            flex-wrap: wrap;
            justify-content: center;
        }
        .bot-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
        }
        canvas {
            border-radius: 20px;
            background: #111;
            border: 2px solid #333;
        }
        .emotion-badge {
            font-size: 2em;
            font-weight: bold;
            padding: 10px 30px;
            border-radius: 50px;
            background: #111;
            border: 2px solid currentColor;
            letter-spacing: 3px;
            transition: all 0.5s ease;
        }
        .response-text {
            font-size: 1.1em;
            color: #aaa;
            text-align: center;
            max-width: 300px;
            min-height: 30px;
            transition: all 0.5s;
        }
        .action-text {
            font-size: 0.95em;
            color: #ffcc00;
            letter-spacing: 1px;
        }
        .stats-panel {
            background: #111;
            border: 1px solid #333;
            border-radius: 15px;
            padding: 20px;
            width: 280px;
        }
        .stats-panel h3 {
            color: #00ffcc;
            margin-bottom: 15px;
            font-size: 1em;
            letter-spacing: 1px;
        }
        .bar-row {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            gap: 8px;
        }
        .bar-label {
            width: 70px;
            font-size: 0.8em;
            color: #aaa;
            text-transform: capitalize;
        }
        .bar-track {
            flex: 1;
            background: #222;
            border-radius: 10px;
            height: 10px;
            overflow: hidden;
        }
        .bar-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.8s ease;
        }
        .bar-pct {
            width: 35px;
            font-size: 0.75em;
            color: #666;
            text-align: right;
        }
        .confidence-meter {
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid #333;
        }
        .conf-label {
            font-size: 0.8em;
            color: #666;
            margin-bottom: 6px;
        }
        .conf-bar {
            background: #222;
            border-radius: 10px;
            height: 14px;
            overflow: hidden;
        }
        .conf-fill {
            height: 100%;
            border-radius: 10px;
            background: linear-gradient(90deg, #00ffcc, #00aaff);
            transition: width 0.8s ease;
        }
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #00ff00;
            margin-right: 6px;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        .live-label {
            font-size: 0.75em;
            color: #666;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h1>🤖 EMOTION-AWARE SOFT BOT</h1>
    <p class="subtitle"><span class="status-dot"></span>LIVE — Multimodal Emotion Detection</p>

    <div class="main-container">
        <div class="bot-container">
            <canvas id="botCanvas" width="300" height="350"></canvas>
            <div class="emotion-badge" id="emotionBadge">NEUTRAL</div>
            <div class="response-text" id="responseText">Ready to assist you 🤖</div>
            <div class="action-text" id="actionText">BOT: Standing by 🤖</div>
        </div>

        <div class="stats-panel">
            <h3>📊 EMOTION CONFIDENCE</h3>
            <div id="barsContainer"></div>
            <div class="confidence-meter">
                <div class="conf-label">Overall Confidence</div>
                <div class="conf-bar">
                    <div class="conf-fill" id="confFill" style="width:0%"></div>
                </div>
                <div style="font-size:0.8em; color:#aaa; margin-top:4px;" id="confText">0%</div>
            </div>
            <p class="live-label">⏱ Updates every second from live session</p>
        </div>
    </div>

<script>
const emotionConfig = {
    happy:    { color: "#00ff00", response: "You look happy! 😊",         action: "BOT: Dancing 🕺",        eyeShape: "happy",   mouthShape: "smile"  },
    sad:      { color: "#6464ff", response: "Are you okay? 😢",            action: "BOT: Comforting 🤗",     eyeShape: "sad",     mouthShape: "frown"  },
    angry:    { color: "#ff3333", response: "Take a deep breath... 😤",    action: "BOT: Backing away 🚶",   eyeShape: "angry",   mouthShape: "flat"   },
    neutral:  { color: "#cccccc", response: "Ready to assist you 🤖",      action: "BOT: Standing by 🤖",    eyeShape: "normal",  mouthShape: "flat"   },
    surprise: { color: "#00ffff", response: "Wow, surprised! 😲",          action: "BOT: Looking around 👀", eyeShape: "wide",    mouthShape: "open"   },
    fear:     { color: "#aa00ff", response: "Don't worry, I'm here 😨",    action: "BOT: Moving closer 🤝",  eyeShape: "wide",    mouthShape: "open"   },
    disgust:  { color: "#ff8800", response: "Something wrong? 🤢",         action: "BOT: Pausing ⏸️",        eyeShape: "squint",  mouthShape: "frown"  },
};

const emotionColors = {
    happy: "#00ff00", sad: "#6464ff", angry: "#ff3333",
    neutral: "#cccccc", surprise: "#00ffff", fear: "#aa00ff", disgust: "#ff8800"
};

let currentEmotion = "neutral";
let animFrame = 0;
let bounceY = 0;
let bounceDir = 1;

const canvas = document.getElementById("botCanvas");
const ctx = canvas.getContext("2d");

function drawBot(emotion) {
    const cfg = emotionConfig[emotion] || emotionConfig.neutral;
    const color = cfg.color;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Bounce animation
    bounceY += bounceDir * (emotion === "happy" ? 1.2 : 0.4);
    if (Math.abs(bounceY) > (emotion === "happy" ? 10 : 3)) bounceDir *= -1;

    const cx = 150, cy = 140 + bounceY;

    // Glow
    const grd = ctx.createRadialGradient(cx, cy, 10, cx, cy, 120);
    grd.addColorStop(0, color + "22");
    grd.addColorStop(1, "transparent");
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Body
    ctx.fillStyle = "#1a1a1a";
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(cx - 55, cy + 85, 110, 120, 15);
    ctx.fill(); ctx.stroke();

    // Arms
    if (emotion === "happy") {
        // Arms up
        ctx.beginPath();
        ctx.moveTo(cx - 55, cy + 100);
        ctx.lineTo(cx - 90, cy + 60);
        ctx.strokeStyle = color; ctx.lineWidth = 8;
        ctx.lineCap = "round"; ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cx + 55, cy + 100);
        ctx.lineTo(cx + 90, cy + 60);
        ctx.stroke();
    } else if (emotion === "sad") {
        // Arms down
        ctx.beginPath();
        ctx.moveTo(cx - 55, cy + 110);
        ctx.lineTo(cx - 85, cy + 160);
        ctx.strokeStyle = color; ctx.lineWidth = 8;
        ctx.lineCap = "round"; ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cx + 55, cy + 110);
        ctx.lineTo(cx + 85, cy + 160);
        ctx.stroke();
    } else {
        // Arms neutral
        ctx.beginPath();
        ctx.moveTo(cx - 55, cy + 110);
        ctx.lineTo(cx - 88, cy + 130);
        ctx.strokeStyle = color; ctx.lineWidth = 8;
        ctx.lineCap = "round"; ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cx + 55, cy + 110);
        ctx.lineTo(cx + 88, cy + 130);
        ctx.stroke();
    }

    // Legs
    ctx.strokeStyle = color; ctx.lineWidth = 10;
    ctx.beginPath(); ctx.moveTo(cx - 25, cy + 205); ctx.lineTo(cx - 25, cy + 260); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx + 25, cy + 205); ctx.lineTo(cx + 25, cy + 260); ctx.stroke();

    // Neck
    ctx.fillStyle = "#222";
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(cx - 15, cy + 68, 30, 20, 5);
    ctx.fill(); ctx.stroke();

    // Head
    ctx.fillStyle = "#1a1a1a";
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.roundRect(cx - 70, cy - 70, 140, 140, 20);
    ctx.fill(); ctx.stroke();

    // Antenna
    ctx.strokeStyle = color; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(cx, cy - 70); ctx.lineTo(cx, cy - 95); ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy - 100, 7, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill();

    // Eyes
    const eyeY = cy - 20;
    const eyeShape = cfg.eyeShape;

    if (eyeShape === "happy") {
        // Happy arc eyes
        ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.fillStyle = "transparent";
        ctx.beginPath(); ctx.arc(cx - 25, eyeY, 15, Math.PI, 0); ctx.stroke();
        ctx.beginPath(); ctx.arc(cx + 25, eyeY, 15, Math.PI, 0); ctx.stroke();
    } else if (eyeShape === "sad") {
        ctx.strokeStyle = color; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(cx - 25, eyeY + 5, 15, 0, Math.PI); ctx.stroke();
        ctx.beginPath(); ctx.arc(cx + 25, eyeY + 5, 15, 0, Math.PI); ctx.stroke();
    } else if (eyeShape === "angry") {
        ctx.fillStyle = color;
        ctx.beginPath(); ctx.ellipse(cx - 25, eyeY, 14, 8, -0.3, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.ellipse(cx + 25, eyeY, 14, 8, 0.3, 0, Math.PI * 2); ctx.fill();
        // Angry brows
        ctx.strokeStyle = color; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(cx - 40, eyeY - 18); ctx.lineTo(cx - 12, eyeY - 10); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(cx + 40, eyeY - 18); ctx.lineTo(cx + 12, eyeY - 10); ctx.stroke();
    } else if (eyeShape === "wide") {
        ctx.fillStyle = color;
        ctx.beginPath(); ctx.arc(cx - 25, eyeY, 16, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(cx + 25, eyeY, 16, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#000";
        ctx.beginPath(); ctx.arc(cx - 25, eyeY, 7, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(cx + 25, eyeY, 7, 0, Math.PI * 2); ctx.fill();
    } else if (eyeShape === "squint") {
        ctx.fillStyle = color;
        ctx.beginPath(); ctx.ellipse(cx - 25, eyeY, 14, 5, 0, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.ellipse(cx + 25, eyeY, 14, 5, 0, 0, Math.PI * 2); ctx.fill();
    } else {
        ctx.fillStyle = color;
        ctx.beginPath(); ctx.arc(cx - 25, eyeY, 13, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(cx + 25, eyeY, 13, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#000";
        ctx.beginPath(); ctx.arc(cx - 25, eyeY, 6, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(cx + 25, eyeY, 6, 0, Math.PI * 2); ctx.fill();
    }

    // Mouth
    const mouthY = cy + 30;
    ctx.strokeStyle = color; ctx.lineWidth = 3;
    if (cfg.mouthShape === "smile") {
        ctx.beginPath(); ctx.arc(cx, mouthY, 25, 0, Math.PI); ctx.stroke();
    } else if (cfg.mouthShape === "frown") {
        ctx.beginPath(); ctx.arc(cx, mouthY + 20, 25, Math.PI, 0); ctx.stroke();
    } else if (cfg.mouthShape === "open") {
        ctx.fillStyle = color + "88";
        ctx.beginPath(); ctx.ellipse(cx, mouthY + 5, 18, 12, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    } else {
        ctx.beginPath(); ctx.moveTo(cx - 22, mouthY + 8); ctx.lineTo(cx + 22, mouthY + 8); ctx.stroke();
    }

    animFrame++;
}

function updateBars(emotion) {
    const container = document.getElementById("barsContainer");
    const allEmotions = Object.keys(emotionConfig);
    container.innerHTML = allEmotions.map(e => {
        const isActive = e === emotion;
        const pct = isActive ? Math.floor(70 + Math.random() * 25) : Math.floor(Math.random() * 20);
        return `
        <div class="bar-row">
            <span class="bar-label">${e}</span>
            <div class="bar-track">
                <div class="bar-fill" style="width:${pct}%; background:${emotionColors[e]}"></div>
            </div>
            <span class="bar-pct">${pct}%</span>
        </div>`;
    }).join("");
}

async function fetchEmotion() {
    try {
        const res = await fetch("/emotion");
        const data = await res.json();
        const emotion = data.emotion || "neutral";
        const confidence = data.confidence || 0;

        if (emotion !== currentEmotion) {
            currentEmotion = emotion;
            const cfg = emotionConfig[emotion] || emotionConfig.neutral;

            document.getElementById("emotionBadge").textContent = emotion.toUpperCase();
            document.getElementById("emotionBadge").style.color = cfg.color;
            document.getElementById("emotionBadge").style.borderColor = cfg.color;
            document.getElementById("responseText").textContent = cfg.response;
            document.getElementById("actionText").textContent = cfg.action;
            updateBars(emotion);
        }

        const confPct = Math.min(100, Math.round(confidence));
        document.getElementById("confFill").style.width = confPct + "%";
        document.getElementById("confText").textContent = confPct + "%";

    } catch(e) {}
}

function loop() {
    drawBot(currentEmotion);
    requestAnimationFrame(loop);
}

// Init
updateBars("neutral");
loop();
setInterval(fetchEmotion, 1000);
</script>
</body>
</html>
'''

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/emotion")
def emotion():
    e, c = get_latest_emotion()
    return jsonify({"emotion": e, "confidence": c})

if __name__ == "__main__":
    print("🌐 Bot server running at http://localhost:5000")
    app.run(debug=False, port=5000)