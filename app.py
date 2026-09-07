from flask import Flask, render_template, request
import os
from transformers import pipeline
import torch

torch.set_num_threads(1)

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==============================
# AI EMOTION MODEL
# ==============================

print("Loading emotion model...")

emotion_classifier = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er"
)

print("Emotion model loaded successfully!")


# ==============================
# HOME PAGE
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# ANALYSIS PAGE
# ==============================

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")


# ==============================
# AUDIO UPLOAD + EMOTION ANALYSIS
# ==============================

@app.route("/upload", methods=["POST"])
def upload():

    # Check audio file
    if "audio" not in request.files:
        return "No audio file selected"

    audio = request.files["audio"]

    # Check filename
    if audio.filename == "":
        return "No audio file selected"

    # Save audio
    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        audio.filename
    )

    audio.save(file_path)

    print("Audio saved:", file_path)


    # ==============================
    # AI PREDICTION
    # ==============================

    try:
        results = emotion_classifier(file_path)

    except Exception as e:
        return f"Error analyzing audio: {str(e)}"


    # ==============================
    # EMOTION NAME
    # ==============================

    emotion_names = {
        "neu": "Neutral",
        "hap": "Happy",
        "ang": "Angry",
        "sad": "Sad"
    }


    # ==============================
    # CREATE EMOTION RESULTS
    # ==============================

    emotion_html = ""

    for result in results:

        label = result["label"]
        score = result["score"] * 100

        emotion = emotion_names.get(
            label,
            label
        )

        emotion_html += f"""
        <div class="emotion-card">

            <div class="emotion-header">

                <span class="emotion-name">
                    {emotion}
                </span>

                <span class="emotion-score">
                    {score:.2f}%
                </span>

            </div>

            <div class="bar">

                <div
                    class="fill"
                    style="width: {score}%">
                </div>

            </div>

        </div>
        """


    # ==============================
    # TOP EMOTION
    # ==============================

    top_label = results[0]["label"]

    top_emotion = emotion_names.get(
        top_label,
        top_label
    )

    top_score = results[0]["score"] * 100


    # ==============================
    # RESULT PAGE
    # ==============================

    return f"""
<!DOCTYPE html>

<html>

<head>

    <title>Emotion Analysis</title>

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <style>

        * {{
            box-sizing: border-box;
        }}

        body {{

            margin: 0;

            padding: 20px;

            font-family: Arial, sans-serif;

            background:
                linear-gradient(
                    135deg,
                    #eef2ff,
                    #f5f3ff
                );

            min-height: 100vh;
        }}


        .container {{

            max-width: 700px;

            margin: 40px auto;
        }}


        .result-box {{

            background: white;

            padding: 35px;

            border-radius: 25px;

            box-shadow:
                0 15px 40px
                rgba(0,0,0,0.12);
        }}


        h1 {{

            text-align: center;

            color: #312e81;

            margin-bottom: 10px;
        }}


        .audio-name {{

            text-align: center;

            color: #64748b;

            margin-bottom: 30px;
        }}


        .main-result {{

            text-align: center;

            background: #eef2ff;

            padding: 30px;

            border-radius: 20px;

            margin-bottom: 30px;
        }}


        .main-emotion {{

            font-size: 42px;

            font-weight: bold;

            color: #4f46e5;
        }}


        .main-score {{

            font-size: 20px;

            color: #475569;

            margin-top: 10px;
        }}


        h2 {{

            color: #1e293b;

            margin-bottom: 20px;
        }}


        .emotion-card {{

            margin-bottom: 22px;
        }}


        .emotion-header {{

            display: flex;

            justify-content:
                space-between;

            margin-bottom: 8px;
        }}


        .emotion-name {{

            font-size: 18px;

            font-weight: bold;

            color: #1e293b;
        }}


        .emotion-score {{

            color: #475569;

            font-weight: bold;
        }}


        .bar {{

            width: 100%;

            height: 15px;

            background: #e2e8f0;

            border-radius: 20px;

            overflow: hidden;
        }}


        .fill {{

            height: 100%;

            background:
                linear-gradient(
                    90deg,
                    #6366f1,
                    #8b5cf6
                );

            border-radius: 20px;
        }}


        .back {{

            display: block;

            width: fit-content;

            margin: 35px auto 0;

            padding: 14px 25px;

            background: #4f46e5;

            color: white;

            text-decoration: none;

            border-radius: 12px;

            font-weight: bold;
        }}


        .back:hover {{

            background: #3730a3;
        }}


        @media (max-width: 600px) {{

            body {{
                padding: 12px;
            }}

            .container {{
                margin: 15px auto;
            }}

            .result-box {{
                padding: 22px;
            }}

            h1 {{
                font-size: 27px;
            }}

            .main-emotion {{
                font-size: 34px;
            }}

        }}

    </style>

</head>


<body>

    <div class="container">

        <div class="result-box">

            <h1>
                🎙️ Emotion Analysis
            </h1>


            <div class="audio-name">

                Audio:
                <strong>
                    {audio.filename}
                </strong>

            </div>


            <div class="main-result">

                <div class="main-emotion">

                    {top_emotion}

                </div>


                <div class="main-score">

                    Confidence:
                    {top_score:.2f}%

                </div>

            </div>


            <h2>
                Emotion Scores
            </h2>


            {emotion_html}


            <a
                href="/"
                class="back">

                🔄 Analyze Another Audio

            </a>

        </div>

    </div>

</body>

</html>
"""


# ==============================
# RUN FLASK WITH HTTPS
# ==============================

if __name__ == "__main__":
    import os
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
