from flask import Flask, request, jsonify, render_template
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from transformers import pipeline
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
AUDIO_FOLDER = 'audio'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

def extract_audio(video_path, audio_output_path):
    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_output_path)
        return True, f"Audio extracted and saved to {audio_output_path}"
    except Exception as e:
        return False, str(e)

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(audio_path)
    with audio_file as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return True, text
    except sr.UnknownValueError:
        return False, "Google Speech Recognition could not understand the audio"
    except sr.RequestError as e:
        return False, f"Could not request results from Google Speech Recognition service; {e}"

def detect_hate_speech(text):
    classifier = pipeline("text-classification", model="unitary/toxic-bert")
    results = classifier(text)
    return results

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({"error": "No selected file"}), 400

    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    audio_output_path = os.path.join(AUDIO_FOLDER, os.path.splitext(os.path.basename(video_path))[0] + '.wav')
    success, message = extract_audio(video_path, audio_output_path)

    if not success:
        return jsonify({"error": message}), 500

    success, text = transcribe_audio(audio_output_path)
    if not success:
        return jsonify({"error": text}), 500

    results = detect_hate_speech(text)

    # Print results to the console
    print("Transcription: ", text)
    print("Hate Speech Detection Results: ", results)

    # Check if any label is "toxic" with a score above a certain threshold
    hate_speech_detected = any(result['label'] == 'toxic' and result['score'] > 0.5 for result in results)
    if hate_speech_detected:
        hate_speech_status = "Hate speech detected."
    else:
        hate_speech_status = "No hate speech detected."

    return render_template('result.html', transcription=text, hate_speech_status=hate_speech_status, results=results)

if __name__ == '__main__':
    app.run(debug=True)
