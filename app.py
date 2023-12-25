import atexit
from collections import Counter
import cv2
from flask import Flask, render_template, Response, send_from_directory
from deepface import DeepFace
import matplotlib.pyplot as plt
import numpy as np
import io

app = Flask(__name__)

# List to store detected emotions
emotions_list = []

# Emojis corresponding to each emotion
emotion_emojis = {
    'happy': '',
    'sad': '',
    'angry': '',
    'surprise': '',
    'fear': '',
    'neutral': '',
    'disgust': '',
}

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to release the webcam when the application exits
def release_webcam():
    cap.release()
    cv2.destroyAllWindows()

# Register the release_webcam function to be called at exit
atexit.register(release_webcam)

# Function to analyze frame and update emotions_list
def analyze_frame(frame):
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    emotion = result[0]['dominant_emotion']
    emotions_list.append(emotion)
    return emotion

# Function to generate video frames
def generate_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        emotion = analyze_frame(frame)

        # Display emotion information on the frame
        cv2.putText(frame, f'Emotion: {emotion}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Convert the frame to bytes
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Function to generate and save the emotion distribution chart
def generate_emotion_chart(emotion_counter):
    emotions = list(emotion_counter.keys())
    counts = list(emotion_counter.values())

    # Create a pie chart with emojis
    fig, ax = plt.subplots()
    ax.pie(counts, labels=emotions, autopct='%1.1f%%', startangle=90)

    # Place emojis on the pie chart
    for i, (emotion, count) in enumerate(zip(emotions, counts)):
        emoji_character = emotion_emojis.get(emotion, '❓')  # Default emoji if not found
        angle = 360 * sum(counts[:i]) / sum(counts)  # Calculate the angle for the current emotion
        x = 0.5 + 0.35 * np.cos(np.radians(angle))  # Place the emoji in a circular pattern
        y = 0.5 + 0.35 * np.sin(np.radians(angle))
        ax.text(x, y, emoji_character, fontsize=40, va='center', ha='center')

    # Save the chart in memory
    chart_image = io.BytesIO()
    plt.savefig(chart_image, format='png')
    plt.close()

    # Return the chart image as bytes
    return chart_image.getvalue()

# Route to serve the webcam feed with emotion information
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to serve the emotions distribution as a pie chart with emojis
@app.route('/emotions_chart')
def emotions_chart():
    emotion_counter = Counter(emotions_list)
    chart_image = generate_emotion_chart(emotion_counter)
    
    # Return the chart image as a response
    return Response(chart_image, mimetype='image/png')

# Route to display the webcam feed with emotion information and the emotions chart
@app.route('/')
def index():
    return render_template('index.html')

# Route to serve the static emotions chart image
@app.route('/static/<path:filename>')
def static_file(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
