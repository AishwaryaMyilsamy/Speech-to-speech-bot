from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from orchestrator import generate_response
from speech_processing import transcribe_audio, synthesize_speech
import os

app = Flask(__name__)
socketio = SocketIO(app)


is_active = False

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('audio')
def handle_audio(data):
    global is_active

    # Save the received audio data
    with open("input.wav", "wb") as f:
        f.write(data)

    # Transcribe the audio into text
    user_input = transcribe_audio("input.wav")
    print(f"User said: {user_input}") 

    if not is_active and "hi rachel" in user_input.lower():
        is_active = True
        emit('response', {'text': "Hi! I'm Rachel. How can I brighten your day?"})
        return

    if "bye rachel" in user_input.lower():
        is_active = False
        emit('response', {'text': "Goodbye! Have a great day!"})
        return

    if is_active:
        response = generate_response(user_input)
        audio_path = synthesize_speech(response)
        emit('response', {'text': response, 'audio': audio_path})

if __name__ == '__main__':
    socketio.run(app, debug=True)
