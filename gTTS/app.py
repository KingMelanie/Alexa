import sounddevice as sd
import numpy as np
import whisper
from gtts import gTTS
import os

def record_audio(duration=10, fs=44100):
    """Records audio from the microphone."""
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Recording ended.")
    return recording.flatten()

def transcribe_audio(audio_data, model_name="tiny"):
    """Transcribes audio using OpenAI's Whisper model."""
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_data)
    return result['text']

def text_to_speech(text, lang='en', filename='output.mp3'):
    """Converts text to speech."""
    print("Converting text to speech...")
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(filename)
    print(f"Saved speech to {filename}")

# Main execution
if __name__ == "__main__":
    # Record audio for 5 seconds
    audio_data = record_audio(duration=5)

    # Transcribe the recorded audio
    transcription = transcribe_audio(audio_data, model_name="tiny")
    print("Transcription:", transcription)

    # Convert the transcription to speech and save it as an MP3
    text_to_speech(transcription, filename='transcription.mp3')
