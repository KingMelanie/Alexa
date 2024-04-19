import sounddevice as sd
import numpy as np
from openai import OpenAI
import wave
import os

client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))

def record_audio(filename='output.wav', duration=10, fs=44100):
    """Records audio from the microphone and saves it as a WAV file."""
    print("Recording...")
    data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Recording finished.")
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(data.tobytes())


def transcribe_audio(filename='output.wav', model_name="whisper-1"):
    """Transcribes audio using OpenAI's Whisper model."""
    with open(filename, 'rb') as f:
        response = client.audio.transcriptions.create(file=f, model=model_name)
        return response.text


def text_to_speech(text, filename='output.mp3'):
    """Converts text to speech using OpenAI's TTS model."""
    response = client.audio.speech.create(
        model="tts-1",
        input={"text": text},
        voice="alloy",
    )
    response.stream_to_file(filename)

# Main execution
if __name__ == "__main__":
    record_audio(duration=5)  # Record 5 seconds of audio
    transcription = transcribe_audio()
    print("Transcription:", transcription)
    text_to_speech(transcription, filename='transcription.mp3')
