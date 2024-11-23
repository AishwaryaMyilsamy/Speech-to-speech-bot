from faster_whisper import WhisperModel
from TTS.api import TTS

whisper_model = WhisperModel("base", device="cuda")

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

def transcribe_audio(audio_path):
  
    segments, _ = whisper_model.transcribe(audio_path)
    transcription = " ".join(segment.text for segment in segments)
    return transcription

def synthesize_speech(text, output_path="response.wav"):
    
    tts.tts_to_file(text=text, file_path=output_path)
    return output_path
