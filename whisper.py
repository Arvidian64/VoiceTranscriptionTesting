from faster_whisper import WhisperModel

# stt = sound-to-text
stt_model = WhisperModel(model_size_or_path="large-v3", device="gpu", compute_type="int8")


def transcribe_audio(audio_path: str) -> str:
    segments, info = stt_model.transcribe(audio_path)
    return " ".join(segments.text.strip() for segments in segments)


def main():
    pass

if __name__ == "__main__":
    main()
