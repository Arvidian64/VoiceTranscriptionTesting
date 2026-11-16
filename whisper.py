from faster_whisper import WhisperModel
import jiwer
import io
import os
import site


def register_cuda_dlls():
    # Add pip-installed NVIDIA cuBLAS/cuDNN bin folders to the Windows DLL search path.
    # Install using command: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
    for sp in site.getsitepackages():
        for lib in ("cublas", "cudnn", "cuda_runtime"):
            bin_dir = os.path.join(sp, "nvidia", lib, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]

register_cuda_dlls()

text_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])

def evaluate_model_against_subset(model: WhisperModel, subset) -> float:
    if len(subset) == 0:
        return float("nan")

    references, hypotheses = [], []

    for x, sample in enumerate(subset):

        audio_input = (
            io.BytesIO(sample["audio"]["bytes"])
            if sample["audio"].get("bytes") is not None
            else sample["audio"]["path"]
        )


        segments, _ = model.transcribe(audio_input)
        predicted_text = " ".join(seg.text.strip() for seg in segments)

        ref_norm = text_transform(sample["transcription"])
        hyp_norm = text_transform(predicted_text)

        # prevents ZeroDivisionError when reference becomes empty
        if ref_norm.strip():
            references.append(ref_norm)
            hypotheses.append(hyp_norm)
        else:
            print("Empty reference")

        if (x+1) % 20 == 0:
            print(f"finished with {x+1} comparison sets")

            print(references)
            print(hypotheses)

    return jiwer.wer(references, hypotheses)

def main():
    import fleurs

    # You need an external huggingface token unless you want to be rate limited
    with open("tokens.txt") as f:
        token = f.read().strip()

    fleurs.os.environ["HF_TOKEN"] = token

    short_clips, long_clips = fleurs.load_and_filter_dataset(7,14,"en_us")

    # stt = sound-to-text
    stt_model = WhisperModel(model_size_or_path="tiny", device="cuda", compute_type="int8")

    wer_score = evaluate_model_against_subset(stt_model, short_clips)

    print(wer_score)

if __name__ == "__main__":
    main()
