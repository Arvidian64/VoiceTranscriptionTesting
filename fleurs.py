import os
from datasets import Audio, load_dataset


def load_and_filter_dataset(shorter_than: int, longer_than: int, language_code):
    dataset = load_dataset("google/fleurs", language_code, split="train")
    dataset = dataset.cast_column("audio", Audio(decode=False))

    # Sample rate definierar sekunder
    sample_rate = 16000
    max_short_samples = shorter_than * sample_rate
    min_long_samples = longer_than * sample_rate

    #Filtrerar datan till två subset
    short_clips = dataset.filter(lambda x: x["num_samples"] <= max_short_samples)
    long_clips = dataset.filter(lambda x: x["num_samples"] >= min_long_samples)

    return short_clips, long_clips



def main():
    # You need an external huggingface token unless you want to be rate limited
    with open("tokens.txt") as f:
        token = f.read().strip()

    os.environ["HF_TOKEN"] = token

    shorter_than = 7
    longer_than = 14

    languages = [
    "en_us",        # English
    "sv_se",        # Swedish
    "hi_in",        # Hindi
    "es_419",       # Spanish
    "ar_eg",        # Arabic
    "fr_fr",        # French
    "bn_in",        # Bengali
    "pt_br",        # Portuguese
    "id_id",        # Indonesian
    "ur_pk",        # Urdu
    ]

    for language in languages:
        short_clips, long_clips = load_and_filter_dataset(shorter_than, longer_than, language_code=language)
        print("For language code: ", language)
        print(f"<= {shorter_than}s clips: {len(short_clips)}")
        print(f">= {longer_than}s clips: {len(long_clips)}")

if __name__ == "__main__":
    main()