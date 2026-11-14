import os
from datasets import Audio, load_dataset


def load_and_filter_dataset(longer_than: int, shorter_than: int):
    # You need an external huggingface token unless you want to be rate limited
    with open("tokens.txt") as f:
        token = f.read()

    os.environ["HF_TOKEN"] = token

    languages = ["en_us", "urd"]

    dataset = load_dataset("google/fleurs", "en_us", split="train")
    dataset = dataset.cast_column("audio", Audio(decode=False))

    # Använder sample rate för att definiera 5 respektive 25 sekunder
    sample_rate = 16000
    max_short_samples = shorter_than * sample_rate
    min_long_samples = longer_than * sample_rate

    print("num samples: ", dataset[0]["num_samples"])
    print(dataset.features)

    #Filtrerar datan till två subset
    short_clips = dataset.filter(lambda x: x["num_samples"] >= max_short_samples)
    long_clips = dataset.filter(lambda x: x["num_samples"] <= min_long_samples)

    short_clips = short_clips.cast_column("audio", Audio(decode=True))
    long_clips = long_clips.cast_column("audio", Audio(decode=True))

    return short_clips, long_clips



def main():
    shorter_than = 6
    longer_than = 18

    short_clips, long_clips = load_and_filter_dataset(shorter_than, longer_than)
    print(f"<= {shorter_than}s clips: {len(short_clips)}")
    print(f">= {longer_than}s clips: {len(long_clips)}")

if __name__ == "__main__":
    main()