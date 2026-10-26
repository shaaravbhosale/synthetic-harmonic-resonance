import csv
import glob
import os
from datetime import datetime

import librosa

from key_identifier import NOTES, find_best_key


def process_inputs_to_csv(
    inputs_dir: str = os.path.join("..", "inputs"),
    outputs_dir: str = os.path.join("..", "outputs"),
    csv_filename: str | None = None,
) -> str:
    """
    Batch process reads all .mp3 files in inputs_dir by invoking find_best_key twice per file
    (with and without a perfect fifth), and write results to a CSV in outputs_dir.

    CSV columns:
      File, PerfectFifth, BestKey, C, C#, D, D#, E, F, F#, G, G#, A, A#, B

    Returns the path to the generated CSV file.
    """
    os.makedirs(outputs_dir, exist_ok=True)

    if csv_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        csv_filename = f"key_identification_batch_{timestamp}.csv"

    csv_path = os.path.join(outputs_dir, csv_filename)

    mp3_paths = sorted(glob.glob(os.path.join(inputs_dir, "*.mp3")))

    header = [
        "File",
        "PerfectFifth",
        "BestKey",
    ] + NOTES

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        if not mp3_paths:
            # Still create an empty CSV with just the header
            return csv_path

        for audio_path in mp3_paths:
            try:
                y, sr = librosa.load(audio_path, sr=None, mono=True)
            except Exception as e:
                # Write an error line for traceability, scores left blank
                writer.writerow([
                    os.path.basename(audio_path),
                    "ERROR_LOADING",
                    str(e),
                ] + [""] * len(NOTES))
                continue

            for add_pf in (False, True):
                scores = find_best_key(
                    y=y,
                    sr=sr,
                    audio_path=audio_path,
                    add_perfect_fifth=add_pf,
                )
                # Determine best key
                if scores:
                    best_key = max(scores.items(), key=lambda kv: kv[1])[0]
                else:
                    best_key = ""

                row = [
                    os.path.basename(audio_path),
                    "True" if add_pf else "False",
                    best_key,
                ] + [scores.get(note, "") for note in NOTES]

                writer.writerow(row)

    return csv_path


if __name__ == "__main__":
    # Default relative paths assume running from the package directory
    csv_out = process_inputs_to_csv(
        inputs_dir=os.path.join("..", "inputs"),
        outputs_dir=os.path.join("..", "outputs"),
    )
    print(f"Batch processing complete. CSV written to: {csv_out}")
