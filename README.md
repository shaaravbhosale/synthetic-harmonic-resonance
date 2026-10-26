# Synthetic Harmonic Resonance

This project provides tools for analyzing, processing, and generating synthetic harmonic resonance from audio files. It includes key identification, audio conversion, and synthetic sound generation for musical analysis and experimentation.

## Features
- Key identification for audio files
- Conversion between audio formats (e.g., m4a to mp3)
- Generation of synthetic harmonic and flute sounds
- Batch processing of input audio files
- Output of processed audio and plots

## Directory Structure
- `inputs/` — Source audio files for processing
- `outputs/audio/perfect_fifth/` — Generated audio files with perfect fifth synthetic added
- `outputs/audio/no_perfect_fifth/` — Generated audio files without perfect fifth synthetic
- `outputs/plots/perfect_fifth/` — Plots and visualizations with perfect fifth synthetic
- `outputs/plots/no_perfect_fifth/` — Plots and visualizations without perfect fifth synthetic
- `outputs/synthetic-sounds/` — Synthetic sound outputs

- `synthetic_harmonic_resonance/` — Core Python modules:
  - `key_identifier.py` — Key detection logic
  - `m4a_to_mp3.py` — Audio format conversion
  - `synthetic_harmonic_generator.py` — Synthetic sound generation
  - `flute_110Hz.wav`, `harmonic_110Hz.wav` — Example synthetic sounds
- `requirements.txt` — Python dependencies

  
> **Note:** If the `outputs/` directory or its subdirectories (`audio/perfect_fifth`, `audio/no_perfect_fifth`, `plots/perfect_fifth`, `plots/no_perfect_fifth`, `synthetic-sounds`) do not exist, please create them before running the scripts to avoid errors.

## Installation
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- Place your audio files in the `inputs/` directory.
- Run the relevant Python scripts from the `synthetic_harmonic_resonance/` folder. For example:
  ```bash
  python synthetic_harmonic_resonance/key_identifier.py
  ```
- Generated outputs will appear in the `outputs/` folders, organized by whether perfect fifth synthetic is added or not.

### Batch Processing (inputs folder)
You can batch process all `.mp3` files placed in the `inputs/` folder and write a CSV summary to the `outputs/` directory. For each input file, two rows will be produced in the CSV: one with perfect fifth disabled and one with perfect fifth enabled.

- To run batch processing from the project root:
  ```bash
  python -m synthetic_harmonic_resonance.batch_processor
  ```
  This will create a CSV named like `key_identification_batch_YYYYMMDD-HHMMSS.csv` under `outputs/`.

- CSV columns are:
  - File: name of the input file
  - PerfectFifth: True/False
  - BestKey: the note with the highest score
  - Scores for each note in order: C, C#, D, D#, E, F, F#, G, G#, A, A#, B

Notes:
- Only `.mp3` files are processed.
- The script will also generate hybrid audio and chroma plots as part of `find_best_key` execution, placed in `outputs/audio/...` and `outputs/plots/...` respectively.

## Comparison with Other Approaches

| File                                  | KS       | FFT | Synthetic Harmonic overlay <BR> with perfect fifth | Synthetic Harmonic overlay <BR> without perfect fifth |
|---------------------------------------|----------|-----|----------------------------------------------------|-------------------------------------------------------|
| Aaj_ki_Raat.mp3                       | G minor  | C3  | F                                                  | F                                                     |
| Apni To Jaise Taise.mp3               | E minor  | G♯2 | G#                                                 | G#                                                    |
| Durge Durgat Bhari.mp3                | G# major | F♯1 | E                                                  | B                                                     |
| Ek pyar ka nagma hai.mp3              | F# minor | F♯4 | A                                                  | A                                                     |
| Hum Bane tum Bane Ek duje ke liye.mp3 | D major  | F2  | A#                                                 | A#                                                    |
| Hum_Mile_Na_Mile_Kisna.mp3            | B minor  | D♯1 | A                                                  | A                                                     |
| Hum_Tumhe_Chahate_Hai.mp3             | C major  | C4  | F                                                  | G                                                     |
| Kaise Muze Tum Mil Gaye.mp3           | D major  | A♯1 | A#                                                 | F                                                     |
| Kesariya.mp3                          | C major  | G1  | G                                                  | G                                                     |
| Krishna Theme OMG.mp3                 | B major  | C♯4 | C#                                                 | C#                                                    |
| Lambi Judai.mp3                       | C# major | E2  | B                                                  | B                                                     |
| Man Udhan Varyache.mp3                | E major  | B1  | G#                                                 | G#                                                    |
| Vaishnav-Jan-To-Shreya-Ghoshal.mp3    | B major  | F2  | C#                                                 | C#                                                    |

## Dependencies
See `requirements.txt` for a full list. Common dependencies may include:
- numpy
- scipy
- librosa
- pydub

## Contributing
Pull requests and suggestions are welcome. Please open an issue for major changes.

## License
This project is for educational and research purposes.

## Contact
For questions or feedback, please contact the repository owner.
