import logging
import os
from typing import List
from scipy.spatial.distance import cosine

import librosa
import numpy as np
from matplotlib import pyplot as plt
import soundfile as sf

NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
HOP_LENGTH = 4096  # large hop for stable chroma over time; you can reduce for finer resolution
# octave to synthesize fundamentals (choose low so harmonics overlap song spectrum)
SYNTH_OCTAVE = 2   # you can try 2, 3, 4

N_HARMONICS = 6
SYNTH_GAIN = 0.8   # relative to mean amplitude of audio
CHUNK_SECONDS = 20
logging.basicConfig(format='%(asctime)s %(levelname)s [%(filename)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
AUDIO_PATH = "../inputs/Vaishnav-Jan-To-Shreya-Ghoshal.mp3"

CHORDS = {
    "A":   {"major": [0, 4, 7],  "minor": [0, 3, 7]},
    "A#":  {"major": [1, 5, 8],  "minor": [1, 4, 8]},
    "B":   {"major": [2, 6, 9],  "minor": [2, 5, 9]},
    "C":   {"major": [3, 7, 10], "minor": [3, 6, 10]},
    "C#":  {"major": [4, 8, 11], "minor": [4, 7, 11]},
    "D":   {"major": [5, 9, 0],  "minor": [5, 8, 0]},
    "D#":  {"major": [6, 10, 1], "minor": [6, 9, 1]},
    "E":   {"major": [7, 11, 2], "minor": [7, 10, 2]},
    "F":   {"major": [8, 0, 3],  "minor": [8, 11, 3]},
    "F#":  {"major": [9, 1, 4],  "minor": [9, 0, 4]},
    "G":   {"major": [10, 2, 5], "minor": [10, 1, 5]},
    "G#":  {"major": [11, 3, 6], "minor": [11, 2, 6]}
}


def plot_chroma_heatmaps(orig_chroma, hybrid_chroma, sr, hop_length, hybrid_note, file_name):
    # Plot and save original chroma
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    img1 = librosa.display.specshow(orig_chroma, y_axis='chroma', x_axis='time', sr=sr, hop_length=hop_length, ax=ax1)
    ax1.set_title("Original Song - Chroma CQT")
    plt.colorbar(img1, ax=ax1, location='right')
    plt.tight_layout()
    plt.savefig(f"../outputs/plots/{file_name}_original.png")
    plt.close(fig1)

    # Plot and save hybrid chroma
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    img2 = librosa.display.specshow(hybrid_chroma, y_axis='chroma', x_axis='time', sr=sr, hop_length=hop_length, ax=ax2)
    ax2.set_title(f"Hybrid with Sa = {hybrid_note} - Chroma CQT")
    plt.colorbar(img2, ax=ax2, location='right')
    plt.tight_layout()
    plt.savefig(f"../outputs/plots/{file_name}_{hybrid_note}_hybrid.png")
    plt.close(fig2)

def note_to_freq(note: str, octave: int) -> float:
    """Convert note name (e.g., 'A', 'C#') and octave to frequency in Hz.
       Uses A4 = 440 Hz, MIDI standard. octave is integer (e.g., 4 for A4).
    """
    note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    idx = note_names.index(note)
    # midi number formula: C4 = 60 -> midi = note_index + 12*(octave + 1)
    midi = idx + 12 * (octave + 1)
    freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
    return freq

def make_synthetic(freq: float, t: np.ndarray, n_harmonics: int, gain: float, mean_amp: float) -> np.ndarray:
    """Create harmonic-rich synthetic tone of the same length as t."""
    harmonics = np.zeros_like(t)
    for i in range(n_harmonics):
        harmonic_freq = freq * (i + 1)
        # Avoid aliasing: if harmonic_freq > Nyquist, break
        # (We still add lower harmonics; break prevents aliasing producing weirdness)
        harmonics += np.sin(2 * np.pi * harmonic_freq * t) * (1.0 / (i + 1))
    synthetic = gain * mean_amp * harmonics
    # optional windowing/clip to avoid loud spikes
    return synthetic

def make_flute_synthetic(freq: float, t: np.ndarray, n_harmonics: int, gain: float, mean_amp: float) -> np.ndarray:
    """Create a flute-like synthetic tone of the same length as t."""
    harmonics = np.zeros_like(t)
    for i in range(n_harmonics):
        harmonic_freq = freq * (i + 1)
        # Avoid aliasing: if harmonic_freq > Nyquist, break
        if harmonic_freq > (0.5 / (t[1] - t[0])):
            break
        # Flute-like harmonic structure: weaker higher harmonics
        harmonics += (1.0 / (i + 1) ** 2) * np.sin(2 * np.pi * harmonic_freq * t)

    # Apply an amplitude envelope (attack, sustain, decay)
    attack_time = 0.1  # seconds
    decay_time = 0.2   # seconds
    sustain_level = 0.7
    envelope = np.ones_like(t)
    attack_samples = int(attack_time * len(t) / t[-1])
    decay_samples = int(decay_time * len(t) / t[-1])

    # Attack phase
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    # Decay phase
    envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, sustain_level, decay_samples)
    # Sustain phase
    envelope[attack_samples + decay_samples:] = sustain_level
    # Add random amplitude variations
    random_variation = 1 + 0.2 * np.random.uniform(-1, 1, size=t.shape)  # Random factor between 0.9 and 1.1
    envelope *= random_variation
    # Apply envelope to harmonics
    synthetic = gain * mean_amp * harmonics * envelope
    synthetic /= np.max(np.abs(synthetic))  # Normalize
    synthetic *= 0.5  # Scale to mean amplitude
    return synthetic

def chunk_mean_chroma(chroma: np.ndarray, sr: int, hop_length: int, chunk_seconds: int) -> List[np.ndarray]:
    """Return list of 12-d mean chroma vectors, one per chunk."""
    n_frames = chroma.shape[1]
    duration = (n_frames * hop_length) / sr
    seconds_per_frame = duration / n_frames
    frames_per_chunk = max(1, int(round(chunk_seconds / seconds_per_frame)))
    chunks = []
    for start in range(0, n_frames, frames_per_chunk):
        end = min(n_frames, start + frames_per_chunk)
        chunk_vec = np.mean(chroma[:, start:end], axis=1)
        # normalize to unit L2 to prepare for cosine similarity
        norm = np.linalg.norm(chunk_vec)
        if norm > 0:
            chunk_vec = chunk_vec / norm
        chunks.append(chunk_vec)
    return chunks

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # both are unit-norm vectors ideally
    # protect against small numerical issues
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return 1.0 - cosine(a, b)  # 1 - cosine distance

def score_hybrid_against_orig(orig_chroma: np.ndarray, hybrid_chroma: np.ndarray,
                              sr: int, hop_length: int, chunk_seconds: int, sigma: float,
                              file_name:str, note_name: str,
                              skip: bool = False) -> dict[str, float]:
    # Determine total duration
    n_frames = orig_chroma.shape[1]
    duration = (n_frames * hop_length) / sr
    # If longer than 4 minutes, skip the first minute
    if skip and duration > 240:
        frames_to_skip = int(60 * sr / hop_length)
        orig_chroma = orig_chroma[:, frames_to_skip:]
        hybrid_chroma = hybrid_chroma[:, frames_to_skip:]
    # chunk means
    o_chunks = chunk_mean_chroma(orig_chroma, sr, hop_length, chunk_seconds)
    h_chunks = chunk_mean_chroma(hybrid_chroma, sr, hop_length, chunk_seconds)
    # Align number of chunks (if different due to rounding, compare min length)
    n_chunks = min(len(o_chunks), len(h_chunks))
    cos_sims = []
    for i in range(n_chunks):
        cos_sims.append(cosine_similarity(o_chunks[i], h_chunks[i]))
    mean_cos = float(np.mean(cos_sims)) if cos_sims else 0.0
    return {
        'mean_cosine': mean_cos
    }

def find_best_key(y, sr,
                  audio_path: str,
                  notes: List[str] = NOTES,
                  synth_octave: int = SYNTH_OCTAVE,
                  n_harmonics: int = N_HARMONICS,
                  synth_gain: float = SYNTH_GAIN,
                  chunk_seconds: int = CHUNK_SECONDS,
                  hop_length: int = HOP_LENGTH,
                  add_perfect_fifth: bool = False  # New parameter
                  ) -> dict[str, float]:
    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    mean_amp = np.mean(np.abs(y)) + 1e-12
    duration = len(y) / sr
    t = np.arange(len(y)) / sr  # ensures same length as y
    # compute original chroma (CQT-based)
    chroma_orig = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    results = {}
    # Prepare output directories for hybrid audio files and plots
    audio_subdir = "perfect_fifth" if add_perfect_fifth else "no_perfect_fifth"
    plot_subdir = "perfect_fifth" if add_perfect_fifth else "no_perfect_fifth"
    out_audio_dir = os.path.join("..", "outputs", "audio", audio_subdir)
    out_plot_dir = os.path.join("..", "outputs", "plots", plot_subdir)
    os.makedirs(out_audio_dir, exist_ok=True)
    os.makedirs(out_plot_dir, exist_ok=True)
    logger.info(f"Notes to evaluate: {notes}")
    for note in notes:
        f0 = note_to_freq(note, synth_octave)
        synthetic = make_synthetic(f0, t, n_harmonics, synth_gain, mean_amp)
        # Optionally generate synthetic signal for the perfect fifth
        if add_perfect_fifth:
            perfect_fifth_note = NOTES[(NOTES.index(note) + 7) % 12]  # Perfect fifth note
            perfect_fifth_freq = note_to_freq(perfect_fifth_note, synth_octave)  # Perfect fifth frequency
            synthetic_perfect_fifth = make_synthetic(perfect_fifth_freq, t, n_harmonics, synth_gain, mean_amp)
            synthetic_perfect_fifth *= 0.75
        else:
            synthetic_perfect_fifth = 0
        # ensure same length
        if synthetic.shape != y.shape:
            synthetic = librosa.util.fix_length(synthetic, size=len(y))
        if isinstance(synthetic_perfect_fifth, np.ndarray) and synthetic_perfect_fifth.shape != y.shape:
            synthetic_perfect_fifth = librosa.util.fix_length(synthetic_perfect_fifth, size=len(y))
        hybrid = y + synthetic + synthetic_perfect_fifth
        # Normalize to avoid clipping when saving
        peak = np.max(np.abs(hybrid)) + 1e-12
        if peak > 1.0:
            hybrid_to_save = hybrid * (0.99 / peak)
        else:
            hybrid_to_save = hybrid
        # Save hybrid audio for this note
        out_audio_path = os.path.join(out_audio_dir, f"{file_name}_{note}.wav")
        try:
            sf.write(out_audio_path, hybrid_to_save, sr, subtype='PCM_16')
            logger.info(f"Saved hybrid audio: {out_audio_path}")
        except Exception as e:
            logger.error(f"Failed to save hybrid audio for {note}: {e}")
        chroma_hybrid = librosa.feature.chroma_cqt(y=hybrid, sr=sr, hop_length=hop_length)
        # Save plot in the correct subfolder
        plot_chroma_heatmaps(chroma_orig, chroma_hybrid, sr, hop_length, note, os.path.join(plot_subdir, file_name))
        score = score_hybrid_against_orig(chroma_orig, chroma_hybrid, sr, hop_length, CHUNK_SECONDS, 0.0, file_name, note, True)
        results[note] = score['mean_cosine']
    return results

if __name__ == "__main__":
    logger.info(f"Running key identification for ${AUDIO_PATH} ...")
    y, sr = librosa.load(AUDIO_PATH, sr=None, mono=True)
    scores = find_best_key(y=y, sr=sr, audio_path=AUDIO_PATH, add_perfect_fifth=True)
    logger.info(f"Scores by note: ${scores}")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for note, score in sorted_scores:
        logger.info(f"Note: {note}, Score: {score:.4f}")
    best_note, best_score = sorted_scores[0]
    logger.info(f"Best matching note: {best_note} with score {best_score:.4f}")