import librosa
import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks

def detect_key_fft(y_vocals, sr, threshold=0.1):
    """
    Detect the dominant note using FFT peak detection.

    Args:
        y_vocals (ndarray): The isolated vocal track
        sr (int): Sample rate
        threshold (float, optional): Threshold for peak detection. Defaults to 0.1.

    Returns:
        tuple: (dominant_freq, dominant_note) where:
            - dominant_freq is the dominant frequency in Hz
            - dominant_note is the corresponding musical note
    """
    # Compute the FFT
    yf = fft(y_vocals)
    freqs = np.fft.fftfreq(len(y_vocals), 1 / sr)

    # Only consider positive frequencies
    positive_mask = freqs > 0
    yf_pos = np.abs(yf[positive_mask])
    freqs_pos = freqs[positive_mask]

    # Find peaks in the FFT magnitude
    peak_idx, properties = find_peaks(yf_pos, height=threshold * np.max(yf_pos))

    # If no peaks found, return None
    if len(peak_idx) == 0:
        return None, None

    # Get the dominant frequency (highest peak)
    dominant_idx = peak_idx[np.argmax(properties['peak_heights'])]
    dominant_freq = freqs_pos[dominant_idx]

    # Convert frequency to musical note
    dominant_note = librosa.hz_to_note(dominant_freq)

    return dominant_freq, dominant_note


def detect_key_ks(y_vocals, sr):
    """
    Detect the key using the Krumhansl-Schmuckler algorithm.
    This is a simplified implementation that uses librosa's key detection.
    For a more comprehensive implementation, consider using music21 or PyMusicKit.

    Args:
        y_vocals (ndarray): The isolated vocal track
        sr (int): Sample rate

    Returns:
        str: The detected key (e.g., "C major", "A minor")
    """
    # Compute the chromagram
    chroma = librosa.feature.chroma_cqt(y=y_vocals, sr=sr)

    # Compute the average chroma vector
    chroma_avg = np.mean(chroma, axis=1)

    # Define the key profiles (Krumhansl-Schmuckler)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    # Compute correlation with all possible keys
    correlations_major = []
    correlations_minor = []

    for i in range(12):  # 12 possible keys (C, C#, D, etc.)
        # Rotate the chroma vector to match the key
        rotated_chroma = np.roll(chroma_avg, i)

        # Compute correlation with major and minor profiles
        corr_major = np.corrcoef(rotated_chroma, major_profile)[0, 1]
        corr_minor = np.corrcoef(rotated_chroma, minor_profile)[0, 1]

        correlations_major.append(corr_major)
        correlations_minor.append(corr_minor)

    # Find the key with the highest correlation
    max_corr_major = np.max(correlations_major)
    max_corr_minor = np.max(correlations_minor)

    if max_corr_major > max_corr_minor:
        key_idx = np.argmax(correlations_major)
        mode = "major"
    else:
        key_idx = np.argmax(correlations_minor)
        mode = "minor"

    # Convert index to key name
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key = f"{key_names[key_idx]} {mode}"

    return key


def detect_key(y_vocals, sr):
    """
    Detect the key and dominant note of the track using multiple methods.

    Args:
        y_vocals (ndarray): The isolated vocal track
        sr (int): Sample rate

    Returns:
        tuple: (key, dominant_note) where:
            - key is the detected key (e.g., "C major")
            - dominant_note is the dominant note (e.g., "C4")
    """
    # Detect dominant note using FFT
    dominant_freq, dominant_note = detect_key_fft(y_vocals, sr)

    # Detect key using Krumhansl-Schmuckler algorithm
    key = detect_key_ks(y_vocals, sr)

    return key, dominant_note

if __name__ == "__main__":
    # Example usage
    audio_path = "../inputs/track_m.mp3"
    y_vocals, sr = librosa.load(audio_path, sr=None)

    key, dominant_note = detect_key(y_vocals, sr)
    print(f"Detected Key using KS: {key}")
    print(f"Dominant Note using FFT and Peaks: {dominant_note}")

