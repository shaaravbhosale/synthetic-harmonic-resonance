import numpy as np
import soundfile as sf

def generate_harmonic_waveform(duration, sample_rate, fundamental_freq, num_harmonics):
    """
    Generate a waveform with harmonics.
    Args:
        duration (float): Duration in seconds.
        sample_rate (int): Samples per second.
        fundamental_freq (float): Fundamental frequency in Hz.
        num_harmonics (int): Number of harmonics to include.
    Returns:
        np.ndarray: Generated waveform.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = np.zeros_like(t)
    for n in range(1, num_harmonics + 1):
        amplitude = 1.0 / 2  # decreasing amplitude for higher harmonics
        waveform += amplitude * np.sin(2 * np.pi * fundamental_freq * n * t)
    waveform /= np.max(np.abs(waveform))
    return waveform


def generate_flute_waveform(duration, sample_rate, fundamental_freq, num_harmonics):
    """
    Generate a flute-like waveform with harmonics and an amplitude envelope.
    Args:
        duration (float): Duration in seconds.
        sample_rate (int): Samples per second.
        fundamental_freq (float): Fundamental frequency in Hz.
        num_harmonics (int): Number of harmonics to include.
    Returns:
        np.ndarray: Generated flute-like waveform.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = np.zeros_like(t)

    # Harmonic structure for a flute-like sound
    for n in range(1, num_harmonics + 1):
        amplitude = 1.0 / (n ** 2)  # Flute has weaker higher harmonics
        waveform += amplitude * np.sin(2 * np.pi * fundamental_freq * n * t)

    # Apply an amplitude envelope (attack, sustain, decay)
    attack_time = 0.1  # seconds
    decay_time = 0.2  # seconds
    sustain_level = 0.7
    envelope = np.ones_like(t)
    attack_samples = int(attack_time * sample_rate)
    decay_samples = int(decay_time * sample_rate)

    # Attack phase
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    # Decay phase
    envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, sustain_level, decay_samples)
    # Sustain phase
    envelope[attack_samples + decay_samples:] = sustain_level

    # Apply envelope to waveform
    waveform *= envelope
    waveform /= np.max(np.abs(waveform))  # Normalize

    return waveform
# Parameters
duration = 5.0            # seconds
sample_rate = 44100       # samples per second
fundamental_freq = 110.0  # Hz
num_harmonics = 5         # number of harmonics to include

# Generate waveform with harmonics
waveform = generate_harmonic_waveform(duration, sample_rate, fundamental_freq, num_harmonics)
flute_waveform = generate_flute_waveform(duration, sample_rate, fundamental_freq, num_harmonics)
# Save to WAV file
sf.write('harmonic_110Hz.wav', waveform, sample_rate)
sf.write('flute_110Hz.wav', flute_waveform, sample_rate)
print(f"Shape of waveform: {waveform.shape}")
