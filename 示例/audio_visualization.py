import numpy as np
import matplotlib.pyplot as plt
import librosa  # For note-to-frequency conversion

# ------------------------------
# Constants
# ------------------------------
# Standard note frequencies (A4 = 440 Hz as reference)
STANDARD_NOTES = {
    # Original 4th octave notes
    'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13,
    'E4': 329.63, 'F4': 349.23, 'F#4': 369.99, 'G4': 392.00,
    'G#4': 415.30, 'A4': 440.00, 'A#4': 466.16, 'B4': 493.88,
    'C5': 523.25, 'D5': 587.33, 'E5': 659.25,  # E5 = 2×E4 frequency
    'F5': 698.46, 'G5': 783.99                 # F5 = 2×F4, G5 = 2×G4
}

def generate_note(frequency, t, amplitude=0.7, harmonic_strength=0.3):
    """Original note generation function (unchanged)"""
    # Core fundamental frequency
    fundamental = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Add harmonics (multiples of fundamental frequency)
    harmonic = harmonic_strength * amplitude * np.sin(2 * np.pi * 2 * frequency * t)
    harmonic += harmonic_strength * 0.5 * amplitude * np.sin(2 * np.pi * 3 * frequency * t)
    
    return fundamental + harmonic

# Common audio sample rates
DEFAULT_SAMPLE_RATE = 44100  # Standard for most audio
LOW_SAMPLE_RATE = 22050      # For faster processing

# ------------------------------
# Helper Functions
# ------------------------------
def get_peak_amplitude(xf, yf_mag, target_freq, tolerance=2.0):
    """Find the maximum amplitude within a small range around a target frequency"""
    mask = np.abs(xf - target_freq) <= tolerance
    if np.any(mask):
        return np.max(yf_mag[mask])
    return 0.0

def create_time_segment(full_time_array, end_time=0.1):
    """
    Create a short time segment from a full time array.
    
    Parameters:
        full_time_array: np.array - Complete time array (from 0 to duration)
        end_time: float - End time of the segment in seconds (default: 0.1s)
    
    Returns:
        np.array - Time segment from 0 to end_time
    """
    return full_time_array[full_time_array <= end_time]

# ------------------------------
# Signal Generation
# ------------------------------
def generate_note(frequency, t, amplitude=0.7, harmonic_strength=0.3):
    """Generate a musical note with fundamental frequency and harmonics."""
    fundamental = amplitude * np.sin(2 * np.pi * frequency * t)
    harmonic = harmonic_strength * amplitude * np.sin(2 * np.pi * 2 * frequency * t)
    harmonic += harmonic_strength * 0.5 * amplitude * np.sin(2 * np.pi * 3 * frequency * t)
    return fundamental + harmonic

def create_envelope(duration, sample_rate, attack_ratio=0.2, decay_ratio=0.2, max_attack=0.05, max_decay=0.05):
    """
    Create an ADSR-style envelope for a note.
    
    Parameters:
        duration: Note duration in seconds
        sample_rate: Audio sample rate (Hz)
        attack_ratio: % of duration used for attack (0-1)
        decay_ratio: % of duration used for decay (0-1)
        max_attack: Maximum attack time in seconds
        max_decay: Maximum decay time in seconds
    """
    total_samples = int(sample_rate * duration)
    
    # Calculate envelope segment lengths
    attack_length = min(max_attack, duration * attack_ratio)
    decay_length = min(max_decay, duration * decay_ratio)
    attack_samples = int(sample_rate * attack_length)
    decay_samples = int(sample_rate * decay_length)
    sustain_samples = total_samples - attack_samples - decay_samples
    
    # Create segments
    attack = np.linspace(0, 1, attack_samples) if attack_samples > 0 else np.array([])
    sustain = np.ones(sustain_samples) if sustain_samples > 0 else np.array([])
    decay = np.linspace(1, 0.9, decay_samples) if decay_samples > 0 else np.array([])
    
    return np.concatenate([attack, sustain, decay])

def synthesize_melody(melody_notes, notes_dict, sample_rate, amplitude=0.7, harmonic_strength=0.3):
    """
    Generate a complete audio signal from a melody.
    
    Parameters:
        melody_notes: List of (note_name, duration) tuples
        notes_dict: Dictionary mapping note names to frequencies
        sample_rate: Audio sample rate (Hz)
    """
    note_signals = []
    
    for note, duration in melody_notes:
        t_note = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        note_signal = generate_note(
            frequency=notes_dict[note],
            t=t_note,
            amplitude=amplitude,
            harmonic_strength=harmonic_strength
        )
        envelope = create_envelope(duration, sample_rate)
        note_signals.append(note_signal * envelope)
    
    # Final processing
    song = np.concatenate(note_signals)
    song = song / np.max(np.abs(song))  # Normalize
    return add_fadeout(song, sample_rate, fade_duration=0.3)

def generate_chord(frequencies, t, amplitude=0.5, harmonic_strength=0.05):
    """Generate a chord by summing multiple note signals"""
    chord = np.zeros_like(t)
    for freq in frequencies:
        chord += generate_note(freq, t, amplitude, harmonic_strength)
    # Normalize to prevent clipping
    return chord / np.max(np.abs(chord))

def add_noise(signal, noise_level=0.01):
    """Add white noise to a signal (noise_level controls amplitude)"""
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise

# ------------------------------
# Time-Domain Visualization
# ------------------------------
def plot_time_segment(signal, time_segment, title, color='blue', downsample=1):
    """
    Plot a short time segment (e.g., 0-0.1s) of an audio signal
    """
    samples_segment = len(time_segment)
    plt.figure(figsize=(12, 3))
    plt.plot(
        time_segment[::downsample],
        signal[:samples_segment][::downsample],
        linewidth=1.0,
        color=color
    )
    plt.title(title, fontsize=12)
    plt.xlabel("Time (s)", fontsize=10)
    plt.ylabel("Amplitude", fontsize=10)
    plt.xlim(0, max(time_segment))  # Use segment's max instead of hardcoding
    plt.grid(alpha=0.3)
    plt.show()

# ------------------------------
# Frequency-Domain Visualization
# ------------------------------
def plot_spectrogram(signal, sample_rate, title, n_fft=2048, hop_length=512):
    """Plot a spectrogram of the signal (time-frequency analysis)"""
    plt.figure(figsize=(12, 4))
    # Compute spectrogram using librosa
    X = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    X_mag = librosa.amplitude_to_db(abs(X))
    librosa.display.specshow(
        X_mag,
        sr=sample_rate,
        x_axis='time',
        y_axis='hz',
        hop_length=hop_length
    )
    plt.colorbar(label='Amplitude (dB)')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

# ------------------------------
# Combined Time/Frequency Visualization
# ------------------------------
def plot_signal_and_fft(signal, t, time_segment, sample_rate, title, 
                        fundamental_freq, xlim=(0, 3000), downsample=5):
    """
    Side-by-side plots of time-domain signal and frequency spectrum (FFT)
    with harmonic markers
    """
    # Prepare time-domain data
    samples_segment = len(time_segment)
    signal_segment = signal[:samples_segment]
    
    # Compute FFT
    n = len(signal)
    yf = np.fft.fft(signal)
    xf = np.fft.fftfreq(n, 1/sample_rate)[:n//2]
    yf_mag = 2.0/n * np.abs(yf[:n//2])
    
    # Create subplots
    fig, (ax_time, ax_freq) = plt.subplots(1, 2, figsize=(14, 4))
    
    # Time-domain plot
    ax_time.plot(
        time_segment[::downsample],
        signal_segment[::downsample],
        linewidth=1.0
    )
    ax_time.set_title(f"Time Domain: {title}")
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Amplitude")
    ax_time.set_xlim(0, max(time_segment))  # Use segment's max
    ax_time.grid(alpha=0.3)
    
    # Frequency-domain plot (harmonics behind spectrum)
    # 1. Draw harmonic lines first (behind)
    for i in range(1, 6):
        harm_freq = fundamental_freq * i
        harm_amp = get_peak_amplitude(xf, yf_mag, harm_freq)
        relative_strength = (harm_amp / np.max(yf_mag)) * 100 if np.max(yf_mag) > 0 else 0
        alpha = 0.3 + (relative_strength / 100) * 0.7
        
        ax_freq.axvline(
            x=harm_freq, 
            color='#4287f5', 
            linestyle='--', 
            alpha=alpha,
            linewidth=1.5,
            zorder=1
        )
    
    # 2. Draw spectrum (in front)
    ax_freq.plot(xf, yf_mag, linewidth=0.8, label='Frequency Spectrum', zorder=2)
    
    # Configure frequency plot
    ax_freq.set_title(f"Frequency Spectrum: {title}")
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel("Magnitude (log scale)")
    ax_freq.set_xlim(xlim)
    
    # Log scale settings
    min_magnitude = np.min(yf_mag[yf_mag > 0])
    ax_freq.set_ylim(min_magnitude * 0.5, np.max(yf_mag) * 1.2)
    ax_freq.set_yscale('log')
    ax_freq.grid(True, alpha=0.5, which="both")
    
    # 3. Add labels (on top)
    for i in range(1, 6):
        harm_freq = fundamental_freq * i
        harm_amp = get_peak_amplitude(xf, yf_mag, harm_freq)
        relative_strength = (harm_amp / np.max(yf_mag)) * 100 if np.max(yf_mag) > 0 else 0
        
        if relative_strength > 0.1:
            ax_freq.text(
                harm_freq, ax_freq.get_ylim()[1] * 0.95,
                f"{i}x ({relative_strength:.1f}%)",
                rotation=90,
                va='top',
                ha='center',
                fontsize=8,
                color='#1a56db',
                alpha=alpha,
                zorder=3
            )
    
    ax_freq.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    
    # Return dominant frequency
    peak_idx = np.argmax(yf_mag[:int(xlim[1] * n / sample_rate)])
    return xf[peak_idx]

# ------------------------------
# Note Frequency Utilities
# ------------------------------
def get_note_frequencies(notes_list):
    """
    Convert a list of note names (e.g., ['C4', 'A4']) to their frequencies (Hz)
    using librosa's tuning (A4 = 440 Hz)
    """
    return {note: librosa.note_to_hz(note) for note in notes_list}

def note_to_freq(note):
    """Convert a single note (e.g., 'A4') to its frequency in Hz"""
    return librosa.note_to_hz(note)

def freq_to_note(frequency):
    """Estimate the closest musical note to a given frequency"""
    return librosa.hz_to_note(frequency)

def add_fadeout(signal, sample_rate, fade_duration=0.3):
    """Add a linear fade-out to the end of an audio signal."""
    fade_samples = int(sample_rate * fade_duration)
    if fade_samples > len(signal):
        fade_samples = len(signal)
    fade = np.linspace(1, 0, fade_samples)
    signal[-fade_samples:] *= fade
    return signal

import matplotlib.pyplot as plt
import seaborn as sns

def plot_waveform_segments(signal, num_segments, segment_lyrics, title="Waveform Segments"):
    """Plot audio signal divided into labeled segments."""
    sns.set_style("whitegrid")
    total_samples = len(signal)
    segment_size = total_samples // num_segments
    
    fig, axes = plt.subplots(num_segments, 1, figsize=(18, 3 * num_segments))
    fig.suptitle(title, fontsize=16, y=0.99)
    
    for i in range(num_segments):
        start = i * segment_size
        end = start + segment_size if i < num_segments - 1 else total_samples
        color = 'forestgreen' if i % 2 == 0 else 'darkgreen'
        
        axes[i].plot(signal[start:end], color=color, linewidth=0.7)
        axes[i].set_title(f"Segment {i+1}: {segment_lyrics[i]}", fontsize=10)
        axes[i].set_xlabel("Samples")
        axes[i].set_ylabel("Amplitude")
        axes[i].grid(alpha=0.2)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    return fig

def create_fade_curve(fade_duration, sample_rate):
    """Create a linear fade curve for audio endings"""
    fade_samples = int(sample_rate * fade_duration)
    return np.linspace(1, 0, fade_samples)

# Module version
MODULE_VERSION = "4.0"  # Incremented from 4.0
