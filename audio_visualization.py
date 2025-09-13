import numpy as np
import matplotlib.pyplot as plt
import librosa  # For note-to-frequency conversion
from scipy.fft import fft, fftfreq
import seaborn as sns
import warnings

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


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
def generate_note(frequency, t, amplitude=0.7, harmonic_strength=0.3, harmonic_strengths=None, fade_duration=0.0):
    """
    Generate a musical note with fundamental frequency, harmonics, and optional fadeout.
    
    Parameters:
        frequency: Fundamental frequency (Hz)
        t: Time array (seconds)
        amplitude: Volume amplitude (0-1)
        harmonic_strength: Strength of harmonic overtones (0-1) for simple harmonics
                           (used if harmonic_strengths is None)
        harmonic_strengths: Array of strengths for individual harmonics (1st, 2nd, 3rd...),
                           overrides harmonic_strength if provided
        fade_duration: Duration of linear fadeout at the end (seconds, default=0.0)
    """
    # Core fundamental frequency
    fundamental = amplitude * np.sin(2 * np.pi * frequency * t)
    signal = fundamental  # Start with just the fundamental
    
    # Handle harmonic input - prioritize array if provided
    if harmonic_strengths is not None:
        # Add harmonics using the provided strength array
        for i, strength in enumerate(harmonic_strengths[1:], start=2):  # Skip 0th index (fundamental)
            harmonic = strength * amplitude * np.sin(2 * np.pi * i * frequency * t)
            signal += harmonic
    else:
        # Default simple harmonic profile using single strength value
        harmonic = harmonic_strength * amplitude * np.sin(2 * np.pi * 2 * frequency * t)
        harmonic += harmonic_strength * 0.5 * amplitude * np.sin(2 * np.pi * 3 * frequency * t)
        signal += harmonic
    
    # Normalize to prevent clipping
    signal = signal / np.max(np.abs(signal))
    
    # Apply fadeout if specified (and valid)
    if fade_duration > 0 and fade_duration < t[-1]:  # Ensure fade < total duration
        sample_rate = 1 / (t[1] - t[0])  # Calculate from time array
        signal = add_fadeout(signal, sample_rate, fade_duration=fade_duration)
    
    return signal


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
def synthesize_melody(
    melody_notes, 
    notes_dict, 
    sample_rate, 
    amplitude=0.7, 
    harmonic_strength=0.3,
    fadeout_duration=0.3  # Explicitly added fadeout parameter for the final song
):
    """
    Generate a complete audio signal from a melody.
    
    Parameters:
        melody_notes: List of (note_name, duration) tuples
        notes_dict: Dictionary mapping note names to frequencies
        sample_rate: Audio sample rate (Hz)
        amplitude: Volume amplitude (0-1)
        harmonic_strength: Strength of harmonic overtones (0-1)
        fadeout_duration: Duration of the final fadeout (seconds) for the entire song
    """
    note_signals = []
    
    for note, duration in melody_notes:
        # Generate time array for this note
        t_note = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Generate raw note signal with harmonics
        note_signal = generate_note(
            frequency=notes_dict[note],
            t=t_note,
            amplitude=amplitude,
            harmonic_strength=harmonic_strength
        )
        
        # Apply envelope (ADSR) to individual note to prevent clicks between notes
        envelope = create_envelope(duration, sample_rate)
        note_signals.append(note_signal * envelope)
    
    # Combine all notes into a single song signal
    song = np.concatenate(note_signals)
    
    # Normalize to prevent clipping
    song = song / np.max(np.abs(song))
    
    # Apply final fadeout to the entire song using the explicit parameter
    return add_fadeout(song, sample_rate, fade_duration=fadeout_duration)
    
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


def calculate_fft(signal, sample_rate, normalize=True):
    """
    Calculate the Fast Fourier Transform (FFT) of a time-domain signal and return
    the frequency axis and corresponding magnitude spectrum.
    
    Parameters:
        signal (np.ndarray): Time-domain audio signal (1D array of samples).
        sample_rate (int/float): Sample rate of the signal (Hz).
        normalize (bool): If True, normalize the magnitude spectrum by the number
                         of samples to get accurate amplitude values (default: True).
    
    Returns:
        xf (np.ndarray): Frequency axis (Hz) corresponding to the FFT output.
        yf_mag (np.ndarray): Magnitude spectrum of the signal (amplitude).
    
    Notes:
        - Returns only the positive frequency half (0 to sample_rate/2).
        - Applies a Hann window to reduce spectral leakage.
    """
    # Validate input
    if not isinstance(signal, np.ndarray) or signal.ndim != 1:
        raise ValueError("Signal must be a 1D numpy array.")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be a positive number.")
    
    # Get signal length
    n = len(signal)
    
    # Apply Hann window to reduce spectral leakage (critical for clean FFT)
    window = np.hanning(n)
    signal_windowed = signal * window
    
    # Compute FFT
    yf = fft(signal_windowed)
    
    # Compute frequency axis (only positive frequencies)
    xf = fftfreq(n, 1 / sample_rate)[:n//2]  # n//2 gives positive half
    
    # Calculate magnitude spectrum
    yf_mag = np.abs(yf[:n//2])  # Take magnitude of positive frequencies
    
    # Normalize if requested (scales amplitude to match time-domain values)
    if normalize:
        yf_mag = 2.0 / n * yf_mag  # Factor of 2 accounts for negative frequencies
    
    return xf, yf_mag

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
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans',  'Arial']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    warnings.filterwarnings('ignore')
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

# ------------------------------
# Chord Visualization Functions
# ------------------------------
def plot_note_component(note, signal, axes, row_idx, fundamental_freq, 
                       time_segment, sample_rate,  # Added sample_rate parameter
                       fundamental_color='#e63946', harmonic_color='#457b9d'):
    """
    Plot a note's time and frequency domains in a specified subplot grid.
    
    Parameters:
        note: Name of the note (e.g., 'C4')
        signal: Audio signal array
        axes: Subplot axes grid (from plt.subplots())
        row_idx: Row index in the subplot grid
        fundamental_freq: Fundamental frequency of the note (Hz)
        time_segment: Time segment array for x-axis
        sample_rate: Audio sample rate (Hz) - new parameter
        fundamental_color: Color for fundamental frequency marker
        harmonic_color: Color for harmonic markers
    """

# Add these functions to your existing audio_visualization.py

def generate_pure_tone(frequency, duration=1.0, sample_rate=DEFAULT_SAMPLE_RATE, amplitude=0.5):
    """Generate a clean sine wave with no harmonics (for pure tone comparisons)"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * frequency * t)

def generate_combined_tone(
    frequencies, 
    duration=1.5, 
    sample_rate=DEFAULT_SAMPLE_RATE, 
    amplitude=0.3,
    fade_duration=0.3  # New keyword argument for fade-out
):
    """
    Combine multiple pure tones into a single chord signal with optional fade-out.
    
    Args:
        frequencies: List of fundamental frequencies (Hz)
        duration: Total duration of the chord (seconds)
        sample_rate: Audio sample rate (Hz)
        amplitude: Base amplitude for each tone (0-1)
        fade_duration: Duration of linear fade-out at the end (seconds, default=0.3)
    
    Returns:
        np.ndarray: Normalized combined signal with optional fade-out
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    combined = np.zeros_like(t)
    
    # Add each frequency component
    for freq in frequencies:
        combined += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Normalize to prevent clipping
    combined = combined / np.max(np.abs(combined))
    
    # Apply fade-out if specified (and fade_duration is valid)
    if fade_duration > 0 and fade_duration < duration:
        combined = add_fadeout(combined, sample_rate, fade_duration=fade_duration)
    
    return combined

def detect_peaks(xf, yf_mag, threshold=0.1, num_peaks=5):
    """
    Detect dominant frequency peaks in an FFT spectrum
    
    Args:
        xf: Frequency array from FFT
        yf_mag: Magnitude array from FFT
        threshold: Minimum amplitude (relative to max) to consider as peak
        num_peaks: Maximum number of peaks to return
    
    Returns:
        list: Top dominant frequencies (sorted by amplitude)
    """
    # Normalize to find relative thresholds
    max_amplitude = np.max(yf_mag)
    peak_mask = yf_mag >= (max_amplitude * threshold)
    
    # Find peak indices and sort by amplitude
    peak_indices = np.where(peak_mask)[0]
    peak_indices = sorted(peak_indices, key=lambda i: yf_mag[i], reverse=True)
    
    # Return top peaks with their frequencies
    return [xf[i] for i in peak_indices[:num_peaks]]

def plot_chord_components(chord_notes, note_signals, time_segment, sample_rate, title="Chord Components"):
    """
    Plot time/frequency domains for individual chord components and combined signal
    
    Args:
        chord_notes: List of note names in the chord
        note_signals: Dictionary of {note: signal}
        time_segment: Short time segment for time-domain plots
        sample_rate: Audio sample rate
        title: Plot title
    """
    # Create combined chord signal
    combined_signal = np.zeros_like(next(iter(note_signals.values())))
    for note in chord_notes:
        combined_signal += note_signals[note]
    combined_signal = combined_signal / np.max(np.abs(combined_signal))
    
    # Create subplots grid
    fig, axes = plt.subplots(len(chord_notes) + 1, 2, figsize=(16, 4 * (len(chord_notes) + 1)))
    fig.suptitle(title, fontsize=16)
    
    # Plot individual components
    for row, note in enumerate(chord_notes):
        signal = note_signals[note]
        signal_segment = signal[:len(time_segment)]
        
        # Time domain
        axes[row, 0].plot(time_segment[::2], signal_segment[::2], color='teal', linewidth=1)
        axes[row, 0].set_title(f"{note} - Time Domain")
        axes[row, 0].set_xlabel("Time (s)")
        axes[row, 0].set_xlim(0, max(time_segment))
        
        # Frequency domain
        xf, yf_mag = calculate_fft(signal, sample_rate)
        axes[row, 1].plot(xf, yf_mag, linewidth=0.8)
        axes[row, 1].axvline(x=librosa.note_to_hz(note), color='#e63946', linestyle='-', alpha=0.8)
        axes[row, 1].set_title(f"{note} - Frequency Domain")
        axes[row, 1].set_xlabel("Frequency (Hz)")
        axes[row, 1].set_xlim(0, 1000)
    
    # Plot combined signal in last row
    combined_segment = combined_signal[:len(time_segment)]
    axes[-1, 0].plot(time_segment[::2], combined_segment[::2], color='purple', linewidth=1)
    axes[-1, 0].set_title("Combined Chord - Time Domain")
    axes[-1, 0].set_xlabel("Time (s)")
    
    xf_comb, yf_comb = calculate_fft(combined_signal, sample_rate)
    axes[-1, 1].plot(xf_comb, yf_comb, linewidth=0.8)
    for note in chord_notes:
        axes[-1, 1].axvline(x=librosa.note_to_hz(note), color='#e63946', linestyle='-', alpha=0.8)
    axes[-1, 1].set_title("Combined Chord - Frequency Domain")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    axes[-1, 1].set_xlim(0, 1000)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    
def transpose_melody(melody, semitones):
    """
    Transpose a melody (list of note-duration tuples) by a number of semitones.
    
    Args:
        melody: List of (note_name, duration) tuples (e.g., [('E4', 1), ('F4', 1)])
        semitones: Number of semitones to shift (+ = up, - = down)
    
    Returns:
        list: Transposed melody with same durations
    """
    transposed = []
    for note, duration in melody:
        # Convert to MIDI, transpose, convert back to note name
        midi = librosa.note_to_midi(note)
        transposed_note = librosa.midi_to_note(midi + semitones)
        transposed.append((transposed_note, duration))
    return transposed

def match_chord_durations_to_melody(chord_progression, melody, bpm):
    """
    Adjust chord durations to exactly match the total length of the melody.
    
    Args:
        chord_progression: List of (chord_name, measures) tuples
        melody: List of (note_name, beats) tuples
        bpm: Tempo in beats per minute
    
    Returns:
        list: Chord progression with adjusted durations (in seconds)
    """
    # Calculate total melody duration in seconds
    total_melody_beats = sum(beats for _, beats in melody)
    total_melody_seconds = (total_melody_beats * 60) / bpm
    
    # Calculate total measures in chord progression
    total_chord_measures = sum(measures for _, measures in chord_progression)
    
    # Adjust each chord's duration to fill the melody length
    adjusted = []
    for chord_name, measures in chord_progression:
        chord_ratio = measures / total_chord_measures
        adjusted_duration = total_melody_seconds * chord_ratio
        adjusted.append((chord_name, adjusted_duration))
    return adjusted

def generate_timbre(frequency, t, instrument='violin', envelope=None, harmonic_strengths=None):
    """
    Generate a note with customizable harmonic profiles and optional ADSR envelope.
    
    Args:
        frequency: Fundamental frequency (Hz)
        t: Time array (seconds)
        instrument: 'violin', 'piano', 'flute', or 'trumpet' (predefined harmonic strengths)
                    Ignored if harmonic_strengths is provided
        envelope: Optional amplitude envelope array (same length as t)
        harmonic_strengths: Optional list/array of harmonic amplitudes 
                            (first element = fundamental, then overtones)
    
    Returns:
        np.ndarray: Note signal with specified timbre and envelope
    """
    # Predefined instrument profiles (used if harmonic_strengths not provided)
    predefined_timbres = {
        'violin': [1.0, 0.7, 0.5, 0.3, 0.2],  # Strong higher harmonics
        'piano': [1.0, 0.5, 0.3, 0.1, 0.05],  # Muted higher harmonics
        'flute': [1.0, 0.1, 0.05, 0.02, 0.01],  # Dominant fundamental
        'trumpet': [1.0, 0.9, 0.8, 0.7, 0.9]  # Brassy with strong upper harmonics (especially 5th)
    }
    
    # Use custom harmonics if provided, otherwise use instrument defaults
    if harmonic_strengths is not None:
        amps = np.asarray(harmonic_strengths)
        if np.any(amps < 0):
            raise ValueError("Harmonic strengths cannot be negative")
    else:
        if instrument not in predefined_timbres:
            raise ValueError(f"Unknown instrument: {instrument}. Choose from {list(predefined_timbres.keys())}")
        amps = np.asarray(predefined_timbres[instrument])
    
    # Build signal by summing harmonics
    signal = np.zeros_like(t)
    for i, amp in enumerate(amps, 1):  # i = harmonic number (1 = fundamental)
        signal += amp * np.sin(2 * np.pi * (frequency * i) * t)
    
    # Apply amplitude envelope if provided
    if envelope is not None:
        if len(envelope) != len(signal):
            raise ValueError(f"Envelope length ({len(envelope)}) must match time array length ({len(signal)})")
        signal *= envelope
    
    # Normalize to prevent clipping
    max_amplitude = np.max(np.abs(signal))
    return signal / max_amplitude if max_amplitude > 0 else signal

def dynamic_mixer(melody_signal, chord_signal, melody_strength=1.0, chord_strength=0.7):
    """
    Mix melody and chords with dynamic range compression to avoid clipping.
    
    Args:
        melody_signal: Melody audio signal
        chord_signal: Chord accompaniment signal
        melody_strength: Relative volume of melody (0-1)
        chord_strength: Relative volume of chords (0-1)
    
    Returns:
        np.ndarray: Mixed signal with safe amplitude levels
    """
    # Pad shorter signal to match lengths
    max_len = max(len(melody_signal), len(chord_signal))
    melody_padded = np.pad(melody_signal, (0, max_len - len(melody_signal)), mode='constant')
    chord_padded = np.pad(chord_signal, (0, max_len - len(chord_signal)), mode='constant')
    
    # Apply volume weights
    mixed = (melody_padded * melody_strength) + (chord_padded * chord_strength)
    
    # Compress dynamic range to prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed * 0.9 / peak  # Scale to 90% of maximum safe amplitude
    return mixed

def repeat_section(melody, chord_progression, repeats=1):
    """
    Repeat a melody and its chord progression multiple times.
    
    Args:
        melody: List of (note_name, duration) tuples
        chord_progression: List of (chord_name, measures) tuples
        repeats: Number of times to repeat the section
    
    Returns:
        tuple: (repeated_melody, repeated_chords)
    """
    repeated_melody = melody * repeats
    repeated_chords = chord_progression * repeats
    return repeated_melody, repeated_chords

def generate_melody(melody, note_freqs, sample_rate, bpm, instrument='violin'):
    """
    Generate a complete melody signal from a list of (note, duration) tuples.
    
    Args:
        melody: List of (note_name, beats) tuples
        note_freqs: Dictionary mapping note names to frequencies (Hz)
        sample_rate: Audio sample rate (Hz)
        bpm: Tempo in beats per minute
        instrument: Instrument timbre ('violin', 'piano', 'flute')
    
    Returns:
        np.ndarray: Normalized melody signal
    """
    beat_duration = 60 / bpm
    melody_signals = []
    
    for note_name, beats in melody:
        duration = beats * beat_duration
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Use instrument-specific timbre
        note_signal = generate_timbre(
            frequency=note_freqs[note_name],
            t=t,
            instrument=instrument
        )
        
        # Smooth transitions
        note_signal = add_fadeout(note_signal, sample_rate, fade_duration=0.05)
        melody_signals.append(note_signal)
    
    full_melody = np.concatenate(melody_signals)
    return full_melody / np.max(np.abs(full_melody))  # Normalize

def generate_accompaniment(progression, chords_dict, note_freqs, 
                          sample_rate, bpm, melody):
    """
    Generate a chord accompaniment perfectly synced to a melody's duration.
    
    Args:
        progression: List of (chord_name, measures) tuples
        chords_dict: Dictionary mapping chord names to lists of note names
        note_freqs: Dictionary mapping note names to frequencies (Hz)
        sample_rate: Audio sample rate (Hz)
        bpm: Tempo in beats per minute
        melody: List of (note_name, beats) tuples (for duration sync)
    
    Returns:
        np.ndarray: Normalized chord accompaniment signal
    """
    # Sync chord durations to melody length
    adjusted_progression = match_chord_durations_to_melody(
        chord_progression=progression,
        melody=melody,
        bpm=bpm
    )
    
    chord_signals = []
    for chord_name, duration in adjusted_progression:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        chord_freqs = [note_freqs[note] for note in chords_dict[chord_name]]
        
        # Generate combined chord tone
        chord_signal = generate_combined_tone(
            frequencies=chord_freqs,
            duration=duration,
            sample_rate=sample_rate,
            amplitude=0.35
        )
        
        # Smooth transitions between chords
        chord_signal = add_fadeout(chord_signal, sample_rate, fade_duration=0.4)
        chord_signals.append(chord_signal)
    
    full_accomp = np.concatenate(chord_signals)
    return full_accomp / np.max(np.abs(full_accomp))  # Normalize to prevent clipping

def create_adsr_envelope(t, attack, decay, sustain_level, release):
    """
    Generate a reusable ADSR (Attack-Decay-Sustain-Release) envelope.
    
    Parameters:
    t (array): Time array (in seconds)
    attack (float): Duration of attack phase in seconds (0 to peak amplitude)
    decay (float): Duration of decay phase in seconds (peak to sustain level)
    sustain_level (float): Amplitude level during sustain phase (0-1 range)
    release (float): Duration of release phase in seconds (sustain to 0)
    
    Returns:
    array: ADSR envelope with same length as input time array
           Values range from 0 to 1 (normalized amplitude)
    """
    # Calculate total duration from time array
    total_duration = t[-1] if len(t) > 0 else attack + decay + release
    
    # Validate parameters
    if attack < 0 or decay < 0 or release < 0:
        raise ValueError("ADSR phase durations cannot be negative")
    if not (0 <= sustain_level <= 1):
        raise ValueError("Sustain level must be between 0 and 1")
    if (attack + decay + release) > total_duration:
        raise ValueError("Sum of ADSR phases exceeds total duration")
    
    # Create empty envelope array
    envelope = np.zeros_like(t)
    
    # Define phase boundaries
    attack_end = attack
    decay_end = attack + decay
    release_start = total_duration - release
    
    # Attack phase: linear increase from 0 to 1
    attack_mask = t <= attack_end
    envelope[attack_mask] = t[attack_mask] / attack_end if attack > 0 else 1.0
    
    # Decay phase: linear decrease from 1 to sustain_level
    decay_mask = (t > attack_end) & (t <= decay_end)
    if decay > 0:
        decay_progress = (t[decay_mask] - attack_end) / decay
        envelope[decay_mask] = 1 - (1 - sustain_level) * decay_progress
    else:
        envelope[decay_mask] = sustain_level
    
    # Sustain phase: constant at sustain_level
    sustain_mask = (t > decay_end) & (t <= release_start)
    envelope[sustain_mask] = sustain_level
    
    # Release phase: linear decrease from sustain_level to 0
    release_mask = t > release_start
    if release > 0:
        release_progress = (t[release_mask] - release_start) / release
        envelope[release_mask] = sustain_level * (1 - release_progress)
    else:
        envelope[release_mask] = 0.0
    
    return envelope

def plot_adsr_envelope(
    envelope, 
    sample_rate, 
    attack_end,    # User-defined attack end time (s)
    decay_end,     # User-defined decay end time (s)
    release_start, # User-defined release start time (s)
    title="ADSR Envelope", 
    instrument=None
):
    """
    ADSR visualization with explicit user-defined boundaries.
    You directly specify where each phase starts/ends for perfect alignment.
    """
    # Calculate time array and total duration
    total_samples = len(envelope)
    duration = total_samples / sample_rate
    time = np.linspace(0, duration, total_samples)
    
    # Validate boundaries to prevent errors
    if not (0 < attack_end < decay_end < release_start < duration):
        raise ValueError("Invalid boundaries! Must follow: 0 < attack_end < decay_end < release_start < duration")
    
    # Plot envelope with user-defined regions
    plt.figure(figsize=(10, 4))
    plt.plot(time, envelope, color='darkred', linewidth=2.5, zorder=3)
    
    # Colored regions exactly matching user input
    plt.axvspan(0, attack_end, color='#4287f5', alpha=0.3, label='Attack', zorder=2)
    plt.axvspan(attack_end, decay_end, color='#fca311', alpha=0.3, label='Decay', zorder=2)
    plt.axvspan(decay_end, release_start, color='#38b000', alpha=0.3, label='Sustain', zorder=2)
    plt.axvspan(release_start, duration, color='#e63946', alpha=0.3, label='Release', zorder=2)
    
    # Explicit boundary markers
    plt.axvline(x=attack_end, color='black', linestyle='--', alpha=0.7)
    plt.axvline(x=decay_end, color='black', linestyle='--', alpha=0.7)
    plt.axvline(x=release_start, color='black', linestyle='--', alpha=0.7)
    
    # Annotations
    plt.title(f"{title} {f'({instrument})' if instrument else ''}", fontsize=12)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.ylim(0, max(1.05, np.max(envelope) * 1.05))
    plt.grid(alpha=0.2)
    plt.legend()
    plt.show()
    
    # Report user-defined boundaries (exact match)
    print(f"ADSR Phases (user-defined boundaries):")
    print(f"Attack: 0 to {attack_end:.3f}s")
    print(f"Decay: {attack_end:.3f}s to {decay_end:.3f}s")
    print(f"Sustain: {decay_end:.3f}s to {release_start:.3f}s")
    print(f"Release: {release_start:.3f}s to {duration:.3f}s")
    
def plot_chord_spectrum_with_peaks(xf, yf_mag, chord_notes, note_freqs, 
                                  title="Chord Frequency Spectrum",
                                  xlim=(200, 1800), max_peaks=8):
    """
    Visualize chord frequency spectrum with labeled fundamental frequencies and dominant peaks.
    
    Args:
        xf: Frequency array from FFT (Hz)
        yf_mag: Magnitude spectrum from FFT
        chord_notes: List of note names in the chord
        note_freqs: Dictionary mapping note names to frequencies (Hz)
        title: Plot title
        xlim: Frequency range to display (min, max)
        max_peaks: Maximum number of dominant peaks to label
    """
    # Detect dominant peaks using module's built-in function
    dominant_peaks = detect_peaks(xf, yf_mag, threshold=0.05, num_peaks=max_peaks)
    
    # Create figure with consistent styling
    plt.figure(figsize=(14, 6))
    
    # Plot spectrum
    plt.plot(xf, yf_mag, color='purple', linewidth=0.8, label='Chord Spectrum')
    
    # Highlight fundamental frequencies with unique labels
    fundamentals = []
    for note in chord_notes:
        freq = note_freqs[note]
        if xlim[0] < freq < xlim[1]:  # Only show if within display range
            plt.axvline(
                x=freq, 
                color='#2a9d8f', 
                linestyle='-', 
                linewidth=2, 
                alpha=0.9,
                label=f"{note} ({freq:.1f} Hz)" if note not in fundamentals else ""
            )
            fundamentals.append(note)
    
    # Label detected peaks with musical notes
    max_magnitude = np.max(yf_mag)
    for peak in dominant_peaks:
        if xlim[0] < peak < xlim[1]:  # Only label peaks in display range
            detected_note = freq_to_note(peak)
            # Get peak amplitude for vertical positioning
            peak_amp = get_peak_amplitude(xf, yf_mag, peak)
            y_pos = max_magnitude * 0.95 if peak_amp < 0.7 * max_magnitude else peak_amp * 1.1
            
            plt.text(
                peak, y_pos, 
                f"{detected_note}\n{peak:.1f} Hz",
                color='#e63946', 
                fontweight='bold', 
                ha='center', 
                va='top', 
                rotation=90,
                fontsize=8
            )
    
    # Formatting
    plt.title(title, fontsize=14)
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Magnitude (log scale)", fontsize=12)
    plt.xlim(xlim)
    plt.yscale('log')
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()


# Module version
MODULE_VERSION = 16.0