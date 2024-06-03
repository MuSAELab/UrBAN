import numpy as np
from scipy import signal
import time

def signal_to_frames(signal_data, window_size, hop_size, sample_rate):
    """
    Convert signal_data into a matrix of frames where each row is a time slice.

    Parameters:
    - signal_data (np.ndarray): Input audio signal.
    - window_size (int): Size of the analysis window (in samples).
    - hop_size (int): Number of samples between successive frames.
    - sample_rate (int): Sampling rate of the signal.

    Returns:
    - np.ndarray: STFT matrix representing the signal frames.
    """
    _, _, stft_matrix = signal.stft(signal_data, fs=sample_rate, nperseg=window_size, noverlap=hop_size, return_onesided=True, nfft=window_size * 8)
    return stft_matrix.T

def frames_to_signal(frame_matrix, window_size, hop_size, sample_rate):
    """
    Convert frame_matrix into a 1D signal using Overlap-Add.

    Parameters:
    - frame_matrix (np.ndarray): STFT matrix representing the signal frames.
    - window_size (int): Size of the analysis window (in samples).
    - hop_size (int): Number of samples between successive frames.
    - sample_rate (int): Sampling rate of the signal.

    Returns:
    - np.ndarray: Reconstructed signal from the frame matrix.
    """
    _, reconstructed_signal = signal.istft(frame_matrix.T, fs=sample_rate, nperseg=window_size, noverlap=hop_size, input_onesided=True, nfft=window_size * 8)
    return reconstructed_signal

def estimate_noise_snr(stft_matrix, snr_threshold=1.5):
    """
    Estimate the magnitude and power spectrum of the noise for each frame.

    Parameters:
    - stft_matrix (np.ndarray): STFT matrix representing the signal frames.
    - snr_threshold (float): SNR threshold for noise estimation.

    Returns:
    - np.ndarray: Estimated magnitude of the noise for each frame.
    """
    estimated_magnitude = np.zeros_like(stft_matrix)

    num_frames = 10  # Number of frames to use for estimating a-posteriori SNR

    # Compute the squared magnitude for all frames
    squared_magnitude = np.abs(stft_matrix) ** 2

    for frame_index in range(stft_matrix.shape[0]):
        if frame_index < num_frames:
            # Use noisy spectra for the first 10 iterations
            estimated_magnitude[frame_index] = np.abs(stft_matrix[frame_index])
        else:
            # A-posteriori SNR
            posterior_snr = squared_magnitude[frame_index] / np.mean(squared_magnitude[frame_index - num_frames:frame_index], axis=0)
            if np.abs(stft_matrix[frame_index].mean()) < 0.000003:
                estimated_magnitude[frame_index] = 0
            else:
                smoothing_factor = 25
                alpha = 1 / (1 + np.exp(-smoothing_factor * (posterior_snr - snr_threshold)))
                estimated_magnitude[frame_index] = alpha * estimated_magnitude[frame_index - 1] + (1 - alpha) * np.abs(stft_matrix[frame_index])

    return estimated_magnitude


def spectral_subtraction_magnitude(stft_matrix, estimated_magnitude):
    """
    Perform spectral subtraction using the estimated noise magnitude.

    Parameters:
    - stft_matrix (np.ndarray): STFT matrix representing the signal frames.
    - estimated_magnitude (np.ndarray): Estimated magnitude of the noise for each frame.

    Returns:
    - np.ndarray: Cleaned spectrogram after spectral subtraction.
    """
    estimated_magnitude_clean = np.maximum(abs(stft_matrix) - estimated_magnitude, 0)
    estimated_phase = np.angle(stft_matrix)
    clean_spectrogram = estimated_magnitude_clean * np.exp(1j * estimated_phase)
    return clean_spectrogram

def spectral_subtraction(signal_data, snr_threshold=1.5, window_time=30e-3, sample_rate=16000):
    """
    Perform spectral subtraction on the input signal.

    Parameters:
    - signal_data (np.ndarray): Input audio signal.
    - snr_threshold (float): SNR threshold for noise estimation.
    - window_time (float): Length of the analysis window in seconds.
    - sample_rate (int): Sampling rate of the signal.

    Returns:
    - np.ndarray: Reconstructed signal after spectral subtraction.
    """
    window_size = round(sample_rate * window_time)  
    hop_size = window_size // 2
    
    start_time = time.time()
    stft_matrix = signal_to_frames(signal_data, window_size, hop_size, sample_rate)

    estimated_magnitude = estimate_noise_snr(stft_matrix, snr_threshold)

    clean_spectrogram = spectral_subtraction_magnitude(stft_matrix, estimated_magnitude)

    # Reconstruction of the speech signal estimate
    reconstructed_signal = frames_to_signal(clean_spectrogram, window_size, hop_size, sample_rate)
    end_time = time.time()
    print(f"Time taken for spectral subtraction for this file: {end_time - start_time} seconds")

    return reconstructed_signal
