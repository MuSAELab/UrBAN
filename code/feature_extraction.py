import numpy as np
import pandas as pd

import librosa
import scipy.fftpack
from librosa import util
from librosa.util.exceptions import ParameterError

def data_ls_to_string(ls, hive):
    file_names = []
    ls_string = []
    for i in range(len(ls)):
        ls_string.append(ls[i].strftime('%Y-%m-%d %H:%M:%S'))
        
        file_names.append(ls_string[i][8:10] + '-' + ls_string[i][5:7] + '-' +ls_string[i][0:4] + '_' + ls_string[i][11:13] + 'h' + ls_string[i][14:16] + '_' + 'HIVE' + '-' + str(int(hive)) + '.wav')
    return file_names

def spectral_descriptors(signal, sample_rate, n_fft, descriptor):
    """
    Calculate spectral descriptors for an audio signal.

    Parameters:
    - signal (np.ndarray): Audio signal.
    - sample_rate (int): Sampling rate of the audio signal.
    - n_fft (int): Number of FFT components.
    - descriptor (str): Spectral descriptor to compute. Options are:
        'centroid', 'spread', 'skewness', 'kurtosis', 'flatness',
        'rolloff', 'slope', 'entropy', 'crest', 'flux', 'zcr', 'rms'.

    Returns:
    - np.ndarray: Computed spectral descriptor.
    """
    
    # Compute the frequency array and reshape it for later calculations
    freq = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft).reshape((-1, 1))
    
    # Compute the Short-Time Fourier Transform (STFT) of the signal
    X = abs(librosa.stft(signal, n_fft=n_fft, hop_length=512, pad_mode='reflect'))

    # Spectral Centroid
    if descriptor == 'centroid':
        return librosa.feature.spectral_centroid(y=signal, sr=sample_rate, n_fft=n_fft, hop_length=512)
    
    # Spectral Spread
    elif descriptor == 'spread':
        cent = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, n_fft=n_fft, hop_length=512)
        spread = np.sqrt(np.sum(((freq - cent)**2) * librosa.util.normalize(X, norm=1, axis=0), axis=0, keepdims=True))
        return spread
    
    # Spectral Skewness
    elif descriptor == 'skewness':
        cent = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, n_fft=n_fft, hop_length=512)
        spread = np.sqrt(np.sum(((freq - cent)**2) * librosa.util.normalize(X, norm=1, axis=0), axis=0, keepdims=True))
        skewness = np.sum(((freq - cent)**3) * librosa.util.normalize(X, norm=1, axis=0) / (spread)**3, axis=0, keepdims=True)
        return skewness
    
    # Spectral Kurtosis
    elif descriptor == 'kurtosis':
        cent = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, n_fft=n_fft, hop_length=512)
        spread = np.sqrt(np.sum(((freq - cent)**2) * librosa.util.normalize(X, norm=1, axis=0), axis=0, keepdims=True))
        kurtosis = np.sum(((freq - cent)**4) * librosa.util.normalize(X, norm=1, axis=0) / (spread)**4, axis=0, keepdims=True)
        return kurtosis
    
    # Spectral Flatness
    elif descriptor == 'flatness':
        return librosa.feature.spectral_flatness(y=signal, n_fft=n_fft, hop_length=512)
    
    # Spectral Rolloff
    elif descriptor == 'rolloff':
        return librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, n_fft=n_fft, hop_length=512, roll_percent=0.85)
    
    # Spectral Slope
    elif descriptor == 'slope':
        slope = np.sum((freq - np.mean(freq)) * (X - np.mean(X, axis=0)), axis=0, keepdims=True) / np.sum((freq - np.mean(freq))**2)
        return slope
    
    # Spectral Entropy
    elif descriptor == 'entropy':
        eps = np.finfo(float).eps  # Small epsilon value to avoid log(0) issues
        entropy = -np.sum((X + eps) * np.log(X + eps), axis=0, keepdims=True) / np.log(np.max(freq) - np.min(freq))
        return entropy
    
    # Spectral Crest
    elif descriptor == 'crest':
        crest = np.max(X, axis=0) / (np.sum(X, axis=0) / (np.max(freq) - np.min(freq)))
        return crest
    
    # Spectral Flux
    elif descriptor == 'flux':
        flux = np.sum(np.abs(np.diff(X, axis=1)), axis=0)
        return flux
    
    # Zero Crossing Rate
    elif descriptor == 'zcr':
        return librosa.feature.zero_crossing_rate(y=signal, frame_length=n_fft, hop_length=512)
    
    # Root Mean Square Energy
    elif descriptor == 'rms':
        return librosa.feature.rms(y=signal, frame_length=n_fft, hop_length=512, center=True)
    
    else:
        raise ValueError(f"Descriptor '{descriptor}' is not recognized.")
        
        
        
        
def linear_filter_banks(nfilts=20,
                        nfft=512,
                        fs=16000,
                        low_freq=None,
                        high_freq=None,
                        scale="constant"):
    """
    Calculate linear filter banks.

    Parameters:
    - nfilts (int): Number of filters in the filterbank. Default is 20.
    - nfft (int): FFT size. Default is 512.
    - fs (int): Sample rate/sampling frequency of the signal. Default is 16000 Hz.
    - low_freq (int): Lowest band edge of linear filters. Default is 0 Hz.
    - high_freq (int): Highest band edge of linear filters. Default is fs/2.
    - scale (str): Choose if max bins amplitudes ascend, descend, or are constant (=1). Default is "constant".

    Returns:
    - np.ndarray: Array of size (nfilts, nfft//2 + 1) containing the filterbank.
                  Each row holds 1 filter.
    """
    
    # Set default values for high_freq and low_freq if they are not provided
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # Compute points evenly spaced in frequency (points are in Hz)
    linear_points = np.linspace(low_freq, high_freq, nfilts + 2)

    # Convert from Hz to FFT bin number
    bins = np.floor((nfft + 1) * linear_points / fs).astype(int)
    fbank = np.zeros([nfilts, nfft // 2 + 1])

    # Initialize the scaler based on the scale type
    if scale == "descendant" or scale == "constant":
        c = 1
    else:
        c = 0

    # Compute amplitudes of filter banks
    for j in range(nfilts):
        b0, b1, b2 = bins[j], bins[j + 1], bins[j + 2]

        # Adjust the scaler value based on the scale type
        if scale == "descendant":
            c -= 1 / nfilts
            c = max(c, 0)

        elif scale == "ascendant":
            c += 1 / nfilts
            c = min(c, 1)

        # Compute filter banks
        fbank[j, b0:b1] = c * (np.arange(b0, b1) - b0) / (b1 - b0)
        fbank[j, b1:b2] = c * (b2 - np.arange(b1, b2)) / (b2 - b1)

    return np.abs(fbank)




def manual_lfcc(
    *,
    y,
    sr: float = 22050,
    n_fft: int,
    hop_length: int,
    n_lfcc: int = 13,
    dct_type: int = 2,
    norm: str = "ortho",
    lifter: float = 0
) -> np.ndarray:
    """
    Compute Linear Frequency Cepstral Coefficients (LFCCs) from an audio signal.

    Parameters:
    - y (np.ndarray): Audio signal.
    - sr (float): Sampling rate of the audio signal. Default is 22050 Hz.
    - n_fft (int): Number of FFT components.
    - hop_length (int): Number of samples between successive frames.
    - n_lfcc (int): Number of LFCCs to return. Default is 13.
    - dct_type (int): Type of Discrete Cosine Transform (DCT) to use. Default is type 2.
    - norm (str): Normalization to use in DCT. Default is "ortho".
    - lifter (float): Liftering coefficient. Default is 0. Liftering applies a sinusoidal window to the LFCCs.

    Returns:
    - np.ndarray: Computed LFCCs of shape (n_lfcc, T), where T is the number of frames.
    """
    
    # Generate linear filter banks
    f_b = linear_filter_banks(nfilts=26,
                              nfft=1600,
                              fs=16000,
                              low_freq=None,
                              high_freq=None,
                              scale="constant")
    
    # Compute the Short-Time Fourier Transform (STFT) of the audio signal
    X = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    S = librosa.power_to_db(np.abs(X)**2)
    
    # Apply the filter banks to the power spectrogram
    features = np.dot(S.T, f_b.T)

    # Apply the Discrete Cosine Transform (DCT) to get LFCCs
    M: np.ndarray = scipy.fftpack.dct(features.T, axis=-2, type=dct_type, norm=norm)[..., :n_lfcc, :]
    
    # Apply liftering if required
    if lifter > 0:
        # Shape lifter for broadcasting
        LI = np.sin(np.pi * np.arange(1, 1 + n_lfcc, dtype=M.dtype) / lifter)
        LI = util.expand_to(LI, ndim=S.ndim, axes=-2)
        M *= 1 + (lifter / 2) * LI
        return M
    elif lifter == 0:
        return M
    else:
        raise ParameterError(f"Lifter value {lifter} must be a non-negative number.")
        
        
        
        
def feature_extraction(feature, sample_rate, n_fft, hop_length, dict_hives, hives, year, enhancement=False):
    """
    Extract various features from audio files.

    Parameters:
    - feature (str): Type of feature to extract. Options are:
        'mfccs', 'lfccs', 'spectral_shape_descriptors', 'nectar_hand_crafted'.
    - sample_rate (int): Sampling rate of the audio files.
    - n_fft (int): Number of FFT components.
    - hop_length (int): Number of samples between successive frames.
    - dict_hives (dict): Dictionary containing information about hives.
    - hives (list): List of hives to process.
    - year (int): Year of the audio files.
    - enhancement (bool): Whether to apply spectral subtraction for noise reduction. Default is False.

    Returns:
    - pd.DataFrame: DataFrame containing extracted features.
    """
    


    n = 0
    df = pd.DataFrame()

    
    shape_descriptors = ['centroid', 'spread', 'skewness', 'kurtosis', 'flatness', 'rolloff', 'crest','flux', 'entropy']
    
    for hive in hives: 

        print("--------------------------------- // Hive number {} //---------- -------".format(hive))

        ls_date = []

        for date in pd.date_range(start=dict_hives[hive].index.min(),
                                  end=dict_hives[hive].index.max(),
                                  freq='15min'): 
            if date not in ls_date:
                ls_date.append(date)

        file_names = data_ls_to_string(ls_date, hive)


        for i in range(len(file_names)):

            try:

                signal_audio, sample_rate = librosa.load(os.path.join("../data/Nectar/full_" + str(year), file_names[i]), 
                                                         sr=sample_rate)
                
                print("-------------- ------------// File {} //------------- -------".format(file_names[i]))
                         
                if enhancement:
                         
                         
                    signal_audio = spectral_subtraction(signal_audio, snr_threshold=1.5, window_time=30e-3,
                                                        sample_rate=sample_rate)    
                         
                df.loc[n, 'date'] =  ls_date[i]
                df.loc[n, 'tag'] = hive
                df.loc[n, 'fob'] = dict_hives[hive].loc[ls_date[i], "fob"]
                df.loc[n, 'raw_audio'] = np.mean(signal_audio)
                
                
                if feature == 'mfccs':
                    
                    MFCCs = librosa.feature.mfcc(y=signal_audio, sr=sample_rate, n_fft=n_fft,
                                                         hop_length=hop_length,
                                                         n_mfcc=13, n_mels=26)
                    
                    
                    for m in range(0,13):
                        df.loc[n, str('mfccs_' + str(m))] = np.mean(np.array(MFCCs)[m,:])
            
                
                elif feature == 'lfccs':
                    LFCCs = manual_lfcc(y=signal_audio, sr=sample_rate, n_fft=n_fft,
                                                         hop_length=hop_length,
                                                         n_lfcc=13)
                    
                    for m in range(0,13):
                        df.loc[n, str('lfccs_' + str(m))] = np.mean(np.array(LFCCs)[m,:])

                    
                elif feature == 'spectral_shape_descriptors':
                
                    for c in shape_descriptors:

                        df.loc[n, c] = spectral_descriptors(signal_audio, sample_rate, n_fft=2048, descriptor=c).mean()

                         
                elif feature == 'nectar_hand_crafted':
                
                    spect_1 = librosa.stft(signal_audio[1*60*15625:1*60*15625+29*512],  n_fft=512, hop_length=512, center=False,
                                           window='hann').T
                    spect_2 = librosa.stft(signal_audio[6*60*15625:6*60*15625+29*512],  n_fft=512, hop_length=512, center=False,
                                           window='hann').T
                    spect_3 = librosa.stft(signal_audio[11*60*15625:11*60*15625+29*512],  n_fft=512, hop_length=512,
                                           center=False,  window='hann').T




                    hive_power_1 = 10*np.log10((np.mean(np.sum(abs(spect_1[:, 4:18])**2, axis=1))))
                    hive_power_2 = 10*np.log10((np.mean(np.sum(abs(spect_2[:, 4:18])**2, axis=1))))
                    hive_power_3 = 10*np.log10((np.mean(np.sum(abs(spect_3[:, 4:18])**2, axis=1))))

                    df.loc[n,  'hive_power'] = np.mean([hive_power_1, hive_power_2, hive_power_3])


                    audio_band_density_1 = 10*np.log10((np.sum(np.sum(abs(spect_1[:, 4:18])**2, axis=1)))/420)
                    audio_band_density_2 = 10*np.log10((np.sum(np.sum(abs(spect_2[:, 4:18])**2, axis=1)))/420)
                    audio_band_density_3 = 10*np.log10((np.sum(np.sum(abs(spect_3[:, 4:18])**2, axis=1)))/420)

                    df.loc[n,  'audio_density'] = np.mean([audio_band_density_1, audio_band_density_2, audio_band_density_3])


                    audio_band_density_ratio_1 = np.sum(np.sum(abs(spect_1[:, 4:18])**2, axis=1)/np.sum(abs(spect_1[:,
                                                                                                                    4:257])**2,
                                                                                                        axis=1))/30
                    audio_band_density_ratio_2 = np.sum(np.sum(abs(spect_2[:, 4:18])**2, axis=1)/np.sum(abs(spect_2[:,
                                                                                                                    4:257])**2,
                                                                                                        axis=1))/30
                    audio_band_density_ratio_3 = np.sum(np.sum(abs(spect_3[:, 4:18])**2, axis=1)/np.sum(abs(spect_3[:,
                                                                                                                    4:257])**2,
                                                                                                        axis=1))/30

                    df.loc[n,  'audio_density_ratio'] = np.mean([audio_band_density_ratio_1, audio_band_density_ratio_2,
                                                                 
                                                                 audio_band_density_ratio_3])

                    audio_density_variation_1 = 10*np.log10(np.max(np.sum(abs(spect_1[:, 4:257])**2,
                                                                          axis=1))/np.min(np.sum(abs(spect_1[:, 4:257])**2,
                                                                                                 axis=1)))
                    audio_density_variation_2 = 10*np.log10(np.max(np.sum(abs(spect_2[:, 4:257])**2,
                                                                          
                                                                          axis=1))/np.min(np.sum(abs(spect_2[:, 4:257])**2,
                                                                                                 axis=1)))
                    audio_density_variation_3 = 10*np.log10(np.max(np.sum(abs(spect_3[:, 4:257])**2, 
                                                                          axis=1))/np.min(np.sum(abs(spect_3[:, 4:257])**2,
                                                                                                 axis=1)))

                    df.loc[n,  'density_variation'] = np.mean([audio_density_variation_1, audio_density_variation_2,
                                                               audio_density_variation_3])




                    for m in range(1,17):
                        power_1 = abs(spect_1[:, 4:20])**2
                        power_2 = abs(spect_2[:, 4:20])**2
                        power_3 = abs(spect_3[:, 4:20])**2

                        df.loc[n, str('f_' + str(m))] = np.mean([10*np.log10(np.sum(power_1[:,m-1])/30),
                                                                 10*np.log10(np.sum(power_2[:,m-1])/30), 
                                                                 10*np.log10(np.sum(power_3[:,m-1])/30)])

                n = n+1
                         
                         
                         
            except OSError as e:
                pass
            except ValueError:
                pass

    return df




def extract_statistic(df_all, f_columns, year, start, stop):
    """
    Extract statistical features from the given DataFrame.

    Parameters:
    - df_all (pd.DataFrame): DataFrame containing all data.
    - f_columns (list): List of feature columns to compute statistics for.
    - year (int): Year of the data.
    - start (str): Start time for filtering data.
    - stop (str): Stop time for filtering data.

    Returns:
    - pd.DataFrame: DataFrame containing extracted statistical features.
    """
    n = 0

    # Create an empty DataFrame to store extracted features
    df_features = pd.DataFrame(columns=['mean_' + str(i) for i in f_columns] + 
                                      ['std_' + str(i) for i in f_columns]+ 
                                      ['skew_' + str(i) for i in f_columns]+ 
                                      ['kurt_' + str(i) for i in f_columns]+ 
                                      ['box', 'month', 'date', 'tag', 'fob'])

    # Iterate over each hive
    for hive in df_all["tag"].unique():

        print(hive)
        
        # Resample the data to hourly frequency and drop NaN values
        df = df_all[df_all["tag"]==hive].resample('h').mean()
        df = df.dropna(how='all')

        # Iterate over each date in the DataFrame
        for date in np.unique(df.index.date):

            temp = df.loc[str(date)]
            if len(temp) > 23:
                temp = temp.between_time(start, stop)
                df_features.loc[n, 'fob'] = df.loc[str(date)]['fob'][0]
                df_features.loc[n, 'date'] = date
                df_features.loc[n, 'month'] = int(date.month)
                df_features.loc[n, 'tag'] = hive

                # Determine the 'box' value based on the fob
                if df.loc[str(date)]['fob'][0] <= 10:
                    df_features.loc[n, 'box'] = 1
                elif 10 < df.loc[str(date)]['fob'][0] <= 20:
                    df_features.loc[n, 'box'] = 2
                elif 20 < df.loc[str(date)]['fob'][0] <= 30:
                    df_features.loc[n, 'box'] = 3

                # Compute statistical features
                df_features.loc[n, :len(f_columns)] = temp[f_columns].mean().values
                df_features.loc[n, len(f_columns):2*len(f_columns)] = temp[f_columns].std().values
                df_features.loc[n, 2*len(f_columns):3*len(f_columns)] = temp[f_columns].skew().values
                df_features.loc[n, 3*len(f_columns):4*len(f_columns)] = temp[f_columns].apply(pd.DataFrame.kurt).values

                n = n+1

    # Convert 'fob' column to integer
    df_features['fob'] = df_features['fob'].astype(float).round(0).astype(int)
    # Convert other columns to float
    df_features.iloc[:, :4*len(f_columns)] = df_features.iloc[:, :4*len(f_columns)].astype('f') 

    return df_features

