import numpy as np
import torch
from scipy.io import wavfile
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list
from kymatio import Scattering1D
from kymatio.torch import Scattering1D as Scattering1D_Torch

def read_wav(filename):
    rate, data = wavfile.read(filename)
    print(f"Read WAV file: {filename}, Rate: {rate}, Shape: {data.shape}")
    return rate, data

def write_wav(filename, rate, data):
    wavfile.write(filename, rate, data)
    print(f"Wrote WAV file: {filename}, Rate: {rate}, Shape: {data.shape}")

def wavelet_scattering(data, J=6, Q=1, device='cpu'):
    data = torch.tensor(data.astype(np.float32)).to(device)
    scattering = Scattering1D_Torch(J=J, shape=(data.shape[-1],), Q=Q).to(device)
    features = scattering(data)
    print(f"Scattering features shape: {features.shape}")
    return features.cpu().numpy()

def get_significant_features(features, percentage=10):
    flat_features = features.flatten()
    threshold = np.percentile(np.abs(flat_features), 100 - percentage)
    significant_indices = np.abs(flat_features) >= threshold
    significant_features = np.zeros_like(flat_features)
    significant_features[significant_indices] = flat_features[significant_indices]
    print(f"Significant features shape: {significant_features.shape}")
    return significant_features.reshape(features.shape)

def pad_to_equal_length(data1, data2):
    max_len = max(data1.shape[-1], data2.shape[-1])
    padded_data1 = np.pad(data1, ((0, 0), (0, max_len - data1.shape[-1])), 'constant')
    padded_data2 = np.pad(data2, ((0, 0), (0, max_len - data2.shape[-1])), 'constant')
    print(f"Padded data lengths: {padded_data1.shape}, {padded_data2.shape}")
    return padded_data1, padded_data2

def normalize_and_convert(data, original_max):
    data = data / np.max(np.abs(data)) * original_max
    return data.astype(np.int16)

def seriate(data):
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    dist_matrix = pdist(data)
    linkage_matrix = linkage(dist_matrix, method='single')
    order = leaves_list(linkage_matrix)
    return order

def reconstruct_signal(features, order, original_length):
    reordered_features = features[order]
    print(f"Reordered features shape: {reordered_features.shape}")
    if len(reordered_features) == 0 or len(reordered_features[0]) == 0:
        return np.zeros(original_length)
    reconstructed_signal = np.interp(
        np.arange(original_length),
        np.linspace(0, original_length, len(reordered_features)),
        reordered_features.mean(axis=0)
    )
    return reconstructed_signal

def process_chunk(left_chunk, right_chunk, J, Q, percentage, original_max, chunk_size, device='cpu'):
    left_features = wavelet_scattering(left_chunk, J=J, Q=Q, device=device)
    right_features = wavelet_scattering(right_chunk, J=J, Q=Q, device=device)
    
    left_significant = get_significant_features(left_features, percentage)
    right_significant = get_significant_features(right_features, percentage)
    
    left_padded, right_padded = pad_to_equal_length(left_significant, right_significant)
    
    left_padded = left_padded.flatten()
    right_padded = right_padded.flatten()
    
    left_padded = normalize_and_convert(left_padded, original_max)
    right_padded = normalize_and_convert(right_padded, original_max)

    left_order = seriate(left_padded)
    right_order = seriate(right_padded)

    left_reconstructed = reconstruct_signal(left_features, left_order, chunk_size)
    right_reconstructed = reconstruct_signal(right_features, right_order, chunk_size)
    
    output_chunk = np.column_stack((left_reconstructed, right_reconstructed))

    return output_chunk

def main(input_wav, output_wav, percentage=10, J=6, Q=1, chunk_size=2**14, device='cpu'):
    rate, data = read_wav(input_wav)
    
    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)
    
    left_channel = data[:, 0]
    right_channel = data[:, 1]
    
    original_max = np.iinfo(np.int16).max
    
    output_data = []
    
    for start in range(0, len(left_channel), chunk_size):
        end = min(start + chunk_size, len(left_channel))
        left_chunk = left_channel[start:end]
        right_chunk = right_channel[start:end]
        
        output_chunk = process_chunk(left_chunk, right_chunk, J, Q, percentage, original_max, chunk_size, device)
        output_data.append(output_chunk)
    
    output_data = np.vstack(output_data)
    
    write_wav(output_wav, rate, output_data)

if __name__ == "__main__":
    input_wav = "input.wav"
    output_wav = "wavscatter.wav"
    percentage = 100  # Percentage of most significant features to keep
    J = 6  # Scale parameter
    Q = 1  # Quality factor
    chunk_size = 2**16  # Adjust chunk size as needed
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(input_wav, output_wav, percentage, J, Q, chunk_size, device)
