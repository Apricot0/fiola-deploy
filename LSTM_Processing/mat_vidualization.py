import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def load_mat_files(data_dir):
    """
    Loads .mat files from the specified directory and separates them
    into light and dark categories based on the filename.
    """
    light_signals = []
    dark_signals = []
    light_filenames = []
    dark_filenames = []

    for file in os.listdir(data_dir):
        if file.endswith('.mat'):
            file_path = os.path.join(data_dir, file)
            # Load the .mat file
            mat_data = sio.loadmat(file_path)
            # Get keys that are not meta keys (i.e., keys not starting with '__')
            keys = [key for key in mat_data.keys() if not key.startswith('__')]
            if not keys:
                print(f"No data found in {file}. Skipping...")
                continue
            # Assume the first valid key contains the signal data
            signal = np.array(mat_data[keys[0]]).squeeze()  # Remove singleton dimensions
            # Separate signals based on filename
            if 'light' in file.lower():
                light_signals.append(signal)
                light_filenames.append(file)
            elif 'dark' in file.lower():
                dark_signals.append(signal)
                dark_filenames.append(file)
            else:
                print(f"File '{file}' does not specify 'light' or 'dark'. Skipping...")
    return (light_signals, light_filenames), (dark_signals, dark_filenames)

def plot_signal(signal, title="Signal"):
    """
    Plots the time series of the signal.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

def analyze_signal(signal, title="Signal Analysis"):
    """
    Performs three types of analysis on the signal:
      1. Time series plot.
      2. FFT amplitude spectrum (frequency domain).
      3. Histogram of the signal values.
    """
    n = len(signal)
    fft_vals = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(n, d=1)  # Assuming a sampling interval d=1. Adjust as needed.
    
    plt.figure(figsize=(15, 4))
    
    # 1. Time Series
    plt.subplot(1, 3, 1)
    plt.plot(signal)
    plt.title(f"{title} - Time Series")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # 2. FFT Amplitude Spectrum
    plt.subplot(1, 3, 2)
    # Only plot the positive half of frequencies for real signals
    half_n = n // 2
    plt.plot(fft_freq[:half_n], np.abs(fft_vals)[:half_n])
    plt.title(f"{title} - FFT Spectrum")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # 3. Histogram
    plt.subplot(1, 3, 3)
    plt.hist(signal, bins=30, edgecolor='black')
    plt.title(f"{title} - Value Distribution")
    plt.xlabel("Value")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.show()

def main():
    # Specify the directory where your .mat files are located
    data_dir = "/home/jasper/real-time-examples/Neuroscience/LSTM_Processing"  # <-- Change this to your folder path
    
    # Load and separate files into light and dark groups
    (light_signals, light_filenames), (dark_signals, dark_filenames) = load_mat_files(data_dir)
    
    print(f"Found {len(light_signals)} light recordings and {len(dark_signals)} dark recordings.")
    
    # -------------------------------
    # Visualize and Analyze Light Files
    # -------------------------------
    print("\nDisplaying Light Recordings:")
    for signal, fname in zip(light_signals, light_filenames):
        plot_signal(signal, title=f"Light - {fname}")
    
    print("\nAnalyzing Light Recordings:")
    for signal, fname in zip(light_signals, light_filenames):
        analyze_signal(signal, title=f"Light - {fname}")
    
    # -------------------------------
    # Visualize and Analyze Dark Files
    # -------------------------------
    print("\nDisplaying Dark Recordings:")
    for signal, fname in zip(dark_signals, dark_filenames):
        plot_signal(signal, title=f"Dark - {fname}")
    
    print("\nAnalyzing Dark Recordings:")
    for signal, fname in zip(dark_signals, dark_filenames):
        analyze_signal(signal, title=f"Dark - {fname}")

if __name__ == "__main__":
    main()
