import time
import numpy as np
from scipy.io import wavfile
import scipy.signal as signal
import os
import tempfile

# We will try to import vct_rs. 
# Note: This will only work after 'cargo build --release' and renaming the dll/so.
try:
    import vct_rs
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("Warning: vct_rs not found. Please compile the Rust module first.")

def create_dummy_audio(duration=5, sample_rate=44100, stereo=True):
    t = np.linspace(0, duration, int(sample_rate * duration))
    if stereo:
        l = np.sin(2 * np.pi * 440 * t)
        r = np.sin(2 * np.pi * 880 * t)
        data = np.stack([l, r], axis=1)
    else:
        data = np.sin(2 * np.pi * 440 * t)
    
    data = (data * 32767).astype(np.int16)
    temp_file = os.path.join(tempfile.gettempdir(), "test_input.wav")
    wavfile.write(temp_file, sample_rate, data)
    return temp_file

def python_preprocess(file_path):
    start = time.time()
    sample_rate, data = wavfile.read(file_path)
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    if sample_rate != 16000:
        data = signal.resample(data, int(len(data) * 16000 / sample_rate))
    return time.time() - start

def rust_preprocess(file_path):
    if not RUST_AVAILABLE:
        return None
    start = time.time()
    rate, data = vct_rs.preprocess_audio(file_path)
    return time.time() - start

if __name__ == "__main__":
    dummy_file = create_dummy_audio(duration=5) # 5 seconds
    
    print("Benchmarking Python...")
    py_time = python_preprocess(dummy_file)
    print(f"Python time: {py_time:.4f}s")
    
    if RUST_AVAILABLE:
        print("Benchmarking Rust...")
        rs_time = rust_preprocess(dummy_file)
        print(f"Rust time: {rs_time:.4f}s")
        print(f"Speedup: {py_time/rs_time:.2f}x")
    else:
        print("\nTo test Rust:")
        print("1. Ensure maturin is installed: pip install maturin")
        print("2. Run: maturin develop")
        print("3. Run this script again.")
        
    os.remove(dummy_file)
