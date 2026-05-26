from scipy.io import wavfile
import os

def check_audio(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    rate, data = wavfile.read(path)
    duration = len(data) / rate
    print(f"File: {os.path.basename(path)}")
    print(f"  Rate: {rate} Hz")
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Peak: {np.max(np.abs(data)) if len(data) > 0 else 'N/A'}")

import numpy as np
resource_dir = r"e:\vct\Voice-Cloned-Translator\resource"
check_audio(os.path.join(resource_dir, "8657a6d7-e739-4653-8831-fa981e4bbd95.wav"))
check_audio(os.path.join(resource_dir, "7d274020-bb4b-439d-80c5-adf39a080377.wav"))
