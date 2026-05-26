# best_xtts_clone_fixed.py
# Fixed for latest TTS library (Dec 2025) — removes invalid 'sample_rate' param
# Do not run the supporting reference file is misisng.

from TTS.api import TTS
import librosa
import soundfile as sf
import os

# -------------------------- LOAD MODEL ON CPU --------------------------
print("Loading XTTS v2 on CPU (this takes 15–40 seconds first time)...")
tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    gpu=False,                    # CPU mode
    progress_bar=True
)

# -------------------------- CLEAN REFERENCE AUDIO (still critical!) --------------------------
def prepare_reference(reference_path):
    print("Cleaning reference audio for maximum cloning accuracy...")
    audio, sr = librosa.load(reference_path, sr=24000, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=25)     # Remove silence
    audio = librosa.util.normalize(audio) * 0.95         # Peak normalize
    clean_path = "ref_clean.wav"
    sf.write(clean_path, audio, 24000)
    print(f"Clean reference ready: {clean_path}")
    return clean_path

# -------------------------- BEST CPU INFERENCE SETTINGS (FIXED) --------------------------
def generate_best_clone_cpu(
    text,
    reference_wav,
    language="en",
    output_path="best_clone_fixed.wav"
):
    ref = prepare_reference(reference_wav)

    print("Generating speech (this will take 10–60 seconds depending on text length and CPU)...")
    
    tts.tts_to_file(
        text=text,
        speaker_wav=ref,
        language=language,
        file_path=output_path,
        
        # Core parameters for perfect clones (these are safe/valid)
        temperature=0.30,
        repetition_penalty=12.0,     # ← Critical for no stuttering
        top_k=50,
        top_p=0.8,
        length_penalty=1.0,
        speed=1.0,
        
        # Extra stability for CPU (valid in latest XTTS)
        gpt_cond_len=30,
        max_ref_len=15,
        enable_text_splitting=True,
        
        # FIXED: Removed 'sample_rate' — it's not needed/valid here
    )
    
    print(f"Best possible CPU clone saved: {output_path}")
    print("Play it — you will be shocked how good it is even without GPU!")

# -------------------------- RUN IT --------------------------
if __name__ == "__main__":
    reference = "ttsmaker-file-2025-12-7-21-28-40.wav"   # ← Put your 6–15 second clean recording here
    
    text = """
    Hello, this is the most realistic voice clone you've ever heard.
    The clarity, intonation, and natural breathing are preserved perfectly.
    This is what peak XTTS v2 performance sounds like in 2025.
    """
    # print(f"{text}")
    generate_best_clone_cpu(
        text=text,
        reference_wav=reference,
        language="en",
        output_path="ultra_realistic_cpu_clone_fixed.wav"
    )
