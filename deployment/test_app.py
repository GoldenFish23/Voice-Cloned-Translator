from flask import Flask, jsonify, render_template, request, redirect, url_for, send_file, flash
import whisper, requests

# from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from TTS.api import TTS
import torch, os, tempfile, sys
from scipy.io import wavfile
from scipy.io.wavfile import write as write_wav
import numpy as np
import scipy.signal as signal  # Added for resampling
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
import time

# Attempt to load Rust accelerator
try:
    import vct_rs
    RUST_ACCELERATOR = True
    print("Rust accelerator (vct_rs) loaded.")
except ImportError:
    RUST_ACCELERATOR = False
    print("Warning: Rust accelerator (vct_rs) not found. Falling back to Python defaults.")

# from flask_cors import CORS

os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    import _locale
    _locale._getdefaultlocale = (lambda *args: ['en_US', 'utf-8'])

app = Flask(__name__)
app.secret_key = "secret_key"

# ----------------------------
# Model Initialization
# ----------------------------
print("Loading models...")

# Whisper model for transcription
whisper_model = whisper.load_model("small")  # tiny model for faster performance
print("Whisper model loaded on server.")

# TTS model (loaded once) - Agree to terms to avoid interactive prompt
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
# Note: Agreement is usually handled via environment variable or CLI. 
# If it still prompts, you may need to run 'set TTS_AGREEMENT=1' in your terminal.
print("TTS model loaded on server.")

OLLAMA_URL = "http://localhost:11434/api/generate"
GEMMA_MODEL = "gemma2:2b"

AYA_LANG = {
    "en": "English",
    "hi": "Hindi",
    "de": "German",
    "ja": "Japanese"
}

XTTS_LANG = {
    "en": "en",
    "hi": "hi",
    "de": "de",
    "ja": "ja"
}

print("Models are ready.")

# ----------------------------
# Routes
# ----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Receive recorded audio blob from browser, translate + voice-clone it."""
    if 'audio' not in request.files:
        flash('Error 400: Error Recieving file', 'error')
        return "No audio received", 400

    source_lang = request.form.get('src-lang', 'en') # set default to english language
    target_lang = request.form.get('tar-lang', 'hi') # set default to hindi language
    print(f"Received request: source_lang={source_lang}, target_lang={target_lang}")
    audio_file = request.files['audio']

    if source_lang not in AYA_LANG or target_lang not in AYA_LANG:
        return "Unsupported language", 400

    # Save input audio temporarily
    temp_dir = tempfile.mkdtemp()
    input_audio = os.path.join(temp_dir, "input.wav")
    input_audio_clean = os.path.join(temp_dir, "input_clean.wav")
    output_audio = os.path.join(temp_dir, "output.wav")
    audio_file.save(input_audio)
    
    # Check duration for quality feedback
    sr_chk, data_chk = wavfile.read(input_audio)
    duration = len(data_chk) / sr_chk
    print(f"Input audio duration: {duration:.2f}s")

    # --- Preprocessing Block (Rust-optimized) ---
    global RUST_ACCELERATOR
    speaker_wav = input_audio 
    
    if RUST_ACCELERATOR:
        try:
            print("Using Rust accelerator for preprocessing (Normalization + DC Offset Removal)...")
            start_rs = time.time()
            sample_rate, data = vct_rs.preprocess_audio(input_audio)
            print(f"Rust preprocessing took: {time.time() - start_rs:.4f}s")
            
            # Save the CLEANED version of the LIVE recording to use for cloning
            # This ensures the engine gets a professional-level signal even from a browser mic.
            write_wav(input_audio_clean, sample_rate, data)
            speaker_wav = input_audio_clean
        except Exception as e:
            print(f"Rust accelerator failed: {e}. Falling back to Python.")
            RUST_ACCELERATOR = False 

    if not RUST_ACCELERATOR:
        # Load audio file manually (Python fallback)
        print("Using Python fallback for preprocessing...")
        sample_rate, data = wavfile.read(input_audio)
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        if sample_rate != 16000:
            data = signal.resample(data, int(len(data) * 16000 / sample_rate))
        speaker_wav = input_audio

    # Transcribe (pass np.ndarray at 16kHz)
    print("Starting transcription...")
    start_whisper = time.time()
    result = whisper_model.transcribe(data, language = source_lang)
    text = result['text'].strip()
    print(f"Transcription took: {time.time() - start_whisper:.4f}s")
    print(f"Transcribed text: {text}")

    # Translate
    print("Starting translation...")
    start_trans = time.time()
    
    prompt = f'''You are a professional translator.
    Translate ONLY the sentence below from {AYA_LANG[source_lang]} to {AYA_LANG[target_lang]} without changing tone of sentence.
    Do not add any explanation or extra words.

    Sentence: "{text}"

    Translation:'''

    payload = {
        "model": GEMMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 100
        }
    }
    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        translated_text = "[Translation Error]"
    else:
        result = response.json()
        translated_text = result["response"].strip()
    
    print(f"Translation took: {time.time() - start_trans:.4f}s")
    print(f"Translated text: {translated_text}")

    # Generate cloned TTS (Ultra-Stable Parameters)
    print("Starting XTTS synthesis (Ultra-Stable Tuning)...")
    start_xtts = time.time()
    
    # These parameters are tuned to be more deterministic and "solid"
    # Lower temperature = less "hallucination" in the voice tone
    # Higher repetition penalty = prevents robotic/stuttering artifacts
    wav = tts.tts(
        text=translated_text,
        speaker_wav=speaker_wav, 
        language=XTTS_LANG[target_lang],
        temperature=0.25,          # Lowered for maximum stability
        repetition_penalty=12.0,   # Increased to prevent robotic loops
        top_k=50,                  # Limits vocabulary to top 50
        top_p=0.80,                # Nucleus sampling (slightly tighter)
        gpt_cond_len=12,           # Ideal for clips between 5-15s
        max_ref_len=12,            
        enable_text_splitting=False # Set to False to avoid KeyError: 'hi' with Hindi
    )
    
    # Save as 24kHz (XTTS native rate)
    write_wav(output_audio, 24000, np.array(wav, dtype=np.float32))
    print(f"XTTS synthesis took: {time.time() - start_xtts:.4f}s")

    # Send result audio for in-browser playback
    return send_file(output_audio, mimetype="audio/wav")


if __name__ == '__main__':
    # Disable reloader because loading models twice is too slow/memory intensive
    app.run(debug=True, use_reloader=False, port=5001)
