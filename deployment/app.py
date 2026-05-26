from flask import Flask, jsonify, render_template, request, redirect, url_for, send_file, flash, send_from_directory
import whisper, requests
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from TTS.api import TTS
import torch, os, tempfile, sys, shutil
from scipy.io import wavfile
from scipy.io.wavfile import write as write_wav
import numpy as np
import scipy.signal as signal  # Added for resampling
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
import time
from pydub import AudioSegment
import mimetypes

# Attempt to load Rust accelerator
try:
    import vct_rs
    RUST_ACCELERATOR = True
    print("Rust accelerator (vct_rs) loaded.\n")
except ImportError:
    RUST_ACCELERATOR = False
    print("Warning: Rust accelerator (vct_rs) not found. Falling back to Python defaults.")

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
whisper_type = "medium"
whisper_model = whisper.load_model(whisper_type)  # 'base' is much faster than 'small' on CPU
print(f"Whisper: {whisper_type} - model loaded on server.\n")

# TTS model (loaded once) - Agree to terms to avoid interactive prompt
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
print("TextToSpeech model loaded on server.\n")

OLLAMA_URL = "http://localhost:11434/api/generate"
# GEMMA_MODEL = "gemma3.5:2b" #gemma2:2b, phi4-mini:3.8b was acting poorly, qwen3:4b was over thinking.., gemma3:4b is very big, translategemma:4b
GEMMA_MODEL = "gemma4:31b-cloud"  # finally chosen gemma4:31b-cloud


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

print("Models are ready.\n")

# ----------------------------
# Important Functions
# ----------------------------
def convert_to_wav(input_path):
    """
    Convert audio file to WAV format if needed.
    Returns the path to a WAV file.
    Supports: WAV, MP3, M4A, OGG, FLAC, and other formats pydub can handle (with FFmpeg).
    Falls back to using librosa if pydub/FFmpeg is unavailable.
    """
    wav_path = input_path.replace(os.path.splitext(input_path)[1], ".wav")
    
    try:
        # Try to read as WAV first
        sr, data = wavfile.read(input_path)
        print(f"File is already a valid WAV (sample rate: {sr}Hz)")
        return input_path
    except (ValueError, Exception) as e:
        # Not a valid WAV, attempt conversion
        print(f"Input file is not a valid WAV. Attempting to convert from other formats...")
        print(f"Error details: {str(e)}")
        
        # Try pydub first (preferred if FFmpeg is available)
        try:
            print(f"Loading audio file with pydub...")
            audio = AudioSegment.from_file(input_path)
            print(f"Audio loaded successfully. Duration: {len(audio)/1000:.2f}s, Channels: {audio.channels}, Sample rate: {audio.frame_rate}Hz")
            
            # Ensure mono or stereo audio
            if audio.channels > 2:
                print(f"Converting from {audio.channels} channels to stereo...")
                audio = audio.set_channels(2)
            
            # Export to WAV with standard settings
            print(f"Exporting to WAV format...")
            audio.export(wav_path, format="wav")
            print(f"Successfully converted to WAV: {wav_path}")
            
            # Verify the converted WAV is readable
            sr_verify, data_verify = wavfile.read(wav_path)
            print(f"Verified converted WAV (sample rate: {sr_verify}Hz, samples: {len(data_verify)})")
            
            # Remove original if different
            if wav_path != input_path and os.path.exists(input_path):
                try:
                    os.remove(input_path)
                except:
                    pass
            
            return wav_path
        except (FileNotFoundError, Exception) as pydub_error:
            print(f"Pydub conversion failed ({type(pydub_error).__name__}): {str(pydub_error)}")
            print("Attempting fallback conversion with librosa...")
            
            # Fallback: Use librosa for format detection and conversion
            try:
                import librosa
                
                # Load audio with librosa (auto-detects format)
                print(f"Loading audio file with librosa...")
                data, sr = librosa.load(input_path, sr=None, mono=False)
                print(f"Audio loaded successfully. Sample rate: {sr}Hz, Shape: {data.shape}")
                
                # Convert to mono if stereo
                if len(data.shape) > 1 and data.shape[0] > 1:
                    print(f"Converting from {data.shape[0]} channels to mono...")
                    data = np.mean(data, axis=0)
                
                # Ensure float32 format for scipy wavfile
                if data.dtype != np.float32:
                    data = data.astype(np.float32)
                
                # Normalize audio to prevent clipping
                max_val = np.max(np.abs(data))
                if max_val > 1.0:
                    print(f"Normalizing audio (max value: {max_val:.2f})...")
                    data = data / max_val
                
                # Write WAV file
                print(f"Exporting to WAV format...")
                write_wav(wav_path, sr, data)
                print(f"Successfully converted to WAV: {wav_path}")
                
                # Verify the converted WAV is readable
                sr_verify, data_verify = wavfile.read(wav_path)
                print(f"Verified converted WAV (sample rate: {sr_verify}Hz, samples: {len(data_verify)})")
                
                # Remove original if different
                if wav_path != input_path and os.path.exists(input_path):
                    try:
                        os.remove(input_path)
                    except:
                        pass
                
                return wav_path
            except ImportError:
                raise ValueError(f"Unable to process audio file. Librosa not available for fallback conversion. Error: {type(pydub_error).__name__}: {str(pydub_error)}")
            except Exception as librosa_error:
                print(f"Error converting audio with librosa: {type(librosa_error).__name__}: {str(librosa_error)}")
                raise ValueError(f"Unable to process audio file. Supported formats: WAV, MP3, M4A, OGG, FLAC, WebM, and more. Error: {type(librosa_error).__name__}: {str(librosa_error)}")



# ----------------------------
# Routes (Flask)
# ----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/resource/<path:filename>')
def custom_resource(filename):
    # Serving from the 'resource' folder which is one level above 'deployment'
    # Using absolute path to ensure Windows compatibility
    resource_path = os.path.abspath(os.path.join(app.root_path, '..', 'resource'))
    print(f"Serving resource: {filename} from {resource_path}")
    return send_from_directory(resource_path, filename)

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Receive recorded audio blob from browser, translate + voice-clone it."""
    if 'audio' not in request.files:
        flash('Error 400: Error Receiving file', 'error')
        return "No audio received", 400

    source_lang = request.form.get('src-lang', 'en') 
    target_lang = request.form.get('tar-lang', 'hi') 
    print(f"Received request: source_lang={source_lang}, target_lang={target_lang}\n")
    audio_file = request.files['audio']

    if source_lang not in AYA_LANG or target_lang not in AYA_LANG:
        return "Unsupported language", 400

    # Save input audio temporarily
    temp_dir = tempfile.mkdtemp()
    input_audio = os.path.join(temp_dir, "input.wav")
    input_audio_clean = os.path.join(temp_dir, "input_clean.wav")
    output_audio = os.path.join(temp_dir, "output.wav")
    
    # Save the uploaded file with its original extension first
    temp_input_file = os.path.join(temp_dir, "input" + os.path.splitext(audio_file.filename)[1])
    audio_file.save(temp_input_file)
    
    # Convert to WAV if needed
    try:
        input_audio = convert_to_wav(temp_input_file)
    except ValueError as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        flash(f'Error: {str(e)}', 'error')
        return str(e), 400
    
    # Check duration for quality feedback
    try:
        sr_chk, data_chk = wavfile.read(input_audio)
        duration = len(data_chk) / sr_chk
        print(f"Input audio duration: {duration:.2f}s")
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        error_msg = f"Failed to read converted audio file: {type(e).__name__}: {str(e)}"
        print(f"ERROR: {error_msg}")
        return error_msg, 400

    # --- Preprocessing Block (Rust-optimized) ---
    global RUST_ACCELERATOR
    speaker_wav = input_audio 
    
    if RUST_ACCELERATOR:
        try:
            print("Using Rust accelerator for preprocessing...\n")
            start_rs = time.time()
            sample_rate, data = vct_rs.preprocess_audio(input_audio)
            print(f"Rust preprocessing took: {time.time() - start_rs:.4f}s")
            
            # Save the CLEANED version of the LIVE recording for cloning
            write_wav(input_audio_clean, sample_rate, data)
            speaker_wav = input_audio_clean
        except Exception as e:
            print(f"Rust accelerator failed: {e}. Falling back to Python.\n")
            RUST_ACCELERATOR = False 

    if not RUST_ACCELERATOR:
        print("Using Python fallback for preprocessing...")
        sample_rate, data = wavfile.read(input_audio)
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        if sample_rate != 16000:
            data = signal.resample(data, int(len(data) * 16000 / sample_rate))
        speaker_wav = input_audio

    # Transcribe
    print("Starting transcription...")
    start_whisper = time.time()
    result = whisper_model.transcribe(data, language = source_lang)
    text = result['text'].strip()
    print(f"Transcription took: {time.time() - start_whisper:.4f}s")
    print(f"Transcribed text: {text}")

    # Translate
    print("Starting translation...")
    start_trans = time.time()
    
    # Prompt Engineering
    prompt = f'''prompt = f"""Task: Translate the text below.
    Source Language: {AYA_LANG[source_lang]}
    Target Language: {AYA_LANG[target_lang]}
    Style: Natural, Kanji (Only) for Japanese.
    Constraint: Return ONLY the translated text. 
    Preserve all proper nouns and technical terms exactly. 
    Maintain the original sentence's tense and level of formality.

    Text: "{text}"
    Translation:'''

    # prompt = f'''You are an expert linguist translator. 
    # Translate the text below from {AYA_LANG[source_lang]} to {AYA_LANG[target_lang]}.

    # Guidelines:
    # 1. Preserve the original tone and level of formality.
    # 2. Ensure grammatical accuracy (specifically gender and honorifics in {AYA_LANG[target_lang]}).
    # 3. If a literal translation sounds unnatural, prioritize the "natural" equivalent in {AYA_LANG[target_lang]}.

    # Sentence: "{text}"
    # Translation:'''
    
    # API payload 
    payload = {
        "model": GEMMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.15,   #0.1
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
    
    wav = tts.tts(
        # text=translated_text,
        # speaker_wav=speaker_wav, 
        # language=XTTS_LANG[target_lang],
        # temperature=0.25,          # Lowered for maximum stability
        # repetition_penalty=12.0,   # Increased to prevent robotic loops
        # top_k=50,                  # Limits vocabulary to top 50
        # top_p=0.80,                # Nucleus sampling (slightly tighter)
        # gpt_cond_len=12,           # Ideal for clips between 5-15s
        # max_ref_len=12,            
        # enable_text_splitting=False # Set to False to avoid KeyError: 'hi' with Hindi
        text=translated_text,
        speaker_wav=speaker_wav,
        language=XTTS_LANG[target_lang],
        temperature=0.22,          # Lowered for maximum stability
        repetition_penalty=7.5,   # Increased to prevent robotic loops #12.5
        top_k=50,                  # Limits vocabulary to top 50
        top_p=0.80,                # Nucleus sampling (slightly tighter)
        gpt_cond_len=12,           # Ideal for clips between 5-15s # 12
        max_ref_len=15,            # Consistent with app.py for uniform prosody #15
        enable_text_splitting=False
    )
    
    write_wav(output_audio, 24000, np.array(wav, dtype=np.float32))
    print(f"XTTS synthesis took: {time.time() - start_xtts:.4f}s")

    response = send_file(output_audio, mimetype="audio/wav")
    response.call_on_close(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
    return response

    # Cleanup on processing failure before the response is created
    # try:
    #     shutil.rmtree(temp_dir, ignore_errors=True)
    # except Exception as e:
    #     print(e)


if __name__ == '__main__':
    # Disable reloader because loading models twice is too slow/memory intensive
    app.run(debug=True, use_reloader=False)
