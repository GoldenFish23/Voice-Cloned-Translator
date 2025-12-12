from flask import Flask, jsonify, render_template, request, redirect, url_for, send_file, flash
import whisper, requests

# from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, pipeline
from TTS.api import TTS
import torch, os, tempfile, sys
from scipy.io import wavfile
from scipy.io.wavfile import write as write_wav
import numpy as np
import scipy.signal as signal  # Added for resampling
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
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

# TTS model (loaded once)
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
print("TTS model loaded on server.")

OLLAMA_URL = "http://localhost:11434/api/generate"
GEMMA_MODEL = "gemma2:2b"
# Cache for translation models (to avoid reloading for each request)
# model_name = "CohereForAI/aya-23-8B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#     device_map="auto" if torch.cuda.is_available() else None
# )
# translator = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#     device=0 if torch.cuda.is_available() else -1
# )
# translator = pipeline("translation",
#                       model = "facebook/nllb-200-distilled-600M",
#                       torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#                       device=0 if torch.cuda.is_available() else -1
# )
AYA_LANG = {
    "en": "English",
    "hi": "Hindi",
    "de": "German",
    "ja": "Japanese"
}

NLLB_LANG = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "de": "deu_Latn",
    "ja": "jpn_Jpan"
}

XTTS_LANG = {
    "en": "en",
    "hi": "hi",
    "de": "de",
    "ja": "ja"
}
# translators = {}

print("Models are ready.")

# ----------------------------
# Routes
# ----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Receive recorded audio blob from browser, translate + voice-clone it."""
    if 'audio' not in request.files:
        flash('Error 400: Error Recieving file', 'error')
        return "No audio received", 400

    source_lang = request.form.get('src-lang', 'en') # set default to english language
    target_lang = request.form.get('tar-lang', 'hi') # set default to hindi language
    print(f"Recieved request: source_lang={source_lang}, target_lang={target_lang}")
    audio_file = request.files['audio']

    if source_lang not in NLLB_LANG or target_lang not in NLLB_LANG:
        return "Unsupported language", 400
    # Load translation model on demand (cache if already loaded)
    # model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    # if model_name not in translators:
    #     print(f"Loading translation model for {source_lang} to {target_lang}...")
    #     translator_model = MarianMTModel.from_pretrained(model_name)
    #     translator_tokenizer = MarianTokenizer.from_pretrained(model_name)
    #     translators[model_name] = (translator_model, translator_tokenizer)
    #     print(f"Translation model for {target_lang} loaded.")
    # translator_model, translator_tokenizer = translators[model_name]

    # Save input audio temporarily
    temp_dir = tempfile.mkdtemp()
    input_audio = os.path.join(temp_dir, "input.wav")
    output_audio = os.path.join(temp_dir, "output.wav")
    audio_file.save(input_audio)

    # Load audio file manually (no ffmpeg)
    sample_rate, data = wavfile.read(input_audio)
    # Convert to float32 if needed (normalized to [-1, 1])
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max

    # If stereo, convert to mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Resample to 16kHz if necessary (Whisper expects 16kHz)
    if sample_rate != 16000:
        # num_samples = int(len(data) * 16000 / sample_rate)
        # data = signal.resample(data, num_samples)
        data = signal.resample(data, int(len(data) * 16000 / sample_rate))

    # Transcribe (pass np.ndarray at 16kHz)
    result = whisper_model.transcribe(data, language = source_lang)
    text = result['text'].strip()
    print(f"Transcribed text", text)

    # Translate
    # tokens = translator_tokenizer(text, return_tensors="pt", padding=True)
    # translated_tokens = translator_model.generate(**tokens)
    # translated_text = translator_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    
    # translated_text = translator(
    #     text,
    #     src_lang=NLLB_LANG[source_lang],
    #     tgt_lang=NLLB_LANG[target_lang],
    #     max_length=200
    # )[0]["translation_text"]

    # prompt = f"Translate the following from {AYA_LANG[source_lang]} to {AYA_LANG[target_lang]}: {text}\nTranslation:"
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
    # result = translator(prompt, max_new_tokens=100, num_beams=5, do_sample=False)
    # translated_text = result[0]["generated_text"].split("Translation:")[-1].strip()
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
    else:
        result = response.json()
        # print("Full response:", result)
        translated_text= result["response"].strip()
    # print("Translated :", translated)
    print(f"Translated text: {translated_text}")

    # Generate cloned TTS
    # tts.tts_to_file(
    #     text=translated_text,
    #     speaker_wav=input_audio,
    #     file_path=output_audio,
    #     # language=target_lang
    #     language=XTTS_LANG[target_lang],
    # )
    wav = tts.tts(
        text=translated_text,
        speaker_wav=input_audio,
        language=XTTS_LANG[target_lang]
    )
    write_wav(output_audio, 24000, np.array(wav, dtype=np.float32))
    # tts.tts_to_file(
    #         text=translated_text,
    #         speaker_wav=input_audio,           # Recorded voice
    #         language=XTTS_LANG[target_lang],
    #         file_path=output_audio,

    #         # THESE ARE THE GOLDEN PARAMETERS (2025 community best)
    #         # temperature=0.30,
    #         # repetition_penalty=12.0,           
    #         # top_k=50,
    #         # top_p=0.8,
    #         # length_penalty=1.0,

    #         # # Style & prosody magic
    #         gpt_cond_len=30,                   
    #         max_ref_len=15,                    
    #         speed=1.0,
    #         temperature=0.35,          
    #         length_penalty=1.0,
    #         repetition_penalty=10.0,   
    #         top_k=50,
    #         top_p=0.85,                

    #         # # Style transfer magic
    #         # gpt_cond_len=30,           
    #         # max_ref_len=12,            
    #         # speed=1.0,
    #         # enable_text_splitting=True
    #         enable_text_splitting=False
    #     )
    # Send result audio for in-browser playback
    return send_file(output_audio, mimetype="audio/wav")


if __name__ == '__main__':
    app.run(debug=True)