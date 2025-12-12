# ----------- TEST FILE -----------
# ->from TTS.api import TTS
# import torch
# from TTS.tts.configs.xtts_config import XttsConfig  # ← Import the class
# from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
# from TTS.config.shared_configs import BaseDatasetConfig
# from TTS.api import TTS

# --------------------------------------
import pyaudio, wave, datetime, keyboard, whisper, os
from scipy.io import wavfile
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from gtts import gTTS
from TTS.api import TTS
import torch

# ← ADD THIS LINE IN YOUR SCRIPT, BEFORE TTS()
# torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])

# Load XTTS-v2 (downloads ~2 GB on first run)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
# tts = TTS("tts_models/de/thorsten/tacotron2-DDC")

trial_data = [
    {'lang': 'en', 'text': 'Hello, this is a test of XTTS in English.'},
    {'lang': 'hi', 'text': 'नमस्ते, यह हिंदी में XTTS का परीक्षण है।'},
    {'lang': 'ja', 'text': 'こんにちは、これは日本語でのXTTSのテストです。'},
    {'lang': 'de', 'text': 'Hallo, dies ist ein Test von XTTS auf Deutsch.'},
    ]

translator = pipeline("translation",
                      model = "facebook/nllb-200-distilled-600M",
                      torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                      device=0 if torch.cuda.is_available() else -1
                      )

NLLB_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "de": "deu_Latn",
    "ja": "jpn_Jpan",
}

# translated = translator(
#     text, "ja", "hi"
# )[0]["translation_text"]

# print(translated)

# Synthesize 
for data in trial_data[3]:
    # tts.tts_to_file(
    #     text = data['text'],
    #     speaker_wav="20251031_185342.wav",   # any short .wav of the target voice
    #     language = data['lang'],
    #     file_path=f"test-{data['lang']}-output_final.wav"
    # )
    wav = tts.tts_to_file(
        text=data['text'],
        speaker_wav="Vinay.wav",
        language=data['lang'],
        file_path=f"test-{data['lang']}-output_final.wav",
        # gpt_cond_len=3,                # use first 3 seconds of speaker_wav (increase to 6–8 if clip is longer)
        # gpt_cond_chunk_len=3,          # important for stable cloning
        # temperature=0.65,              # slight variation = more natural
        # speed=1.0,
    )
    # write_wav(output_audio, 24000, np.array(wav, dtype=np.float32))

    # Send result audio for in-browser playback
    # return send_file(output_audio, mimetype="audio/wav")

    print(f"Audio saved: test-{data['lang']}-output_final.wav")