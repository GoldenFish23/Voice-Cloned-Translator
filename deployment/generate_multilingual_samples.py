import requests
from TTS.api import TTS
import torch
import os
import sys
import numpy as np
from scipy.io.wavfile import write as write_wav
import json
import argparse

# Configuration matching app.py
OLLAMA_URL = "http://localhost:11434/api/generate"
GEMMA_MODEL = "gemma4:31b-cloud"

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

def translate(text, source_lang, target_lang):
    if source_lang == target_lang:
        return text
    
    print(f"Translating: '{text}' from {AYA_LANG[source_lang]} to {AYA_LANG[target_lang]}...")
    
    prompt = f'''prompt = f"""Task: Translate the text below.
    Source Language: {AYA_LANG[source_lang]}
    Target Language: {AYA_LANG[target_lang]}
    Style: Natural, Kanji (Only) for Japanese.
    Constraint: Return ONLY the translated text. 
    Preserve all proper nouns and technical terms exactly. 
    Maintain the original sentence's tense and level of formality.

    Text: "{text}"
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
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        translated_text = result["response"].strip().strip('"')
        print(f"Translated to: '{translated_text}'")
        return translated_text
    except Exception as e:
        print(f"Translation error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate multilingual audio samples for a given English script.")
    parser.add_argument("--script", type=str, default="Welcome to the future of voice-cloned translation. This is simple reference audio.", 
                        help="The English sentence to translate and synthesize.")
    parser.add_argument("--reference_wav", type=str, 
                        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resource", "7d274020-bb4b-439d-80c5-adf39a080377.wav")),
                        help="Path to the reference audio for voice cloning.")
    parser.add_argument("--output_dir", type=str, 
                        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "static")),
                        help="Directory to save the generated wav files.")
    
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Initializing TTS model...")
    # XTTS v2 initialization
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    
    # We'll store the translations in a JSON for the UI to pick up if needed
    translations = {}

    for lang in XTTS_LANG.keys():
        print(f"\n--- Processing {AYA_LANG[lang]} ---")
        
        translated_text = translate(args.script, "en", lang)
        if not translated_text:
            continue
            
        translations[lang] = translated_text
        output_path = os.path.join(args.output_dir, f"sample_{lang}.wav")
        
        print(f"Synthesizing XTTS audio for {lang}...")
        try:
            wav = tts.tts(
                # text=translated_text,
                # speaker_wav=args.reference_wav,
                # language=XTTS_LANG[lang],
                # temperature=0.4,           # Increased for better prosody
                # repetition_penalty=3.0,    # Decreased for natural flow
                # top_k=50,
                # top_p=0.85,
                # gpt_cond_len=15,
                # max_ref_len=30,
                # enable_text_splitting=True if lang != 'hi' else False

                text=translated_text,
                speaker_wav=args.reference_wav,
                language=XTTS_LANG[lang],
                temperature=0.22,          # Lowered for maximum stability
                repetition_penalty=7.5,   # Increased to prevent robotic loops #12.5
                top_k=50,                  # Limits vocabulary to top 50
                top_p=0.80,                # Nucleus sampling (slightly tighter)
                gpt_cond_len=12,           # Ideal for clips between 5-15s # 12
                max_ref_len=15,            # Consistent with app.py for uniform prosody #15
                enable_text_splitting=False
            )
            
            write_wav(output_path, 24000, np.array(wav, dtype=np.float32))
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Synthesis error for {lang}: {e}")

    # Save translations to a file for easy UI access
    with open(os.path.join(args.output_dir, "samples.json"), "w", encoding="utf-8") as f:
        json.dump({"original": args.script, "translations": translations}, f, ensure_ascii=False, indent=2)
    print("\nGeneration complete. Translations saved to samples.json")

if __name__ == "__main__":
    main()
