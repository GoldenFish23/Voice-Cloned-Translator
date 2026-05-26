import json
import requests
import sys

OLLAMA_API = "http://localhost:11434"
OLLAMA_URL = f"{OLLAMA_API}/api/generate"
MODELS = [
    # "qwen2.5:3b",
    "qwen3:1.7b",
    "qwen3.5:2b",
    "gemma4:31b-cloud"
]

AYA_LANG = {
    "en": "English",
    "hi": "Hindi",
    "de": "German",
    "ja": "Japanese"
}

LANGUAGE_PAIRS = [
    ("en", "hi"),
    ("hi", "en"),
    ("en", "ja"),
    ("ja", "en"),
    ("hi", "ja"),
    ("ja", "hi"),
    ("en", "de"),
    ("de", "en")
]

BASE_SENTENCE = "Meine Mutter ist krank, ich brauche eine Pause."


def get_available_models() -> list[str]:
    try:
        response = requests.get(f"{OLLAMA_API}/api/models", timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            return [item.get("name") or item.get("model") or item for item in data if item]
        return []
    except Exception as exc:
        print(f"Warning: unable to query Ollama models: {exc}")
        return []


def build_prompt(source_lang: str, target_lang: str, text: str) -> str:
    source_name = AYA_LANG.get(source_lang, source_lang)
    target_name = AYA_LANG.get(target_lang, target_lang)

    return f'''prompt = f"""Task: Translate the text below.
    Source Language: {AYA_LANG[source_lang]}
    Target Language: {AYA_LANG[target_lang]}
    Style: Natural, Kanji (Only) for Japanese.
    Constraint: Return ONLY the translated text. 
    Preserve all proper nouns and technical terms exactly. 
    Maintain the original sentence's tense and level of formality.

    Text: "{text}"
    Translation:'''


def translate_text(model: str, source_lang: str, target_lang: str, text: str) -> str:
    prompt = build_prompt(source_lang, target_lang, text)
    payload = {
        "model": model,
        "prompt": prompt,
        "think": False,
        "stream": False,
        "options": {
            "temperature": 0.15,
            "num_predict": 100
        }
    }

    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"{response.status_code}: {response.text}")

    data = response.json()
    response_text = data.get("response")
    if not isinstance(response_text, str):
        response_text = str(response_text or "")

    response_text = response_text.strip()
    if not response_text:
        print(
            f"WARNING: empty translation from {model} {source_lang}->{target_lang}."
            f" Raw response: {json.dumps(data, ensure_ascii=False)}"
        )

    return response_text


def run_tests(models=None):
    models = models or MODELS
    available_models = get_available_models()
    if available_models:
        print(f"Available Ollama models: {available_models}")
    else:
        print("Warning: could not fetch available Ollama models; continuing with configured list.")

    for model in models:
        if available_models and model not in available_models:
            print(f"\n=== Model: {model} ===")
            print(f"SKIPPED: {model} is not currently available in Ollama.")
            continue

        print(f"\n=== Model: {model} ===")
        for source_lang, target_lang in LANGUAGE_PAIRS:
            try:
                translation = translate_text(model, source_lang, target_lang, BASE_SENTENCE)
                print(f"{source_lang}->{target_lang}: {translation}")
            except Exception as exc:
                print(f"{source_lang}->{target_lang}: ERROR: {exc}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        requested_models = sys.argv[1:]
        print(f"Testing requested model(s): {requested_models}")
        run_tests(requested_models)
    else:
        print("Testing all configured models. To test a single model, run: python test_tm.py qwen2.5:3b")
        run_tests()
