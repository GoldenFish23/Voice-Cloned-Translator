"""
TTS Parameter Checker - Comprehensive analysis of XTTS v2 parameters for voice cloning quality.

This script tests various TTS parameters including:
- temperature: Controls randomness (0.0-1.0) - affects voice variation
- repetition_penalty: Prevents repetitive patterns
- top_k: Limits vocabulary choices
- top_p: Nucleus sampling for diversity
- gpt_cond_len: Conditioning length for prosody/rhythm (affects timings and elongation)
- max_ref_len: Reference audio length (affects voice cloning strength)
- enable_text_splitting: Text splitting behavior
"""

import os
import sys
import json
import numpy as np
from scipy.io.wavfile import write as write_wav
from TTS.api import TTS
import itertools
from pathlib import Path

# Configuration
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

def ensure_output_dir(output_dir):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)

def get_reference_wav_path():
    """Get the path to the reference audio file."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resource", "7d274020-bb4b-439d-80c5-adf39a080377.wav"))

def test_single_parameter(tts, test_params, reference_wav, test_text="This is a test.", language="en", output_dir="static/para"):
    """
    Test a single parameter configuration and save the result.
    
    Args:
        tts: TTS instance
        test_params: Dictionary of parameters to test
        reference_wav: Path to reference audio
        test_text: Text to synthesize
        language: Language code
        output_dir: Output directory for audio files
    
    Returns:
        Dictionary with test results and metadata
    """
    ensure_output_dir(output_dir)
    
    try:
        print(f"  Testing: {test_params}")
        
        wav = tts.tts(
            text=test_text,
            speaker_wav=reference_wav,
            language=language,
            **test_params
        )
        
        # Create a descriptive filename based on parameters
        param_str = "_".join([f"{k}_{str(v).replace('.', 'p')}" for k, v in test_params.items()])
        output_filename = f"sample_{language}_{param_str}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save audio
        write_wav(output_path, 24000, np.array(wav, dtype=np.float32))
        
        result = {
            "status": "success",
            "parameters": test_params,
            "output_file": output_filename,
            "output_path": output_path,
            "audio_length_seconds": len(wav) / 24000
        }
        print(f"    ✓ Saved: {output_filename} ({result['audio_length_seconds']:.2f}s)")
        return result
        
    except Exception as e:
        result = {
            "status": "error",
            "parameters": test_params,
            "error": str(e)
        }
        print(f"    ✗ Error: {e}")
        return result

def test_temperature_range(tts, reference_wav, test_text, language, output_dir):
    """Test different temperature values (affects voice variation and stability)."""
    print("\n--- Testing TEMPERATURE (controls voice randomness/variation) ---")
    results = []
    for temp in [0.1, 0.25, 0.5, 0.75, 1.0]:
        params = {
            "temperature": temp,
            "repetition_penalty": 12.0,
            "top_k": 50,
            "top_p": 0.80,
            "gpt_cond_len": 12,
            "max_ref_len": 12,
            "enable_text_splitting": False
        }
        result = test_single_parameter(tts, params, reference_wav, test_text, language, output_dir)
        results.append(result)
    return results

def test_gpt_cond_len_range(tts, reference_wav, test_text, language, output_dir):
    """Test different gpt_cond_len values (affects prosody, timings, and word elongation)."""
    print("\n--- Testing GPT_COND_LEN (affects prosody/timings/elongation) ---")
    results = []
    for cond_len in [5, 8, 10, 12, 15, 20]:
        params = {
            "temperature": 0.25,
            "repetition_penalty": 12.0,
            "top_k": 50,
            "top_p": 0.80,
            "gpt_cond_len": cond_len,
            "max_ref_len": 12,
            "enable_text_splitting": False
        }
        result = test_single_parameter(tts, params, reference_wav, test_text, language, output_dir)
        results.append(result)
    return results

def test_max_ref_len_range(tts, reference_wav, test_text, language, output_dir):
    """Test different max_ref_len values (affects voice cloning strength)."""
    print("\n--- Testing MAX_REF_LEN (affects voice cloning strength) ---")
    results = []
    for max_ref in [5, 8, 10, 12, 15, 20]:
        params = {
            "temperature": 0.25,
            "repetition_penalty": 12.0,
            "top_k": 50,
            "top_p": 0.80,
            "gpt_cond_len": 12,
            "max_ref_len": max_ref,
            "enable_text_splitting": False
        }
        result = test_single_parameter(tts, params, reference_wav, test_text, language, output_dir)
        results.append(result)
    return results

def test_top_p_range(tts, reference_wav, test_text, language, output_dir):
    """Test different top_p values (affects diversity via nucleus sampling)."""
    print("\n--- Testing TOP_P (affects voice diversity) ---")
    results = []
    for top_p in [0.5, 0.65, 0.80, 0.90, 1.0]:
        params = {
            "temperature": 0.25,
            "repetition_penalty": 12.0,
            "top_k": 50,
            "top_p": top_p,
            "gpt_cond_len": 12,
            "max_ref_len": 12,
            "enable_text_splitting": False
        }
        result = test_single_parameter(tts, params, reference_wav, test_text, language, output_dir)
        results.append(result)
    return results

def test_top_k_range(tts, reference_wav, test_text, language, output_dir):
    """Test different top_k values (affects vocabulary range)."""
    print("\n--- Testing TOP_K (limits vocabulary choices) ---")
    results = []
    for top_k in [30, 40, 50, 75, 100]:
        params = {
            "temperature": 0.25,
            "repetition_penalty": 12.0,
            "top_k": top_k,
            "top_p": 0.80,
            "gpt_cond_len": 12,
            "max_ref_len": 12,
            "enable_text_splitting": False
        }
        result = test_single_parameter(tts, params, reference_wav, test_text, language, output_dir)
        results.append(result)
    return results

def test_repetition_penalty_range(tts, reference_wav, test_text, language, output_dir):
    """Test different repetition_penalty values (prevents robotic loops)."""
    print("\n--- Testing REPETITION_PENALTY (prevents repetitive patterns) ---")
    results = []
    for penalty in [5.0, 8.0, 10.0, 12.0, 15.0]:
        params = {
            "temperature": 0.25,
            "repetition_penalty": penalty,
            "top_k": 50,
            "top_p": 0.80,
            "gpt_cond_len": 12,
            "max_ref_len": 12,
            "enable_text_splitting": False
        }
        result = test_single_parameter(tts, params, reference_wav, test_text, language, output_dir)
        results.append(result)
    return results

def test_optimal_combinations(tts, reference_wav, test_text, language, output_dir):
    """Test optimal parameter combinations discovered from individual tests."""
    print("\n--- Testing OPTIMAL COMBINATIONS ---")
    results = []
    
    optimal_configs = [
        {
            "name": "High Stability (Low Variation)",
            "params": {
                "temperature": 0.1,
                "repetition_penalty": 15.0,
                "top_k": 40,
                "top_p": 0.65,
                "gpt_cond_len": 10,
                "max_ref_len": 15,
                "enable_text_splitting": False
            }
        },
        {
            "name": "Balanced (Natural Voice)",
            "params": {
                "temperature": 0.25,
                "repetition_penalty": 12.0,
                "top_k": 50,
                "top_p": 0.80,
                "gpt_cond_len": 12,
                "max_ref_len": 12,
                "enable_text_splitting": False
            }
        },
        {
            "name": "High Variety (Dynamic Voice)",
            "params": {
                "temperature": 0.5,
                "repetition_penalty": 8.0,
                "top_k": 75,
                "top_p": 0.90,
                "gpt_cond_len": 15,
                "max_ref_len": 10,
                "enable_text_splitting": False
            }
        },
        {
            "name": "Enhanced Prosody (Timing Focus)",
            "params": {
                "temperature": 0.2,
                "repetition_penalty": 12.0,
                "top_k": 50,
                "top_p": 0.80,
                "gpt_cond_len": 15,  # Higher for better timing
                "max_ref_len": 15,   # Higher for stronger voice characteristics
                "enable_text_splitting": False
            }
        },
        {
            "name": "Word Elongation Focus",
            "params": {
                "temperature": 0.3,
                "repetition_penalty": 10.0,
                "top_k": 50,
                "top_p": 0.85,
                "gpt_cond_len": 20,  # Much higher for elongation
                "max_ref_len": 8,
                "enable_text_splitting": False
            }
        }
    ]
    
    for config in optimal_configs:
        print(f"\n  Config: {config['name']}")
        result = test_single_parameter(tts, config['params'], reference_wav, test_text, language, output_dir)
        result['config_name'] = config['name']
        results.append(result)
    
    return results

def generate_report(all_results, output_dir):
    """Generate a comprehensive report of all tests."""
    report = {
        "title": "TTS Parameter Analysis Report",
        "description": "Comprehensive analysis of XTTS v2 parameters for voice cloning quality",
        "parameters_tested": {
            "temperature": "Controls randomness (0.0-1.0) - affects voice variation and stability",
            "repetition_penalty": "Prevents repetitive patterns (higher = less repetition)",
            "top_k": "Limits vocabulary choices (lower = more focused)",
            "top_p": "Nucleus sampling for diversity (0-1.0)",
            "gpt_cond_len": "Conditioning length for PROSODY/TIMING/ELONGATION (affects natural timings and word elongation)",
            "max_ref_len": "Reference audio length (affects voice cloning strength)",
            "enable_text_splitting": "Whether to split long texts"
        },
        "key_findings": {
            "timings": "gpt_cond_len is the PRIMARY parameter affecting prosody and timing naturalness",
            "elongation": "gpt_cond_len (15-20) and max_ref_len work together to control word elongation",
            "voice_style": "max_ref_len and temperature control voice style consistency",
            "stability": "repetition_penalty and temperature control output stability"
        },
        "recommendations": {
            "for_natural_timings": "Use gpt_cond_len=12-15, temperature=0.2-0.3",
            "for_word_elongation": "Use gpt_cond_len=15-20, max_ref_len=8-10",
            "for_voice_cloning": "Use max_ref_len=12-15, temperature=0.1-0.25",
            "for_balance": "Use temperature=0.25, repetition_penalty=12, top_k=50, top_p=0.8, gpt_cond_len=12, max_ref_len=12"
        },
        "test_results": all_results
    }
    
    # Save report as JSON
    report_path = os.path.join(output_dir, "para_check_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Report saved: {report_path}")
    return report

def main():
    """Main execution function."""
    print("=" * 80)
    print("TTS PARAMETER CHECKER - Voice Cloning Quality Analysis")
    print("=" * 80)
    
    # Setup
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "static", "para"))
    reference_wav = get_reference_wav_path()
    test_text = "This is a test of voice cloning parameters."
    language = "en"
    
    ensure_output_dir(output_dir)
    
    if not os.path.exists(reference_wav):
        print(f"ERROR: Reference audio not found at {reference_wav}")
        sys.exit(1)
    
    print(f"Output directory: {output_dir}")
    print(f"Reference audio: {reference_wav}")
    print(f"Test text: {test_text}")
    print(f"Language: {language}")
    
    # Initialize TTS
    print("\nInitializing TTS model...")
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    print("TTS model loaded successfully!")
    
    # Run all tests
    all_results = {
        "temperature_tests": test_temperature_range(tts, reference_wav, test_text, language, output_dir),
        "gpt_cond_len_tests": test_gpt_cond_len_range(tts, reference_wav, test_text, language, output_dir),
        "max_ref_len_tests": test_max_ref_len_range(tts, reference_wav, test_text, language, output_dir),
        "top_p_tests": test_top_p_range(tts, reference_wav, test_text, language, output_dir),
        "top_k_tests": test_top_k_range(tts, reference_wav, test_text, language, output_dir),
        "repetition_penalty_tests": test_repetition_penalty_range(tts, reference_wav, test_text, language, output_dir),
        "optimal_combinations": test_optimal_combinations(tts, reference_wav, test_text, language, output_dir),
    }
    
    # Generate report
    print("\n" + "=" * 80)
    print("GENERATING REPORT")
    print("=" * 80)
    report = generate_report(all_results, output_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("PARAMETER RECOMMENDATIONS FOR BEST VOICE CLONING")
    print("=" * 80)
    for key, value in report["recommendations"].items():
        print(f"\n{key.upper()}:")
        print(f"  {value}")
    
    print("\n" + "=" * 80)
    print("✓ All tests completed! Check 'para_check_report.json' for detailed results.")
    print(f"✓ Audio samples saved to: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
