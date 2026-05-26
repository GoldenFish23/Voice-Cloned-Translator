use pyo3::prelude::*;
use numpy::{PyArray1, ToPyArray};
use hound; // wav loading
use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction}; // resampling

#[pyfunction]
fn preprocess_audio(py: Python, input_path: String) -> PyResult<(u32, Py<PyArray1<f32>>)> {
    // 1. Load Audio
    let mut reader = hound::WavReader::open(&input_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open wav: {}", e))
    })?;
    
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels;
    
    // 2. Read samples and Normalize to f32
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            reader.samples::<i16>()
                .map(|s| s.unwrap_or(0) as f32 / 32768.0)
                .collect()
        }
        hound::SampleFormat::Float => {
            reader.samples::<f32>()
                .map(|s| s.unwrap_or(0.0))
                .collect()
        }
    };

    // 3. Stereo to Mono conversion
    let mut mono_samples = if channels > 1 {
        let mut mono = Vec::with_capacity(samples.len() / channels as usize);
        for chunk in samples.chunks_exact(channels as usize) {
            let sum: f32 = chunk.iter().sum();
            mono.push(sum / channels as f32);
        }
        mono
    } else {
        samples
    };

    if mono_samples.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Audio file is empty"));
    }

    // 4. DC Offset Removal (Subtract Mean)
    let mean: f32 = mono_samples.iter().sum::<f32>() / mono_samples.len() as f32;
    for s in mono_samples.iter_mut() {
        *s -= mean;
    }

    // 5. Peak Normalization (Scale to 90% of max)
    let max_peak = mono_samples.iter()
        .map(|s| s.abs())
        .fold(0.0f32, f32::max);
    
    if max_peak > 0.0001 {
        let scale = 0.9 / max_peak;
        for s in mono_samples.iter_mut() {
            *s *= scale;
        }
    }

    // 6. Resample to 16kHz if necessary
    if sample_rate != 16000 {
        // Optimized Sinc parameters for speed (sinc_len: 128 is a good balance)
        let params = SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 128,
            window: WindowFunction::BlackmanHarris2,
        };
        
        let mut resampler = SincFixedIn::<f32>::new(
            16000.0 / sample_rate as f64,
            2.0, 
            params,
            mono_samples.len(),
            1,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Resampler error: {:?}", e)))?;

        let resampled = resampler.process(&[mono_samples], None).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Resampling failed: {:?}", e))
        })?;
        
        mono_samples = resampled[0].clone();
    }

    // Wrap in NumPy array for fast Python access
    let py_array = mono_samples.to_pyarray(py).to_owned();
    Ok((16000, py_array))
}

#[pymodule]
fn vct_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(preprocess_audio, m)?)?;
    Ok(())
}
