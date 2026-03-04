

from google.colab import drive
drive.mount('/content/drive')

!pip install -U transformers accelerate datasets --quiet

!pip install -q transformers accelerate torch librosa soundfile sentencepiece

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "openai/whisper-large-v3"

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True
).to(device)

model.eval()

print("✅ Model loaded and ready")

import torch
from transformers import AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

processor = AutoProcessor.from_pretrained("openai/whisper-large-v2")
print("✅ Processor loaded")

!pip uninstall -y peft transformers accelerate datasets torchaudio
!pip install --no-cache-dir \
  transformers==4.35.2 \
  accelerate==0.24.1 \
  datasets==2.14.6 \
  torchaudio \
  jiwer

# =====================================================
# WHISPER LARGE V3 - FINAL STABLE TAMIL TRANSCRIBER
# =====================================================

import os
import re
import gc
import torch
import librosa
import soundfile as sf
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ======================
# PATHS (UNCHANGED)
# ======================
MODEL_NAME = "openai/whisper-large-v3"
DATA_PATH = "/content/drive/MyDrive/TamilDialect/Test"
OUTPUT_FILE = "/content/drive/MyDrive/TamilDialect/outputs/final_output.txt"

# ======================
# GPU CLEANUP
# ======================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# ======================
# LOAD MODEL
# ======================
print("Loading Whisper model...")

processor = WhisperProcessor.from_pretrained(MODEL_NAME)

model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    low_cpu_mem_usage=True
).to(device)

model.eval()
print("✅ Whisper Large v3 loaded")

# ======================
# AUDIO LOADER (FAST + SAFE)
# ======================
def load_audio(path):
    try:
        audio, sr = librosa.load(path, sr=16000, mono=True)
    except Exception:
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    if audio is None or len(audio) < 1600:
        raise ValueError("Empty or invalid audio")

    audio = audio.astype(np.float32)
    audio = audio / (np.max(np.abs(audio)) + 1e-7)
    return audio

def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

# ======================
# PROCESS FILES
# ======================
files = sorted([f for f in os.listdir(DATA_PATH) if f.lower().endswith(".wav")])
print("Found files:", len(files))

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for wav in files:
        try:
            path = os.path.join(DATA_PATH, wav)
            audio = load_audio(path)

            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device)

            if device == "cuda":
                input_features = input_features.half()

            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    forced_decoder_ids=processor.get_decoder_prompt_ids(
                        language="ta",
                        task="transcribe"
                    ),
                    max_new_tokens=256,
                    num_beams=3,              # Better Tamil accuracy
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3
                )

            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            text = clean_text(text)

            print(f"{wav} => {text}")
            out.write(f"{wav}|{text}\n")

            del audio, inputs, input_features, predicted_ids
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"{wav} => SKIPPED ({str(e)[:60]})")
            out.write(f"{wav}|\n")

print("\n✅ DONE")
print("Saved to:", OUTPUT_FILE)
