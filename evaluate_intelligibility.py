"""
TTS objective intelligibility benchmark — a round-trip proxy for quality
that doesn't need human raters: synthesize text, transcribe the synthesized
audio with STT, and compute WER against the original input text. Low
round-trip WER means a listener (or at least a well-trained ASR model)
can recover exactly what was supposed to be said — genuine evidence of
intelligibility, though NOT a substitute for human-judged naturalness/MOS
(a robotic-sounding but clearly-enunciated clip could still score well
here). Q18 in Tecgrw/application/GAP_REMEDIATION_PLAN.md still needs a
real human MOS pass — this is a different, complementary signal, not that.

Reuses the 20 real TTS clips already synthesized for the MOS listening
pass (Tecgrw/application/Api_server/eval/mos_samples/ + its manifest) — one
per RAG health topic — rather than generating a new sample. Transcribes
with tecgrw_stt's fine-tuned checkpoint (our best available STT model) so
the number reflects Tekana's actual STT capability, not an artificially
weak one.

Usage:
    py -3 evaluate_intelligibility.py
"""
import csv
import os

import jiwer
import soundfile as sf
import torch
import torchaudio.transforms as T
from transformers import AutoProcessor, Wav2Vec2ForCTC

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.abspath(os.path.join(
    HERE, "..", "application", "Api_server", "eval", "mos_samples_manifest.csv"))
CLIPS_DIR = os.path.dirname(MANIFEST_PATH) + os.sep + "mos_samples"
STT_CHECKPOINT = os.path.join(HERE, "..", "tecgrw_stt", "runs", "adapter_v1")
STT_BASELINE = os.path.abspath(os.path.join(
    HERE, "..", "application", "stt_service", "Automatic_speech_recognition"))
TARGET_SR = 16000

NORMALIZE = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def load_stt(path, label):
    print(f"[intelligibility] Loading STT ({label}): {path}")
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(path, local_files_only=True, dtype=torch.float32)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return processor, model


def transcribe(processor, model, wav_path, resampler_cache):
    audio, sr = sf.read(wav_path, dtype="float32")
    if sr != TARGET_SR:
        if sr not in resampler_cache:
            resampler_cache[sr] = T.Resample(orig_freq=sr, new_freq=TARGET_SR)
        audio = resampler_cache[sr](torch.from_numpy(audio).unsqueeze(0)).squeeze(0).numpy()
    device = next(model.parameters()).device
    inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(ids)[0]


def main():
    if not os.path.isfile(MANIFEST_PATH):
        print(f"[intelligibility] ERROR: manifest not found at {MANIFEST_PATH} — "
              f"run Api_server/scripts/generate_mos_samples.py in the application repo first.")
        return

    use_finetuned = os.path.isdir(STT_CHECKPOINT)
    stt_path = STT_CHECKPOINT if use_finetuned else STT_BASELINE
    label = "tecgrw_stt fine-tuned adapter" if use_finetuned else "production baseline (fine-tuned checkpoint not found)"
    processor, model = load_stt(stt_path, label)

    rows = list(csv.DictReader(open(MANIFEST_PATH, encoding="utf-8")))
    print(f"[intelligibility] {len(rows)} clips (one per RAG topic)\n")

    resampler_cache = {}
    refs, hyps = [], []
    for row in rows:
        wav_path = os.path.join(CLIPS_DIR, row["wav_file"])
        if not os.path.isfile(wav_path):
            print(f"[intelligibility]   MISSING: {wav_path}, skipping")
            continue
        hyp = transcribe(processor, model, wav_path, resampler_cache)
        refs.append(row["text"])
        hyps.append(hyp)
        clip_wer = jiwer.wer(row["text"], hyp, reference_transform=NORMALIZE, hypothesis_transform=NORMALIZE)
        flag = "  <-- high error" if clip_wer > 0.3 else ""
        print(f"[{row['clip_id']}] WER={clip_wer:.2f}{flag}")
        print(f"  text: {row['text'][:70]}")
        print(f"  stt:  {hyp[:70]}")

    overall = jiwer.wer(refs, hyps, reference_transform=NORMALIZE, hypothesis_transform=NORMALIZE)
    print(f"\n[intelligibility] ===== RESULT =====")
    print(f"[intelligibility] STT used: {label}")
    print(f"[intelligibility] Clips: {len(refs)}")
    print(f"[intelligibility] Round-trip WER (TTS -> STT vs. original text): {overall:.4f} ({overall*100:.2f}%)")
    print("[intelligibility] Reminder: this is an intelligibility proxy, not naturalness/MOS — "
          "a real human listening pass is still the open item for that.")


if __name__ == "__main__":
    main()
