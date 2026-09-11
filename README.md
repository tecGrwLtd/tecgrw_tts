# Tecgrw Kinyarwanda TTS (DeepKIN)

This repository contains the pipeline for training and evaluating a Kinyarwanda Text-to-Speech model based on DeepKIN (MB-iSTFT-VITS2).

## Project Structure

- `train_deepkin.ipynb`: Main notebook for training. Now configured for 1.2M iterations.
- `run_synthesis.py`: Test synthesis script. Uses Speaker ID 0 (finetuned).
- `batch_evaluate.py`: Tool for synthesizing multiple sentences from a text file.
- `tonal_marker_prototype.py`: Experimental tool for adding "Amasaku" (tonal marks) to text for better prosody.
- `requirements.txt`: Core dependencies for the project.

## Tonal Improvements ("Amasaku")

Kinyarwanda is a tonal language. Standard text often omits tonal marks, which can lead to unnatural speech. We have extended the model's symbol set to support:
- High tone (á, é, í, ó, ú)
- Falling tone (â, ê, î, ô, û)

To use these, use the `tonal_marker_prototype.py` or manually add marks to your test sentences.

## Objective intelligibility benchmark (2026-09-11)

Human MOS scoring (naturalness) is still an open item — see
`Tecgrw/application/GAP_REMEDIATION_PLAN.md` Q18 — but `evaluate_intelligibility.py` (new)
gives a real, computable proxy in the meantime: synthesize known text, transcribe the audio
back with STT, and score WER against the original. Low round-trip WER means what's spoken is
actually recoverable — real evidence of intelligibility, though it's not naturalness/MOS (a
robotic but clearly-enunciated clip could still score well here).

Ran against the 20 real clips already synthesized for the MOS listening pass (one per RAG
health topic, `Tecgrw/application/Api_server/eval/mos_samples/`), transcribed with
`tecgrw_stt`'s fine-tuned adapter (our best available STT):

| | Round-trip WER |
|---|---:|
| All 20 clips | 24.12% |
| **13 clips, pure Kinyarwanda text** | **16.59%** |
| 7 clips with digits or parenthetical English (`(hormones)`, `(growth spurt)`) | 35.62% |

**The overall number is misleading on its own — split it before citing it.** A third of the
sample has digits written as numerals ("8", "13") that DeepKIN correctly speaks as Kinyarwanda
number words, which STT then correctly transcribes as those words — a text-vs-speech mismatch
in the *reference*, not a TTS quality problem. Same story for parenthetical English medical
terms embedded in otherwise-Kinyarwanda RAG content ("(hormones)", "(growth spurt)") — these
are a `rag_docs` content-authoring issue (mixing English into what's meant to be pure
Kinyarwanda output, see `system.py` rule 1), not evidence the voice model mispronounced
anything. The clean-text 16.59% is the more honest intelligibility number; the confounded
35.62% is really flagging a RAG-content cleanup opportunity, not a TTS defect.

## Usage

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Synthesis
```bash
python run_synthesis.py
```

### 3. Batch Evaluation
```bash
python batch_evaluate.py --input sentences.txt --output_dir results
```

## Dataset
Located in `kinyarwanda_tts_dataset/`. 
- `train_data.psv`: Processed metadata with Speaker ID 0.
