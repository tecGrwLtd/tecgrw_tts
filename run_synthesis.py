import os
import sys
import torch
import torchaudio

# Add DeepKIN to sys.path
DEEPKIN_PATH = os.path.abspath("deepkin_test/ac-ai-models/DeepKIN-AgAI")
if DEEPKIN_PATH not in sys.path:
    sys.path.append(DEEPKIN_PATH)

from deepkin.models.flex_tts import FlexKinyaTTS
from deepkin.data.kinya_norm import text_to_sequence
from deepkin.modules.tts_commons import intersperse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_file = os.path.abspath("finetuned_deepkin.pt")

print(f"Loading model from {model_file} on {device}...")
kinya_tts = FlexKinyaTTS.from_pretrained(device, model_file)
kinya_tts.eval()

test_sentences = [
    "Ikiremwamuntu cyose kivukana umudendezo kandi kingana mu cyubahiro n'uburenganzira.",
    "Buri wese afite uburenganzira bwo kubaho, kwidegembya, n'umutekano w'umuntu ku giti cye.",
    "Ese uzi ko amagambo meza adufasha kubaka umuryango ushikamye kandi wunze ubumwe? Ni ngombwa ko twubaka ejo hazaza heza!",
    "Uyu munsi twizeje kuzamura ireme ry'uburezi mu Rwanda, bityo buri mwana akagira iherezo ryiza.",
    "Amagambo meza adufasha kubaka umuryango ushikamye, wunze ubumwe, kandi uteye imbere mu muco."
]

os.makedirs("test_samples_multi", exist_ok=True)

with torch.no_grad():
    for i, text in enumerate(test_sentences):
        print(f"Generating Audio {i+1}: {text}")
        text_seq = intersperse(text_to_sequence(text, norm=True), 0)
        
        # Use sid=1 as per notebook
        audio_data = kinya_tts(text_seq, sid=1)
        
        output_path = f"test_samples_multi/sample_{i+1}.wav"
        
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)
            
        torchaudio.save(output_path, audio_data.cpu(), 24000)
        print(f"Saved: {output_path}")

print("Synthesis complete.")
