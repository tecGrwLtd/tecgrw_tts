import json
import os

notebook_path = r"c:\Users\kevin\Desktop\Tecgrw\tecgrw_tts\train_deepkin.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell with the training script (Cell 5)
# It has "ea020b3f" as id in the original file I saw
target_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell.get('id') == 'ea020b3f':
        target_idx = i
        break

if target_idx == -1:
    # Fallback search by content
    for i, cell in enumerate(nb['cells']):
        if "TRAINER_SCRIPT" in "".join(cell.get('source', [])):
            target_idx = i
            break

if target_idx != -1:
    print(f"Found training cell at index {target_idx}. Replacing...")
    
    new_cells = [
        {
            "cell_type": "markdown",
            "id": "config-hdr",
            "metadata": {},
            "source": ["## 3. Training Configuration\n", "We define the hyperparameters directly in the notebook for full control."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "config-code",
            "metadata": {},
            "outputs": [],
            "source": [
                "from types import SimpleNamespace\n",
                "import os\n",
                "import torch\n",
                "\n",
                "# Hyperparameters for Fine-Tuning\n",
                "hp = SimpleNamespace(\n",
                "    model_variant=\"flex_tts:base\",\n",
                "    batch_size=2,\n",
                "    accumulation_steps=2,\n",
                "    num_iters=1010000, # Increased from 1000500 to allow more training\n",
                "    warmup_iter=500,\n",
                "    peak_lr=0.0001,\n",
                "    train_log_steps=10,\n",
                "    checkpoint_steps=130,\n",
                "    num_losses=8,\n",
                "    enable_amp=True,\n",
                "    use_bfloat16=True,\n",
                "    sampling_rate=24000,\n",
                "    filter_length=1024,\n",
                "    hop_length=256,\n",
                "    win_length=1024,\n",
                "    n_mel_channels=80,\n",
                "    mel_fmin=0.0,\n",
                "    mel_fmax=None,\n",
                "    add_blank=True,\n",
                "    data_dir=os.path.abspath(\"kinyarwanda_tts_dataset\"),\n",
                "    train_file=os.path.join(os.path.abspath(\"kinyarwanda_tts_dataset\"), \"train_data.psv\"),\n",
                "    save_path=os.path.abspath(\"finetuned_deepkin.pt\")\n",
                ")\n",
                "\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "print(f\"Using device: {device}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "dataloader-hdr",
            "metadata": {},
            "source": ["## 4. Custom Data Loading (The Silence Fix)\n", "This section implements the fixed data loader directly in the notebook to ensure the float32 normalization bug is resolved."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "dataloader-code",
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "from scipy.io.wavfile import read\n",
                "from deepkin.modules.tts_mel import mel_spectrogram_torch\n",
                "from deepkin.data.kinya_norm import text_to_sequence\n",
                "from deepkin.modules.tts_commons import intersperse\n",
                "\n",
                "def load_wav_to_torch(full_path):\n",
                "    sampling_rate, data = read(full_path)\n",
                "    return torch.FloatTensor(data.astype(np.float32)), sampling_rate, data.dtype\n",
                "\n",
                "class TextAudioSpeakerLoader(torch.utils.data.Dataset):\n",
                "    def __init__(self, audiopaths_sid_text, hparams):\n",
                "        with open(audiopaths_sid_text, encoding='utf-8') as f:\n",
                "            self.audiopaths_sid_text = [l.strip().split(\"|\") for l in f]\n",
                "        self.sampling_rate = hparams.sampling_rate\n",
                "        self.filter_length = hparams.filter_length\n",
                "        self.hop_length = hparams.hop_length\n",
                "        self.win_length = hparams.win_length\n",
                "        self.n_mel_channels = hparams.n_mel_channels\n",
                "        self.mel_fmin = hparams.mel_fmin\n",
                "        self.mel_fmax = hparams.mel_fmax\n",
                "        self.add_blank = hparams.add_blank\n",
                "        self.data_dir = hparams.data_dir\n",
                "\n",
                "    def get_audio(self, filename):\n",
                "        full_path = os.path.join(self.data_dir, \"wavs\", filename if filename.endswith(\".wav\") else filename + \".wav\")\n",
                "        if not os.path.exists(full_path):\n",
                "            full_path = os.path.join(self.data_dir, filename)\n",
                "            \n",
                "        audio, sampling_rate, dtype = load_wav_to_torch(full_path)\n",
                "        \n",
                "        if sampling_rate != self.sampling_rate:\n",
                "            raise ValueError(f\"{sampling_rate} SR doesn't match target {self.sampling_rate} SR\")\n",
                "        \n",
                "        # DOUBLE NORMALIZATION FIX\n",
                "        if dtype == np.float32:\n",
                "            audio_norm = audio\n",
                "        else:\n",
                "            audio_norm = audio / 32768.0\n",
                "            \n",
                "        audio_norm = audio_norm.unsqueeze(0)\n",
                "        spec = mel_spectrogram_torch(audio_norm, self.filter_length,\n",
                "                                     self.n_mel_channels, self.sampling_rate, self.hop_length,\n",
                "                                     self.win_length, self.mel_fmin, self.mel_fmax,\n",
                "                                     center=False)\n",
                "        spec = torch.squeeze(spec, 0)\n",
                "        return spec, audio_norm\n",
                "\n",
                "    def get_text(self, text):\n",
                "        text_norm = text_to_sequence(text)\n",
                "        if self.add_blank:\n",
                "            text_norm = intersperse(text_norm, 0)\n",
                "        return torch.LongTensor(text_norm)\n",
                "\n",
                "    def __getitem__(self, index):\n",
                "        audiopath, sid, text = self.audiopaths_sid_text[index]\n",
                "        text = self.get_text(text)\n",
                "        spec, wav = self.get_audio(audiopath)\n",
                "        sid = torch.LongTensor([int(sid)])\n",
                "        return text, spec, wav, sid\n",
                "\n",
                "    def __len__(self):\n",
                "        return len(self.audiopaths_sid_text)\n",
                "\n",
                "class TextAudioSpeakerCollate:\n",
                "    def __call__(self, batch):\n",
                "        batch.sort(key=lambda x: x[1].size(1), reverse=True)\n",
                "        text_padded = torch.nn.utils.rnn.pad_sequence([x[0] for x in batch], batch_first=True)\n",
                "        max_spec_len = max([x[1].size(1) for x in batch])\n",
                "        max_wav_len = max([x[2].size(1) for x in batch])\n",
                "        spec_padded = torch.zeros(len(batch), batch[0][1].size(0), max_spec_len)\n",
                "        wav_padded = torch.zeros(len(batch), 1, max_wav_len)\n",
                "        text_lengths = torch.LongTensor([len(x[0]) for x in batch])\n",
                "        spec_lengths = torch.LongTensor([x[1].size(1) for x in batch])\n",
                "        wav_lengths = torch.LongTensor([x[2].size(1) for x in batch])\n",
                "        sid = torch.LongTensor([x[3] for x in batch])\n",
                "        for i, (_, spec, wav, _) in enumerate(batch):\n",
                "            spec_padded[i, :, :spec.size(1)] = spec\n",
                "            wav_padded[i, :, :wav.size(1)] = wav\n",
                "        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "trainer-hdr",
            "metadata": {},
            "source": ["## 5. In-Notebook Trainer\n", "Launching the training loop directly here for maximum visibility."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "trainer-code",
            "metadata": {},
            "outputs": [],
            "source": [
                "from torch.utils.data import DataLoader\n",
                "from deepkin.models.flex_tts import FlexKinyaTTS\n",
                "from deepkin.train.flex_train_tools import create_optimizer_and_lr_scheduler, load_model_state, save_model_state\n",
                "from deepkin.utils.arguments import FlexArguments\n",
                "from tqdm import tqdm\n",
                "import gc\n",
                "\n",
                "# 1. Initialize Loader\n",
                "train_dataset = TextAudioSpeakerLoader(hp.train_file, hp)\n",
                "train_loader = DataLoader(train_dataset, num_workers=0, shuffle=True, \n",
                "                          batch_size=hp.batch_size, pin_memory=True, collate_fn=TextAudioSpeakerCollate())\n",
                "\n",
                "# 2. Initialize Model\n",
                "train_args = FlexArguments()\n",
                "train_args.model_variant = hp.model_variant\n",
                "train_args.enable_amp = hp.enable_amp\n",
                "train_args.use_bfloat16 = hp.use_bfloat16\n",
                "model = FlexKinyaTTS(train_args).to(device)\n",
                "model_cache = SimpleNamespace()\n",
                "\n",
                "# 3. Setup Optimizer\n",
                "optimizer, scaler, lr_scheduler = create_optimizer_and_lr_scheduler(0, train_args, model, device)\n",
                "\n",
                "# 4. Load Checkpoint\n",
                "total_steps = 0\n",
                "best_valid_loss = 999999.0\n",
                "if os.path.exists(hp.save_path):\n",
                "    print(f\"Loading checkpoint from {hp.save_path}...\")\n",
                "    best_valid_loss, total_steps = load_model_state(train_args, model, optimizer, scaler, lr_scheduler, {'cuda:0': f'cuda:{device.index or 0}'})\n",
                "    print(f\"Resuming from step {total_steps}\")\n",
                "\n",
                "# 5. Training Loop\n",
                "model.train()\n",
                "current_iters = total_steps // hp.accumulation_steps\n",
                "print(f\"Training for {hp.num_iters - current_iters} more iterations...\")\n",
                "pbar = tqdm(total=hp.num_iters, initial=current_iters)\n",
                "\n",
                "try:\n",
                "    accumulated_steps = 0\n",
                "    while current_iters < hp.num_iters:\n",
                "        for batch in train_loader:\n",
                "            batch = [item.to(device) for item in batch]\n",
                "            with torch.cuda.amp.autocast(enabled=hp.enable_amp, dtype=torch.bfloat16 if hp.use_bfloat16 else torch.float16):\n",
                "                losses = model(total_steps, train_args, model_cache, batch)\n",
                "            \n",
                "            total_loss = torch.stack(losses).sum() / hp.accumulation_steps\n",
                "            scaler.scale(total_loss).backward()\n",
                "            \n",
                "            accumulated_steps += 1\n",
                "            total_steps += 1\n",
                "            \n",
                "            if accumulated_steps % hp.accumulation_steps == 0:\n",
                "                scaler.step(optimizer)\n",
                "                scaler.update()\n",
                "                optimizer.zero_grad()\n",
                "                lr_scheduler.step()\n",
                "                current_iters += 1\n",
                "                accumulated_steps = 0\n",
                "                pbar.update(1)\n",
                "                \n",
                "                if current_iters % hp.train_log_steps == 0:\n",
                "                    y = batch[4] # wav\n",
                "                    pbar.set_description(f\"Loss: {total_loss.item()*hp.accumulation_steps:.4f} | GT Max: {y.abs().max().item():.4f}\")\n",
                "\n",
                "                if current_iters % hp.checkpoint_steps == 0:\n",
                "                    save_model_state(hp.save_path, train_args, model, optimizer, scaler, lr_scheduler, total_steps, best_valid_loss, 0)\n",
                "            \n",
                "            if current_iters >= hp.num_iters: break\n",
                "\n",
                "except KeyboardInterrupt:\n",
                "    print(\"Saving checkpoint...\")\n",
                "    save_model_state(hp.save_path, train_args, model, optimizer, scaler, lr_scheduler, total_steps, best_valid_loss, 0)\n",
                "finally:\n",
                "    pbar.close()"
            ]
        }
    ]
    
    # Replace Cell 5 (and potentially Cell 6 which was Evaluation but let's just insert for now)
    # nb['cells'][target_idx:target_idx+1] = new_cells
    
    # Actually, the user had an Evaluation cell after. I should keep that but update it to use hp variables.
    # Let's just replace the training cell for now.
    nb['cells'][target_idx:target_idx+1] = new_cells
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully.")
else:
    print("Could not find training cell.")
