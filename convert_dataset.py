import csv
import os

def convert_metadata(input_csv, output_psv, wavs_dir):
    print(f"Converting {input_csv} to {output_psv}...")
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        #LJSpeech style: filename|text|normalized_text
        reader = csv.reader(f, delimiter='|')
        rows = list(reader)
        
    converted_rows = []
    for row in rows:
        if len(row) < 3:
            continue
            
        file_id = row[0]
        # Text is the 3rd column in LJSpeech LJSpeech format often has (filename|text|normalized_text)
        # In this dataset it seems like (file_id|original_text|normalized_text)
        normalized_text = row[2]
        
        # DeepKIN format: AUDIO_FILE|SPEAKER_ID|NORMALIZED_TEXT
        # AUDIO_FILE must be absolute path
        wav_path = os.path.abspath(os.path.join(wavs_dir, f"{file_id}.wav"))
        
        if os.path.exists(wav_path):
            converted_rows.append(f"{wav_path}|0|{normalized_text}")
        else:
            print(f"Warning: {wav_path} not found.")

    with open(output_psv, 'w', encoding='utf-8') as f:
        f.write('\n'.join(converted_rows))
        
    print(f"Done! Converted {len(converted_rows)} items.")

if __name__ == "__main__":
    DATASET_DIR = r"C:\Users\kevin\Desktop\Tecgrw\tecgrw_tts\kinyarwanda_tts_dataset"
    convert_metadata(
        os.path.join(DATASET_DIR, "metadata_punctuated.csv"),
        os.path.join(DATASET_DIR, "train_data.psv"),
        os.path.join(DATASET_DIR, "wavs")
    )
