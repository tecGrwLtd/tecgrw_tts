import json
import sys

path = r'c:\Users\kevin\Desktop\Tecgrw\tecgrw_tts\train_deepkin.ipynb'
print(f"Patching {path}")
try:
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    patched = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if '!{python_cmd}' in source:
                new_source = source.replace('!{python_cmd}', '''import subprocess
import sys
import threading

# Run the training command and stream output directly to notebook and log file
print("Starting training process... Streaming output below:\\n", flush=True)

process = subprocess.Popen(
    python_cmd, 
    shell=True, 
    stdout=subprocess.PIPE, 
    stderr=subprocess.STDOUT, 
    text=True, 
    encoding='utf-8', 
    bufsize=1
)

def stream_output():
    with open("deepkin_training.log", "w", encoding="utf-8") as log_file:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()

thread = threading.Thread(target=stream_output)
thread.start()

try:
    process.wait()
except KeyboardInterrupt:
    print("\\nTraining interrupted by user. Terminating process...")
    process.terminate()
    process.wait()

thread.join()
print(f"\\nTraining finished with exit code: {process.returncode}")''')
                
                # Convert back to ipynb list of lines
                lines = new_source.split('\n')
                cell_source = []
                for idx, line in enumerate(lines):
                    if idx < len(lines) - 1:
                        cell_source.append(line + '\n')
                    else:
                        if line: # only append the last line if it's not empty, or if empty wait we don't need to append empty unless it's the only line
                            cell_source.append(line)
                
                cell['source'] = cell_source
                patched = True
                break

    if patched:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Notebook patched successfully. The training command will now stream output directly to the notebook and create a separate log file.")
    else:
        print("Could not find '!{python_cmd}' in the notebook.")
except Exception as e:
    print(f"Error: {e}")
