import nbformat

notebook_path = "train_deepkin.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

old_text = '--train_log_steps=10 --validation_steps=130 --checkpoint_steps=130"'
new_text = '--train_log_steps=10 --validation_steps=130 --checkpoint_steps=130 --num_iters=1020000"'

replaced = False
for cell in nb.cells:
    if "flex_trainer.py" in cell.source and old_text in cell.source:
        cell.source = cell.source.replace(old_text, new_text)
        replaced = True
        print("Patched flex_trainer arguments with num_iters.")

if replaced:
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Notebook updated successfully.")
else:
    print("No changes made. Text not found.")
