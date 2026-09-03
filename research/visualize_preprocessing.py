from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# configuration
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "sample_data"
PLOT_DIR = ROOT / "plots"
MODEL_NAME = "t5-small"

DATA_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# 1) create tiny example CSVs (train/test/val)
train = [
    {"dialogue": "Alice: Hi Bob. How are you?\nBob: I'm fine, thanks!", "summary": "Alice greets Bob and asks after his wellbeing."},
    {"dialogue": "Customer: My order didn't arrive.\nAgent: I'm sorry, we'll check that for you.", "summary": "Customer reports missing order; agent offers help."},
    {"dialogue": "Interviewer: Tell me about your experience.\nCandidate: I worked on NLP projects.", "summary": "Candidate describes NLP experience."}
]

val = [
    {"dialogue": "Teacher: Please submit homework.\nStudent: I'll upload it tonight.", "summary": "Student will submit homework tonight."}
]

test = [
    {"dialogue": "User: The app crashed on start.\nSupport: Can you send the logs?", "summary": "Support requests logs to diagnose crash."}
]

# save CSVs
pd.DataFrame(train).to_csv(DATA_DIR / "train.csv", index=False)
pd.DataFrame(val).to_csv(DATA_DIR / "validation.csv", index=False)
pd.DataFrame(test).to_csv(DATA_DIR / "test.csv", index=False)

print(f"Wrote sample CSVs to {DATA_DIR}")

# 2) load CSVs into Hugging Face Dataset
datasets = {}
for split in ["train", "validation", "test"]:
    df = pd.read_csv(DATA_DIR / f"{split}.csv")
    datasets[split] = Dataset.from_pandas(df)

ds = DatasetDict(datasets)
print("Loaded DatasetDict with splits:", list(ds.keys()))

# 3) tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 4) feature conversion (same logic as your method)
def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch['dialogue'], max_length=1024, truncation=True, padding=False)
    target_encodings = tokenizer(example_batch['summary'], max_length=128, truncation=True, padding=False)

    labels = target_encodings['input_ids']
    # replace pad token id in the labels with -100
    labels = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels]

    return {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': labels
    }

# 5) map over dataset (batched)
processed = ds.map(convert_examples_to_features, batched=True, remove_columns=ds['train'].column_names)
print("Processed dataset features keys:", processed['train'].column_names)

# 6) analyze token-length distributions
def label_length(labels_row):
    # labels_row is a list of ints (may contain -100)
    return len([l for l in labels_row if l != -100])

input_lens = {s: [len(x) for x in processed[s]['input_ids']] for s in processed}
label_lens = {s: [label_length(x) for x in processed[s]['labels']] for s in processed}

# 7) visualize
for split in processed:
    plt.figure(figsize=(6,3))
    plt.hist(input_lens[split], bins=10, alpha=0.7)
    plt.title(f"Input token length distribution ({split})")
    plt.xlabel("tokens")
    plt.ylabel("count")
    plt.tight_layout()
    out = PLOT_DIR / f"{split}_input_len.png"
    plt.savefig(out)
    plt.close()

    plt.figure(figsize=(6,3))
    plt.hist(label_lens[split], bins=10, alpha=0.7, color='orange')
    plt.title(f"Label token length distribution ({split})")
    plt.xlabel("tokens")
    plt.ylabel("count")
    plt.tight_layout()
    out2 = PLOT_DIR / f"{split}_label_len.png"
    plt.savefig(out2)
    plt.close()

print(f"Saved plots to {PLOT_DIR}")

# 8) show an example tokenization
example_idx = 0
print("Example dialogue:", ds['train'][example_idx]['dialogue'])
print("Example summary:", ds['train'][example_idx]['summary'])
enc = tokenizer(ds['train'][example_idx]['dialogue'])
print("Tokenized input ids (first 40):", enc['input_ids'][:40])
print("Decoded back (first 200 chars):", tokenizer.decode(enc['input_ids'])[:200])

# 9) optional: save processed dataset to disk
processed.save_to_disk(ROOT / "processed_sample")
print("Saved processed dataset to:", ROOT / "processed_sample")
