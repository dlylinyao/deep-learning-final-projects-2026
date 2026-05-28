import os
import glob
import zipfile
import pickle
import torch
import numpy as np
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys


# 1. Get official label order
def get_codes(codefile):
    codes = {}
    i = 0
    with open(codefile, "r", encoding="utf-8") as cf:
        for line in cf:
            if not line.startswith(";"):

                code = line.strip().split("\t")[0]
                codes[code] = i
                i += 1
    return codes


print(" Loading official label dictionary...")
CODEMAP = get_codes("topic_codes.txt")  # [cite: 2]
num_official_classes = len(CODEMAP)

# 2. Load your trained results
print(" Loading model and config ...")
with open("mlb_binarizer.pkl", "rb") as f:
    mlb = pickle.load(f)

# Point to your best checkpoint
model_path = "./results_512/checkpoint-15000"
if not os.path.exists(model_path):
    print(f"Error: path {model_path} not found, please check checkpoint folder name")
    sys.exit()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Current device: {device}")
model.eval()


model_to_official_map = {
    i: CODEMAP[label] for i, label in enumerate(mlb.classes_) if label in CODEMAP
}

# 3. Traverse files using global dict logic like get_data.py
doc_predictions = {}

# corpus_dir = 'eval_corpus/REUTERS_CORPUS_2'
corpus_dir = "text_test_dataset"
pattern = os.path.join(corpus_dir, "*.zip")

print(f"Scanning zip files under {corpus_dir}...")
zip_files = sorted(glob.glob(pattern))

if not zip_files:
    print(f"Error: no .zip files found under {corpus_dir}!")
    sys.exit()

for zfile in zip_files:
    print(f"Predicting: {os.path.basename(zfile)}")
    with zipfile.ZipFile(zfile, "r") as zf:
        for xmlfile in zf.namelist():
            if xmlfile.endswith("/"):
                continue  # skip directory

            with zf.open(xmlfile, "r") as xf:

                content = xf.read().decode("utf-8", errors="ignore")
                bs = BeautifulSoup(content, "lxml")
                title = bs.title.text if bs.title else ""
                body = " ".join([p.text for p in bs.find_all("p")])
                text = (title + " " + body).strip() or "empty"

                # Model inference
                inputs = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512
                ).to(device)
                with torch.no_grad():
                    logits = model(**inputs).logits

                # Convert predictions
                probs = torch.sigmoid(logits)
                # preds = (probs > 0.5).int().cpu().numpy()[0]
                # threshold = 0.45
                threshold = 0.4
                # threshold = 0.55
                # threshold = 0.35
                # threshold = 0.3
                preds = (probs > threshold).int().cpu().numpy()[0]

                official_vec = np.zeros(num_official_classes, dtype=int)
                for my_idx, is_pos in enumerate(preds):
                    if is_pos == 1 and my_idx in model_to_official_map:
                        official_vec[model_to_official_map[my_idx]] = 1

                doc_predictions[xmlfile] = official_vec

# 4. Strictly sort by filename to match ground_truth.txt order
print("Performing global sorting and saving results...")
sorted_filenames = sorted(doc_predictions.keys())  # [cite: 3]
final_matrix = np.array([doc_predictions[fname] for fname in sorted_filenames])

# Save using official format
np.savetxt("my_predictions_512.txt", final_matrix, fmt="%i")
print(f"Done! Predicted {len(sorted_filenames)} documents.")
