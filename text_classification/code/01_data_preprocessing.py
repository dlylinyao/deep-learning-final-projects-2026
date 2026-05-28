import os
import pandas as pd
import pickle
from bs4 import BeautifulSoup
from sklearn.preprocessing import MultiLabelBinarizer
import time

DATA_DIR = "./all_training_data"

OUTPUT_PICKLE = "reuters_clean_all.pkl"
OUTPUT_MLB = "mlb_binarizer.pkl"


def parse_reuters_xml(file_path):

    try:
        with open(file_path, "rb") as f:
            soup = BeautifulSoup(f, "xml")

        headline_node = soup.find("headline")
        headline = headline_node.text.strip() if headline_node else ""

        text_node = soup.find("text")
        text_body = text_node.text.strip() if text_node else ""

        full_text = f"{headline} {text_body}".strip()

        labels = []
        codes_section = soup.find("codes", class_="bip:topics:1.0")
        if codes_section:
            for code_tag in codes_section.find_all("code"):
                labels.append(code_tag.get("code"))

        return full_text, labels

    except Exception as e:
        return None, None


def load_dataset(data_dir):
    data = []
    file_count = 0
    start_time = time.time()

    print(f" Start parsing all XML files in {data_dir}...")
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".xml"):
                file_path = os.path.join(root, file)
                text, labels = parse_reuters_xml(file_path)

                if text and labels:
                    data.append({"text": text, "labels": labels})

                file_count += 1
                if file_count % 5000 == 0:
                    print(
                        f"  Parsed {file_count} articles... Time: {time.time() - start_time:.1f} s"
                    )

    print(f" Parsing completed! Total valid articles: {len(data)}")
    return pd.DataFrame(data)


if __name__ == "__main__":
    # 1. Load and parse data
    df = load_dataset(DATA_DIR)

    # 2. Multi-label binarization
    print("Processing labels (MultiLabelBinarizer)...")
    mlb = MultiLabelBinarizer()
    df["labels_vec"] = list(mlb.fit_transform(df["labels"]))

    # 3. Save model and data
    print("Saving cache files...")
    with open(OUTPUT_MLB, "wb") as f:
        pickle.dump(mlb, f)

    df.to_pickle(OUTPUT_PICKLE)
    print(f" Preprocessing done! Data saved to {OUTPUT_PICKLE}")
