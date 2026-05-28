import os
import zipfile
import time

SOURCE_ZIP_DIR = "./reuters-training-corpus/REUTERS_CORPUS_2"

TARGET_DATA_DIR = "./all_training_data"
if not os.path.exists(TARGET_DATA_DIR):
    os.makedirs(TARGET_DATA_DIR)

print(f"Start scanning folder: {SOURCE_ZIP_DIR}")
zip_files = [f for f in os.listdir(SOURCE_ZIP_DIR) if f.endswith(".zip")]
total_files = len(zip_files)
print(f"Found {total_files} zip files to extract...\n")

start_time = time.time()

for index, file_name in enumerate(zip_files, 1):
    zip_path = os.path.join(SOURCE_ZIP_DIR, file_name)
    print(f"[{index}/{total_files}] Extracting: {file_name} ...")

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(TARGET_DATA_DIR)
    except Exception as e:
        print(f"Error extracting {file_name}: {e}")

end_time = time.time()
print("\n" + "=" * 40)
print(f"All extraction completed! Total time: {end_time - start_time:.2f} seconds")
print(f"All data saved in: {TARGET_DATA_DIR}")
print("=" * 40)
