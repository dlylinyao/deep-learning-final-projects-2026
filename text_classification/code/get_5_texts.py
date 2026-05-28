import pickle
import pandas as pd

target_indices = [1667, 1677, 2337, 2343, 2826]
file_path = "reuters_clean_all.pkl"


try:
    with open(file_path, "rb") as f:
        data = pickle.load(f)

        for idx in target_indices:
            print(f"======  Line {idx} ======")

            if isinstance(data, list):
                if isinstance(data[idx], str):
                    print(data[idx])
                elif isinstance(data[idx], dict) and "text" in data[idx]:
                    print(data[idx]["text"])
                else:
                    print(data[idx])

            elif isinstance(data, pd.DataFrame):

                if "text" in data.columns:
                    print(data.iloc[idx]["text"])
                elif "body" in data.columns:
                    print(data.iloc[idx]["body"])
                else:
                    print(data.iloc[idx])

            print("\n")

except FileNotFoundError:
    print(
        f"File not found: {file_path}. Please make sure it is in the same folder as this script."
    )
except Exception as e:
    print(f"An error occurred while reading: {e}")