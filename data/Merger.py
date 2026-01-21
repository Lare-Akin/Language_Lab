import pandas as pd
from pathlib import Path

# Define base path
base_path = Path(r"E:\BG_Learning_Lare\Sandpit\data")

# Define input files
files_to_merge = {
    "Bulgarian_Phrases.csv": "Bulgarian_Phrases",
    "bulgarian_reference.csv": "Bulgarian_reference",
    "Learn_From_Human_Conversation.csv": "Learning_From_Human_Conversation"
}

# Load and tag each file
frames = []
for filename, classification in files_to_merge.items():
    file_path = base_path / filename
    try:
        df = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin1", on_bad_lines="skip")
    
    df.columns = df.columns.str.strip()
    df["Classification"] = classification
    frames.append(df)

# Merge all dataframes
merged_df = pd.concat(frames, ignore_index=True)

# Optional: remove duplicates
merged_df.drop_duplicates(inplace=True)

# Save to new CSV
output_path = base_path / "Learning_Resource_Database.csv"
merged_df.to_csv(output_path, index=False, encoding="utf-8")

print("✅ Merged CSV saved to:", output_path)