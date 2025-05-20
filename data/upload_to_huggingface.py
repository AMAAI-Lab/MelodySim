from datasets import Audio, Dataset
from huggingface_hub import HfApi, login
import json
import os
import shutil

audio_folder_path = "/path/to/melodysim_dataset"
jsonl_output_folder_path = "/output/path/to/jsonl"

login(token="...")  # Replace with your actual API token

# Load the dataset from the audio folder
audio_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(audio_folder_path) for f in filenames if f.endswith(('.wav', '.mp3'))]
relative_paths = [os.path.relpath(f, audio_folder_path) for f in audio_files]
dataset = Dataset.from_dict({"audio": audio_files, "relative_path": relative_paths}).cast_column("audio", Audio())

# Print the dataset to debug
print(f"Loaded dataset: {dataset}")

# Create the output directory if it does not exist
os.makedirs(jsonl_output_folder_path, exist_ok=True)

# Convert existing JSON files to JSON Lines format
for track_folder in os.listdir(audio_folder_path):
    track_folder_path = os.path.join(audio_folder_path, track_folder)
    if os.path.isdir(track_folder_path):
        for filename in os.listdir(track_folder_path):
            if filename.endswith(".json"):
                json_file_path = os.path.join(track_folder_path, filename)
                jsonl_file_path = os.path.join(jsonl_output_folder_path, f"{track_folder}_{filename.replace('.json', '.jsonl')}")
                with open(json_file_path, "r") as json_file:
                    data = json.load(json_file)
                    # print(data)
                    with open(jsonl_file_path, "w") as jsonl_file:
                        jsonl_file.write(json.dumps(data) + "\n")

                # Print the path to the JSON Lines file to verify
                print(f"JSON Lines file created at: {jsonl_file_path}")

# Print the dataset to verify
print(dataset)

# Define the repository details
repo_id = "amaai-lab/melodySim"
repo_url = f"https://huggingface.co/datasets/{repo_id}"

# Initialize the HfApi
api = HfApi()

# Push the dataset to the Hugging Face Hub directly
dataset.push_to_hub(repo_id)

# Print the repository URL to verify
print(f"Dataset pushed to: {repo_url}")