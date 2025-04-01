import torch
import numpy as np
import os
import soundfile as sf  # For saving audio files
from espnet2.bin.asr_inference import Speech2Text
from espnet2.bin.asr_inference import Speech2Text

# Define the cache directory and model paths
cache_dir = os.path.expanduser("~/.cache/espnet_model_zoo")
model_dir = os.path.join(
    cache_dir, "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best")
model_file = os.path.join(model_dir, "model.pth")
model_config = os.path.join(model_dir, "config.yaml")
print("Loaded")

# Load the model configuration and weights
model = Speech2Text.from_pretrained(
    "espnet/shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"
)

# Example usage


def enroll(name, file, model):
    """Enroll a user with an audio file using ESPnet model
    inputs: str (Name of the person to be enrolled and registered)
            str (Path to the audio file of the person to enroll)
            model (Pre-loaded ESPnet model)
    outputs: None"""

    # Directories for embeddings and audio files
    embedding_dir = 'embeddings'
    audio_dir = 'audio_files'

    # Create directories if they don't exist
    os.makedirs(embedding_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    print("Processing enroll sample....")
    # Use ESPnet model to extract embeddings

    speech, rate = sf.read(file)
    text, *_ = model(speech)[0]
    print(text)
#         result = model(file)
#         print(result)
#         if not result:
#             raise ValueError("Error processing the input audio file.")

#         # Assuming 'get_embedding' extracts embeddings from the model output
#         # result[1] might contain the embeddings or decoded output
#         enroll_embs = np.array(result[1].tolist())
#         speaker = name
#     except Exception as e:
#         print(
#             f"Error processing the input audio file. Ensure the path is correct. Error: {e}")
#         return

#     try:
#         # Save embeddings
#         np.save(os.path.join(embedding_dir, speaker + ".npy"), enroll_embs)
#         print("Successfully enrolled the user")
#     except Exception as e:
#         print(f"Unable to save the user into the database. Error: {e}")

#     try:
#         # Read the input audio file and save it (using soundfile library)
#         audio_data, sample_rate = sf.read(file)
#         audio_path = os.path.join(audio_dir, speaker + ".wav")
#         sf.write(audio_path, audio_data, sample_rate)
#         print(f"Successfully saved the audio file for {speaker}")
#     except Exception as e:
#         print(f"Unable to save the audio file. Error: {e}")


# # Example usage
enroll("ARR", "C:\\Users\\Balas\\Music\\00001.wav", model)
