import pandas as pd
import ast
import os
import numpy as np
from scipy.spatial.distance import euclidean
import torch
from espnet2.tasks.ssl import SSLTask
import soundfile as sf
from torch.nn.utils.rnn import pad_sequence
import torch.utils.checkpoint as checkpoint


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the XEUS model
def load_xeus_model(checkpoint_path):
    xeus_model, _ = SSLTask.build_model_from_file(
        None, checkpoint_path, device
    )
    xeus_model.eval()  # Set model to evaluation mode
    return xeus_model

# Extract embeddings using the XEUS model
def extract_embeddings(wav_file, xeus_model):
    wavs, _ = sf.read(wav_file)
    wav_lengths = torch.LongTensor([len(wavs)]).to(device)
    wavs = pad_sequence([torch.Tensor(wavs)], batch_first=True).to(device)

    with torch.no_grad():
        feats = checkpointed_forward(xeus_model, wavs, wav_lengths)

    feats = feats.mean(dim=1)  # Average over the sequence dimension if needed
    feats = feats.flatten()  # Flatten to 1-D vector
    return feats.cpu().detach().numpy()  # Extract embeddings from the model output

# Checkpointed forward pass to handle memory efficiently
def checkpointed_forward(model, wavs, wav_lengths):
    def forward_function(wavs, wav_lengths):
        return model.encode(wavs, wav_lengths, use_mask=False, use_final_output=False)[0][-1]

    return checkpoint.checkpoint(forward_function, wavs, wav_lengths)

# Recognize a speaker from an audio file using enrolled embeddings
def recognize_speaker(wav_file, xeus_model, enrolled_embeddings_dir, threshold=10):
    test_embs = extract_embeddings(wav_file, xeus_model)

    # Load enrolled embeddings
    embeds = os.listdir(enrolled_embeddings_dir)
    if not embeds:
        print("No enrolled users found")
        return None

    # Compare test embeddings to all enrolled users
    distances = {}
    for emb in embeds:
        enroll_embs = np.load(os.path.join(enrolled_embeddings_dir, emb))
        speaker = emb.replace(".npy", "")
        distance = euclidean(test_embs, enroll_embs)
        distances[speaker] = distance

    # Find the closest match
    recognized_speaker = min(distances, key=distances.get)
    if distances[recognized_speaker] < threshold:
        return recognized_speaker
    else:
        return None

# Main function to recognize all audio files in the CSV and save results
def recognize_and_save_results(csv_file_path, xeus_checkpoint, enrolled_embeddings_dir):
    # Load the XEUS model
    xeus_model = load_xeus_model(xeus_checkpoint)

    # Load the original CSV file
    df = pd.read_csv(csv_file_path)

    # List to store results for the new CSV
    results_list = []

    # Loop over each row to process the audio files
    for idx, row in df.iterrows():
        speaker_name = row['Speaker']
        audio_files = ast.literal_eval(row['Audio'])  # Safely convert string to list

        # Loop through each audio sample and try to recognize the speaker
        for audio_file in audio_files:
            if not os.path.exists(audio_file):
                print(f"  Audio file {audio_file} not found, skipping.")
                continue

            recognized_speaker = recognize_speaker(audio_file, xeus_model, enrolled_embeddings_dir)

            # Determine if recognition result is correct (1) or incorrect (0)
            result = 1 if recognized_speaker == speaker_name else 0

            # Add the result for this audio file to the results list
            results_list.append({
                'Speaker': speaker_name,
                'Audio_File': audio_file,
                'Recognized_Speaker': recognized_speaker,
                'Result': result
            })

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results_list)

    # Save the results to a new CSV file
    result_csv_file_path = csv_file_path.replace('.csv', '_recognition_results21.csv')
    results_df.to_csv(result_csv_file_path, index=False)

    print(f"Recognition results saved at: {result_csv_file_path}")


# Example usage
csv_file_path = "D:\\Voice Authetication ESPNET\\updated_speaker_info.csv"  # Replace with actual path to your CSV file
xeus_checkpoint = "D:\\Voice Authetication ESPNET\\XEUS\\model\\xeus_checkpoint.pth"  # Replace with actual XEUS model checkpoint
enrolled_embeddings_dir = "D:\\Voice Authetication ESPNET\\embeddings2"  # Replace with actual path to enrolled embeddings

recognize_and_save_results(csv_file_path, xeus_checkpoint, enrolled_embeddings_dir)
