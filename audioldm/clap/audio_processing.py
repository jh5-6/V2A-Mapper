import os
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
import torchaudio
from audioldm.clap.encoders import CLAPAudioEmbeddingClassifierFreev2
from multiprocessing import Pool, cpu_count

def process_audio_file(args):
    audio_file, audio_classifier = args
    audio, _ = torchaudio.load(audio_file, normalize=True, num_frames=48000)
    embedding = audio_classifier(audio).cpu().numpy().squeeze()
    return embedding, os.path.basename(audio_file)

def main(audio_folder_path, csv_output_path, pretrained_path):
    # Get a list of all audio files in the folder
    audio_files = [os.path.join(audio_folder_path, file) for file in os.listdir(audio_folder_path) if file.endswith('.wav')]

    # Instantiate your CLAPAudioEmbeddingClassifierFreev2 model
    audio_classifier = CLAPAudioEmbeddingClassifierFreev2(pretrained_path=pretrained_path)
    audio_classifier.eval()

    # Process audio files using multiprocessing
    with Pool(cpu_count()) as pool, tqdm(total=len(audio_files), position=0, desc="Processing Files") as pbar:
        args = [(file, audio_classifier) for file in audio_files]
        results = list(pool.imap_unordered(process_audio_file, args))
        pbar.update(len(audio_files))

    # Unpack the results
    embeddings, file_names = zip(*results)

    # Create a DataFrame and save to CSV
    df = pd.DataFrame({"file_name": file_names, "embedding": embeddings})
    df.to_csv(csv_output_path, index=False)
    print(f"Results saved to {csv_output_path}")


if __name__ == "__main__":
    audio_folder_path = "D:/vggsound/vggsound/train/audio"
    csv_output_path = "C:/Users/Desktop/Downloads/train_clap.csv"
    pretrained_path = "audioldm-full-s-v2.ckpt"

    main(audio_folder_path, csv_output_path, pretrained_path)