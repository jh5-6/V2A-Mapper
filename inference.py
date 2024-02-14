import os
import soundfile as sf 
from PIL import Image
from audioldm import image_to_audio, build_model

DATA_TYPE = "VGGSound"
INPUT_IMAGE_FOLDER = "D:/vggsound/vggsound/test/frame"
OUPUT_AUDIO_FOLDER = "D:/V2A-Mapper/V2A-Mapper/output_audio/VGGSound"
MODEL_NAME = "audioldm-s-full-v2"
audioldm=build_model(model_name=MODEL_NAME)

def image2audio(file_name, images, duration=10, guidance_scale=2.5, random_seed=42, n_candidates=3):
    waveform = image_to_audio(
        latent_diffusion=audioldm,
        images=images,
        seed=random_seed,
        duration=duration,
        guidance_scale=guidance_scale,
        n_candidate_gen_per_text=int(n_candidates),
    )  # [bs, 1, samples]

    for i, wave in enumerate(waveform):
        filename = f"{OUPUT_AUDIO_FOLDER}/{file_name}.wav"
        sf.write(filename, wave[0], 16000, 'PCM_16') 

    return waveform

if DATA_TYPE == "ImageHear" :
    for file in os.listdir(INPUT_IMAGE_FOLDER):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(INPUT_IMAGE_FOLDER, file)
            image = Image.open(image_path)
            image2audio(os.path.basename(file).split(".")[0],[image])
else:
    for folder in os.listdir(INPUT_IMAGE_FOLDER):
        images = []
        folder_path = os.path.join(INPUT_IMAGE_FOLDER, folder)
        for file in os.listdir(folder_path):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, file)
                image = Image.open(image_path)
                images.append(image)
        if len(images) != 0:
            image2audio(os.path.basename(folder_path), images)
