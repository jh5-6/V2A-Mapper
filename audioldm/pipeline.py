import os
import yaml
import torch

from torch import autocast
from audioldm import LatentDiffusion, seed_everything
from audioldm.utils import default_audioldm_config, get_duration, get_bit_depth, get_metadata, download_checkpoint
from audioldm.audio import wav_to_fbank, TacotronSTFT, read_wav_file
from audioldm.latent_diffusion.ddim import DDIMSampler
from einops import repeat

def make_batch_for_image_to_audio(image, waveform=None, fbank=None, batchsize=1):
    image = [image] * batchsize
    if batchsize < 1:
        print("Warning: Batchsize must be at least 1. Batchsize is set to .")
    
    if(fbank is None):
        fbank = torch.zeros((batchsize, 1024, 64))  # Not used, here to keep the code format
    else:
        fbank = torch.FloatTensor(fbank)
        fbank = fbank.expand(batchsize, 1024, 64)
        assert fbank.size(0) == batchsize
        
    stft = torch.zeros((batchsize, 1024, 512))  # Not used

    if(waveform is None):
        waveform = torch.zeros((batchsize, 160000))  # Not used
    else:
        waveform = torch.FloatTensor(waveform)
        waveform = waveform.expand(batchsize, -1)
        assert waveform.size(0) == batchsize
        
    fname = [""] * batchsize  # Not used
    
    batch = (
        fbank,
        stft,
        None,
        fname,
        waveform,
        image,
    )
    return batch

def round_up_duration(duration):
    return int(round(duration/2.5) + 1) * 2.5

def build_model(
    ckpt_path=None,
    config=None,
    model_name="audioldm-s-full-v2"
):
    print("Load AudioLDM: %s", model_name)
    
    if(ckpt_path is None):
        ckpt_path = get_metadata()[model_name]["path"]
    
    if(not os.path.exists(ckpt_path)):
        download_checkpoint(model_name)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if config is not None:
        assert type(config) is str
        config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    else:
        config = default_audioldm_config(model_name)

    # Use image as condition instead of using waveform during training
    config["model"]["params"]["device"] = device
    config["model"]["params"]["cond_stage_key"] = "image"

    # No normalization here
    latent_diffusion = LatentDiffusion(**config["model"]["params"])

    resume_from_checkpoint = ckpt_path

    checkpoint = torch.load(resume_from_checkpoint, map_location=device)
    '''Original. Here is a bug that, an unexpected key "cond_stage_model.model.text_branch.embeddings.position_ids" exists in the checkpoint file. '''
    '''2023.10.17 Fix the bug by setting the paramer "strict" as "False" to ignore the unexpected key. '''
    latent_diffusion.load_state_dict(checkpoint["state_dict"], strict=False)

    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.to(device)

    latent_diffusion.cond_stage_model.embed_mode = "image"
    return latent_diffusion

def duration_to_latent_t_size(duration):
    return int(duration * 25.6)

def set_cond_audio(latent_diffusion):
    latent_diffusion.cond_stage_key = "waveform"
    latent_diffusion.cond_stage_model.embed_mode="audio"
    return latent_diffusion

def set_cond_image(latent_diffusion):
    latent_diffusion.cond_stage_key = "image"
    latent_diffusion.cond_stage_model.embed_mode="image"
    return latent_diffusion

def image_to_audio(
    latent_diffusion,
    images,
    original_audio_file_path = None,
    seed=42,
    ddim_steps=200,
    duration=10,
    batchsize=1,
    guidance_scale=2.5,
    n_candidate_gen_per_text=3,
    config=None,
):
    seed_everything(int(seed))
    waveform = None
    if(original_audio_file_path is not None):
        waveform = read_wav_file(original_audio_file_path, int(duration * 102.4) * 160)
        
    batch = make_batch_for_image_to_audio(images, waveform=waveform, batchsize=batchsize)

    latent_diffusion.latent_t_size = duration_to_latent_t_size(duration)  

    print("Generate audio using image %s" % images)
    latent_diffusion = set_cond_image(latent_diffusion)
        
    with torch.no_grad():
        waveform = latent_diffusion.generate_sample(
            [batch],
            unconditional_guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            n_candidate_gen_per_text=n_candidate_gen_per_text,
            duration=duration,
        )
    return waveform