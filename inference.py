import torch
from style_transfer_pipeline import AudioLDM2Pipeline
import os
import scipy
from config import get_config
import argparse

def main(config):
    os.makedirs(config["output_dir"], exist_ok=True)
    
    pipeline_trained = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-large", torch_dtype=torch.float16)
    pipeline_trained = pipeline_trained.to("cuda")
    layer_num = 0
    cross = [None, None, 768, 768, 1024, 1024, None, None]
    unet = pipeline_trained.unet
    
    positive_text_prompt = config["positive_text_prompt"]
    negative_text_prompt = config["negative_text_prompt"]

    for i in range(len(positive_text_prompt)):
        waveform = pipeline_trained(
            audio_path=config["audio_prompt_file"],
            noise_scale = config["noise_scale"],
            prompt=positive_text_prompt[i] * config["output_num_files"],
            negative_prompt=negative_text_prompt * config["output_num_files"],
            num_inference_steps=50,
            guidance_scale=config["guidance_scale"],
            num_waveforms_per_prompt=1,
            audio_length_in_s=config["audio_length_in_s"],
        ).audios
        for j in range(config["output_num_files"]):
            file_path = os.path.join(config["output_dir"], f"{positive_text_prompt[i][0]}_{j}.wav")
            scipy.io.wavfile.write(file_path, rate=16000, data=waveform[j])



if __name__ == "__main__":
    config = get_config()  # Pass the parsed arguments to get_config
    print(config)
    main(config)
