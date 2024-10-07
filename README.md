# SDEdit-AudioLDM2
This is the SDEdit implementation for AudioLDM2 use in [AP-adapter](https://github.com/fundwotsai2001/AP-adapter/tree/master) "Audio Prompt Adapter: Unleashing Music Editing Abilities for Text-to-Music with Lightweight Finetuning" in Proc. Int. Society for Music Information Retrieval Conf. (ISMIR), 2024.. 
## Installation
```
git clone https://github.com/fundwotsai2001/SDEdit-AudioLDM2.git
pip install -r requirements.txt
```
## Inference
You can edit the config.py file. Most importantly, if noise_scale = 1.0, it means full steps of noise will be added. If noise_scale = 0, it means no noise will be added. You can tune this parameter based on your preference.
```
python inference.py
```
## Key change from original AudioLDM2
Basically we add a few lines to encode the mel and add desired degree of noise. All other parts remains the same.
```
# Get mel from the input audio
mel = wav_to_mel(audio_path,
                10,
                augment_data=False,
                mix_data=None,
                snr=None)
mel = mel.unsqueeze(0).to(device).to(torch.float16)

# Decode mel
latents = self.vae.encode(mel).latent_dist.sample()
latents = latents * self.vae.config.scaling_factor
noise = torch.randn_like(latents)

# Decide the noise level
shallow_reverse_step = int(num_inference_steps * (1 - noise_scale))
timesteps = timesteps[shallow_reverse_step:]
timesteps_tensor = torch.tensor([timesteps[0]], dtype=torch.int32)

# Add the coresponding noise to the latent
noisy_sample = self.scheduler.add_noise(latents,noise,timesteps_tensor)
```
## Acknowledgments
This code is heavily based on [AudioLDM2](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm2#diffusers.AudioLDM2UNet2DConditionModel), [Diffusers](https://github.com/huggingface/diffusers)
