# SDEdit-AudioLDM2
This is the SDEdit implementation for AudioLDM2 use in [AP-adapter](https://github.com/fundwotsai2001/AP-adapter/tree/master). 
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
