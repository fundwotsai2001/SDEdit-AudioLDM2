def get_config():
    return {
        "output_dir": "SDEdit_test",
        "output_num_files": 1,
        "audio_prompt_file": "piano.wav",
        "guidance_scale": 7.5,   
        "noise_scale": 0.55,
        #######################################################
        # You can change the positive_text_prompt whatever you like,
        # But the negative_text_prompt should be the instrument in the 
        # original music.
        "positive_text_prompt": [
            ["a recording of a marimba solo"],
            # ["a recording of an acoustic guitar solo"],
            # ["a recording of a harp solo"],
            # ["a recording of an cello solo"],
            # ["a recording of a flute solo"],
            ],
        "negative_text_prompt": ["Low quality"],
        "audio_length_in_s" : 10
        #######################################################
    }
    