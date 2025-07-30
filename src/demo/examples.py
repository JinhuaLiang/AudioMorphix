# import torch


# def coord2mask(t_on, t_off, f_low, f_up):
#     mask = torch.zeros(1024, 64)
#     mask[t_on:t_off, f_low:f_up] = 1
#     return mask


# Reference region:
# t_on 0 t_off 1000 \
# f_low 0 f_up 30 \
mix_example = [[
    "./examples/lawn_mower.wav",  # Background audio path
    "Lawn mower can be heard",  # Background caption
    "./examples/doremi.wav",  # Foreground audio path
    "A woman sings",  # Foreground caption
    # coord2mask(0, 1000, 0, 30),  # (t_on,t_off,f_low,f_up)
    0,  # dt
    16,  # df
    1,  # resize_scale_t
    1,  # resize_scale_f
    0,  # w_content
    80,  # w_edit
    1.2,  # guidance_scale
    0.4,  # sde_strength
    1.5   # energy_scale
]]


# Reference region:
# t_on 0 t_off 100 \
# f_low 0  f_up 60 \
remove_example = [[
    "./examples/tick_noise_with_laughter.wav",  # Background audio path
    "Scrape, Tick, Noise, Tuning fork, Breathing, Generic impact sounds with the sound of Door, Clicking, Female speech, woman speaking, Conversation, Laughter, Breathing, Human voice",  # Background caption
    "./examples/tick_noise-noised.wav",  # Foreground audio path
    "Door, Clicking, Female speech, woman speaking, Conversation, Laughter, Breathing, Human voice can be heard",  # Foreground caption
    0,  # dt
    0,  # df
    1,  # resize_scale_t
    1,  # resize_scale_f
    5,  # w_content
    40, # w_edit
    0.1, # w_contrast
    1.2,  # guidance_scale
    0.1,  # sde_strength
    1.5,  # energy_scale
]]


moveandrescale_example = [
    # Reference region:
    # t_on 258 t_off 768 \
    # f_low 0  f_up 64 \
    ["./examples/acoustic_guitar.wav",  # Background audio path
     "Acoustic guitar",  # Background caption
     20,  # dt
     0,  # df
     2.0,  # resize_scale_t
     1.0,   # resize_scale_f
     20.0,  # w_content
     40.0, # w_edit
     0.1, # w_contrast
     0.1, # w_inpaint
     1.2,  # guidance_scale
     0.0,  # sde_strength
     1.5,  # energy_scale
     ],
    # Reference region:
    # t_on 0 t_off 512 \
    # f_low 10  f_up 64 \
    ["./examples/high_pitch.wav",  # Background audio path
     "High pitch",  # Background caption
     0,  # dt
     -10,  # df
     1.0,  # resize_scale_t
     0.95,   # resize_scale_f
     20.0,  # w_content
     40.0, # w_edit
     0.1, # w_contrast
     0.1, # w_inpaint
     1.2,  # guidance_scale
     0.0,  # sde_strength
     1.5,  # energy_scale
    ]]