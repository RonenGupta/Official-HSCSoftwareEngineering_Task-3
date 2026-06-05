import torch

NOTIFICATIONS_ENABLED = True
SOUNDSENABLED = False
CURRENTVOLUME = 0.5

USER_DB = "users.json"
MUSIC_FOLDER = "music"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
