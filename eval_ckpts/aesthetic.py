import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import clip
import requests
import numpy as np
from PIL import Image


root = os.environ.get('AESTHETIC_ROOT', os.path.expanduser('~/cache/aesthetic'))
model_dict = dict()


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


def initialize_model(device):

    clip_model, preprocess = clip.load(
        "ViT-L/14", device=device, jit=False, download_root=root)
    mlp = MLP(768)

    if device == "cpu":
        clip_model.float()
    else:
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

    # have clip.logit_scale require no grad.
    clip_model.logit_scale.requires_grad_(False)

    weights_fname = "sac+logos+ava1-l14-linearMSE.pth"
    loadpath = os.path.join(root, weights_fname)

    if not os.path.exists(loadpath):
        url = (
            "https://github.com/christophschuhmann/"
            f"improved-aesthetic-predictor/blob/main/{weights_fname}?raw=true"
        )
        r = requests.get(url)

        with open(loadpath, "wb") as f:
            f.write(r.content)

    mlp.load_state_dict(torch.load(loadpath, map_location=torch.device("cpu")))
    mlp.to(device)
    mlp.requires_grad_(False)
    mlp.eval()
    model_dict["clip_model"] = clip_model
    model_dict["mlp"] = mlp
    model_dict["preprocess"] = preprocess


def score(imgs, prompts=None, device="cuda:0"):
    if not model_dict:
        initialize_model(device)

    clip_model = model_dict["clip_model"]
    mlp = model_dict["mlp"]
    preprocess = model_dict["preprocess"]
    rewards = []
    for img in imgs:
        # image encode
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if isinstance(img, str):
            img = Image.open(img)
        image = preprocess(img).unsqueeze(0).to(device)
        image_features = F.normalize(clip_model.encode_image(image)).float()
        # score
        rewards.append(mlp(image_features).detach().cpu())
    rewards = torch.cat(rewards, dim=0).squeeze()
    return rewards


def clear():
    model_dict.clear()
