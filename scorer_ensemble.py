import ImageReward as imagereward
import clip
import numpy as np
import os
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from clint.textui import progress
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from transformers import CLIPTextModel, CLIPTokenizer

SCORERS = dict()


def register_score(scorer_cls):
    try:
        SCORERS[scorer_cls.name] = scorer_cls
    except AttributeError:
        SCORERS[scorer_cls.__name__.lower()] = scorer_cls
    return scorer_cls


@register_score
class ImageRewardScorer:
    """Image Reward Scorer"""
    name = "image_reward"

    def __init__(
            self,
            device="cuda",
            weight_dtype=torch.float32,
            **kwargs
    ):
        self.device = device
        self.weight_dtype = torch.float32

    def load(self):
        self.reward_tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.reward_model = imagereward.load("ImageReward-v1.0")
        self.reward_model.to(device=self.device, dtype=self.weight_dtype)
        self.reward_model.requires_grad_(False)

    def unload(self):
        try:
            del self.reward_tokenizer
            del self.reward_model
        except AttributeError:
            pass

    def is_loaded(self):
        return hasattr(self, "reward_tokenizer") and hasattr(self, "reward_model")

    def calculate_score(self, imgs, prompts):
        blip_rewards = []
        for img, prompt in zip(imgs, prompts):
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            blip_rewards.append(self._get_image_reward(img, prompt)[0])
        return torch.stack(blip_rewards, dim=0).cpu().squeeze()

    def _get_image_reward(self, pil_image, prompt):
        """Gets rewards using ImageReward model."""
        image = (
            self.reward_model.preprocess(pil_image).unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
        )
        image_embeds = self.reward_model.blip.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        text_input = self.reward_model.blip.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)
        text_output = self.reward_model.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        txt_features = text_output.last_hidden_state[:, 0, :]
        rewards = self.reward_model.mlp(txt_features)
        rewards = (rewards - self.reward_model.mean) / self.reward_model.std
        return rewards, txt_features

    def __call__(self, imgs, prompts):
        if not self.is_loaded():
            self.load()
        return self.calculate_score(imgs, prompts)


@register_score
class HPSv2Scorer:
    """Human Preference Score v2 Scorer"""
    name = "hpsv2"

    def __init__(self, device, **kwargs):
        self.device = device
        self.environ_root = os.environ.get('HPS_ROOT')
        self.root_path = os.path.expanduser('~/.cache/hpsv2') if self.environ_root is None else self.environ_root
        self.cp = os.path.join(self.root_path, 'HPS_v2_compressed.pt')

        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path, exist_ok=True)

    def load(self):
        model, _, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=self.device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        self.model = model
        self.preprocess_val = preprocess_val

        # check if the checkpoint exists
        if not os.path.exists(self.cp):
            print('Downloading HPS_v2_compressed.pt ...')
            url = 'https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt'
            r = requests.get(url, stream=True)
            with open(self.cp, 'wb') as HPSv2:
                total_length = int(r.headers.get('content-length'))
                for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1):
                    if chunk:
                        HPSv2.write(chunk)
                        HPSv2.flush()
            print('Download HPS_2_compressed.pt to {} successfully.'.format(self.root_path + '/'))
        checkpoint = torch.load(self.cp, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.tokenizer = get_tokenizer('ViT-H-14')
        model = self.model.to(self.device)
        model.eval()

    def unload(self):
        try:
            del self.model
            del self.tokenizer
        except AttributeError:
            pass

    def is_loaded(self):
        return hasattr(self, "model") and hasattr(self, "tokenizer")

    def calculate_score(self, imgs, prompts):
        results = []
        for img, prompt in zip(imgs, prompts):
            # Load your image and prompt
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            with torch.no_grad():
                # Process the image
                if isinstance(img, str):
                    image = self.preprocess_val(
                        Image.open(img)
                    ).unsqueeze(0).to(device=self.device, non_blocking=True)
                elif isinstance(img, Image.Image):
                    image = self.preprocess_val(img).unsqueeze(0).to(device=self.device, non_blocking=True)
                else:
                    raise TypeError('The type of parameter img_path is illegal.')
                # Process the prompt
                text = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
                # Calculate the HPS
                with torch.cuda.amp.autocast():
                    outputs = self.model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T
                    hps_score = torch.diagonal(logits_per_image).cpu()
            results.append(hps_score[0].squeeze())
        return torch.stack(results, dim=0)

    def __call__(self, imgs, prompts):
        if not self.is_loaded():
            self.load()
        return self.calculate_score(imgs, prompts)


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


@register_score
class AestheticScore(nn.Module):
    name = "aesthetic"

    def __init__(self, device='cpu', **kwargs):
        super().__init__()
        self.device = device

    def load(self):
        self.root_path = os.path.expanduser('~/.cache/aesthetic')
        self.clip_model, self.preprocess = clip.load(
            "ViT-L/14", device=self.device, jit=False, download_root=self.root_path)
        self.mlp = MLP(768)

        if self.device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(
                self.clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # have clip.logit_scale require no grad.
        self.clip_model.logit_scale.requires_grad_(False)

        weights_fname = "sac+logos+ava1-l14-linearMSE.pth"
        loadpath = os.path.join(self.root_path, weights_fname)

        if not os.path.exists(loadpath):
            url = (
                "https://github.com/christophschuhmann/"
                f"improved-aesthetic-predictor/blob/main/{weights_fname}?raw=true"
            )
            r = requests.get(url)

            with open(loadpath, "wb") as f:
                f.write(r.content)

        self.mlp.load_state_dict(torch.load(loadpath, map_location=torch.device("cpu")))
        self.mlp.to(self.device)
        self.mlp.requires_grad_(False)
        self.mlp.eval()

    def unload(self):
        try:
            del self.clip_model
            del self.mlp
        except AttributeError:
            pass
    
    def is_loaded(self):
        return hasattr(self, "clip_model") and hasattr(self, "mlp")
    
    def calculate_score(self, imgs, prompts=None):
        rewards = []
        for img in imgs:
            # image encode
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            image = self.preprocess(img).unsqueeze(0).to(self.device)
            image_features = F.normalize(self.clip_model.encode_image(image)).float()
            # score
            rewards.append(self.mlp(image_features).detach().cpu())
        rewards = torch.cat(rewards, dim=0).squeeze()
        return rewards

    def __call__(self, imgs, prompts=None):
        if not self.is_loaded():
            self.load()
        return self.calculate_score(imgs, prompts=prompts)


class ScorerEnsemble:
    """Wrapper class to provide evaluation scores"""

    def __init__(self, scorers, pref_source, **kwargs):
        self.scorers = dict()
        try:
            for scorer in scorers:
                self.scorers[scorer] = SCORERS[scorer](**kwargs)
        except KeyError:
            raise NotImplementedError(f"Only supports {', '.join(SCORERS.keys())}.")
        assert pref_source in self.scorers
        self.pref_source = pref_source

    def train_mode(self):
        """Use for training the policy, we only need the scorer corresponding to `self.pref_source`.
            Unload the weights of all other scorers to save memory
        """
        for key in self.scorers.keys():
            if key != self.pref_source:
                self.scorers[key].unload()

    def test_mode(self):
        """Use for evaluating the generated image. We need all scorers.
            Load the weights of all scorers, if haven't.
        """
        for scorer in self.scorers.values():
            if not scorer.is_loaded():
                scorer.load()

    def get_all_scores(self, imgs, prompts):
        """Calculate all evaluation scores. Use in the final evaluation after training"""
        scores = dict()
        for scorer in self.scorers:
            scores[scorer] = self.scorers[scorer](imgs, prompts)
        return scores

    def get_pref_source_scores(self, imgs, prompts):
        """Only calculate and return the preference-source score to save compute. Use in the training process"""
        scores = dict()
        scores["PrefSourceScore"] = self.scorers[self.pref_source](imgs, prompts)
        return scores
