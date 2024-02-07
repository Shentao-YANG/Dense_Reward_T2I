import os
import requests
from clint.textui import progress
from typing import Union
from PIL import Image
import torch

from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

environ_root = os.environ.get('HPS_ROOT')
root_path = os.path.expanduser('~/.cache/hpsv2') if environ_root is None else environ_root
name = 'hpsv2'
url = 'https://github.com/tgxs002/HPSv2'
os.environ['NO_PROXY'] = 'huggingface.co'
available_models = None
model_dict = {}


def initialize_model(device, cp: str = os.path.join(root_path, 'HPS_v2_compressed.pt')):
    if not model_dict:
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=device,
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

        # check if the checkpoint exists
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        if cp == os.path.join(root_path, 'HPS_v2_compressed.pt') and not os.path.exists(cp):
            print('Downloading HPS_v2_compressed.pt ...')
            url = 'https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt'
            r = requests.get(url, stream=True)
            with open(os.path.join(root_path, 'HPS_v2_compressed.pt'), 'wb') as HPSv2:
                total_length = int(r.headers.get('content-length'))
                for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1):
                    if chunk:
                        HPSv2.write(chunk)
                        HPSv2.flush()
            print('Download HPS_2_compressed.pt to {} sucessfully.'.format(root_path + '/'))

        checkpoint = torch.load(cp, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()

        model_dict['model'] = model
        model_dict['preprocess_val'] = preprocess_val
        model_dict['tokenizer'] = get_tokenizer('ViT-H-14')


def _score(
        img_path: Union[list, str, Image.Image],
        prompt: str,
        cp: str = os.path.join(root_path, 'HPS_v2_compressed.pt'),
        device: str = "cuda"
) -> list:
    initialize_model(device, cp=cp)
    model = model_dict['model']
    preprocess_val = model_dict['preprocess_val']
    tokenizer = model_dict['tokenizer']

    if isinstance(img_path, list):
        result = []
        for one_img_path in img_path:
            # Load your image and prompt
            with torch.no_grad():
                # Process the image
                if isinstance(one_img_path, str):
                    image = preprocess_val(Image.open(one_img_path)).unsqueeze(0).to(device=device, non_blocking=True)
                elif isinstance(one_img_path, Image.Image):
                    image = preprocess_val(one_img_path).unsqueeze(0).to(device=device, non_blocking=True)
                else:
                    raise TypeError('The type of parameter img_path is illegal.')
                # Process the prompt
                text = tokenizer([prompt]).to(device=device, non_blocking=True)
                # Calculate the HPS
                with torch.cuda.amp.autocast():
                    outputs = model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T

                    hps_score = torch.diagonal(logits_per_image).cpu().numpy()
            result.append(hps_score[0])
        return result
    elif isinstance(img_path, str):
        # Load your image and prompt
        with torch.no_grad():
            # Process the image
            image = preprocess_val(Image.open(img_path)).unsqueeze(0).to(device=device, non_blocking=True)
            # Process the prompt
            text = tokenizer([prompt]).to(device=device, non_blocking=True)
            # Calculate the HPS
            with torch.cuda.amp.autocast():
                outputs = model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T

                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
        return [hps_score[0]]
    elif isinstance(img_path, Image.Image):
        # Load your image and prompt
        with torch.no_grad():
            # Process the image
            image = preprocess_val(img_path).unsqueeze(0).to(device=device, non_blocking=True)
            # Process the prompt
            text = tokenizer([prompt]).to(device=device, non_blocking=True)
            # Calculate the HPS
            with torch.cuda.amp.autocast():
                outputs = model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T

                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
        return [hps_score[0]]
    else:
        raise TypeError('The type of parameter img_path is illegal.')


def score(imgs_path: Union[list, str, Image.Image], prompt: str, device: str) -> list:
    """Score the image and prompt

    Args:
        imgs_path (Union[list, str, Image.Image]): paths to generated image(s)
        prompt (str): corresponding prompt

    Returns:
        list: matching scores for images and prompt
    """

    res = _score(imgs_path, prompt, device=device)
    return res


def clear():
    model_dict.clear()
