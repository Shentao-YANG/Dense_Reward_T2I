import numpy as np
from diffusers import DDIMScheduler
from diffusers import StableDiffusionPipeline
import torch
import torch.distributed as tdist
from typing import Union
from stable_diffusion_pipeline_traj import StableDiffusionPipelineTraj
import time
import random
import gc
import ast
import json

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@torch.inference_mode()
def generate(
        model_path, prompt, save_dir, device="cuda:0",
        seed: Union[list[int], int] = 42, return_traj=False,
):

    if return_traj:
        pipe = StableDiffusionPipelineTraj.from_pretrained("runwayml/stable-diffusion-v1-5")
    else:
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if model_path is not None:
        pipe.unet.load_attn_procs(torch.load(model_path))
    pipe.to(device)
    if isinstance(seed, int):
        seed = [seed]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for s in seed:
        img_path = f"{save_dir}/{s}_img.png"
        traj_path = f"{save_dir}/{s}_traj.png"
        skip = os.path.exists(img_path)
        skip &= not return_traj | os.path.exists(traj_path)
        if skip:
            continue
        generator = torch.Generator(device).manual_seed(s)
        outputs = pipe(prompt=prompt, eta=1.0, output_type="pil", generator=generator)
        img = outputs.images[0]
        img.save(img_path)
        if return_traj:
            traj = outputs.trajs[0]
            traj.save(traj_path)


@torch.inference_mode()
def eval_dir(img_dir, prompt, metric, device="cuda", rank=0, world_size=1):
    imgs = sorted([
        f for f in os.listdir(img_dir)
        if f.endswith(".png")
    ])[rank::world_size]
    img_paths = list(map(lambda f: os.path.join(img_dir, f), imgs))
    if metric == "image_reward":
        import ImageReward as RM
        model = None
        while model is None:
            try:
                model = RM.load("ImageReward-v1.0", device=device)
            except FileNotFoundError:   # error caused by racing in reading the file from multiple processes
                time.sleep(0.1 * random.random())
        model.requires_grad_(False)
        scores = model.score(prompt, img_paths)
    elif metric == "hpsv2":
        import hpsv2_mp as hpsv2
        scores = hpsv2.score(img_paths, prompt, device=device)
        hpsv2.clear()
    elif metric == "aesthetic":
        import aesthetic as aes
        scores = aes.score(img_paths)       # aes does not require prompt to score
        aes.clear()
    if isinstance(scores, torch.Tensor):
        scores = scores.numpy()
    if isinstance(scores, np.ndarray):
        scores = np.atleast_1d(scores).tolist()
    elif isinstance(scores, float):
        scores = [scores]

    return dict(zip(imgs, scores))


def eval(img_dir, prompt, metrics, rank=0, world_size=1):
    json_path = f"{img_dir}/eval_results.json"
    if os.path.exists(json_path):
        print(f"\n{json_path} already exists!!!\n")
        return None
    eval_results = []
    score_dicts = dict()
    for metric in metrics:
        score_dicts[metric] = eval_dir(img_dir, prompt, metric, device, rank, world_size)
    for k in score_dicts[metrics[0]].keys():
        record = dict(image=k, prompt=prompt)
        for metric in metrics:
            record[metric] = score_dicts[metric][k]
        eval_results.append(str(record) + "\n")
    with open(f"{img_dir}/eval_results.json.{rank}", "w") as f:
        f.writelines(eval_results)
    tdist.barrier()
    if rank == 0:
        eval_results = []
        for rank in range(world_size):
            with open(f"{img_dir}/eval_results.json.{rank}") as f:
                eval_results.extend(f.readlines())
            os.remove(f"{img_dir}/eval_results.json.{rank}")
        score_dict = {metric: [] for metric in metrics}
        score_dict["prompt"] = prompt
        for res in eval_results:
            temp = ast.literal_eval(res.strip())
            for metric in metrics:
                score_dict[metric].append(temp[metric])
        for metric in metrics:
            score_dict[metric] = dict(mean=np.mean(score_dict[metric]).item(), std=np.std(score_dict[metric]).item())
        with open(json_path, "w") as f:
            json.dump(score_dict, f, indent=2)
    tdist.barrier()


def main(
        model_path, prompt, save_dir, device, seed, return_traj=False, do_eval=True, metrics=("image_reward", ),
        rank=0, world_size=1
):
    generate(
        model_path, prompt, save_dir, device,
        seed=seed, return_traj=return_traj)
    tdist.barrier()
    gc.collect()
    torch.cuda.empty_cache()
    if do_eval:
        eval(save_dir, prompt, metrics, rank, world_size)


if __name__ == "__main__":

    from argparse import ArgumentParser
    import os

    def comma_separated_str(str):
        return str.lower().strip().split(",")

    parser = ArgumentParser()
    parser.add_argument("-s", "--seeds", type=str, default="0-100",
                        help='range of random seeds to generate images')
    parser.add_argument("-t", "--type", choices=("seen_only", "unseen_only", "both_seen_unseen"),
                        help="generate (and evaluate) seen prompts only, unseen prompts only, or both seen and unseen prompts")
    parser.add_argument("--model_path", type=str, default="./ckpts",
                        help="path to the saved checkpoints")
    parser.add_argument("--return_traj", action="store_true",
                        help="whether to store the corresponding generation trajectory for each image")
    parser.add_argument("--eval_generated_imgs", default=1, type=int, choices=(0, 1),
                        help="whether to evaluate the generated images using the specified metrics")
    parser.add_argument("--metrics", type=comma_separated_str, default="image_reward",
                        help="evaluation metrics to evaluate the generated images")
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()
    args.eval_generated_imgs = args.eval_generated_imgs == 1

    tdist.init_process_group(backend="nccl")

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = f"cuda:{rank}"
    print(f"Rank: {rank} / World size: {world_size}")

    seen_prompts = (
        "A_cat_and_a_dog.",
        "A_dog_on_the_moon.",
        "A_green_colored_rabbit.",
        "Four_wolves_in_the_park.",
    )
    unseen_prompts = (
        "A_cat_and_a_cup.",
        "A_lion_on_the_moon.",
        "A_green_colored_cat.",
        "Four_birds_in_the_park.",
    )

    start_seed, end_seed = args.seeds.split("-")
    start_seed = int(start_seed)
    end_seed = int(end_seed)

    for i in range(len(seen_prompts)):
        model_path = f"{args.model_path}/{seen_prompts[i].rstrip('.')}/pytorch_model.bin"
        if args.type == "seen_only":
            prompts = [seen_prompts[i]]
        elif args.type == "unseen_only":
            prompts = [unseen_prompts[i]]
        elif args.type == "both_seen_unseen":
            prompts = [seen_prompts[i], unseen_prompts[i]]
        for prompt in prompts:
            main(
                model_path, prompt.replace("_", " "), f"{args.outdir}/{prompt}", device=device,
                seed=list(range(start_seed + rank, end_seed, world_size)), return_traj=args.return_traj,
                do_eval=args.eval_generated_imgs, metrics=args.metrics, rank=rank, world_size=world_size)
