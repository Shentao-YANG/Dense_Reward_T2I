# coding=utf-8

"""Helper functions."""

from collections import defaultdict
import itertools
import os
import torch
import numpy as np
import json
from PIL import Image
from datetime import datetime
from accelerate import Accelerator
import hpsv2


def print_dict(d): print(json.dumps(d, indent=2), flush=True)


def prints(s, warning=False):
    if warning:
        print(f"[WARNING !!!] {s}", flush=True)
    else:
        print(s, flush=True)


def make_banner(s, symbol="-", front=False, back=False):
    len_s = len(s)
    output = ""

    if front:
        output += (symbol * len_s + "\n")
    output += s
    if back:
        output += ("\n" + symbol * len_s)

    return output


def print_banner(s, symbol="-", front=False, back=False):
    print(make_banner(s, symbol, front, back), flush=True)


def get_snr_schedule(scheduler, num_timesteps, device, vmax=None):
    scheduler.set_timesteps(num_timesteps, device=device)
    timesteps = scheduler.timesteps
    alphas_cumprod = scheduler.alphas_cumprod.to(timesteps.device)[timesteps]
    snrs = alphas_cumprod / (1 - alphas_cumprod)
    if vmax is not None:
        snrs = torch.clamp(snrs, None, vmax)  # eg, min(SNR, 5)
    return snrs.float().cpu().numpy()


def image_folder_loader(folder):
    for f in sorted(os.listdir(folder)):    # sort to ensure the ordering
        if f.endswith(("png", "jpg", "jpeg", "gif", "bmp")):
            yield Image.open(os.path.join(folder, f))


def prompt_folder_loader(folder, idx_prompt_map):
    for f in sorted(os.listdir(folder)):    # sort to ensure the ordering
        if f.endswith(("png", "jpg", "jpeg", "gif", "bmp")):
            yield idx_prompt_map[f]


class ImageFolder:
    def __init__(self, folder):
        self.image_folder = folder

    def __iter__(self):
        return image_folder_loader(self.image_folder)


class PromptFolder:
    def __init__(self, folder):
        self.prompt_folder = folder
        with open(os.path.join(folder, "idx_prompt_map.json")) as f:
            self.idx_prompt_map = json.load(f)

    def __iter__(self):
        return prompt_folder_loader(self.prompt_folder, self.idx_prompt_map)


def evaluate_saved_imgs(eval_dir, scorer_ensemble, accelerator: Accelerator = None):
    start_time = datetime.now()
    folders = [
        os.path.join(eval_dir, f)
        for f in os.listdir(eval_dir)
        if f.startswith("iter")
    ]
    folders.sort(key=lambda x: int(x.split('iter')[1]))
    fname = os.path.join(eval_dir, "eval_results.json")
    eval_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    if accelerator is not None:
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        folders = folders[rank::world_size]
        tmp_dir = os.path.join(eval_dir, "tmp")
        if accelerator.is_main_process:
            os.makedirs(tmp_dir, exist_ok=True)
        accelerator.wait_for_everyone()
        fname = os.path.join(tmp_dir, f"eval_results_{rank}.json")

    for folder in folders:
        iter_num = folder.split("iter")[1]
        if not os.path.isdir(folder):
            print_banner(f"[WARNING] '{folder}' is NOT a folder")
            continue
        for prompt in os.listdir(folder):
            subfolder = os.path.join(folder, prompt)
            if not os.path.isdir(subfolder):
                print_banner(f"[WARNING] '{subfolder}' is NOT a folder")
                continue
            imgs = ImageFolder(subfolder)
            prompts = itertools.repeat(prompt)
            score_dict = scorer_ensemble.get_all_scores(imgs, prompts)
            for k, v in score_dict.items():
                eval_results[prompt][iter_num][k]['Mean'] = v.mean().item()
                eval_results[prompt][iter_num][k]['Std'] = v.std().item()

        print_banner(f"Finished Evaluating '{folder}'. Used time so far: {datetime.now() - start_time}",
                     front=True, back=True)

    with open(fname, "w") as json_file:
        json.dump(eval_results, json_file)

    if accelerator is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # merge the eval_results created by each individual process
            eval_results_unsorted = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
            result_list = os.listdir(tmp_dir)
            for result in result_list:
                with open(os.path.join(tmp_dir, result)) as f:
                    data = json.load(f)
                    for prompt in data.keys():
                        for iter_num in data[prompt].keys():
                            eval_results_unsorted[prompt][iter_num] = data[prompt][iter_num]

            # sort the results for each `prompt` by `iter_num`
            eval_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
            for prompt, prompt_res in eval_results_unsorted.items():
                # get and sort the list of all `iter_num`
                iter_num_sorted = sorted(list(prompt_res.keys()), key=lambda x: int(x))
                for iter_num in iter_num_sorted:
                    eval_results[prompt][iter_num] = prompt_res[iter_num]

            fname = os.path.join(eval_dir, "eval_results.json")
            with open(fname, "w") as json_file:
                json.dump(eval_results, json_file)
        accelerator.wait_for_everyone()

    def print_results():
        print_banner(f"Finished Evaluating '{eval_dir}'. "
                     f"Total time: {datetime.now() - start_time}", front=True, back=True)
        print_banner("Evaluation Results: ", symbol=("*" * 5), front=True)
        print_dict(eval_results)

    if accelerator is not None:
        if accelerator.is_main_process:
            print_results()
        accelerator.wait_for_everyone()
    else:
        print_results()


def evaluate_saved_imgs_multiprompts(eval_dir, scorer_ensemble, accelerator: Accelerator = None):
    start_time = datetime.now()
    folders = [
        os.path.join(eval_dir, f)
        for f in os.listdir(eval_dir)
        if f.startswith("iter")
    ]
    folders.sort(key=lambda x: int(x.split('iter')[1]))
    fname = os.path.join(eval_dir, "eval_results.json")
    eval_results = defaultdict(lambda: defaultdict(dict))

    # load style-prompt dict
    hpsv2_prompts = hpsv2.benchmark_prompts('all')
    for k, v in hpsv2_prompts.items():
        hpsv2_prompts[k] = set(v)

    def get_hpsv2_style(prompt):
        for k, v in hpsv2_prompts.items():
            if prompt in v:
                return k
        print_banner(f"WARNING: '{prompt}' is not in `hpsv2_prompts` !!!")
        return None

    hpsv2_fname = os.path.join(eval_dir, "hpsv2_eval_results.json")
    hpsv2_per_style_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    if accelerator is not None:
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        folders = folders[rank::world_size]
        tmp_dir = os.path.join(eval_dir, "tmp")
        hpsv2_tmp_dir = os.path.join(eval_dir, "hpsv2_tmp")
        if accelerator.is_main_process:
            os.makedirs(tmp_dir, exist_ok=True)
            os.makedirs(hpsv2_tmp_dir, exist_ok=True)
        accelerator.wait_for_everyone()
        fname = os.path.join(tmp_dir, f"eval_results_{rank}.json")
        hpsv2_fname = os.path.join(hpsv2_tmp_dir, f"hpsv2_eval_results_{rank}.json")

    for folder in folders:
        iter_num = folder.split("iter")[1]
        if not os.path.isdir(folder):
            print_banner(f"[WARNING] '{folder}' is NOT a folder")
            continue
        subfolder = os.path.join(folder, "hpsv2Test")
        if not os.path.isdir(subfolder):
            print_banner(f"[WARNING] '{subfolder}' is NOT a folder")
            continue
        imgs = ImageFolder(subfolder)
        prompts = PromptFolder(subfolder)
        score_dict = scorer_ensemble.get_all_scores(imgs, prompts)

        hpsv2_scores = score_dict["hpsv2"]
        for pmp, img_score in zip(prompts, hpsv2_scores.tolist()):
            pmp_style = get_hpsv2_style(pmp)
            if pmp_style is not None:
                hpsv2_per_style_results[iter_num][pmp_style]["all"].append(img_score)

        for style in hpsv2_per_style_results[iter_num].keys():
            hpsv2_per_style_results[iter_num][style]["mean"] = np.mean(hpsv2_per_style_results[iter_num][style]["all"]).item()
            hpsv2_per_style_results[iter_num][style]["Std"] = np.std(hpsv2_per_style_results[iter_num][style]["all"]).item()
        hpsv2_per_style_results[iter_num]["Overall"]["mean"] = hpsv2_scores.mean().item()
        hpsv2_per_style_results[iter_num]["Overall"]["Std"] = hpsv2_scores.std().item()

        for k, v in score_dict.items():
            eval_results[iter_num][k]['Mean'] = v.mean().item()
            eval_results[iter_num][k]['Std'] = v.std().item()

        print_banner(f"Finished Evaluating '{folder}'. Used time so far: {datetime.now() - start_time}", front=True, back=True)

    with open(fname, "w") as json_file:
        json.dump(eval_results, json_file)

    with open(hpsv2_fname, "w") as json_file:
        json.dump(hpsv2_per_style_results, json_file)

    if accelerator is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # merge the eval_results created by each individual process
            eval_results_unsorted = defaultdict(lambda: defaultdict(dict))
            result_list = os.listdir(tmp_dir)
            for result in result_list:
                with open(os.path.join(tmp_dir, result)) as f:
                    data = json.load(f)
                    for iter_num in data.keys():
                        eval_results_unsorted[iter_num] = data[iter_num]

            hpsv2_per_style_results_unsorted = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            result_list = os.listdir(hpsv2_tmp_dir)
            for result in result_list:
                with open(os.path.join(hpsv2_tmp_dir, result)) as f:
                    data = json.load(f)
                    for iter_num in data.keys():
                        hpsv2_per_style_results_unsorted[iter_num] = data[iter_num]

            # sort the results for each `prompt` by `iter_num`
            eval_results = defaultdict(lambda: defaultdict(dict))
            hpsv2_per_style_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

            # get and sort the list of all `iter_num`
            iter_num_sorted = sorted(list(eval_results_unsorted.keys()), key=lambda x: int(x))
            for iter_num in iter_num_sorted:
                eval_results[iter_num] = eval_results_unsorted[iter_num]
                hpsv2_per_style_results[iter_num] = hpsv2_per_style_results_unsorted[iter_num]

            fname = os.path.join(eval_dir, "eval_results.json")
            hpsv2_fname = os.path.join(eval_dir, "hpsv2_eval_results.json")
            with open(fname, "w") as json_file:
                json.dump(eval_results, json_file)
            with open(hpsv2_fname, "w") as json_file:
                json.dump(hpsv2_per_style_results, json_file)

        accelerator.wait_for_everyone()

    def print_results():
        print_banner(f"Finished Evaluating '{eval_dir}'. "
                     f"Total time: {datetime.now() - start_time}", front=True, back=True)
        print_banner("Evaluation Results: ", symbol=("*" * 5), front=True)
        print_dict(eval_results)

    # drop per-image HPSv2 scores for printing purpose
    for iter_num_dict in hpsv2_per_style_results.values():
        for iter_num_style_dict in iter_num_dict.values():
            iter_num_style_dict.pop("all", None)

    def print_hpsv2_per_style_results():
        print_banner(f"HPSv2 Test Prompts Per-style Results: ", symbol=("-" * 4), front=True)
        print_dict(hpsv2_per_style_results)

    if accelerator is not None:
        if accelerator.is_main_process:
            print_results()
            print_hpsv2_per_style_results()
        accelerator.wait_for_everyone()
    else:
        print_results()
        print_hpsv2_per_style_results()
