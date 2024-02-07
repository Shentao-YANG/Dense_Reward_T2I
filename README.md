# A Dense Reward View on Aligning Text-to-Image Diffusion with Preference

## Dependency

To install the required packages, please run the following command:
```angular2html
bash install_packages.sh
```

## Experiments

### Single Prompt Experiments
As a minimal example, our single prompt experiments can be run by the following command
```angular2html
accelerate launch train_t2i.py --expid="single"  
```

We provide our checkpoints in `./ckpts`.
To evaluate our checkpoints, please use the following command
```angular2html
cd eval_ckpts
torchrun --nproc_per_node $NUM_GPUS_TO_USE --standalone main --type="both_seen_unseen" --eval_generated_imgs=1 --metrics="image_reward,aesthetic" --outdir="./outputs"
```
Note:
- `$NUM_GPUS_TO_USE` is the number of gpus you want to use.
- Set `--eval_generated_imgs=0` if evaluating the generated images is not needed, *i.e.* only want to generate images.
- Add the flag `--return_traj` if you want to store the generation trajectory corresponding to each image as well.


### Multiple Prompt Experiments
As a minimal example, our multiple prompt experiments can be run by the following command
```angular2html
accelerate launch train_t2i.py --single_flag=0 --expid="multiple" 
```
*Declaimer: The HPSv2 train set has not been officially released at this moment. We are currently in the process of consulting with the HPSv2's authors on including those prompts in our repository. For now, we temporarily use the drawbench prompts as a substitution* 



## Acknowledgement

This codebase builds on the following codebases:
* [**DPOK**](https://github.com/google-research/google-research/tree/master/dpok)
* [**ImageReward**](https://github.com/THUDM/ImageReward)
* [**HPSv2**](https://github.com/tgxs002/HPSv2)
* [**Hugging Face**](https://github.com/huggingface/diffusers/blob/v0.23.1/examples/text_to_image/train_text_to_image_lora.py)










