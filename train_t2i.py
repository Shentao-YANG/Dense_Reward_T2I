# coding=utf-8

import json
import logging
import os
from typing import List
import shutil
import math
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import datasets
from datasets import load_dataset
import diffusers
from diffusers import UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
if diffusers.__version__ < "0.20.0":
    from diffusers.models.cross_attention import LoRACrossAttnProcessor
else:
    from diffusers.models.attention_processor import LoRAAttnProcessor as LoRACrossAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import deprecate
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from pipeline_stable_diffusion_extended import StableDiffusionPipelineExtended
from scheduling_ddim_extended import DDIMSchedulerExtended
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import torch.utils.checkpoint
import transformers
import utils
from parse_args import parse_args

from replay_buffer import ReplayBufferTorch
from scorer_ensemble import ScorerEnsemble
from preference_based_policy_learner import PreferenceBasedPolicyTrainer

logger = get_logger(__name__, log_level="INFO")


def _update_output_dir(args):
    """Modifies `args.output_dir` using configurations in `args`.
    """
    if args.single_flag == 1:
        data_log = args.single_prompt.replace(" ", "_") + "/"
    else:
        data_log = args.prompt_path.split("/")[-2] + "_"
        data_log += args.prompt_category + "/"
    args.output_dir += f"/exp{args.expid}/{data_log}"


def main():
    args = parse_args()
    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is"
                " deprecated. Please make sure to use `--variant=non_ema` instead."
            ),
        )
    # Change log dir
    _update_output_dir(args)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        logging_dir=logging_dir, total_limit=args.checkpoints_total_limit
    )
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    args.g_batch_size = math.ceil(args.g_batch_size / accelerator.num_processes)
    args.gradient_accumulation_steps = math.ceil(args.total_train_batch_size / (accelerator.num_processes * args.p_batch_size))

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        # delete the old debugging logs
        if args.expid == "_debug" and os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)

        logger.info(utils.make_banner(f"Output save to: {args.output_dir}", front=True, back=True))
        os.makedirs(args.output_dir, exist_ok=True)
        start_time = datetime.now()

    # Load scheduler and models.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if accelerator.is_main_process:
        accelerator.print(utils.make_banner(f"SET torch_dtype = {weight_dtype} !!!", front=True, back=True))

    if args.sft_initialization == 0:
        pipe = StableDiffusionPipelineExtended.from_pretrained(
            args.pretrained_model_name_or_path, torch_dtype=weight_dtype
        )
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.non_ema_revision,
        )
    else:
        pipe = StableDiffusionPipelineExtended.from_pretrained(
            args.sft_path, torch_dtype=weight_dtype
        )
        unet = UNet2DConditionModel.from_pretrained(
            args.sft_path, subfolder="unet", revision=args.non_ema_revision
        )

    pipe.scheduler = DDIMSchedulerExtended.from_config(pipe.scheduler.config)
    vae = pipe.vae
    unet.requires_grad_(False)
    unet.eval()
    text_encoder = pipe.text_encoder
    pipe.set_progress_bar_config(disable=True)

    # disable safety checker to save memory
    # pipe.safety_checker = None

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # `pi_init` in our paper: pretrain model to calculate (the kl wrt the initial model)
    unet_copy = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.non_ema_revision,
    )
    # freeze unet copy
    unet_copy.requires_grad_(False)
    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    unet_copy.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.revision,
        )
        ema_unet = EMAModel(
            ema_unet.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=ema_unet.config,
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you"
                    " observe problems during training, please update xFormers to at least 0.0.17. See"
                    " https://huggingface.co/docs/diffusers/main/en/optimization/xformers"
                    " for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            if accelerator.is_main_process:
                accelerator.print(utils.make_banner("Using xFormers memory efficient attention!!", front=True, back=True))
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # Define lora layers
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRACrossAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=args.lora_rank,
        )

    unet.set_attn_processor(lora_attn_procs)
    # lora_layers = AttnProcsLayers(unet.attn_processors)

    # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
    # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
    # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
    class _Wrapper(AttnProcsLayers):
        def forward(self, *args, **kwargs):
            return unet(*args, **kwargs)

    lora_layers = _Wrapper(unet.attn_processors)

    # Enable TF32 for faster training on Ampere GPUs,
    if args.gradient_checkpointing:
        if accelerator.is_main_process:
            accelerator.print(utils.make_banner("Using gradient checkpointing!!!", front=True, back=True))
        unet.enable_gradient_checkpointing()

    if any(x in torch.cuda.get_device_name() for x in ("A", "30", "40")) or args.allow_tf32:
        if accelerator.is_main_process:
            accelerator.print(utils.make_banner("Using TF32 on Ampere GPUs!!!", front=True, back=True))
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate
                * args.gradient_accumulation_steps
                * args.p_batch_size
                * accelerator.num_processes
        )

    # Initialize the optimizer
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.p_batch_size
        accelerator.state.deepspeed_plugin.deepspeed_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
        accelerator.state.deepspeed_plugin.gradient_accumulation_steps = args.gradient_accumulation_steps

        from deepspeed.ops.adam import DeepSpeedCPUAdam
        optimizer_cls = DeepSpeedCPUAdam
        args.use_8bit_adam = False
    else:
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError as exc:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by"
                    " running `pip install bitsandbytes`"
                ) from exc

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

    if accelerator.is_main_process:
        accelerator.print(utils.make_banner(f"Optimizer class: {optimizer_cls}", front=True, back=True))

    optimizer = optimizer_cls(
        lora_layers.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if args.seed is not None and args.single_flag != 1:
        # multi-prompt: large batch size on multiple GPU
        set_seed(args.seed, device_specific=True)
        # single prompt: small batch size on single GPU so no need to reset seed

    # In distributed training, the load_dataset function guarantees that only one
    # local process can concurrently download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        with open(args.prompt_path) as json_file:
            prompt_dict = json.load(json_file)
        if args.prompt_category != "all":
            prompt_category = [e for e in args.prompt_category.split(",")]
        prompt_list = []
        for prompt in prompt_dict:
            category = prompt_dict[prompt]["category"]
            if args.prompt_category != "all":
                if category in prompt_category:
                    prompt_list.append(prompt)
            else:
                prompt_list.append(prompt)
        if accelerator.is_main_process:
            accelerator.print(utils.make_banner(f"Loaded {len(prompt_list)} prompts in categories: `{args.prompt_category}`! Examples:\n", front=True, back=False))
            accelerator.print(*prompt_list[:3], sep="\n")

    # Map-style prompt dataset
    def _my_data_iterator(data: List[str], batch_size, num_processes):
        class PromptDataset(Dataset):
            def __init__(self, prompts):
                data_len = len(prompts)
                self.lcm = math.lcm(data_len, batch_size * num_processes)
                self.prompts = prompts

            def __len__(self):
                return self.lcm

            def __getitem__(self, idx):
                _idx = idx % len(self.prompts)
                return self.prompts[_idx]

        return DataLoader(PromptDataset(data), batch_size=batch_size, shuffle=True)

    data_iterator = _my_data_iterator(
        prompt_list, batch_size=args.g_batch_size, num_processes=accelerator.num_processes)
    data_iterator = accelerator.prepare(data_iterator)

    if accelerator.state.deepspeed_plugin is not None:
        from deepspeed.runtime.lr_schedules import WarmupLR
        assert args.lr_scheduler in ("constant", "constant_with_warmup")
        lr_scheduler = WarmupLR(
            optimizer=optimizer,
            warmup_max_lr=args.learning_rate,
            warmup_num_steps=args.lr_warmup_steps,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

    if accelerator.is_main_process:
        accelerator.print(utils.make_banner(f"using `lr_scheduler = {lr_scheduler}`", front=True, back=True))

    # Prepare everything with our `accelerator`.
    if args.multi_gpu:
        unet, optimizer, lr_scheduler = accelerator.prepare(
            unet, optimizer, lr_scheduler
        )
    else:
        lora_layers, optimizer, lr_scheduler = accelerator.prepare(
            lora_layers, optimizer, lr_scheduler
        )
    if args.use_ema:
        ema_unet.to(accelerator.device)

    if accelerator.is_main_process:
        accelerator.init_trackers(project_name="text2image", config=vars(args))

    pipe.scheduler = DDIMSchedulerExtended.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    data_iter_loader = iter(data_iterator)
    pipe.unet = unet

    replay_buffer = ReplayBufferTorch(weight_dtype=weight_dtype, args=args)

    if args.reward_flag == 0:
        pref_source = "image_reward"
    elif args.reward_flag == 1:
        pref_source = "hpsv2"
    else:
        raise NotImplementedError(f"NOT support `reward_flag`={args.reward_flag}")
    scorer_ensemble = ScorerEnsemble(
        scorers=["image_reward", "hpsv2", "aesthetic"],
        pref_source=pref_source,
        device=accelerator.device,
        weight_dtype=weight_dtype,
    )

    if args.pl_weights_name == "gamma":
        # [γ^0, γ^1, ..., γ^49]
        if accelerator.is_main_process:
            accelerator.print(utils.make_banner("Using gamma-weighting scheme for the policy !!!", front=True, back=True))
        policy_loss_weights = (torch.tensor(args.pl_gamma)
                               .pow(torch.tensor(range(replay_buffer.num_denosing_steps), dtype=torch.float))
                               .numpy())
    elif args.pl_weights_name == "reci_snr":    # 1/snr to emphasis early stage of the reverse chain
        if accelerator.is_main_process:
            accelerator.print(utils.make_banner("Using Reciprocal SNR weighting scheme for the policy !!!", front=True, back=True))
        policy_loss_weights = (1. / utils.get_snr_schedule(pipe.scheduler, 50, accelerator.device))
    elif args.pl_weights_name == "reci_min_snr":    # 1/min_snr to emphasis early stage of the reverse chain
        vmax = args.min_snr_vmax
        if accelerator.is_main_process:
            accelerator.print(utils.make_banner(f"Using Reciprocal Min-SNR weighting scheme for the policy with vmax={vmax} !!!", front=True, back=True))
        policy_loss_weights = (1. / utils.get_snr_schedule(pipe.scheduler, 50, accelerator.device, vmax=vmax))
    else:
        raise NotImplementedError(f"Unsupported policy-loss weighting-scheme: {args.pl_weights_name}")

    policy_trainer = PreferenceBasedPolicyTrainer(
        pipe=pipe,
        wrapped_unet=lora_layers,
        initial_unet=unet_copy,
        scorer_ensemble=scorer_ensemble,
        replay_buffer=replay_buffer,
        accelerator=accelerator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        prompt_list=prompt_list,
        data_iter_loader=data_iter_loader,
        data_iterator=data_iterator,
        policy_loss_weights=policy_loss_weights,
        args=args
    )

    scorer_ensemble.train_mode()
    policy_trainer.train_model()

    if policy_trainer.policy_update_steps % args.test_img_gen_freq != 0:
        # generate test img after training, which has not been done before
        policy_trainer.generate_eval_imgs_during_training()

    scorer_ensemble.test_mode()
    if args.single_flag == 1:
        utils.evaluate_saved_imgs(os.path.join(args.output_dir, "saved_imgs"), scorer_ensemble, accelerator=accelerator)
    else:   # multiple prompts
        utils.evaluate_saved_imgs_multiprompts(os.path.join(args.output_dir, "saved_imgs"), scorer_ensemble, accelerator=accelerator)

    if accelerator.is_main_process:
        accelerator.print(utils.make_banner(f"Finished Exp-{args.expid} !!! Used time {datetime.now() - start_time} !!!",
                                            symbol="*", front=True, back=True))


if __name__ == "__main__":
    main()
