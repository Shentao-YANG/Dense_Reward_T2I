import argparse
import os
from utils import print_banner


def get_default_parser():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script."
    )
    parser.add_argument("--expid", type=str, default=None, help="Experiment ID to organize results/logs.")
    # <------------------------------------------------------------> #
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=10000,
        help=(
            "Total number of training steps to perform.  If provided, overrides"
            " num_train_epochs."
        ),
    )
    parser.add_argument(
        "--p_step",
        type=int,
        default=2500,
        help="The number of steps to update the policy per sampling step",
    )
    parser.add_argument(
        "--test_img_gen_freq",
        type=int,
        default=None,
        help=(
            "Frequency to generate and store the text image during the training process, in number of training steps"
        ),
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine",'
            ' "cosine_with_restarts", "polynomial", "constant",'
            ' "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=2e-3,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--g_step", type=int, default=1, help="(NOT USED) The number of sampling steps"
    )
    parser.add_argument(
        "--g_batch_size",
        type=int,
        default=1000,
        help="Total number of prompts added to the replay buffer each time of calling `collect_rollout`, across ALL processes. "
             "Will be changed to `g_batch_size /= num_processes` in the start of `main()`",
    )
    parser.add_argument(
        "--num_traj_for_pref_comp",
        type=int,
        default=5,
        help="Number of trajectories from the same prompt in calculating the ground-truth preference. "
             "Total number of trajectories in the replay buffer is (max_replay_buffer_size x num_traj_for_pref_comp). "
             "Default: the same as `pl_loss_num_traj` ('K choose K')",
    )
    parser.add_argument(
        "--max_replay_buffer_size",
        type=int,
        default=1000,
        help="Number of prompts in the replay buffer. "
             "Total number of trajectories in the replay buffer is (max_replay_buffer_size x num_traj_for_pref_comp)",
    )
    parser.add_argument(
        "--init_buffer_size",
        type=int,
        default=1000,
        help="Initial size of the replay buffer before policy training",
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=100,
        help="Total number of samples to generate per prompt in each call to `generate_test_img`, across ALL processes",
    )
    parser.add_argument(
        "--num_steps_est_logits",
        type=int,
        default=3,
        help="Number of steps to estimate the logits, when formulating logits as expectation"
    )
    parser.add_argument(
        "--total_train_batch_size",
        type=int,
        default=4,
        help="Total training batch size, across ALL processes. "
             "`gradient_accumulation_steps` will be calculated and add to `args` in the start of `main()`"
    )
    parser.add_argument(
        "--p_batch_size",
        type=int,
        default=1,
        help=(
            "batch size for policy update per gpu, before gradient accumulation;"
            " total batch size per gpu = gradient_accumulation_steps * p_batch_size"
        ),
    )
    parser.add_argument("--lora_rank", type=int, default=4, help="rank for LoRA")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Learning rate for policy",
    )
    parser.add_argument(
        "--reg_to_pi_old",
        type=int,
        default=0,
        help="Whether to add KL regularization to pi_old",
    )

    parser.add_argument(
        "--reg_to_pi_init",
        type=int,
        default=1,
        help="Whether to add KL regularization to pi_init",
    )

    parser.add_argument(
        "--pl_loss_num_traj",
        type=int,
        default=2,
        help="Number of trajectories from the same prompt to calculate the Plackett-Luce loss for policy training",
    )

    parser.add_argument(
        "--pl_loss_temp",
        type=float,
        default=10.,
        help="Regularization (a.k.a. temperature or KL) parameter in the Plackett-Luce loss for policy training",
    )

    parser.add_argument(
        "--pl_weights_name",
        type=str,
        default="gamma",
        help="Weighting scheme for the step-wise log density rario. (default: power of the discount factor)",
    )

    parser.add_argument(
        "--pl_gamma",
        type=float,
        default=0.9,
        help="The discount factor in the MDP",
    )
    parser.add_argument(
        "--reward_flag",
        type=int,
        default=0,
        help="0: ImageReward, 1: HPSv2",
    )
    parser.add_argument(
        "--no_reg_warmup_ratio",
        type=float,
        default=-1,
        help="Control the number of steps where the policy is trained without regularization;"
             "to avoid logits being to small, since initially pi_theta = pi_init."
             "Use -1 to de-function",
    )

    parser.add_argument(
        "--use_cfg_in_train",
        type=int,
        default=1,
        help="Whether to use cfg in policy training, affecting replay buffer and training logits",
    )
    parser.add_argument(
        "--pl_loss_num_tuples",
        type=int,
        default=1,
        help="In training the policy, the number of comparison tuples from each prompt. "
             "Should be smaller than (`num_traj_for_pref_comp` choose `pl_loss_num_traj`), which may be too slow for training",
    )
    parser.add_argument(
        "--min_snr_vmax",
        type=float,
        default=5.,
        help="Maximum value in the Min-SNR schedule",
    )
    parser.add_argument(
        "--num_rollout_trajs",
        type=int,
        default=None,
        help="Number of rollout trajectories in `collect_rollout()`; larger than `num_traj_for_pref_comp` means doing exploration",
    )
    parser.add_argument(
        "--rollout_trajs_record_start",
        type=int,
        default=0,
        help="Recording start from the trajectory with the (X+1)-th smallest reward",
    )
    # <------------------------------------------------------------> #

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help=(
            "Path to pretrained model or model identifier from"
            " huggingface.co/models."
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on"
            " (could be your own, possibly private, dataset). It can also be a"
            " path pointing to a local copy of a dataset in your filesystem, or"
            " to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help=(
            "The config of the Dataset, leave as None if there's only one config."
        ),
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow"
            " the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In"
            " particular, a `metadata.jsonl` file must exist to provide the"
            " captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help=(
            "The column of the dataset containing a caption or a list of captions."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help=(
            "The output directory where the model predictions and checkpoints will be written."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the"
            " train/validation dataset will be resized to this resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=True,
        help=(
            "Whether to center crop the input images to the resolution. If not"
            " set, the images will be randomly cropped. The images will be"
            " resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=True,
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=(
            "Whether or not to use gradient checkpointing to save memory at the"
            " expense of slower backward pass."
        ),
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help=(
            "Scale the learning rate by the number of GPUs, gradient accumulation"
            " steps, and batch size."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up"
            " training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", default=False, help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch,"
            " tag or git identifier of the local or remote repository specified"
            " with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the"
            " data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory."
            " Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16"
            " (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU."
            "  Default to the value of accelerate config of the current system or"
            " the flag passed with the `accelerate.launch` command. Use this"
            " argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            "The integration to report the results and logs to. Supported"
            ' platforms are `"tensorboard"` (default), `"wandb"` and'
            ' `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=None,
        help=(
            "Save a checkpoint of the training state every X updates. These"
            " checkpoints are only suitable for resuming training using"
            " `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the"
            " `Accelerator` `ProjectConfiguration`. See Accelerator::save_state"
            " https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a"
            ' path saved by `--checkpointing_steps`, or `"latest"` to'
            " automatically select the last available checkpoint."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    parser.add_argument(
        "--sft_path",
        type=str,
        default="./checkpoints/models/finetune_model",
        help="path to the pretrained supervised finetuned model",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=None,
        help="save model every save_interval steps",
    )
    parser.add_argument(
        "--clip_norm", type=float, default=1., help="norm for gradient clipping"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="./dataset/drawbench/data_meta.json",
        help="path to the prompt dataset",
    )
    parser.add_argument(
        "--prompt_category",
        type=str,
        default="all",
        help="all or specific categories with comma [e.g., color,count]",
    )
    parser.add_argument(
        "--single_flag",
        type=int,
        default=1,
        help="flag for training on a single prompt"
    )
    parser.add_argument(
        "--single_prompt",
        type=str,
        default="A green colored rabbit.",
    )

    parser.add_argument(
        "--unseen_prompt",
        type=str,
        default="A green colored cat.",
    )

    parser.add_argument(
        "--multiprompt_eval_ratio",
        type=float,
        default=1.,
        help="Percentage of the multiprompt dataset used for evaluation. Set to `< 1.` if the dataset is too large."
    )

    parser.add_argument(
        "--sft_initialization",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--multi_gpu",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--log_ratio_clip",
        type=float,
        default=1e-4,      # original setting
        help="Bounds for soft-clipping the density ratios on the log space"
    )

    parser.add_argument(
        "--logging_interval",
        type=int,
        default=None,
        help="Logging evey X steps during the policy learning process"
    )
    return parser


def parse_args():
    """Parse command line flags."""
    parser = get_default_parser()
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    if args.num_traj_for_pref_comp is None:
        # default: `K choose K`
        args.num_traj_for_pref_comp = args.pl_loss_num_traj

    if args.single_flag == 0:
        print_banner("SET training hyperparameters to the Multi-prompts settings")
        args.lora_rank = 32
        args.log_ratio_clip = 5e-4
        args.max_train_steps = 40000
        args.p_step = 4000
        args.g_batch_size = 2000
        args.max_replay_buffer_size = 2000
        args.init_buffer_size = 2000
        args.num_steps_est_logits = 1
        args.pl_loss_temp = 12.5
        args.total_train_batch_size = 32
        args.learning_rate = 2e-5
        args.adam_weight_decay = 1.5e-3
        args.clip_norm = 0.05
        args.reward_flag = 1
        args.num_rollout_trajs = 12

    if args.test_img_gen_freq is None:
        args.test_img_gen_freq = args.max_train_steps // 10

    if args.save_interval is None:
        args.save_interval = args.test_img_gen_freq

    if args.logging_interval is None:
        args.logging_interval = args.max_train_steps // 1000

    if args.checkpointing_steps is None:
        args.checkpointing_steps = args.max_train_steps // 5

    if args.num_rollout_trajs is None:
        args.num_rollout_trajs = args.num_traj_for_pref_comp

    return args
