import torch
import torch.nn.functional as F
from reward_loss import listMLELoss
import copy
import dataclasses
from utils import make_banner, print_banner
from datetime import datetime
from functools import partial
import os
from math import ceil
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import numpy as np
from accelerate import Accelerator
import hpsv2
import random
import json
import shutil


@dataclasses.dataclass(frozen=False)
class TrainPolicyLogData:
    # Moving average of training loss and grad norm
    avg_p_loss: float = 0.
    avg_grad_norm: float = 0.
    step_p_loss: float = 0.


COLLECTIVE_FN = "broadcast"


def zeros_multigpu(*shape, world_size):
    return [torch.zeros(shape, device=f"cuda:{i}") for i in range(world_size)]


def all_gather_multigpu(x, world_size, dim=0):
    tensor_list = zeros_multigpu(x.shape, world_size=world_size)
    dist.all_gather(tensor_list, x)
    return torch.cat(tensor_list, dim=dim)


class PreferenceBasedPolicyTrainer:
    def __init__(
            self,
            pipe,
            wrapped_unet,
            initial_unet,
            scorer_ensemble,
            replay_buffer,
            accelerator: Accelerator,
            optimizer,
            lr_scheduler,
            prompt_list,
            data_iter_loader,
            data_iterator,
            policy_loss_weights,
            args
    ):
        self.pipe = pipe
        self.wrapped_unet = wrapped_unet
        self.scorer_ensemble = scorer_ensemble
        self.replay_buffer = replay_buffer
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.generator = torch.Generator(device=accelerator.device) \
            .manual_seed(12700 + accelerator.process_index)

        self.np_generator = np.random.default_rng(12700 + accelerator.process_index)

        self.prompt_list = prompt_list
        self.data_iter_loader = data_iter_loader
        self.data_iterator = data_iterator
        self.unet = self.pipe.unet
        self.is_ddp = isinstance(self.unet, DistributedDataParallel)
        self.initial_unet = initial_unet

        self.policy_loss_weights = policy_loss_weights / (policy_loss_weights.sum() + 1e-8)
        assert isinstance(self.policy_loss_weights, np.ndarray)
        assert len(self.policy_loss_weights.shape) == 1

        self.args = args
        self.train_log = TrainPolicyLogData()

        # soft-clipping on the log space, should be around log(1) = 0
        self.soft_clip = partial(torch.clamp, min=(-args.log_ratio_clip), max=args.log_ratio_clip)

        self.policy_update_steps = 0
        self.policy_update_steps_after_rollout = 0
        self.data_collection_times = self.args.max_train_steps // self.args.p_step
        self.world_size = self.accelerator.num_processes
        self.rank = self.accelerator.process_index
        self.local_rank = self.accelerator.local_process_index

        self.no_reg_pi_init_warmup_steps = max(self.args.no_reg_warmup_ratio * self.args.max_train_steps, -1)
        self.no_reg_pi_old_warmup_steps = max(self.args.no_reg_warmup_ratio * self.args.p_step, -1)
        self.rollout_store_idx = torch.tensor(np.linspace(self.args.rollout_trajs_record_start, self.args.num_rollout_trajs-1, self.args.num_traj_for_pref_comp),
                                              dtype=torch.long)

        self.cfg_guide_scale = 7.5 if self.args.use_cfg_in_train else 1.

        if self.accelerator.is_main_process:
            self.accelerator.print(
                make_banner(f"Initialized `PreferenceBasedPolicyTrainer`! Pref Src: {self.scorer_ensemble.pref_source}; "
                            f"Device: {self.rank + 1}/{self.world_size}"))

    def resume_from_checkpoint(self) -> None:
        if self.args.resume_from_checkpoint != "latest":
            path = os.path.basename(self.args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(self.args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            if self.accelerator.is_main_process:
                print_banner(
                    f"\n[WARNING!!!] Checkpoint '{self.args.resume_from_checkpoint}' does not exist!!! Starting"
                    " a new training run!!!\n", symbol="*", front=True, back=True
                )
            self.args.resume_from_checkpoint = None
        else:
            path = os.path.join(self.args.output_dir, path)
            if self.accelerator.is_main_process:
                random_state_path = os.path.join(path, "random_states_0.pkl")
                if os.path.isfile(random_state_path):
                    os.remove(random_state_path)    # o.w. all processes will have same random state
                self.accelerator.print(make_banner(f"Resuming from checkpoint: '{path}' with {os.listdir(path)}; "
                                                   f"Current Step: {self.policy_update_steps}", front=True, back=True))
            self.accelerator.wait_for_everyone()
            self.accelerator.load_state(path)
            self.accelerator.wait_for_everyone()    # wait of all processes finishing loading state
            if self.accelerator.is_main_process:
                shutil.rmtree(path)     # will be overwritten in the training process o.w.
            self.accelerator.wait_for_everyone()

    def save_model(self, count):
        """Saves UNET model."""
        save_path = os.path.join(self.args.output_dir, f"save_{count}")
        print(f"Saving model to {save_path}")
        if self.is_ddp:
            unet_to_save = copy.deepcopy(self.accelerator.unwrap_model(self.unet)).to(
                torch.float32
            )
            unet_to_save.save_attn_procs(save_path)
        else:
            unet_to_save = copy.deepcopy(self.unet).to(torch.float32)
            unet_to_save.save_attn_procs(save_path)

    def get_batch(self):
        if self.args.single_flag == 1:  # training with single prompt only
            batch = [self.args.single_prompt for _ in range(self.args.num_rollout_trajs)]
            batch_list = [batch for _ in range(self.args.g_batch_size)]
        else:
            batch = next(self.data_iter_loader, None)
            if batch is None:
                self.data_iter_loader = iter(self.data_iterator)
                batch = next(self.data_iter_loader, None)
                assert batch is not None
            batch_list = []
            for i in range(len(batch)):
                # `batch_list`: [[p1,p1,p1],[p2,p2,p2],...]
                batch_list.append([batch[i] for _ in range(self.args.num_rollout_trajs)])
                # `num_rollout_trajs`: for each prompt we collect `num_rollout_trajs` trajectories

        return batch_list

    def broadcast_buffer(self, tensor_dict):
        shape_dict = dict()
        dtype_dict = dict()
        for k in tensor_dict.keys():
            if isinstance(tensor_dict[k], list):
                tensor_dict[k] = torch.stack(tensor_dict[k], dim=0)
            shape_dict[k] = tensor_dict[k].shape
            dtype_dict[k] = tensor_dict[k].dtype
        for src_rank in range(self.world_size):
            new_data = dict()
            for k in (
                    "latents_list",
                    "reward_list",
                    "unconditional_prompt_embeds",
                    "guided_prompt_embeds",
                    "log_prob_list"
            ):
                if self.rank == src_rank:
                    comm_tensor = tensor_dict[k].to(self.local_rank)
                else:
                    comm_tensor = torch.zeros(
                        shape_dict[k], device=self.local_rank, dtype=dtype_dict[k])
                dist.broadcast(comm_tensor, src=src_rank)
                new_data[k] = comm_tensor.cpu()

            self.replay_buffer.add_samples(**new_data)
            del new_data, comm_tensor
            if src_rank == self.rank:
                del tensor_dict

    def collect_rollout(self, batch):
        """Collects trajectories."""
        if not isinstance(batch, list):
            batch = [batch, ]
        for _ in range(self.args.g_step):
            for bch in batch:
                with torch.no_grad():
                    (
                        image,
                        latents_list,
                        unconditional_prompt_embeds,
                        guided_prompt_embeds,
                        log_prob_list,  # log-prob of the sampled trajectory under the *sampling* policy
                        _,
                    ) = self.pipe.forward_collect_traj_ddim(
                        prompt=bch, is_ddp=self.is_ddp, output_type="pil", generator=self.generator,
                        guidance_scale=self.cfg_guide_scale
                    )     # `guidance_scale` should match `get_pl_loss_logit` and `args.use_cfg_in_train`
                    reward_list = self.scorer_ensemble.get_pref_source_scores(image, bch)["PrefSourceScore"]

                    if self.args.num_rollout_trajs > self.args.num_traj_for_pref_comp:
                        # only record `num_traj_for_pref_comp` trajectories
                        selected_trajs = reward_list.sort().indices[self.rollout_store_idx]
                        latents_list = [x[selected_trajs] for x in latents_list]
                        unconditional_prompt_embeds = unconditional_prompt_embeds[selected_trajs]
                        guided_prompt_embeds = guided_prompt_embeds[selected_trajs]
                        log_prob_list = [x[selected_trajs] for x in log_prob_list]
                        reward_list = reward_list[selected_trajs]

                    assert reward_list.shape == (self.args.num_traj_for_pref_comp,)

                    if dist.is_available() and torch.cuda.is_available() and dist.is_initialized():
                        if self.world_size > 1:
                            if COLLECTIVE_FN == "all_gather":
                                latents_list = all_gather_multigpu(
                                    torch.stack(latents_list, dim=0).to(self.rank), world_size=self.world_size, dim=1
                                ).cpu()
                                reward_list = all_gather_multigpu(reward_list.to(self.rank), world_size=self.world_size).cpu()
                                unconditional_prompt_embeds = all_gather_multigpu(
                                    unconditional_prompt_embeds.to(self.rank), world_size=self.world_size).cpu()
                                guided_prompt_embeds = all_gather_multigpu(
                                    guided_prompt_embeds.to(self.rank), world_size=self.world_size).cpu()
                                log_prob_list = all_gather_multigpu(torch.stack(log_prob_list, dim=0).to(self.rank),
                                                                    world_size=self.world_size).cpu()
                                self.replay_buffer.add_samples(
                                    latents_list=latents_list,
                                    reward_list=reward_list,
                                    unconditional_prompt_embeds=unconditional_prompt_embeds,
                                    guided_prompt_embeds=guided_prompt_embeds,
                                    log_prob_list=log_prob_list
                                )
                            elif COLLECTIVE_FN == "broadcast":
                                self.accelerator.wait_for_everyone()
                                self.broadcast_buffer(dict(
                                    latents_list=latents_list,
                                    reward_list=reward_list,
                                    unconditional_prompt_embeds=unconditional_prompt_embeds,
                                    guided_prompt_embeds=guided_prompt_embeds,
                                    log_prob_list=log_prob_list,
                                ))
                            else:
                                raise NotImplementedError
                        else:
                            latents_list = torch.stack(latents_list, dim=0)
                            log_prob_list = torch.stack(log_prob_list, dim=0)
                            self.replay_buffer.add_samples(
                                latents_list=latents_list,
                                reward_list=reward_list,
                                unconditional_prompt_embeds=unconditional_prompt_embeds,
                                guided_prompt_embeds=guided_prompt_embeds,
                                log_prob_list=log_prob_list
                            )

    def get_pl_loss_logit(self, batch):
        num_steps_est_logits = self.args.num_steps_est_logits
        final_reward = batch["final_reward"].cuda()  # (p_batch_size, pl_loss_num_traj)
        assert final_reward.shape == (self.args.p_batch_size, self.args.pl_loss_num_traj)

        logits = []

        for traj_idx in range(batch["timestep"].shape[1]):
            batch_guided_prompt_embeds = batch["guided_prompt_embeds"][:, traj_idx]

            if self.args.use_cfg_in_train:
                batch_unconditional_prompt_embeds = batch["unconditional_prompt_embeds"][:, traj_idx]
                batch_promt_embeds = torch.cat(
                    [batch_unconditional_prompt_embeds, batch_guided_prompt_embeds]
                )
            else:
                batch_promt_embeds = batch_guided_prompt_embeds

            log_diff = 0.  # expectation of log(density ratio) over the entire trajectory
            sampled_time_steps = self.np_generator.choice(batch["timestep"].shape[2],
                                                          size=num_steps_est_logits,
                                                          replace=True,
                                                          p=self.policy_loss_weights
                                                          )

            for time_idx in sampled_time_steps:
                batch_state = batch["state"][:, traj_idx, time_idx]
                batch_next_state = batch["next_state"][:, traj_idx, time_idx]
                batch_timestep = batch["timestep"][:, traj_idx, time_idx]
                batch_log_pi_old_step_t = batch["log_pi_old"][:, traj_idx, time_idx].cuda()

                # calculate loss from the custom function
                log_prob, log_prob_init = self.pipe.forward_calculate_logprob(
                    prompt_embeds=batch_promt_embeds.cuda(),
                    latents=batch_state.cuda(),
                    next_latents=batch_next_state.cuda(),
                    ts=batch_timestep.cuda(),
                    unet=self.wrapped_unet,
                    unet_copy=self.initial_unet,
                    is_ddp=self.is_ddp,
                    generator=self.generator,
                    guidance_scale=self.cfg_guide_scale     # should match `collect_rollout` and `args.use_cfg_in_train`
                )

                log_diff_step_t = 0.
                if self.args.reg_to_pi_init:
                    if self.policy_update_steps <= self.no_reg_pi_init_warmup_steps:   # no regularization
                        log_pi_theta_minus_log_pi_init = log_prob       # no clipping since we do not have log-density-ratio here
                    else:   # with regularization
                        log_pi_theta_minus_log_pi_init = self.soft_clip(log_prob - log_prob_init)  # torch.Size([p_batch_size]); clipping on the log space
                    log_diff_step_t += log_pi_theta_minus_log_pi_init
                if self.args.reg_to_pi_old:
                    if self.policy_update_steps_after_rollout <= self.no_reg_pi_old_warmup_steps:   # no regularization
                        log_pi_theta_minus_log_pi_old = log_prob
                    else:   # with regularization
                        log_pi_theta_minus_log_pi_old = self.soft_clip(log_prob - batch_log_pi_old_step_t)
                    log_diff_step_t += log_pi_theta_minus_log_pi_old

                assert log_diff_step_t.requires_grad

                log_diff += log_diff_step_t / num_steps_est_logits  # torch.Size([p_batch_size])
                assert log_diff.requires_grad

            logits.append(log_diff * self.args.pl_loss_temp)

        logits = torch.column_stack(logits)
        assert logits.isfinite().all()
        assert logits.requires_grad
        assert logits.shape == final_reward.shape == (self.args.p_batch_size, self.args.pl_loss_num_traj)

        return logits, final_reward

    def calculate_pl_loss_and_backward(self):
        batch = self.replay_buffer.sample_pref_data()

        idx_tensor = torch.tensor(range(self.args.num_traj_for_pref_comp))
        assert len(idx_tensor) == batch["state"].shape[1]

        all_combs = torch.combinations(idx_tensor, r=self.args.pl_loss_num_traj)
        all_combs = all_combs[torch.randperm(all_combs.shape[0])]
        all_combs = all_combs[:self.args.pl_loss_num_tuples]

        assert all_combs.shape[1] == self.args.pl_loss_num_traj
        num_combs = float(len(all_combs))
        assert 1 <= num_combs <= self.args.pl_loss_num_tuples

        total_loss = 0.

        for comparison_idx in all_combs:
            batch_for_logit = {k: v[:, comparison_idx] for k, v in batch.items()}
            logits, final_reward = self.get_pl_loss_logit(batch=batch_for_logit)  # (p_batch_size, pl_loss_num_traj)
            loss = listMLELoss(y_pred=logits, y_true=final_reward)
            # loss show be average over all combinations -> divide by `num_combs`
            loss = loss / (num_combs * self.args.gradient_accumulation_steps)
            assert loss.isfinite()
            assert loss.requires_grad

            self.accelerator.backward(loss)

            total_loss += loss.item()

        return total_loss

    def train_policy_func(self):
        """Trains the policy for one step."""
        loss = self.calculate_pl_loss_and_backward()

        # logging
        self.train_log.avg_p_loss = self.train_log.avg_p_loss * ((self.policy_update_steps - 1.) / self.policy_update_steps) \
                                    + loss / self.policy_update_steps
        self.train_log.step_p_loss += loss

    def _print_training_info(self):
        if self.accelerator.is_main_process:
            self.accelerator.print("*" * 30 + " Running training " + "*" * 30)
            self.accelerator.print(f"  Max Train Steps = {self.args.max_train_steps}")
            self.accelerator.print(
                f"  Total data-collection times = {self.data_collection_times}"
            )
            self.accelerator.print(f"  # policy-training steps per data collection = {self.args.p_step}")
            self.accelerator.print(
                f"  Instantaneous batch size per device = {self.args.p_batch_size}"
            )
            self.accelerator.print(
                f"  # processes = {self.accelerator.num_processes}"
            )
            self.accelerator.print(
                f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}"
            )
            self.accelerator.print(
                "  Total train batch size (w. parallel, distributed & accumulation) ="
                f" {self.args.p_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps}"
            )
            self.accelerator.print(f"  `args.total_train_batch_size` = {self.args.total_train_batch_size}")
            self.accelerator.print(f"  no_reg_pi_init_warmup_steps: {self.no_reg_pi_init_warmup_steps}/{self.args.max_train_steps}"
                                   f" ~= {self.no_reg_pi_init_warmup_steps / self.args.max_train_steps * 100.:.1f}%")
            self.accelerator.print(f"  no_reg_pi_old_warmup_steps: {self.no_reg_pi_old_warmup_steps}/{self.args.p_step}"
                                   f" ~= {self.no_reg_pi_old_warmup_steps / self.args.p_step * 100.:.1f}%")
            self.accelerator.print(f"  Model is parallel: {self.is_ddp}")
            self.accelerator.print(f"  Use CFG in training: {self.args.use_cfg_in_train == 1}")
            self.accelerator.print(f"  CFG Guidance Scale: {self.cfg_guide_scale}")
            self.accelerator.print(f"  Clip log(ratio): [min, max]={(-self.args.log_ratio_clip), self.args.log_ratio_clip}")
            self.accelerator.print(f"  num_rollout_trajs: {self.args.num_rollout_trajs}")
            self.accelerator.print(f"  num_traj_for_pref_comp: {self.args.num_traj_for_pref_comp}")

    def generate_eval_imgs_during_training(self):
        self.accelerator.wait_for_everyone()
        if self.args.single_flag == 1:
            self.generate_test_img(self.args.single_prompt)
            if self.args.unseen_prompt is not None:
                self.accelerator.wait_for_everyone()
                self.generate_test_img(self.args.unseen_prompt)
        else:   # multiple prompts
            self.generate_test_img_hpsv2()

        self.accelerator.wait_for_everyone()

    def train_model(self):
        start_time = datetime.now()

        # Train!
        if self.args.resume_from_checkpoint:
            self.resume_from_checkpoint()

        self._print_training_info()

        for _ in range(ceil(self.args.init_buffer_size / (self.world_size * self.args.g_batch_size))):
            # init replay buffer
            batch = self.get_batch()
            self.collect_rollout(batch=batch)

        assert self.replay_buffer.num_steps_can_sample() >= self.args.init_buffer_size      # ">=" due to potential rounding
        if self.accelerator.is_main_process:
            self.accelerator.print(make_banner(f"\nFINISH Initializing replay buffer of {self.replay_buffer.num_steps_can_sample()} prompts !!! "
                                            f"Using time {datetime.now() - start_time} !!!"))

        # generate test imgs before training
        self.generate_eval_imgs_during_training()

        for data_collect_count in range(0, self.data_collection_times):
            # fix batchnorm
            self.unet.eval()

            # policy learning
            for _ in range(self.args.p_step):
                self.policy_update_steps += 1
                self.policy_update_steps_after_rollout += 1
                self.train_log.step_p_loss = 0.
                self.optimizer.zero_grad(set_to_none=True)
                for accum_step in range(self.args.gradient_accumulation_steps):
                    if accum_step < self.args.gradient_accumulation_steps - 1:
                        with self.accelerator.no_sync(self.unet):
                            self.train_policy_func()
                    else:
                        self.train_policy_func()

                if self.accelerator.sync_gradients:
                    norm = self.accelerator.clip_grad_norm_(self.unet.parameters(), self.args.clip_norm)
                    if self.accelerator.state.deepspeed_plugin is None:
                        # `norm` will be none if using deepspeed, so will only record grad_norm when not using deepspeed
                        self.train_log.avg_grad_norm = self.train_log.avg_grad_norm * ((self.policy_update_steps - 1.) / self.policy_update_steps) \
                                                   + norm.item() / self.policy_update_steps

                self.optimizer.step()
                self.lr_scheduler.step()

                if self.accelerator.is_main_process and (self.policy_update_steps % self.args.logging_interval == 0):
                    curr_avg_rew = self.replay_buffer.get_average_reward()
                    self.accelerator.log(
                        {"train_reward": torch.mean(curr_avg_rew).item()},
                        step=self.policy_update_steps,
                    )
                    self.accelerator.log({"grad norm": self.train_log.avg_grad_norm}, step=self.policy_update_steps)
                    self.accelerator.log({"p_loss": self.train_log.avg_p_loss}, step=self.policy_update_steps)
                    self.accelerator.log({"step_p_loss": self.train_log.step_p_loss}, step=self.policy_update_steps)
                    self.accelerator.log({"step_grad_norm": norm}, step=self.policy_update_steps)

                    s = f"policy train:{self.policy_update_steps}/{self.args.max_train_steps}" \
                        f"|data collect:{data_collect_count + 1}/{self.data_collection_times}" \
                        f"|p_loss:{self.train_log.avg_p_loss:.4f}" \
                        f"|grad norm:{self.train_log.avg_grad_norm:.4f}" \
                        f"|step_p_loss:{self.train_log.step_p_loss:.4f}" \
                        f"|step_grad_norm:{norm:.4f}" \
                        f"|train_reward:{[round(x, 2) for x in curr_avg_rew.tolist()]}" \
                        f"|used time:{datetime.now() - start_time}"
                    if self.accelerator.is_main_process:
                        self.accelerator.print(make_banner(s, front=True, back=True))  # only print on the main process

                if self.accelerator.sync_gradients:
                    if self.policy_update_steps % self.args.checkpointing_steps == 0:
                        if self.accelerator.is_main_process:
                            save_path = os.path.join(self.args.output_dir, f"checkpoint-{self.policy_update_steps}")
                            self.accelerator.save_state(output_dir=save_path)
                            if self.accelerator.is_main_process:
                                self.accelerator.print(f"Saved state to {save_path}")

                # Save model per interval
                if self.policy_update_steps % self.args.save_interval == 0:
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_main_process:
                        self.save_model(count=self.policy_update_steps)

                if self.policy_update_steps % self.args.test_img_gen_freq == 0:
                    self.generate_eval_imgs_during_training()

            if self.policy_update_steps < self.args.max_train_steps:
                # Do not collect on the final training step
                if self.accelerator.is_main_process:
                    print_banner(f"[{self.policy_update_steps}/{self.args.max_train_steps}] "
                                 f"Recollect data. Count {data_collect_count + 2}/{self.data_collection_times}", front=True, back=True)
                # collect data once, train policy multiple (`p_step`) optimization steps
                batch = self.get_batch()
                self.collect_rollout(batch=batch)
                self.policy_update_steps_after_rollout = 0  # reset by definition

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.save_model(count=self.policy_update_steps)

        self.accelerator.end_training()

        if self.accelerator.is_main_process:
            self.accelerator.print(make_banner(f"\nFINISH TRAINING !!! Using time {datetime.now() - start_time} !!!"))

        return

    def generate_test_img(self, prompt):
        """ Model evaluation. Only use main process"""
        if self.accelerator.is_main_process:
            save_folder_name = os.path.join(self.args.output_dir, "saved_imgs", f"iter{self.policy_update_steps}", prompt)
            # `args.output_dir` may contain the `single_prompt` used for training policy, but `prompt` can differ from it.

            num_eval_samples = self.args.num_eval_samples
            start_idx = 0

            if not os.path.exists(save_folder_name):
                os.makedirs(save_folder_name, exist_ok=True)

            start_time = datetime.now()

            batch_size = min(self.args.num_traj_for_pref_comp * 4, 8)  # no grad in generation, so use a larger batch_size
            for i in range(0, num_eval_samples, batch_size):
                if (i + batch_size) >= num_eval_samples:
                    batch_size = num_eval_samples - i
                prompts = [prompt for _ in range(batch_size)]
                imgs = self.pipe(prompts, output_type="pil", generator=self.generator)  # numpy array B H W C
                for j, img in enumerate(imgs):
                    fname = os.path.join(save_folder_name, f"{start_idx+i+j}.png")
                    while True:
                        try:
                            img.save(fname)
                            break
                        except FileNotFoundError as e:
                            print_banner(f"\nProcess {self.accelerator.process_index}/{self.world_size}: {e}\n")
                            os.system("sleep 1s")

            self.accelerator.print(make_banner(f"[{self.policy_update_steps}/{self.args.max_train_steps}] "
                                                   f"Finish generating test imgs by main process !!! "
                                                   f"Used time {datetime.now() - start_time} !!!"))

    def generate_test_img_hpsv2(self):
        """ Model evaluation on the HPSv2 test prompts."""
        save_folder_name = os.path.join(self.args.output_dir, "saved_imgs", f"iter{self.policy_update_steps}", "hpsv2Test")

        seed_for_this_eval = (self.policy_update_steps // self.args.test_img_gen_freq) * 42
        # `seed_for_this_eval` should be the same across all process to ensure the splitting of `all_prompts` by process forms a partition
        rd = random.Random(seed_for_this_eval)
        num_prompts_per_style = int(800 * self.args.multiprompt_eval_ratio)

        # gather the prompts for each style into a common list
        all_prompts_dict = hpsv2.benchmark_prompts('all')
        all_prompts = []
        for v in all_prompts_dict.values():
            rd.shuffle(v)
            all_prompts.extend(v[:num_prompts_per_style])

        self.accelerator.print(make_banner(f"Loaded HPSv2 test prompts! Size: {len(all_prompts)}"))

        num_eval_samples = ceil(len(all_prompts) / self.world_size)
        start_idx = num_eval_samples * self.rank
        # slice `all_prompts` for each process
        all_prompts = all_prompts[start_idx:start_idx+num_eval_samples]

        if self.policy_update_steps // self.args.test_img_gen_freq < 1:     # do not print later on
            print_banner(f"Process {self.accelerator.process_index}/{self.world_size} evaluates {len(all_prompts)} prompts")

        if self.accelerator.is_main_process:
            if not os.path.exists(save_folder_name):
                os.makedirs(save_folder_name, exist_ok=True)
                os.makedirs(os.path.join(save_folder_name, "tmp"), exist_ok=True)
        self.accelerator.wait_for_everyone()

        start_time = datetime.now()
        batch_size = min(self.args.num_traj_for_pref_comp * 4, 8)  # no grad in generation, so use a larger batch_size

        idx_prompt_map = {}     # store index-prompt mapping for evaluation

        # iterate over batches
        for i in range(0, len(all_prompts), batch_size):
            prompts = all_prompts[i:i+batch_size]
            imgs = self.pipe(prompts, output_type="pil", generator=self.generator)  # numpy array B H W C
            for j, (img, prompt) in enumerate(zip(imgs, prompts)):
                fname = os.path.join(save_folder_name, f"r{self.rank}_{start_idx+i+j}.png")
                idx_prompt_map[f"r{self.rank}_{start_idx+i+j}.png"] = prompt
                while True:
                    try:
                        img.save(fname)
                        break
                    except FileNotFoundError as e:
                        print_banner(f"\nProcess {self.accelerator.process_index}/{self.world_size}: {e}\n")
                        os.system("sleep 1s")

        with open(os.path.join(save_folder_name, "tmp", f"idx_prompt_{self.rank}.json"), "w") as json_file:
            json.dump(idx_prompt_map, json_file, indent=2)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            # combine the `idx_prompt_map` from each process
            idx_prompt = {}
            for tmp_map_loc in os.listdir(os.path.join(save_folder_name, "tmp")):
                with open(os.path.join(save_folder_name, "tmp", tmp_map_loc)) as f:
                    tmp_map = json.load(f)
                for k, v in tmp_map.items():
                    idx_prompt[k] = v

            with open(os.path.join(save_folder_name, f"idx_prompt_map.json"), "w") as json_file:
                json.dump(idx_prompt, json_file, indent=2)

            self.accelerator.print(make_banner(f"[{self.policy_update_steps}/{self.args.max_train_steps}] "
                                                   f"Finish generating {len(idx_prompt.keys())} test imgs by {self.world_size} processes !!! "
                                                   f"Used time {datetime.now() - start_time} !!!"))

        self.accelerator.wait_for_everyone()
