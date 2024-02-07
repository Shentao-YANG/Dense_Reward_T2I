import torch
from utils import print_banner


class ReplayBufferTorch(object):
    def __init__(self, weight_dtype, args):
        self.device = "cpu"
        self.weight_dtype = weight_dtype
        self.max_buffer_size = args.max_replay_buffer_size
        self.num_denosing_steps = 50
        self.p_batch_size = args.p_batch_size      # batch size for policy update per gpu, before gradient accumulation

        self._top = 0
        self._size = 0
        self.total_entries = 0

        self.buffer_first_two_dim = (self.max_buffer_size, args.num_traj_for_pref_comp)
        self.latent_dim = (4, 64, 64)
        self.prompt_emb_dim = (77, 768)

        self.state = torch.zeros(self.buffer_first_two_dim + (self.num_denosing_steps, ) + self.latent_dim,
                                 device=self.device, dtype=self.weight_dtype)
        self.next_state = torch.zeros(self.buffer_first_two_dim + (self.num_denosing_steps, ) + self.latent_dim,
                                      device=self.device, dtype=self.weight_dtype)
        self.timestep = (torch.LongTensor(range(self.num_denosing_steps), device=self.device)
                         .unsqueeze(0)
                         .repeat_interleave(self.buffer_first_two_dim[1], dim=0)
                         .unsqueeze(0)
                         .repeat_interleave(self.buffer_first_two_dim[0], dim=0))
        assert self.timestep.shape == self.buffer_first_two_dim + (self.num_denosing_steps, )
        # one value for each reverse chain (trajectory)
        self.final_reward = torch.zeros(self.buffer_first_two_dim,
                                        device=self.device, dtype=self.weight_dtype)
        # same for each denoising step
        self.unconditional_prompt_embeds = torch.zeros(self.buffer_first_two_dim + self.prompt_emb_dim,
                                                       device=self.device, dtype=self.weight_dtype)
        self.guided_prompt_embeds = torch.zeros(self.buffer_first_two_dim + self.prompt_emb_dim,
                                                device=self.device, dtype=self.weight_dtype)
        # one value for each denoising step
        self.log_pi_old = torch.zeros(self.buffer_first_two_dim + (self.num_denosing_steps, ),
                                       device=self.device, dtype=self.weight_dtype)

        self.all_attrs = {
            "state": self.state,
            "next_state": self.next_state,
            "timestep": self.timestep,
            "final_reward": self.final_reward,
            "unconditional_prompt_embeds": self.unconditional_prompt_embeds,
            "guided_prompt_embeds": self.guided_prompt_embeds,
            "log_pi_old": self.log_pi_old
        }

        print_banner(f"Initialized `ReplayBufferTorch`, with first two dims {self.buffer_first_two_dim} and latent dims {self.latent_dim}")

    def _advance(self):
        self._top = (self._top + 1) % self.max_buffer_size
        if self._size < self.max_buffer_size:
            self._size += 1
        self.total_entries += 1

        for v in self.all_attrs.values():
            assert not v.requires_grad

    def add_samples(self, latents_list, reward_list, unconditional_prompt_embeds, guided_prompt_embeds, log_prob_list):
        latents_list = latents_list.transpose(0, 1).to(self.device)

        self.state[self._top] = latents_list[:, 0:self.num_denosing_steps, ...]
        self.next_state[self._top] = latents_list[:, 1:(self.num_denosing_steps + 1), ...]
        self.final_reward[self._top] = reward_list.to(self.device)
        self.unconditional_prompt_embeds[self._top] = unconditional_prompt_embeds.to(self.device)
        self.guided_prompt_embeds[self._top] = guided_prompt_embeds.to(self.device)
        self.log_pi_old[self._top] = log_prob_list.transpose(0, 1).to(self.device)

        self._advance()

    def sample_pref_data(self):
        # sample shape: (p_batch_size, num_traj_for_pref_comp, ...)
        indices = torch.randint(low=0, high=self._size, size=(self.p_batch_size,), device=self.device)
        batch = dict()

        for k, v in self.all_attrs.items():
            batch[k] = v[indices]

        return batch

    def top(self):
        return self._top

    def num_steps_can_sample(self):
        return self._size

    def get_average_reward(self):
        sorted_rew, _ = torch.sort(self.final_reward, descending=True, dim=-1)
        return sorted_rew.mean(dim=0)
