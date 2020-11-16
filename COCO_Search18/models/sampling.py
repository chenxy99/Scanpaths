import torch
import numpy as np

class Sampling():
    def __init__(self, convLSTM_length=16, min_length=2, map_width=40, map_height=30, width=320, height=240):
        super(Sampling, self).__init__()
        self.convLSTM_length = convLSTM_length
        self.min_length = min_length
        self.map_width = map_width
        self.map_height = map_height
        self.width = width
        self.height = height
        self.x_granularity = float(self.width / self.map_width)
        self.y_granularity = float(self.height / self.map_height)

    def random_sample(self, all_actions_prob, log_normal_mu, log_normal_sigma2):
        # sampling stage
        batch = all_actions_prob.shape[0]
        probs = all_actions_prob.data.clone()
        probs[:, :self.min_length, 0] = 0
        dist = torch.distributions.categorical.Categorical(probs=probs)
        selected_specific_actions = dist.sample()
        selected_actions_probs = \
            torch.gather(all_actions_prob, dim=2, index=selected_specific_actions.unsqueeze(-1)).squeeze(-1)

        random_rand = torch.randn(log_normal_mu.shape).to(log_normal_mu.get_device())
        duration_samples = torch.exp(random_rand * log_normal_sigma2 + log_normal_mu)

        scanpath_length = all_actions_prob.new_zeros(batch)
        for index in range(self.convLSTM_length):
            scanpath_length[torch.logical_and(
                scanpath_length == 0, selected_specific_actions[:, index] == 0)] = index
        scanpath_length[scanpath_length == 0] = self.convLSTM_length
        scanpath_length = scanpath_length.unsqueeze(-1)

        predicts = {}
        # [N, 1]
        predicts["scanpath_length"] = scanpath_length
        # [N, T]
        predicts["durations"] = duration_samples
        # [N, T]
        predicts["selected_actions_probs"] = selected_actions_probs
        # [N, T]
        predicts["selected_actions"] = selected_specific_actions

        return predicts

    def generate_scanpath(self, images, prob_sample_actions, durations, sample_actions):
        # computer the logprob for action and duration
        action_masks = images.new_zeros(prob_sample_actions.shape)
        duration_masks = images.new_zeros(prob_sample_actions.shape)
        t = durations.data.clone()
        N = images.shape[0]
        predict_fix_vectors = list()
        for index in range(N):
            sample_action = sample_actions[index].cpu().numpy()
            drts = t[index].cpu().numpy()
            fix_vector = []
            for order in range(sample_action.shape[0]):
                if sample_action[order] == 0:
                    action_masks[index, order] = 1
                    break
                else:
                    image_index = sample_action[order] - 1
                    map_pos_x = image_index % self.map_width
                    map_pos_y = image_index // self.map_width
                    pos_x = map_pos_x * self.x_granularity + self.x_granularity / 2
                    pos_y = map_pos_y * self.y_granularity + self.y_granularity / 2
                    drt = drts[order]
                    action_masks[index, order] = 1
                    duration_masks[index, order] = 1
                    fix_vector.append((pos_x, pos_y, drt))
            fix_vector = np.array(fix_vector, dtype={'names': ('start_x', 'start_y', 'duration'),
                                                     'formats': ('f8', 'f8', 'f8')})
            predict_fix_vectors.append(fix_vector)

        return predict_fix_vectors, action_masks, duration_masks
