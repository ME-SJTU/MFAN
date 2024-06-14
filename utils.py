import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


def direct_spat_edge_adj(graph, graph_rel):
    """
    :param graph: spatial-temporal graph with node feature of position and category. Node i -- (c_i, x_i, y_i)
    :param graph_rel: spatial-temporal graph with node feature of velocity and category. Node i -- (c_i, vx_i, v_yi)
                      vx_i and vy_i are represented by relative displacement
    :return: sparse direct spatial adjacency matrix and spatial edge feature tensor
    """

    graph = graph.squeeze()
    graph_rel = graph_rel.squeeze()
    seq_len = graph.shape[0]
    node_num = graph.shape[1]

    V_node = torch.cat((graph, graph_rel[:, :, 1:]), dim=-1)  # [T, N, F=4], F -- [x, y, step, vx, vy]
    mask = torch.ones((seq_len, node_num, node_num))

    v_node_i = V_node.repeat(1, 1, V_node.shape[1]).view(V_node.shape[0], V_node.shape[1], -1, V_node.shape[-1])
    v_node_j = V_node.repeat(1, V_node.shape[1], 1).view(V_node.shape[0], V_node.shape[1], -1, V_node.shape[-1])

    v_feat = v_node_i - v_node_j  # [dx, dy, dvx, dvy]

    v_dist = torch.sqrt(v_feat[:, :, :, 0] ** 2 + v_feat[:, :, :, 1] ** 2)
    zero_vec = torch.zeros_like(v_dist)
    # if distance > 20, mask = 0
    mask = torch.where(v_dist <= 20, mask, zero_vec)

    vd_dot = v_node_i[:, :, :, 2] * v_feat[:, :, :, 0] + v_node_i[:, :, :, 3] * v_feat[:, :, :, 1]
    # if pedestrian j is behind pedestrian i, mask = 0
    mask = torch.where(vd_dot <= 0, mask, zero_vec)

    # calculate the velocity angle between pedestrian i and j
    vij_dot = v_node_i[:, :, :, 2] * v_node_j[:, :, :, 2] + v_node_i[:, :, :, 3] * v_node_j[:, :, :, 3]

    v_norm_i = torch.sqrt(torch.pow(v_node_i[:, :, :, 2], 2) +
                          torch.pow(v_node_i[:, :, :, 3], 2))
    v_norm_j = torch.sqrt(torch.pow(v_node_j[:, :, :, 2], 2) +
                          torch.pow(v_node_j[:, :, :, 3], 2))

    vij_norm = v_norm_i * v_norm_j

    # preserve the denominator being zero
    ones_vec = torch.ones_like(vij_norm)
    vij_norm = torch.where(vij_norm != 0, vij_norm, ones_vec)

    vij_theta = vij_dot / vij_norm

    edge_feat = torch.cat((v_feat, vij_theta.unsqueeze(-1)), dim=-1) * mask.unsqueeze(-1)

    edge_feat = edge_feat.permute(0, 2, 1, 3)
    mask = mask.permute(0, 2, 1)

    return edge_feat, mask


def direct_temp_edge_adj(graph_temp_rel):
    """
    :param graph_temp_rel: temporal-spatial graph with node feature of time step and relative distance. dim -- [N T F]
    :return: sparse directed temporal adjacency matrix and temporal edge feature tensor
    """

    V_node = graph_temp_rel.squeeze()
    seq_len = graph_temp_rel.shape[0]
    temp_node = graph_temp_rel.shape[1]

    mask = torch.tril(torch.ones(seq_len, temp_node, temp_node))

    v_node_i = V_node.repeat(1, 1, V_node.shape[1]).view(V_node.shape[0], V_node.shape[1], -1, V_node.shape[-1])
    v_node_j = V_node.repeat(1, V_node.shape[1], 1).view(V_node.shape[0], V_node.shape[1], -1, V_node.shape[-1])

    v_feat = v_node_i - v_node_j  # [dT, dvx, dvy]

    # calculate the velocity angle of pedestrian between T and T+1
    vij_dot = v_node_i[:, :, :, 1] * v_node_j[:, :, :, 1] + v_node_i[:, :, :, 2] * v_node_j[:, :, :, 2]

    v_norm_i = torch.sqrt(torch.pow(v_node_i[:, :, :, 1], 2) +
                          torch.pow(v_node_i[:, :, :, 2], 2))
    v_norm_j = torch.sqrt(torch.pow(v_node_j[:, :, :, 1], 2) +
                          torch.pow(v_node_j[:, :, :, 2], 2))

    vij_norm = v_norm_i * v_norm_j

    # preserve the denominator being zero
    ones_vec = torch.ones_like(vij_norm)
    vij_norm = torch.where(vij_norm != 0, vij_norm, ones_vec)

    vij_theta = vij_dot / vij_norm

    edge_feat = torch.cat((v_feat, vij_theta.unsqueeze(-1)), dim=-1) * mask.unsqueeze(-1)

    edge_feat = edge_feat.permute(0, 2, 1, 3)
    mask = mask.permute(0, 2, 1)

    return edge_feat, mask


def loc_pos(seq_):
    # seq_ [obs_len N 3]

    obs_len = seq_.shape[0]
    num_agent = seq_.shape[1]

    pos_seq = np.arange(1, obs_len + 1)
    pos_seq = pos_seq[:, np.newaxis, np.newaxis]
    pos_seq = pos_seq.repeat(num_agent, axis=1)

    result = np.concatenate((pos_seq, seq_), axis=-1)

    return result


def seq_to_graph(seq_, seq_rel, pos_enc=False):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    v_obs = seq_.permute(2, 0, 1)
    V = np.zeros((seq_len, max_nodes, 2))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]

    if pos_enc:
        V = loc_pos(V)

    return v_obs, torch.from_numpy(V).type(torch.float)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
            min_ped=1, delim='\t'):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []

        for path in all_files:
            data = read_file(path, delim)

            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    # ipdb.set_trace()
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    # rel_curr_ped_seq[:, 1:] = \
                    #     curr_ped_seq[:, 1:] - np.reshape(curr_ped_seq[:, 0], (2,1))
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        # Convert to Graphs
        self.v_obs = []
        self.v_pred = []
        self.v_spat_edge = []
        self.v_temp_edge = []
        self.A_spat = []
        self.A_temp = []
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end))
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]

            v_obs, v_ = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], True)
            self.v_obs.append(v_.clone())
            v_spat_edge, ST_Mask = direct_spat_edge_adj(v_obs, v_)
            self.v_spat_edge.append(v_spat_edge.clone())
            self.A_spat.append(ST_Mask.clone())

            v_temp_edge, TS_Mask = direct_temp_edge_adj(v_.permute(1, 0, 2))
            self.A_temp.append(TS_Mask.clone())
            self.v_temp_edge.append(v_temp_edge.clone())

            v_obs, v_ = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], False)
            self.v_pred.append(v_.clone())

        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.v_obs[index], self.v_pred[index],
            self.v_spat_edge[index], self.v_temp_edge[index],
            self.A_spat[index], self.A_temp[index]
        ]
        return out
#
#
# data_set = './dataset/eth/'
#
# data_train = TrajectoryDataset(
#         data_set + 'train001/',
#         obs_len=9,
#         pred_len=12,
#         skip=1)
