from coperception.models.det.backbone.Backbone import *
import numpy as np

from coperception.utils.data_util import apply_pose_noise


class DetModelBase(nn.Module):
    """Abstract class. The super class for all detection models.

    Attributes:
        motion_state (bool): To return motion state in the loss calculation method or not
        out_seq_len (int): Length of output sequence
        box_code_size (int): The specification for bounding box encoding.
        category_num (int): Number of categories.
        use_map (bool): use_map
        anchor_num_per_loc (int): Anchor number per location.
        classification (nn.Module): The classification head.
        regression (nn.Module): The regression head.
        agent_num (int): The number of agent (including RSU and vehicles)
        kd_flag (bool): Required for DiscoNet.
        layer (int): Collaborate at which layer.
        p_com_outage (float): The probability of communication outage.
        neighbor_feat_list (list): The list of neighbor features.
        tg_agent (tensor): Features of the current target agent.
    """

    def __init__(
        self,
        config,
        layer=3,
        in_channels=13,
        kd_flag=True,
        p_com_outage=0.0,
        num_agent=5,
        only_v2i=False
    ):
        super(DetModelBase, self).__init__()

        self.motion_state = config.motion_state
        self.out_seq_len = 1 if config.only_det else config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.agent_num = num_agent
        self.kd_flag = kd_flag
        self.layer = layer
        self.p_com_outage = p_com_outage
        self.neighbor_feat_list = []
        self.tg_agent = None
        self.only_v2i = only_v2i

    def agents_to_batch(self, feats):
        """Concatenate the features of all agents back into a bacth.

        Args:
            feats (tensor): features

        Returns:
            The concatenated feature matrix of all agents.
        """
        feat_list = []
        for i in range(self.agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)

        feat_mat = torch.flip(feat_mat, (2,))

        return feat_mat

    def get_feature_maps_and_size(self, encoded_layers: list):
        """Get the features of the collaboration layer and return the corresponding size.

        Args:
            encoded_layers (list): The output from the encoder.

        Returns:
            Feature map of the collaboration layer and the corresponding size.
        """
        feature_maps = encoded_layers[self.layer]

        size_tuple = (
            (1, 32, 256, 256),
            (1, 64, 128, 128),
            (1, 128, 64, 64),
            (1, 256, 32, 32),
            (1, 512, 16, 16),
        )
        # 512
        # size_tuple = (
        #     (1, 32, 512, 512),
        #     (1, 64, 256, 256),
        #     (1, 128, 128, 128),
        #     (1, 256, 64, 64),
        #     (1, 512, 32, 32),
        # )
        # 384
        # size_tuple = (
        #     (1, 32, 384, 384),
        #     (1, 64, 192, 192),
        #     (1, 128, 96, 96),
        #     (1, 256, 48, 48),
        #     (1, 512, 24, 24),
        # )
        size = size_tuple[self.layer]

        feature_maps = torch.flip(feature_maps, (2,))
        return feature_maps, size

    def build_feature_list(self, batch_size: int, feat_maps) -> list:
        """Get the feature maps for each agent

        e.g: [10 512 16 16] -> [2 5 512 16 16] [batch size, agent num, channel, height, width]
            [5 256 32 32] -> 5[1 1 256 32 32]
        Args:
            batch_size (int): The batch size.
            feat_maps (tensor): The feature maps of the collaboration layer.

        Returns:
            A list of feature maps for each agent.
        """
        feature_map = {}
        feature_list = []

        for i in range(self.agent_num):
            feature_map[i] = torch.unsqueeze(
                feat_maps[batch_size * i : batch_size * (i + 1)], 1
            )
            feature_list.append(feature_map[i])

        return feature_list

    @staticmethod
    def build_local_communication_matrix(feature_list: list):
        """Concatendate the feature list into a tensor.

        Args:
            feature_list (list): The input feature list for each agent.

        Returns:
            A tensor of concatenated features.
        """
        return torch.cat(tuple(feature_list), 1)

    def outage(self) -> bool:
        """Simulate communication outage according to self.p_com_outage.

        Returns:
            A bool indicating if the communication outage happens.
        """
        return np.random.choice(
            [True, False], p=[self.p_com_outage, 1 - self.p_com_outage]
        )

    @staticmethod
    def feature_transformation(
            b, j, agent_idx, local_com_mat, all_warp, device, size, trans_matrices,
            noise_level=0.0, noisy_agents=None
    ):
        """Transform the features of the other agent (j) to the coordinate system of the current agent.

        Args:
            b (int): The index of the sample in current batch.
            j (int): The index of the other agent.
            local_com_mat (tensor): The local communication matrix. Features of all the agents.
            all_warp (tensor): The warp matrix for current sample for the current agent.
            device: The device used for PyTorch.
            size (tuple): Size of the feature map.
            trans_matrices (tensor): Transformation matrices with shape (batch, agents, agents, 4, 4).
            noise_level (float, optional): Noise level for the transformation matrix (default: 0.0).
            noisy_agents (list, optional): List of agent indices (j) to add noise to (default: None).

        Returns:
            A tensor of transformed features of agent j.
        """
        nb_agent = torch.unsqueeze(local_com_mat[b, j], 0)

        # 获取原始变换矩阵
        tfm_ji = trans_matrices[b, j, agent_idx]

        # 对指定智能体的变换矩阵添加噪声
        if isinstance(noise_level,float):
            if noisy_agents is not None and j in noisy_agents:
                # 创建噪声矩阵（仅在平移部分添加噪声）
                noise = torch.randn_like(tfm_ji[:3, 3]) * noise_level

                # 只修改平移部分（最后一列的前3行）
                tfm_ji = tfm_ji.clone()  # 避免修改原始矩阵
                tfm_ji[:3, 3] += noise

        # 构建仿射变换矩阵 [2, 3]
        M = torch.hstack((tfm_ji[:2, :2], -tfm_ji[:2, 3:4])).float().unsqueeze(0)  # [1, 2, 3]

        # 应用掩码
        mask = torch.tensor([[[1, 1, 4 / 128], [1, 1, 4 / 128]]], device=M.device)
        M *= mask

        # 生成网格并进行特征采样
        grid = F.affine_grid(M, size=torch.Size(size))
        warp_feat = F.grid_sample(nb_agent, grid).squeeze()

        return warp_feat

    def build_neighbors_feature_list(
        self,
        b,
        agent_idx,
        all_warp,
        num_agent,
        local_com_mat,
        device,
        size,
        trans_matrices,
        collab_agent_list=None, 
        trial_agent_id=None,
        pert=None,
        attacker_list=None,
        eps=None,
        unadv_pert=None
    ) -> None:
        """Append the features of the neighbors of current agent to the neighbor_feat_list list.

        Args:
            b (int): The index of the sample in current batch.
            agent_idx (int): The index of the current agent.
            all_warp (tensor): The warp matrix for current sample for the current agent.
            num_agent (int): The number of agents.
            local_com_mat (tensor): The local communication matrix. Features of all the agents.
            device: The device used for PyTorch.
            size (tuple): Size of the feature map.
        """
        for j in range(num_agent):
            if j != agent_idx:
                if self.only_v2i and agent_idx != 0 and j != 0:
                    continue

                if collab_agent_list is not None:
                    # only fuse with collab agent and trial agent
                    if not (j in collab_agent_list or j == trial_agent_id):
                        continue

                # 第 j 个代理的特征变换到当前代理的坐标系中
                warp_feat = DetModelBase.feature_transformation(
                        b,
                        j,
                        agent_idx,
                        local_com_mat,
                        all_warp,
                        device,
                        size,
                        trans_matrices,
                        noise_level=unadv_pert,
                        noisy_agents=attacker_list,
                    )

                if pert is not None and j in attacker_list:
                    if unadv_pert is None:
                        # clip
                        eta = torch.clamp(pert[j], min=-eps, max=eps)
                    elif unadv_pert is True:
                        # noise on feature if unadv_pert is True
                        eta = pert[j]
                    else:
                        # noise on Point if unadv_pert is False
                        # noise on Trans if unadv_pert is float
                        eta = None
                    # Apply perturbation
                    warp_feat = warp_feat + eta if eta is not None else warp_feat
                    
                self.neighbor_feat_list.append(warp_feat)

                

    def get_decoded_layers(self, encoded_layers, feature_fuse_matrix, batch_size,ego_agent, no_fuse):
        """Replace the collaboration layer of the output from the encoder with fused feature maps.

        Args:
            encoded_layers (list): The output from the encoder.
            feature_fuse_matrix (tensor): The fused feature maps.
            batch_size (int): The batch size.

        Returns:
            A list. Output from the decoder.
        """
        if no_fuse is None:
            for idx in range(len(encoded_layers)):
                if idx != self.layer:
                    ego_tensor = encoded_layers[idx][ego_agent].unsqueeze(0) # [1,C,H,W]
                    encoded_layers[idx] = ego_tensor.expand(encoded_layers[idx].shape[0], -1, -1, -1)
        encoded_layers[self.layer] = feature_fuse_matrix
        decoded_layers = self.decoder(*encoded_layers, batch_size, kd_flag=self.kd_flag)
        return decoded_layers # torch.Size([8, 32, 256, 256])

    def get_cls_loc_result(self, x):
        """Get the classification and localization result.

        Args:
            x (tensor): The output from the last layer of the decoder.

        Returns:
            cls_preds (tensor): Predictions of the classification head.
            loc_preds (tensor): Predications of the localization head.
            result (dict): A dictionary of classificaion, localization, and optional motion state classification result.
        """
        # Cell Classification head
        cls_preds = self.classification(x) # torch.Size([8, 12, 256, 256])
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous() # torch.Size([8, 256, 256, 12])
        cls_preds = cls_preds.view(cls_preds.shape[0], -1, self.category_num)

        # Detection head
        loc_preds = self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(
            -1,
            loc_preds.size(1),
            loc_preds.size(2),
            self.anchor_num_per_loc,
            self.out_seq_len,
            self.box_code_size,
        )

        # loc_pred (N * T * W * H * loc)
        result = {"loc": loc_preds, "cls": cls_preds}

        # MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0], -1, motion_cat)
            result["state"] = motion_cls_preds

        return cls_preds, loc_preds, result


class ClassificationHead(nn.Module):
    """The classificaion head."""

    def __init__(self, config):
        super(ClassificationHead, self).__init__()

        category_num = config.category_num
        channel = 32
        if config.use_map:
            channel += 6
        if config.use_vis:
            channel += 13

        anchor_num_per_loc = len(config.anchor_size)

        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            channel,
            category_num * anchor_num_per_loc,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.bn1 = nn.BatchNorm2d(channel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) # torch.Size([8, 32, 256, 256])
        x = self.conv2(x) # torch.Size([8, 12, 256, 256])

        return x


class SingleRegressionHead(nn.Module):
    """The regression head."""

    def __init__(self, config):
        super(SingleRegressionHead, self).__init__()

        channel = 32
        if config.use_map:
            channel += 6
        if config.use_vis:
            channel += 13

        anchor_num_per_loc = len(config.anchor_size)
        box_code_size = config.box_code_size
        out_seq_len = 1 if config.only_det else config.pred_len

        if config.binary:
            if config.only_det:
                self.box_prediction = nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(channel),
                    nn.ReLU(),
                    nn.Conv2d(
                        channel,
                        anchor_num_per_loc * box_code_size * out_seq_len,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ),
                )
            else:
                self.box_prediction = nn.Sequential(
                    nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(
                        128,
                        anchor_num_per_loc * box_code_size * out_seq_len,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ),
                )

    def forward(self, x):
        box = self.box_prediction(x)

        return box
