# -*- coding: utf-8 -*-
# @Time   : 2022/3/8
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com
r"""
SGL
################################################
Reference:
    Jiancan Wu et al. "SGL: Self-supervised Graph Learning for Recommendation" in SIGIR 2021.

Reference code:
    https://github.com/wujcan/SGL
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree

from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import EmbLoss, BPRLoss
from recbole.utils import InputType

from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender
from recbole_gnn.model.layers import LightGCNConv


class SGLL(GeneralGraphRecommender):
    r"""SGL是一个基于GCN的推荐模型。
        SGL在经典的推荐监督任务的基础上增加了一个辅助的自我监督任务，通过自我判别来强化节点表示学习。
        SGL可以生成一个节点的多个视图，使同一节点的不同视图与其他节点的视图之间的一致性最大化。
        SGL设计了三种操作符来生成视图 — — 节点丢失、边丢失和随机游走 — — 以不同的方式改变图结构。
        我们按照原作者的方法，采用两两训练模式来实现该模型。

        SGL supplements the classical supervised task of recommendation with an auxiliary self supervised task, which
        reinforces node representation learning via self-discrimination.
        Specifically,SGL generates multiple views of a node, maximizing the agreement between different views of the
        same node compared to that of other nodes.
        SGL devises three operators to generate the views — node dropout, edge dropout, and random walk — that change
        the graph structure in different manners.
        We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SGLL, self).__init__(config, dataset)

        # 加载配置参数
        self.latent_dim = config["embedding_size"]  # dim
        self.n_layers = int(config["n_layers"])  # layer
        self.aug_type = config["type"]  # 用于生成视图的运算符
        self.drop_ratio = config["drop_ratio"]  # dropout
        self.ssl_tau = config["ssl_tau"]  # softmax温度系数 https://blog.csdn.net/qq_36560894/article/details/114874268
        self.reg_weight = config["reg_weight"]  # L2正则化权值
        self.ssl_weight = config["ssl_weight"]  # 控制SSL强度的超参数

        self.mf_loss = BPRLoss()  # BPR损失

        self._user = dataset.inter_feat[dataset.uid_field]  # user列
        self._item = dataset.inter_feat[dataset.iid_field]  # item列

        # 定义层和损失（layer 和 LightGCN是一样的）
        self.user_embedding = torch.nn.Embedding(self.n_users, self.latent_dim)  # 用户嵌入层
        self.item_embedding = torch.nn.Embedding(self.n_items, self.latent_dim)  # 项目嵌入层
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def forward(self):
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])  # 固定操作，先获取所有的嵌入
        embeddings_list = [all_embeddings]  # 加入嵌入列表作为第0层嵌入

        # 进行多层gcn的计算
        for layer_idx in range(self.n_layers):
            # all_embeddings = self.gcn_conv(all_embeddings, self.edge_index, self.edge_weight)
            # embeddings_list.append(all_embeddings)
            if layer_idx > 0:
                temp_embeddings = torch.cat(
                    [all_embeddings.unsqueeze(dim=1), embeddings_list[layer_idx - 1].unsqueeze(dim=1)],
                    dim=1
                )
                temp_embeddings = torch.mean(temp_embeddings, dim=1)
                all_embeddings = self.gcn_conv(temp_embeddings, self.edge_index, self.edge_weight)
            else:
                all_embeddings = self.gcn_conv(all_embeddings, self.edge_index, self.edge_weight)
            embeddings_list.append(all_embeddings)

        # 算出最终的嵌入
        lightgcn_all_embeddings = torch.stack(embeddings_list[:self.n_layers + 1], dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, embeddings_list

    def calc_bpr_loss(self, user_emd, item_emd, user_list, pos_item_list, neg_item_list):
        r"""Calculate the pairwise Bayesian Personalized Ranking (BPR) loss and parameter regularization loss.
            计算成对贝叶斯个性化排序(BPR)损失和参数正则化损失。
        Args:
            user_emd (torch.Tensor): Ego embedding of all users after forwarding.
            item_emd (torch.Tensor): Ego embedding of all items after forwarding.
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.
            neg_item_list (torch.Tensor): List of negative examples.

        Returns:
            torch.Tensor: Loss of BPR tasks and parameter regularization.
        """
        u_e = user_emd[user_list]
        pi_e = item_emd[pos_item_list]
        ni_e = item_emd[neg_item_list]
        p_scores = torch.mul(u_e, pi_e).sum(dim=1)
        n_scores = torch.mul(u_e, ni_e).sum(dim=1)
        # BPR损失
        l1 = self.mf_loss(p_scores, n_scores)

        # 计算l2正则化损失
        u_e_p = self.user_embedding(user_list)
        pi_e_p = self.item_embedding(pos_item_list)
        ni_e_p = self.item_embedding(neg_item_list)

        l2 = self.reg_loss(u_e_p, pi_e_p, ni_e_p)

        return l1 + l2 * self.reg_weight

    def calc_ssl_loss(self, user_list, pos_item_list, embed_list):
        r"""计算自监督任务的损失

        Args:
            user_list (torch.Tensor): List of the user.  uid列表（交互的uid） [batch_size, ]
            pos_item_list (torch.Tensor): List of positive examples.        正例 iid列表（交互的iid）
            embed_list: 包含每一层卷积结果的嵌入列表
        Returns:
            torch.Tensor: Loss of self-supervised tasks.
        """
        ssl_losses = []

        for layer in range(self.n_layers - 2, self.n_layers + 1):
            center_embed = embed_list[layer - 2]
            center_user_embed, center_item_embed = torch.split(center_embed, [self.n_users, self.n_items], dim=0)

            # 先拿后面每一层和第一层做对比损失
            # for layer in range(1, len(embed_list)):
            context_embed = embed_list[layer]
            context_user_embed, context_item_embed = torch.split(context_embed, [self.n_users, self.n_items], dim=0)

            u_emd1 = F.normalize(center_user_embed[user_list])  # L2正则  [b, dim]
            u_emd2 = F.normalize(context_user_embed[user_list])  # L2正则  [b, dim]
            v1 = torch.sum(u_emd1 * u_emd2, dim=1)
            v1 = torch.exp(v1 / self.ssl_tau)
            all_user2 = F.normalize(context_user_embed)
            v2 = u_emd1.matmul(all_user2.T)
            v2 = torch.sum(torch.exp(v2 / self.ssl_tau), dim=1)
            user_loss = -torch.sum(torch.log(v1 / v2))

            i_emd1 = F.normalize(center_item_embed[pos_item_list])
            i_emd2 = F.normalize(context_item_embed[pos_item_list])

            v3 = torch.sum(i_emd1 * i_emd2, dim=1)
            all_item2 = F.normalize(context_item_embed)
            v4 = i_emd1.matmul(all_item2.T)
            v3 = torch.exp(v3 / self.ssl_tau)
            v4 = torch.sum(torch.exp(v4 / self.ssl_tau), dim=1)
            item_loss = -torch.sum(torch.log(v3 / v4))

            ssl_loss = self.ssl_weight * (user_loss + self.alpha * item_loss)
            ssl_losses.append(ssl_loss)

        ssl_losses = torch.sum(torch.stack(ssl_losses))

        return ssl_losses

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:  # 计算损失前先清空保存的user和item嵌入向量
            self.restore_user_e, self.restore_item_e = None, None

        user_list = interaction[self.USER_ID]
        pos_item_list = interaction[self.ITEM_ID]
        neg_item_list = interaction[self.NEG_ITEM_ID]

        user_emd, item_emd, embed_list = self.forward()  # 原始图的前向计算（和LightGCN是一样的）

        bpr_loss = self.calc_bpr_loss(user_emd, item_emd, user_list, pos_item_list, neg_item_list)
        ssl_loss = self.calc_ssl_loss(user_list, pos_item_list, embed_list)

        total_loss = bpr_loss + ssl_loss

        return total_loss

    def predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, _ = self.forward()

        user = self.restore_user_e[interaction[self.USER_ID]]
        item = self.restore_item_e[interaction[self.ITEM_ID]]
        return torch.sum(user * item, dim=1)

    def full_sort_predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, _ = self.forward()

        user = self.restore_user_e[interaction[self.USER_ID]]
        return user.matmul(self.restore_item_e.T)
