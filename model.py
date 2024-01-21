#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author:
# @Date  : 2023/9/23 16:16
# @Desc  :
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data_set import DataSet
from utils import BPRLoss, EmbLoss
from lightGCN import LightGCN


class BIPN(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(BIPN, self).__init__()

        self.device = args.device
        self.layers = args.layers
        self.reg_weight = args.reg_weight
        self.log_reg = args.log_reg
        self.node_dropout = args.node_dropout
        self.message_dropout = nn.Dropout(p=args.message_dropout)
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.inter_matrix = dataset.inter_matrix
        self.user_item_inter_set = dataset.user_item_inter_set
        self.test_users = list(dataset.test_interacts.keys())
        self.behaviors = args.behaviors
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self.bhv_embs = nn.Parameter(torch.eye(len(self.behaviors)))
        self.global_Graph = LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, dataset.all_inter_matrix)
        self.behavior_Graph = LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, dataset.inter_matrix[-1])

        self.RZ = nn.Linear(2 * self.embedding_size + len(self.behaviors), 2 * self.embedding_size, bias=False)
        self.U = nn.Linear(2 * self.embedding_size + len(self.behaviors), self.embedding_size, bias=False)

        self.reg_weight = args.reg_weight
        self.layers = args.layers
        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()
        self.cross_loss = nn.BCELoss()

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model
        self.message_dropout = nn.Dropout(p=args.message_dropout)

        self.storage_user_embeddings = None
        self.storage_item_embeddings = None

        self.apply(self._init_weights)

        self._load_model()

    def _init_weights(self, module):

        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)

    def agg_info(self, u_emb, i_emb, bhv_emb):
        in_feature = torch.cat((u_emb, i_emb, bhv_emb), dim=-1)
        RZ = torch.sigmoid(self.RZ(in_feature))
        R, Z = torch.chunk(RZ, 2, dim=-1)
        RU = R * u_emb
        RU = torch.cat((RU, i_emb, bhv_emb), dim=-1)
        u_hat = torch.tanh(self.U(RU))
        u_final = Z * u_hat

        return u_final


    def user_agg_item(self, user_samples, u_emb, ini_item_embs):

        keys = user_samples.tolist()
        user_item_set = self.user_item_inter_set[-1]
        agg_items = [user_item_set[x] for x in keys]
        degree = [len(x) for x in agg_items]
        degree = torch.tensor(degree).unsqueeze(-1).to(self.device)
        max_len = max(len(l) for l in agg_items)
        padded_list = np.zeros((len(agg_items), max_len), dtype=int)
        for i, l in enumerate(agg_items):
            padded_list[i, :len(l)] = l
        padded_list = torch.from_numpy(padded_list).to(self.device)
        mask = (padded_list == 0)
        agg_item_emb = ini_item_embs[padded_list.long()]
        u_in = u_emb.repeat(1, max_len, 1)
        bhv_emb = self.bhv_embs[-1].repeat(u_in.shape[0], u_in.shape[1], 1)

        u_final = self.agg_info(u_in, agg_item_emb, bhv_emb)

        u_final[mask] = 0
        u_final = torch.sum(u_final, dim=1)
        lamb = 1 / (degree + 1e-8)
        u_final = u_final.unsqueeze(1)
        u_final = u_final

        return u_final, lamb

    def forward(self, batch_data):
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = self.global_Graph(all_embeddings)
        user_embedding, item_embedding = torch.split(all_embeddings, [self.n_users + 1, self.n_items + 1])

        buy_embeddings = self.behavior_Graph(all_embeddings)
        user_buy_embedding, item_buy_embedding = torch.split(buy_embeddings, [self.n_users + 1, self.n_items + 1])

        p_samples = batch_data[:, 0, :]
        n_samples = batch_data[:, 1:-1, :].reshape(-1, 4)
        samples = torch.cat([p_samples, n_samples], dim=0)

        u_sample, i_samples, b_samples, gt_samples = torch.chunk(samples, 4, dim=-1)
        u_emb = user_embedding[u_sample.long()].squeeze()
        i_emb = item_embedding[i_samples.squeeze().long()]
        bhv_emb = self.bhv_embs[b_samples.reshape(-1).long()]
        u_final = self.agg_info(u_emb, i_emb, bhv_emb)

        log_loss_scores = torch.sum((u_final * i_emb), dim=-1).unsqueeze(1)
        log_loss = self.cross_loss(torch.sigmoid(log_loss_scores), gt_samples.float())

        pair_samples = batch_data[:, -1, :-1]
        mask = torch.any(pair_samples != 0, dim=-1)
        pair_samples = pair_samples[mask]
        bpr_loss = 0
        if pair_samples.shape[0] > 0:
            user_samples = pair_samples[:, 0].long()
            item_samples = pair_samples[:, 1:].long()
            u_emb = user_embedding[user_samples].unsqueeze(1)
            i_emb = item_embedding[item_samples]

            u_point, lamb = self.user_agg_item(user_samples, u_emb, item_embedding)
            u_gen_emb = u_emb + user_buy_embedding[user_samples].unsqueeze(1)
            i_final = i_emb + item_buy_embedding[item_samples]
            score_point = torch.sum((u_point * i_emb), dim=-1)
            score_gen = torch.sum((u_gen_emb * i_final), dim=-1)
            bpr_scores = (1 - lamb) * score_point + lamb * score_gen
            p_scores, n_scores = torch.chunk(bpr_scores, 2, dim=-1)
            bpr_loss += self.bpr_loss(p_scores, n_scores)
        emb_loss = self.emb_loss(self.user_embedding.weight, self.item_embedding.weight)
        loss = self.log_reg * log_loss + (1 - self.log_reg) * bpr_loss + self.reg_weight * emb_loss

        return loss

    def full_predict(self, users):
        if self.storage_user_embeddings is None:
            all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
            all_embeddings = self.global_Graph(all_embeddings)
            user_embedding, item_embedding = torch.split(all_embeddings, [self.n_users + 1, self.n_items + 1])

            buy_embeddings = self.behavior_Graph(all_embeddings)
            user_buy_embedding, item_buy_embedding = torch.split(buy_embeddings, [self.n_users + 1, self.n_items + 1])

            self.storage_user_embeddings = torch.zeros(self.n_users + 1, self.embedding_size).to(self.device)

            test_users = [int(x) for x in self.test_users]
            tmp_emb_list = []
            for i in range(0, len(test_users), 100):
                tmp_users = test_users[i: i + 100]
                tmp_users = torch.LongTensor(tmp_users)
                tmp_embeddings = user_embedding[tmp_users].unsqueeze(1)
                tmp_embeddings, _ = self.user_agg_item(tmp_users, tmp_embeddings, item_embedding)
                tmp_emb_list.append(tmp_embeddings.squeeze())
            tmp_emb_list = torch.cat(tmp_emb_list, dim=0)
            for index, key in enumerate(test_users):
                self.storage_user_embeddings[key] = tmp_emb_list[index]

            user_item_set = self.user_item_inter_set[-1]
            degree = [len(x) for x in user_item_set]
            degree = torch.tensor(degree).unsqueeze(-1).to(self.device)
            lamb = 1/(degree + 1e-8)

            user_embedding = user_embedding + user_buy_embedding
            user_embedding = lamb * user_embedding
            self.storage_user_embeddings = (1-lamb) * self.storage_user_embeddings


            self.storage_user_embeddings = torch.cat((self.storage_user_embeddings, user_embedding), dim=-1)
            self.storage_item_embeddings = torch.cat((item_embedding, item_embedding + item_buy_embedding), dim=-1)

        user_emb = self.storage_user_embeddings[users.long()]
        scores = torch.matmul(user_emb, self.storage_item_embeddings.transpose(0, 1))

        return scores

