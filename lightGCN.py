#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : lightGCN.py
# @Author: yanms
# @Date  : 2023/9/23 15:10
# @Desc  :
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


class LightGCN(nn.Module):
    '''
    interaction_matrix: coo_matrix
    '''

    def __init__(self, device, n_layers, n_users, n_items, interaction_matrix):
        super(LightGCN, self).__init__()
        self.device = device
        self.n_layers = n_layers
        self.user_count = n_users
        self.item_count = n_items
        self.interaction_matrix = interaction_matrix
        self.A_adj_matrix = self._get_a_adj_matrix()


    def _get_a_adj_matrix(self):
        """
        得到 系数矩阵A~
        :return:
        """
        A = sp.dok_matrix((self.user_count + self.item_count, self.user_count + self.item_count), dtype=float)
        inter_matrix = self.interaction_matrix
        inter_matrix_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_matrix.row, inter_matrix.col + self.user_count), [1] * inter_matrix.nnz))
        data_dict.update(dict(zip(zip(inter_matrix_t.row + self.user_count, inter_matrix_t.col), [1] * inter_matrix_t.nnz)))
        A._update(data_dict)
        sum_list = (A > 0).sum(axis=1)
        diag = np.array(sum_list.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        A_adj = D * A * D
        A_adj = sp.coo_matrix(A_adj)
        row = A_adj.row
        col = A_adj.col
        index = torch.LongTensor([row, col])
        data = torch.FloatTensor(A_adj.data)
        A_sparse = torch.sparse.FloatTensor(index, data, torch.Size(A_adj.shape))
        return A_sparse

    def forward(self, in_embs):

        result = [in_embs]
        for i in range(self.n_layers):
            in_embs = torch.sparse.mm(self.A_adj_matrix.to(self.device), in_embs)
            in_embs = F.normalize(in_embs, dim=-1)
            result.append(in_embs / (i + 1))
            result.append(in_embs)

        result = torch.stack(result, dim=0)
        result = torch.sum(result, dim=0)

        return result