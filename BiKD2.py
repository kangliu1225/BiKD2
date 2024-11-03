# -*- coding: utf-8 -*-
import argparse
import dgl.function as fn
import dgl
import os
import torch
import numpy as np
from load_data import *
from util import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.manual_seed(2024)
np.random.seed(2024)

global_emb_size = 64
dataset_name = os.listdir("../data/")
dataset_name.remove('glove.6B.100d.txt')
dataset_name = dataset_name[0]

class Data(object):

    def __init__(self, dataset_name, dataset_path, device, review_fea_size):
        self._device = device
        self._review_fea_size = review_fea_size

        sent_train_data, sent_valid_data, sent_test_data, _, _, dataset_info = load_sentiment_data(
            dataset_path)

        self._num_user = dataset_info['user_size']
        self._num_item = dataset_info['item_size']

        review_feat_path = f'../data/{dataset_name}/bert-base-uncased_sentence_vectors_dim_{review_fea_size}.pkl'
        self.train_review_feat = torch.load(review_feat_path)

        self.review_feat_updated = {}

        def process_sent_data(info):
            user_id = info['user_id'].to_list()
            # 让物品的id从max user id开始，相当于将用户和物品节点视为一类节点；
            item_id = [int(i) + self._num_user for i in info['item_id'].to_list()]
            rating = info['rating'].to_list()

            return user_id, item_id, rating

        self.train_datas = process_sent_data(sent_train_data)
        self.valid_datas = process_sent_data(sent_valid_data)
        self.test_datas = process_sent_data(sent_test_data)
        self.possible_rating_values = np.unique(self.train_datas[2])

        self.user_item_rating = {}
        self.user_rating_count = {}
        self.user_ratings_test = {}
        self.user_item_ratings = {}

        self.user_items = {}

        def _generate_train_pair_value(data: tuple):
            user_id, item_id, rating = np.array(data[0], dtype=np.int64), np.array(data[1], dtype=np.int64), \
                                       np.array(data[2], dtype=np.int64)

            rating_pairs = (user_id, item_id)
            rating_pairs_rev = (item_id, user_id)
            rating_pairs = np.concatenate([rating_pairs, rating_pairs_rev], axis=1)

            rating_values = np.concatenate([rating, rating],
                                           axis=0)

            for i in range(len(rating)):
                uid, iid = user_id[i], item_id[i]

                if uid not in self.user_item_rating:
                    self.user_item_rating[uid] = []
                    self.user_item_ratings[uid] = {}
                    self.user_items[uid] = []
                self.user_item_rating[uid].append((iid, rating[i]))
                self.user_item_ratings[uid][iid] = rating[i]
                self.user_items[uid].append(iid)

                if uid not in self.user_rating_count:
                    self.user_rating_count[uid] = [0, 0, 0, 0, 0]

                self.user_rating_count[uid][rating[i] - 1] += 1

            return rating_pairs, rating_values

        def _generate_valid_pair_value(data: tuple):
            user_id, item_id, rating = data[0], data[1], data[2]

            rating_pairs = (np.array(user_id, dtype=np.int64),
                            np.array(item_id, dtype=np.int64))

            rating_values = np.array(rating, dtype=np.float32)

            return rating_pairs, rating_values

        def _generate_test_pair_value(data: tuple):
            user_id, item_id, rating = data[0], data[1], data[2]

            rating_pairs = (np.array(user_id, dtype=np.int64),
                            np.array(item_id, dtype=np.int64))

            rating_values = np.array(rating, dtype=np.float32)

            for i in range(len(rating)):
                uid, iid = user_id[i], item_id[i]

                if uid not in self.user_ratings_test:
                    self.user_ratings_test[uid] = []

                self.user_ratings_test[uid].append(rating[i])

            return rating_pairs, rating_values

        print('Generating train/valid/test data.\n')
        self.train_rating_pairs, self.train_rating_values = _generate_train_pair_value(self.train_datas)
        self.valid_rating_pairs, self.valid_rating_values = _generate_valid_pair_value(self.valid_datas)
        self.test_rating_pairs, self.test_rating_values = _generate_test_pair_value(self.test_datas)

        count_mis = 0
        count_same = 0
        count_all = 0
        for uid, items in self.user_ratings_test.items():
            count_all += len(items)
            max_rate_train = np.where(self.user_rating_count[uid] == np.max(self.user_rating_count[uid]))[0]
            for i in items:
                if i - 1 not in max_rate_train:
                    count_mis += 1
                else:
                    count_same += 1

        print(count_mis, count_same, count_all, len(self.test_rating_values))

        ## find and collect extremely distributed samples
        self.extra_dist_pairs = {}
        self.extra_uid, self.extra_iid, self.extra_r_idx = [], [], []
        for uid, l in self.user_rating_count.items():

            max_count = np.max(l)
            max_idx = np.where(l == max_count)[0]

            for i, c in enumerate(l):
                # if c == 0 or abs(max_idx.max() - i) <= 1 or abs(max_idx.min() - i) <= 1:
                if i in max_idx or c == 0:
                    continue

                if c / max_count <= 0.2:
                    if uid not in self.extra_dist_pairs:
                        self.extra_dist_pairs[uid] = []
                    self.extra_dist_pairs[uid].append((i + 1, c))
                    for item in self.user_item_rating[uid]:
                        self.extra_uid.append(uid)
                        self.extra_iid.append(item[0])
                        self.extra_r_idx.append(i)

        self.item_rate_review = {}

        for u, d in self.user_item_ratings.items():
            for i, r in d.items():
                review = self.train_review_feat[(u, i - self._num_user)]
                if i not in self.item_rate_review:
                    self.item_rate_review[i] = {}
                if r not in self.item_rate_review[i]:
                    self.item_rate_review[i][r] = []
                self.item_rate_review[i][r].append(review)

        self.mean_review_feat_list_1 = []
        self.mean_review_feat_list_2 = []
        self.mean_review_feat_list_3 = []
        self.mean_review_feat_list_4 = []
        self.mean_review_feat_list_5 = []
        for key, value in self.train_review_feat.items():
            self.review_feat_updated[(key[0], key[1] + self._num_user)] = value
            self.review_feat_updated[(key[1] + self._num_user, key[0])] = value
            if key[1] + self._num_user not in self.user_item_ratings[key[0]]:
                continue

            if self.user_item_ratings[key[0]][key[1] + self._num_user] == 1:
                self.mean_review_feat_list_1.append(value)

            elif self.user_item_ratings[key[0]][key[1] + self._num_user] == 2:
                self.mean_review_feat_list_2.append(value)

            elif self.user_item_ratings[key[0]][key[1] + self._num_user] == 3:
                self.mean_review_feat_list_3.append(value)

            elif self.user_item_ratings[key[0]][key[1] + self._num_user] == 4:
                self.mean_review_feat_list_4.append(value)

            else:
                self.mean_review_feat_list_5.append(value)

        self.mean_review_feat_1 = torch.mean(torch.stack(self.mean_review_feat_list_1, dim=0), dim=0)
        self.mean_review_feat_2 = torch.mean(torch.stack(self.mean_review_feat_list_2, dim=0), dim=0)
        self.mean_review_feat_3 = torch.mean(torch.stack(self.mean_review_feat_list_3, dim=0), dim=0)
        self.mean_review_feat_4 = torch.mean(torch.stack(self.mean_review_feat_list_4, dim=0), dim=0)
        self.mean_review_feat_5 = torch.mean(torch.stack(self.mean_review_feat_list_5, dim=0), dim=0)

        print('Generating train graph.\n')
        self.train_enc_graph = self._generate_enc_graph()

    def update_graph(self, uid_list, iid_list, r_list):
        uid_list, iid_list, r_list = np.array(uid_list), np.array(iid_list), np.array(r_list)
        rating_pairs = (uid_list, iid_list)
        rating_pairs_rev = (iid_list, uid_list)
        self.train_rating_pairs = np.concatenate([self.train_rating_pairs, rating_pairs, rating_pairs_rev], axis=1)

        self.train_rating_values = np.concatenate([self.train_rating_values, r_list, r_list], axis=0)

        self.train_enc_graph_updated = self._generate_enc_graph()

    def _generate_enc_graph(self):
        user_item_r = np.zeros((self._num_user + self._num_item, self._num_item + self._num_user), dtype=np.float32)
        for i in range(len(self.train_rating_values)):
            user_item_r[[self.train_rating_pairs[0][i], self.train_rating_pairs[1][i]]] = self.train_rating_values[i]
        record_size = self.train_rating_pairs[0].shape[0]
        review_feat_list = [self.review_feat_updated[(self.train_rating_pairs[0][x], self.train_rating_pairs[1][x])] for
                            x in
                            range(record_size)]
        review_feat_list = torch.stack(review_feat_list).to(torch.float32)

        rating_row, rating_col = self.train_rating_pairs

        graph_dict = {}
        for rating in self.possible_rating_values:
            ridx = np.where(self.train_rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]

            graph_dict[str(rating)] = dgl.graph((rrow, rcol), num_nodes=self._num_user + self._num_item)
            graph_dict[str(rating)].edata['review_feat'] = review_feat_list[ridx]

        def _calc_norm(x, d):
            x = x.numpy().astype('float32')
            x[x == 0.] = np.inf
            x = torch.FloatTensor(1. / np.power(x, d))
            return x.unsqueeze(1)

        c = []
        for r_1 in self.possible_rating_values:
            c.append(graph_dict[str(r_1)].in_degrees())
            graph_dict[str(r_1)].ndata['ci_r'] = _calc_norm(graph_dict[str(r_1)].in_degrees(), 0.5)

        c_sum = _calc_norm(torch.sum(torch.stack(c, dim=0), dim=0), 0.5)
        c_sum_mean = _calc_norm(torch.sum(torch.stack(c, dim=0), dim=0), 1)

        for r_1 in self.possible_rating_values:
            graph_dict[str(r_1)].ndata['ci'] = c_sum
            graph_dict[str(r_1)].ndata['ci_mean'] = c_sum_mean

        return graph_dict

    def _train_data(self, batch_size=1024):

        rating_pairs, rating_values = self.train_rating_pairs, self.train_rating_values
        idx = np.arange(0, len(rating_values))
        np.random.shuffle(idx)
        rating_pairs = (rating_pairs[0][idx], rating_pairs[1][idx])
        rating_values = rating_values[idx]

        data_len = len(rating_values)

        users, items = rating_pairs[0], rating_pairs[1]
        u_list, i_list, r_list = [], [], []
        review_list = []
        n_batch = data_len // batch_size + 1

        for i in range(n_batch):
            begin_idx = i * batch_size
            end_idx = begin_idx + batch_size if i != n_batch - 1 else len(self.train_rating_values)
            batch_users, batch_items, batch_ratings = users[begin_idx: end_idx], items[
                                                                                 begin_idx: end_idx], rating_values[
                                                                                                      begin_idx: end_idx]

            u_list.append(torch.LongTensor(batch_users).to('cuda:0'))
            i_list.append(torch.LongTensor(batch_items).to('cuda:0'))
            r_list.append(torch.LongTensor(batch_ratings - 1).to('cuda:0'))

        return u_list, i_list, r_list

    def _test_data(self, flag='valid'):
        if flag == 'valid':
            rating_pairs, rating_values = self.valid_rating_pairs, self.valid_rating_values
        else:
            rating_pairs, rating_values = self.test_rating_pairs, self.test_rating_values
        u_list, i_list, r_list = [], [], []
        for i in range(len(rating_values)):
            u_list.append(rating_pairs[0][i])
            i_list.append(rating_pairs[1][i])
            r_list.append(rating_values[i])
        u_list = torch.LongTensor(u_list).to('cuda:0')
        i_list = torch.LongTensor(i_list).to('cuda:0')
        r_list = torch.FloatTensor(r_list).to('cuda:0')
        return u_list, i_list, r_list

def config():
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('--device', default='0', type=int, help='gpu.')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")

    parser.add_argument('--gcn_dropout', type=float, default=0.5)
    parser.add_argument('--train_max_iter', type=int, default=1000)
    parser.add_argument('--train_optimizer', type=str, default="Adam")
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_early_stopping_patience', type=int, default=50)

    args = parser.parse_args()
    args.model_short_name = 'RGC'
    args.dataset_name = dataset_name
    args.dataset_path = f'../data/{dataset_name}/{dataset_name}.json'
    args.emb_size = 64
    args.emb_dim = 64
    args.origin_emb_dim = 60

    args.gcn_dropout = 0.7
    args.device = torch.device(args.device)
    args.train_max_iter = 1000
    # args.batch_size = 271466
    args.batch_size = 100000

    return args

gloabl_dropout = 0.5
global_review_size = 64

class GCN_interaction(nn.Module):
    def __init__(self):
        super(GCN_interaction, self).__init__()
        self.dropout = nn.Dropout(0.7)
        self.review_w = nn.Linear(64, global_review_size, bias=False, device='cuda:0')


    def forward(self, g, feature):
        g.srcdata['h_r'] = feature
        g.edata['r'] = self.review_w(g.edata['review_feat'])

        g.update_all(lambda edges: {
            'm': (torch.cat([edges.src['h_r'], edges.data['r']], dim=1)) * self.dropout(edges.src['ci'])},
                     fn.sum(msg='m', out='h'))

        rst = g.dstdata['h'] * g.dstdata['ci']

        return rst

class GCN_interaction_dis(nn.Module):
    def __init__(self):
        super(GCN_interaction_dis, self).__init__()
        self.dropout = nn.Dropout(0.7)

    def forward(self, g, feature):
        g.srcdata['h_r'] = feature

        g.update_all(lambda edges: {
            'm': (edges.src['h_r']) * self.dropout(edges.src['ci'])},
                     fn.sum(msg='m', out='h'))

        rst = g.dstdata['h'] * g.dstdata['ci']

        return rst

class GCN_interaction_com(nn.Module):
    def __init__(self):
        super(GCN_interaction_com, self).__init__()
        self.dropout = nn.Dropout(0.7)

    def forward(self, g, feature):
        g.srcdata['h_r'] = feature

        g.update_all(lambda edges: {
            'm': (edges.src['h_r']) * self.dropout(edges.src['ci'])},
                     fn.sum(msg='m', out='h'))

        rst = g.dstdata['h'] * g.dstdata['ci']

        return rst

class GCN_review(nn.Module):
    def __init__(self):
        super(GCN_review, self).__init__()
        self.dropout = nn.Dropout(0.7)
        self.review_w = nn.Linear(64, global_review_size, bias=False, device='cuda:0')

    def forward(self, g):
        g.edata['r'] = self.review_w(g.edata['review_feat'])

        g.update_all(lambda edges: {
            'm': (edges.data['r']) * self.dropout(edges.src['ci'])},
                     fn.sum(msg='m', out='h'))

        rst = g.dstdata['h'] * g.dstdata['ci']

        return rst


class ProjectedDotProductSimilarity(nn.Module):

    def __init__(self, tensor_1_dim, tensor_2_dim, projected_dim,
                 reuse_weight=False, bias=False, activation=None):
        super(ProjectedDotProductSimilarity, self).__init__()
        self.reuse_weight = reuse_weight
        self.projecting_weight_1 = nn.Parameter(torch.Tensor(tensor_1_dim, projected_dim))
        if self.reuse_weight:
            if tensor_1_dim != tensor_2_dim:
                raise ValueError('if reuse_weight=True, tensor_1_dim must equal tensor_2_dim')
        else:
            self.projecting_weight_2 = nn.Parameter(torch.Tensor(tensor_2_dim, projected_dim))
        self.bias = nn.Parameter(torch.Tensor(1)) if bias else None
        self.activation = activation

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.projecting_weight_1)
        if not self.reuse_weight:
            nn.init.xavier_uniform_(self.projecting_weight_2)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, tensor_1, tensor_2):
        projected_tensor_1 = torch.matmul(tensor_1, self.projecting_weight_1)
        if self.reuse_weight:
            projected_tensor_2 = torch.matmul(tensor_2, self.projecting_weight_1)
        else:
            projected_tensor_2 = torch.matmul(tensor_2, self.projecting_weight_2)
        result = (projected_tensor_1 * projected_tensor_2).sum(dim=-1)
        if self.bias is not None:
            result = result + self.bias
        if self.activation is not None:
            result = self.activation(result)
        return result


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()

        self.weight = nn.ParameterDict({
            str(r): nn.Parameter(torch.Tensor(params.num_users + params.num_items, params.emb_dim))
            for r in [1, 2, 3, 4, 5]
        })

        self.weight_com = nn.ParameterDict({
            str(r): nn.Parameter(torch.Tensor(params.num_users + params.num_items, params.emb_dim))
            for r in [1, 2, 3, 4, 5]
        })

        self.weight_dis = nn.ParameterDict({
            str(r): nn.Parameter(torch.Tensor(params.num_users + params.num_items, params.emb_dim))
            for r in [1, 2, 3, 4, 5]
        })

        self.weight_origin = nn.Parameter(torch.Tensor(params.num_users + params.num_items, global_emb_size * 5))

        self.encoder_interaction = nn.ModuleDict({
            str(i + 1): GCN_interaction() for i in range(5)
        })

        self.encoder_review = nn.ModuleDict({
            str(i + 1): GCN_review() for i in range(5)
        })

        self.encoder_interaction_com = nn.ModuleDict({
            str(i + 1): GCN_interaction_com() for i in range(5)
        })

        self.encoder_interaction_dis = nn.ModuleDict({
            str(i + 1): GCN_interaction_dis() for i in range(5)
        })

        self.num_user = params.num_users
        self.num_item = params.num_items

        self.fc_user = nn.Linear(global_review_size * 5 * 2,
                                 global_review_size * 5 * 2)
        self.fc_item = nn.Linear(global_review_size * 5 * 2,
                                 global_review_size * 5 * 2)


        self.fc_user_com_1 = nn.Linear(global_review_size * 5 * 1,
                                     global_review_size * 5 * 1)
        self.fc_item_com_1 = nn.Linear(global_review_size * 5 * 1,
                                     global_review_size * 5 * 1)


        self.fc_user_dis_1 = nn.Linear(global_review_size * 5 * 1,
                                       global_review_size * 5 * 1)
        self.fc_item_dis_1 = nn.Linear(global_review_size * 5 * 1,
                                       global_review_size * 5 * 1)

        self.fc_user_review = nn.Linear(global_review_size * 5 * 1,
                                 global_review_size * 5 * 1)

        self.fc_item_review = nn.Linear(global_review_size * 5 * 1,
                                 global_review_size * 5 * 1)

        self.dropout_1 = nn.Dropout(0.7)
        self.dropout_2 = nn.Dropout(0.7)
        self.dropout_3 = nn.Dropout(0.7)
        self.dropout_4 = nn.Dropout(0.7)

        self.predictor_interaction = nn.Sequential(
            nn.Linear(global_review_size * 5 * 2,
                      global_review_size * 5 * 1, bias=False),
            nn.ReLU(),
            nn.Linear(global_review_size * 5 * 1, 5, bias=False),
        )

        self.predictor_review = nn.Sequential(
            nn.Linear(global_review_size * 5 * 1,
                      global_review_size * 5 * 1, bias=False),
            nn.ReLU(),
            nn.Linear(global_review_size * 5 * 1, 5, bias=False),
        )

        self.predictor_interaction_dis = nn.Sequential(
            nn.Linear(global_review_size * 5 * 1,
                      global_review_size * 5 * 1, bias=False),
            nn.ReLU(),
            nn.Linear(global_review_size * 5 * 1, 5, bias=False),
        )

        self.predictor_interaction_com = nn.Sequential(
            nn.Linear(global_review_size * 5 * 1,
                      global_review_size * 5 * 1, bias=False),
            nn.ReLU(),
            nn.Linear(global_review_size * 5 * 1, 5, bias=False),
        )

        self.predictor_review_dis = nn.Sequential(
            nn.Linear(global_review_size * 5 * 1,
                      global_review_size * 5 * 1, bias=False),
            nn.ReLU(),
            nn.Linear(global_review_size * 5 * 1, 5, bias=False),
        )

        self.new_loss = ProjectedDotProductSimilarity(320, 320, 320)

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, enc_graph_dict, users, items):
        feat_all, feat_all_dis_id, feat_all_dis_review = [], [], []
        feat_all_r, feat_all_com, feat_all_dis = [], [], []
        rate_list = [1, 2, 3, 4, 5]

        for r in rate_list:
            # first layer
            feat_ = self.encoder_interaction[str(r)](enc_graph_dict[str(r)], self.weight[str(r)])
            feat_all.append(feat_)

            feat_com = self.encoder_interaction_com[str(r)](enc_graph_dict[str(r)], self.weight_com[str(r)])
            feat_all_com.append(feat_com)

            feat_dis = self.encoder_interaction_dis[str(r)](enc_graph_dict[str(r)], self.weight_dis[str(r)])
            feat_all_dis.append(feat_dis)

            feat_r = self.encoder_review[str(r)](enc_graph_dict[str(r)])
            feat_all_r.append(feat_r)

        user_feat, item_feat = torch.split(torch.cat(feat_all, dim=-1), [self.num_user, self.num_item], dim=0)
        u_feat = self.fc_user(self.dropout_1(user_feat))
        i_feat = self.fc_item(self.dropout_1(item_feat))
        feat_id = torch.cat([u_feat, i_feat], dim=0)
        user_embeddings_id, item_embeddings_id = feat_id[users], feat_id[items]
        pred_ratings = self.predictor_interaction(user_embeddings_id * item_embeddings_id).squeeze()

        user_feat_com, item_feat_com = torch.split(torch.cat(feat_all_com, dim=-1), [self.num_user, self.num_item], dim=0)
        u_feat = self.fc_user_com_1(self.dropout_2(user_feat_com))
        i_feat = self.fc_item_com_1(self.dropout_2(item_feat_com))
        feat_id = torch.cat([u_feat, i_feat], dim=0)
        user_embeddings_id, item_embeddings_id = feat_id[users], feat_id[items]
        pred_ratings_com = self.predictor_interaction_com(user_embeddings_id * item_embeddings_id).squeeze()

        user_feat_dis, item_feat_dis = torch.split(torch.cat(feat_all_dis, dim=-1), [self.num_user, self.num_item], dim=0)
        u_feat = self.fc_user_dis_1(self.dropout_3(user_feat_dis))
        i_feat = self.fc_item_dis_1(self.dropout_3(item_feat_dis))
        feat_id = torch.cat([u_feat, i_feat], dim=0)
        user_embeddings_id, item_embeddings_id = feat_id[users], feat_id[items]
        pred_ratings_dis = self.predictor_interaction_dis(user_embeddings_id * item_embeddings_id).squeeze()

        user_feat_review, item_feat_review = torch.split(torch.cat(feat_all_r, dim=-1), [self.num_user, self.num_item], dim=0)
        u_feat_review = self.fc_user_review(self.dropout_4(user_feat_review))
        i_feat_review = self.fc_item_review(self.dropout_4(item_feat_review))
        feat_review = torch.cat([u_feat_review, i_feat_review], dim=0)
        user_embeddings_review, item_embeddings_review = feat_review[users], feat_review[items]
        pred_ratings_review = self.predictor_review(user_embeddings_review * item_embeddings_review).squeeze()

        loss_kd_feat_com = - 0.5 * torch.mean(torch.cosine_similarity(user_feat_com, user_feat_review, dim=0)) - \
                           0.5 * torch.mean(torch.cosine_similarity(item_feat_com, item_feat_review, dim=0)) - 0.0 * torch.mean(self.new_loss(user_feat_dis, user_feat_review)) - \
                           0.0 * torch.mean(self.new_loss(item_feat_dis, item_feat_review))
        loss_kd_feat_dis = + 0.5 * torch.mean(torch.cosine_similarity(user_feat_dis, user_feat_review, dim=0)) + \
                           0.5 * torch.mean(torch.cosine_similarity(item_feat_dis, item_feat_review, dim=0))+ 0.0 * torch.mean(self.new_loss(user_feat_dis, user_feat_review)) + \
                           0.0 * torch.mean(self.new_loss(item_feat_dis, item_feat_review))


        return pred_ratings, pred_ratings_review, \
               pred_ratings_com, \
               pred_ratings_dis, \
               loss_kd_feat_com, loss_kd_feat_dis


def evaluate(args, net, dataset, flag='valid', add=False, epoch=256):
    nd_possible_rating_values = torch.FloatTensor([1, 2, 3, 4, 5]).to(args.device)

    u_list, i_list, r_list = dataset._test_data(flag=flag)

    net.eval()

    with torch.no_grad():

        if epoch <= g_epoch:

            pred_ratings, pred_ratings_review, \
            pred_ratings_com, \
            pred_ratings_dis, \
            loss_kd_feat_com, loss_kd_feat_dis = net(dataset.train_enc_graph, u_list, i_list)
        else:
            pred_ratings, pred_ratings_review, \
            pred_ratings_com, \
            pred_ratings_dis, \
            loss_kd_feat_com, loss_kd_feat_dis = net(dataset.train_enc_graph_updated, u_list, i_list)

        real_pred_ratings = (torch.softmax(pred_ratings, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)
        real_pred_ratings_review = (torch.softmax(pred_ratings_review, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)

        real_pred_ratings_com = (torch.softmax(pred_ratings_com, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)
        real_pred_ratings_review_com = (torch.softmax(pred_ratings_review, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)

        real_pred_ratings_dis = (torch.softmax(pred_ratings_dis, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)
        real_pred_ratings_review_dis = (torch.softmax(pred_ratings_review, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)

        r_list = r_list.cpu().numpy()

        real_pred_ratings = real_pred_ratings.cpu().numpy()
        real_pred_ratings_review = real_pred_ratings_review.cpu().numpy()

        real_pred_ratings_dis = real_pred_ratings_dis.cpu().numpy()
        real_pred_ratings_review_dis = real_pred_ratings_review_dis.cpu().numpy()

        real_pred_ratings_com = real_pred_ratings_com.cpu().numpy()
        real_pred_ratings_review_com = real_pred_ratings_review_com.cpu().numpy()

        mse = ((real_pred_ratings - r_list) ** 2.).mean()
        mse_review = ((real_pred_ratings_review - r_list) ** 2.).mean()

        mse_dis = ((real_pred_ratings_dis - r_list) ** 2.).mean()
        mse_review_dis = ((real_pred_ratings_review_dis - r_list) ** 2.).mean()

        mse_com = ((real_pred_ratings_com - r_list) ** 2.).mean()
        mse_review_com = ((real_pred_ratings_review_com - r_list) ** 2.).mean()

        mse_final = (((real_pred_ratings + real_pred_ratings_com + real_pred_ratings_dis)/3 - r_list) ** 2.).mean()

    return mse, mse_review, mse_com, mse_review_com, mse_dis, mse_review_dis, mse_final


g_epoch = 1000


def train(params):
    dataset = Data(params.dataset_name,
                   params.dataset_path,
                   params.device,
                   params.emb_size,
                   )
    print("Loading data finished.\n")

    params.num_users = dataset._num_user
    params.num_items = dataset._num_item

    params.rating_vals = dataset.possible_rating_values

    print(
        f'Dataset information:\n \tuser num: {params.num_users}\n\titem num: {params.num_items}\n\ttrain interaction num: {len(dataset.train_rating_values)}\n')

    net = Net(params)
    net = net.to(params.device)

    rating_loss_net = nn.CrossEntropyLoss()
    kd_loss_net = nn.MSELoss()
    kd_loss_net_2 = nn.L1Loss()

    learning_rate = params.train_lr

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-6)
    print("Loading network finished.\n")

    best_final_test_mse = np.inf
    no_better_valid = 0
    best_iter_final = -1
    result = []

    for r in [1, 2, 3, 4, 5]:
        dataset.train_enc_graph[str(r)] = dataset.train_enc_graph[str(r)].int().to(params.device)

    print("Training and evaluation.")
    u_list, i_list, r_list = dataset._train_data(batch_size=params.batch_size)
    for b in u_list:
        print(b.shape)
    for iter_idx in range(1, 400):
        net.train()


        for idx in range(len(r_list)):
            batch_user = u_list[idx]
            batch_item = i_list[idx]
            batch_rating = r_list[idx]
            if iter_idx <= g_epoch:
                pred_ratings, pred_ratings_review, \
                pred_ratings_com, \
                pred_ratings_dis, \
                loss_kd_feat_com, loss_kd_feat_dis = net(dataset.train_enc_graph, batch_user, batch_item)
            else:
                pred_ratings, pred_ratings_review, \
                pred_ratings_com, \
                pred_ratings_dis, \
                loss_kd_feat_com, loss_kd_feat_dis = net(dataset.train_enc_graph_updated, batch_user,
                                                                   batch_item)


            loss = rating_loss_net(pred_ratings, batch_rating).mean()
            loss_review = rating_loss_net(pred_ratings_review, batch_rating).mean()
            loss_dis = rating_loss_net(pred_ratings_dis, batch_rating).mean()
            loss_com = rating_loss_net(pred_ratings_com, batch_rating).mean()
            loss_sum = rating_loss_net((pred_ratings + pred_ratings_com + pred_ratings_dis)/3, batch_rating).mean()


            loss_kd = kd_loss_net(torch.softmax(pred_ratings, dim=1),
                                  torch.softmax(pred_ratings_review, dim=1)) + \
                      kd_loss_net_2(torch.softmax(pred_ratings, dim=1),
                                    torch.softmax(pred_ratings_review, dim=1))

            loss_kd_com = kd_loss_net(torch.softmax(pred_ratings_com, dim=1),
                                      torch.softmax(pred_ratings_review, dim=1)) + \
                          kd_loss_net_2(torch.softmax(pred_ratings_com, dim=1),
                                        torch.softmax(pred_ratings_review, dim=1))

            loss_kd_dis = kd_loss_net(torch.softmax(pred_ratings_dis, dim=1),
                                      torch.softmax(pred_ratings_review, dim=1)) + \
                          kd_loss_net_2(torch.softmax(pred_ratings_dis, dim=1),
                                        torch.softmax(pred_ratings_review, dim=1))

            loss_final = loss + loss_review + loss_dis + loss_com + 0.5 * loss_sum + \
                         loss_kd + loss_kd_com + loss_kd_dis + \
                         loss_kd_feat_com + loss_kd_feat_dis

            optimizer.zero_grad()

            loss_final.backward()

            optimizer.step()


        mse, mse_review, mse_com, mse_review_com, mse_dis, mse_review_dis, mse_final = evaluate(args=params, net=net, dataset=dataset, flag='test', add=False, epoch=iter_idx)

        if mse_final < best_final_test_mse:
            best_final_test_mse = mse_final

            best_iter_final = iter_idx
            no_better_valid = 0
        else:
            no_better_valid += 1
            if no_better_valid > params.train_early_stopping_patience:
                print("Early stopping threshold reached. Stop training.")
                break

        print(f'Epoch [{iter_idx}] \t'
              f'Test_MSE={mse:.4f}, Test_MSE_review: {mse_review:.4f}'
              f'; Test_MSE_com: {mse_com:.4f}'
              f'; Test_MSE_dis: {mse_dis:.4f}'
              f'\t**Test_MSE_final**: {mse_final:.4f}.')
        result.append(mse)
    print(f'Best Iter Idx={best_iter_final}, Best Final MSE={best_final_test_mse:.4f}.')


if __name__ == '__main__':
    config_args = config()

    train(config_args)
