import scipy.sparse as sp
import torch as th
import dgl
import sys

sys.path.append("..")

from BiKD2.load_data import *
from BiKD2.util import *


class MovieLens(object):

    def __init__(self, dataset_name, dataset_path, device, review_fea_size, symm=True, mix_cpu_gpu=True, use_user_item_doc=False):
        self._device = device
        self._review_fea_size = review_fea_size
        self._symm = symm

        review_feat_path = f'../data/{dataset_name}/bert-base-uncased_sentence_vectors_dim_{review_fea_size}.pkl'
        self.train_review_feat = torch.load(review_feat_path)

        sent_train_data, sent_valid_data, sent_test_data, _, _, dataset_info = load_sentiment_data(dataset_path)

        if use_user_item_doc:
            doc_data = load_data_for_review_based_rating_prediction(dataset_path)
            self.word2id = doc_data['word2id']
            self.embedding = doc_data['embeddings']
            self.user_doc = torch.from_numpy(process_doc(doc_data['user_doc'], self.word2id))
            self.movie_doc = torch.from_numpy(process_doc(doc_data['item_doc'], self.word2id))
            if not mix_cpu_gpu:
                self.user_doc.to(device)
                self.movie_doc.to(device)
        else:
            self.word2id = None
            self.embedding = None
            self.user_doc = None
            self.movie_doc = None

        def process_sent_data(info):
            user_id = info['user_id'].to_list()
            item_id = info['item_id'].to_list()
            rating = info['rating'].to_list()

            return user_id, item_id, rating

        self.train_datas = process_sent_data(sent_train_data)
        self.valid_datas = process_sent_data(sent_valid_data)
        self.test_datas = process_sent_data(sent_test_data)
        self.possible_rating_values = np.unique(self.train_datas[2])

        self._num_user = dataset_info['user_size']
        self._num_movie = dataset_info['item_size']

        self.user_feature = None
        self.movie_feature = None

        self.user_feature_shape = (self._num_user, self._num_user)
        self.movie_feature_shape = (self._num_movie, self._num_movie)

        info_line = "Feature dim: "
        info_line += "\nuser: {}".format(self.user_feature_shape)
        info_line += "\nmovie: {}".format(self.movie_feature_shape)
        print(info_line)

        def _generate_pair_value(data: tuple):
            user_id, item_id, rating = data[0], data[1], data[2]
            rating_pairs = (np.array(user_id, dtype=np.int64),
                            np.array(item_id, dtype=np.int64))
            rating_values = np.array(rating, dtype=np.float32)
            return rating_pairs, rating_values

        train_rating_pairs, train_rating_values = _generate_pair_value(self.train_datas)
        valid_rating_pairs, valid_rating_values = _generate_pair_value(self.valid_datas)
        test_rating_pairs, test_rating_values = _generate_pair_value(self.test_datas)

        self.train_enc_graph = self._generate_enc_graph(train_rating_pairs, train_rating_values, add_support=True)
        self.train_dec_graph = self._generate_dec_graph(train_rating_pairs, review_feat=self.train_review_feat)
        self.train_labels = th.LongTensor(train_rating_values - 1).to(device)
        self.train_truths = th.FloatTensor(train_rating_values).to(device)

        self.valid_enc_graph = self.train_enc_graph
        self.valid_dec_graph = self._generate_dec_graph(valid_rating_pairs, review_feat=self.train_review_feat)
        self.valid_labels = th.LongTensor(valid_rating_values - 1).to(device)
        self.valid_truths = th.FloatTensor(valid_rating_values).to(device)

        self.test_enc_graph = self.train_enc_graph
        self.test_dec_graph = self._generate_dec_graph(test_rating_pairs, review_feat=self.train_review_feat)
        self.test_labels = th.LongTensor(valid_rating_values - 1).to(device)
        self.test_truths = th.FloatTensor(test_rating_values).to(device)

        print("Train enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.train_enc_graph.number_of_nodes('user'), self.train_enc_graph.number_of_nodes('movie'),
            np.sum([self.train_enc_graph.number_of_edges(str(r)) for r in self.possible_rating_values])))
        print("Train dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.train_dec_graph.number_of_nodes('user'), self.train_dec_graph.number_of_nodes('movie'),
            self.train_dec_graph.number_of_edges()))

        print("Valid enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.valid_enc_graph.number_of_nodes('user'), self.valid_enc_graph.number_of_nodes('movie'),
            np.sum([self.valid_enc_graph.number_of_edges(str(r)) for r in self.possible_rating_values])))
        print("Valid dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.valid_dec_graph.number_of_nodes('user'), self.valid_dec_graph.number_of_nodes('movie'),
            self.valid_dec_graph.number_of_edges()))

        print("Test enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.test_enc_graph.number_of_nodes('user'), self.test_enc_graph.number_of_nodes('movie'),
            np.sum([self.test_enc_graph.number_of_edges(str(r)) for r in self.possible_rating_values])))
        print("Test dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.test_dec_graph.number_of_nodes('user'), self.test_dec_graph.number_of_nodes('movie'),
            self.test_dec_graph.number_of_edges()))


    def _process_user_item_review_feat_groupby_rating(self):

        user_id = self.train_datas[0]
        item_id = self.train_datas[1]
        rating = self.train_datas[2]
        ui = list(zip(user_id, item_id))
        ui2r = dict(zip(ui, rating))

        # r -> id -> feature_list
        user_list = [[[] for _ in range(self._num_user)] for r in self.possible_rating_values]
        movie_list = [[[] for _ in range(self._num_movie)] for r in self.possible_rating_values]

        for u, m in zip(user_id, item_id):
            r = ui2r[(u, m)]
            user_list[r-1][u].append(self.train_review_feat[(u, m)])
            movie_list[r-1][m].append(self.train_review_feat[(u, m)])

        def stack(vector_list):
            if len(vector_list) > 0:
                return torch.stack(vector_list).mean(0)
            else:
                return torch.randn(self._review_fea_size)

        user_list = [[stack(x) for x in r] for r in user_list]
        movie_list = [[stack(x) for x in r] for r in movie_list]

        user_list = [torch.stack(x).to(torch.float32) for x in user_list]
        movie_list = [torch.stack(x).to(torch.float32) for x in movie_list]
        user_review_feat_groupby_rating = {k+1: v for k, v in enumerate(user_list)}
        movie_review_feat_groupby_rating = {k+1: v for k, v in enumerate(movie_list)}
        return user_review_feat_groupby_rating, movie_review_feat_groupby_rating

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):
        user_movie_r = np.zeros((self._num_user, self._num_movie), dtype=np.float32)
        user_movie_r[rating_pairs] = rating_values
        record_size = rating_pairs[0].shape[0]
        review_feat_list = [self.train_review_feat[(rating_pairs[0][x], rating_pairs[1][x])] for x in range(record_size)]
        review_feat_list = torch.stack(review_feat_list).to(torch.float32)

        data_dict = dict()
        num_nodes_dict = {'user': self._num_user, 'movie': self._num_movie}
        rating_row, rating_col = rating_pairs
        review_data_dict = dict()
        for rating in self.possible_rating_values:
            ridx = np.where(rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            data_dict.update({
                ('user', str(rating), 'movie'): (rrow, rcol),
                ('movie', 'rev-%s' % str(rating), 'user'): (rcol, rrow)
            })
            review_data_dict[str(rating)] = review_feat_list[ridx]

        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        # 在异质图的边上保存review features
        for rating in self.possible_rating_values:
            graph[str(rating)].edata['review_feat'] = review_data_dict[str(rating)]
            graph['rev-%s' % str(rating)].edata['review_feat'] = review_data_dict[str(rating)]

        assert len(rating_pairs[0]) == sum(
            [graph.number_of_edges(et) for et in graph.etypes]) // 2

        if self.user_doc is not None:
            graph.nodes['user'].data['doc'] = self.user_doc
            graph.nodes['movie'].data['doc'] = self.movie_doc

        def _calc_norm(x):
            x = x.numpy().astype('float32')
            x[x == 0.] = np.inf
            x = th.FloatTensor(1. / np.power(x, 0.5))
            return x.unsqueeze(1)

        user_ci, movie_ci = [], []
        for r in self.possible_rating_values:
            r = str(r)
            user_ci.append(graph['rev-%s' % r].in_degrees())
            movie_ci.append(graph[r].in_degrees())

        user_ci = _calc_norm(sum(user_ci))
        movie_ci = _calc_norm(sum(movie_ci))

        graph.nodes['user'].data.update({'ci': user_ci})
        graph.nodes['movie'].data.update({'ci': movie_ci})

        return graph

    def _generate_dec_graph(self, rating_pairs, review_feat=None):
        ones = np.ones_like(rating_pairs[0])
        user_movie_ratings_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self._num_user, self._num_movie), dtype=np.float32)
        g = dgl.bipartite_from_scipy(user_movie_ratings_coo, utype='_U',
                                     etype='_E', vtype='_V')
        g = dgl.heterograph({('user', 'rate', 'movie'): g.edges()},
                            num_nodes_dict={'user': self._num_user,
                                            'movie': self._num_movie})

        if review_feat is not None:
            ui = list(zip(rating_pairs[0].tolist(), rating_pairs[1].tolist()))
            feat = [review_feat[x] for x in ui]

            feat = torch.stack(feat, dim=0).float()
            g.edata['review_feat'] = feat

        return g


def process_doc(doc, word2id, doc_length=256):
    for k, v in doc.items():
        v = [x['review_text'] for x in v]
        v = ' '.join(v)
        v = v.split()[: doc_length]
        v = ' '.join(v)
        v = parse_word_to_idx(word2id, v)
        v = pad_sentence(v, doc_length)
        doc[k] = v
    result = [doc[i] for i in range(len(doc))]
    result = np.stack(result)
    return result


def parse_word_to_idx(word2id, sentence):
    idx = np.array([word2id[x] for x in sentence.split()], dtype=np.int64)
    return idx


def pad_sentence(sentence, length):
    if sentence.shape[0] < length:
        pad_length = length - sentence.shape[0]
        sentence = np.pad(sentence, (0, pad_length), 'constant', constant_values=0)
    return sentence
