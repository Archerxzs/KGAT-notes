'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
from utility.load_data import Data
from time import time
import scipy.sparse as sp
import random as rd
import collections

'''
train_data/test_data------>[[userId,itemId],[userId,itemId]...]
train_user_dict/test_user_dict------>{userId:[itemId,itemId,...],userId:[itemId,itemId,...]}
kg_data:[(head,relation,tail),(head,relation,tail),...]
kg_dict:{head:(tail,relation),head:(tail,relation)...}
relation_dict:{relation:(head,tail),relation:(head,tail)...}
'''
class KGAT_loader(Data):
    def __init__(self, args, path):
        super().__init__(args, path)

        # generate the sparse adjacency matrices for user-item interaction & relational kg data.
        #adj_list生成user-item、item-item的交互邻接矩阵 一共20个矩阵，前两个是user-item的，互为转置矩阵，后面18个是item-item的，互为转置矩阵
        #adj_r_list是关系的序号，前两个是0和10，代表了user-item的关系id，后面18个item-item的关系id，是1，11，2，12，。。。 不重复 顺序固定
        self.adj_list, self.adj_r_list = self._get_relational_adj_list()

        # generate the sparse laplacian matrices.
        #对每个矩阵进行归一化，生成拉普拉斯矩阵
        self.lap_list = self._get_relational_lap_list()

        # generate the triples dictionary, key is 'head', value is '(tail, relation)'.
        self.all_kg_dict = self._get_all_kg_dict()

        self.all_h_list, self.all_r_list, self.all_t_list, self.all_v_list = self._get_all_kg_data()


    def _get_relational_adj_list(self):
        t1 = time()
        adj_mat_list = []
        adj_r_list = []

        def _np_mat2sp_adj(np_mat, row_pre, col_pre):
            #23566+106389=129955
            n_all = self.n_users + self.n_entities
            # single-direction
            #1.[0,1,2,3,....,23565]（不是有序的，而且存在重复）1289003个
            #2.head∈[0,106388] +23566=>[23566,129954] 464597个
            a_rows = np_mat[:, 0] + row_pre
            #1.[23566,....，71688]（从23566开始，因为加上了23566） 1289003个
            #2.tail∈[0,106387] +23566=>[23566,129953] 464597个
            a_cols = np_mat[:, 1] + col_pre
            #1.[1.0,1.0,....] 1289003个1.0
            #2.[1.0,1.0,....] 464597个1.0
            a_vals = [1.] * len(a_rows)

            #1.[23566,....]（从23566开始，因为加上了23566） 1289003个
            #2.tail∈[0,106387] +23566=>[23566,129953] 464597个
            b_rows = a_cols
            #1.[0,1,2,3,....,23565]（不是有序的，而且存在重复）1289003个
            #2.head∈[0,106388] +23566=>[23566,129954] 464597个
            b_cols = a_rows
            #1.[1.0,1.0,....] 1289003个1.0
            #2.[1.0,1.0,....] 464597个1.0
            b_vals = [1.] * len(b_rows)

            '''
            row  = np.array([0, 3, 1, 0])
            col  = np.array([0, 3, 1, 2])
            data = np.array([4, 5, 7, 9])
            coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
            ===>array([[4, 0, 9, 0],
                        [0, 7, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 5]])
            '''
            #1.a_adj和b_adj都是129955*129955矩阵，有值的都为1
            #1.a_rows∈[0,23565] a_cols∈[23566,71688]
            #2.a_adj和b_adj都是129955*129955矩阵，有值的都为1
            #2.a_rows∈[23566,129954] a_cols∈[23566,129953]
            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
            #1.b_rows∈[23566,71688] b_cols∈[0,23565]
            #2.b_rows∈[23566,129953] b_cols∈[23566,129954]
            b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_all, n_all))

            return a_adj, b_adj
        
        #user-item的邻接矩阵 R和R_inv互逆
        R, R_inv = _np_mat2sp_adj(self.train_data, row_pre=0, col_pre=self.n_users)
        #adj_mat_list[0]=[[129955*129955矩阵(但是只有行为0~23565,列为23565~71689下标的空间有值为1）]]
        adj_mat_list.append(R)
        #adj_r_list[0]=0
        adj_r_list.append(0)

        #adj_mat_list[1]=[[129955*129955矩阵(但是只有行为23565~71689,列为0~23565下标的空间有值为1）]]
        adj_mat_list.append(R_inv)
        #adj_r_list=[0,10]
        adj_r_list.append(self.n_relations + 1)
        print('\tconvert ratings into adj mat done.')

        for r_id in self.relation_dict.keys():#有9个
            #np.array(self.relation_dict[r_id])==>[（head,tail）,(head,tail),...]
            #K:[[129955*129955矩阵(但是rows∈[23566,129954] cols∈[23566,129953]才有值）]]
            #item-item邻接矩阵
            K, K_inv = _np_mat2sp_adj(np.array(self.relation_dict[r_id]), row_pre=self.n_users, col_pre=self.n_users)
            #adj_mat_list[??]=[129955*129955矩阵]
            adj_mat_list.append(K)
            #adj_r_list[??] = [1,2,3,4,5,6,7,8,9]
            adj_r_list.append(r_id + 1)

            # adj_mat_list[??]=[129955*129955矩阵]
            adj_mat_list.append(K_inv)
            # adj_r_list[??] = [11,12,13,14,15,16,17,18,19]
            adj_r_list.append(r_id + 2 + self.n_relations)
        #20 有向图，所以关系也加了一倍
        print('\tconvert %d relational triples into adj mat done. @%.4fs' %(len(adj_mat_list), time()-t1))

        self.n_relations = len(adj_r_list)
        # print('\tadj relation list is', adj_r_list)

        return adj_mat_list, adj_r_list

    def _get_relational_lap_list(self):
        def _bi_norm_lap(adj):
            #对矩阵的每一行做加法，最后变成一维向量
            rowsum = np.array(adj.sum(1))#(129955, 1)
            #(1,129955)
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            #吧没有值的全部赋予0
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            #d_inv_sqrt的值铺在对角线上
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            #每个矩阵做归一化处理
            bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        def _si_norm_lap(adj):
            #统计每行有多少个1,以adj_list[0]为例，虽然矩阵是[129955, 129955]
            #但是只有行坐标为0~23565和纵坐标为23565~71689的空间有值1,
            #所以adj.sum(1)就是计算在矩阵中每行有多少1，上例中第0行为113个，第1行97个，当在第23566行是就是0了
            '''
            [[113.]
            [ 97.]
            [ 94.]
            ...
            [  0.]
            [  0.]
            [  0.]]
            '''
            rowsum = np.array(adj.sum(1))

            #np.power(rowsum, -1)将每个数字都成为自己的倒数，然后flatten(）摊平
            #1/113=0.00884956 1/97=0.01030928...
            #[[0.00884956 0.01030928 0.0106383  ...        inf        inf        inf]]
            d_inv = np.power(rowsum, -1).flatten()
            #将所有inf转为0
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.args.adj_type == 'bi':
            lap_list = [_bi_norm_lap(adj) for adj in self.adj_list]
            print('\tgenerate bi-normalized adjacency matrix.')
        else:
            lap_list = [_si_norm_lap(adj) for adj in self.adj_list]
            print('\tgenerate si-normalized adjacency matrix.')
        return lap_list

    def _get_all_kg_dict(self):
        all_kg_dict = collections.defaultdict(list)
        #对拉普拉斯矩阵进行处理
        for l_id, lap in enumerate(self.lap_list):

            rows = lap.row
            cols = lap.col
            # print("l_id=" + str(l_id))
            # print("lap=" + str(lap))
            # print("rows=" + str(rows))
            # print("cols=" + str(cols))

            for i_id in range(len(rows)):
                head = rows[i_id]
                tail = cols[i_id]
                relation = self.adj_r_list[l_id]

                all_kg_dict[head].append((tail, relation))
            # print("len(all_kg_dict)=" + str(len(all_kg_dict)))
            # print("all_kg_dict.keys()=" + str(all_kg_dict.keys()))
            # print("====")
        return all_kg_dict

    def _get_all_kg_data(self):
        #截取list
        def _reorder_list(org_list, order):
            new_list = np.array(org_list)
            new_list = new_list[order]
            return new_list

        all_h_list, all_t_list, all_r_list = [], [], []
        all_v_list = []

        for l_id, lap in enumerate(self.lap_list):#0-19
            # print("lap.row="+str(lap.row))
            # print("type(lap.row)="+str(type(lap.row)))
            # print("list(lap.row)="+str(list(lap.row)))
            all_h_list += list(lap.row)#[[row],[row],[row],...] 行下标 长度3507140
            all_t_list += list(lap.col)#[[col],[col],[col],...] 列下标 3507140
            all_v_list += list(lap.data)#[[data],[data],[data],...] 具体的数据 3507140
            all_r_list += [self.adj_r_list[l_id]] * len(lap.row) #[[r_id,r_id,r_id,..],[r_id,r_id,r_id,...]] 关系的id 3507140

        assert len(all_h_list) == sum([len(lap.data) for lap in self.lap_list])

        # resort the all_h/t/r/v_list,
        # ... since tensorflow.sparse.softmax requires indices sorted in the canonical lexicographic order
        print('\treordering indices...')
        org_h_dict = dict()

        # print("len(all_h_list)="+str(len(all_h_list)))
        # print("len(all_t_list)="+str(len(all_t_list)))
        # print("len(all_v_list)="+str(len(all_v_list)))
        # print("len(all_r_list)="+str(len(all_r_list)))
        # print("str(all_h_list[0:9])="+str(all_h_list[0:9]))
        # print("str(all_t_list[0:9])="+str(all_t_list[0:9]))
        # print("str(all_v_list[0:9])="+str(all_v_list[0:9]))
        # print("str(all_r_list[0:9])="+str(all_r_list[0:9]))
        for idx, h in enumerate(all_h_list):
            # print("idx="+str(idx))
            # print("h="+str(h))
            if h not in org_h_dict.keys():
                org_h_dict[h] = [[],[],[]]

            org_h_dict[h][0].append(all_t_list[idx])
            org_h_dict[h][1].append(all_r_list[idx])
            org_h_dict[h][2].append(all_v_list[idx])
        print('\treorganize all kg data done.')

        sorted_h_dict = dict()
        for h in org_h_dict.keys():
            org_t_list, org_r_list, org_v_list = org_h_dict[h]
            sort_t_list = np.array(org_t_list)
            # 根据头实体排序 获取到如果按照顺序排列的的下标   比如sort_t_list=[4,3,5] 那么np.argsort(sort_t_list)=[1,0,2]
            sort_order = np.argsort(sort_t_list)

            sort_t_list = _reorder_list(org_t_list, sort_order)
            sort_r_list = _reorder_list(org_r_list, sort_order)
            sort_v_list = _reorder_list(org_v_list, sort_order)

            sorted_h_dict[h] = [sort_t_list, sort_r_list, sort_v_list]#{头：[尾，关系，数据]，头：[...]。。。}
        print('\tsort meta-data done.')

        #OrderedDict根据放入元素的顺序来排序
        od = collections.OrderedDict(sorted(sorted_h_dict.items()))
        new_h_list, new_t_list, new_r_list, new_v_list = [], [], [], []
        print(str(len(od)))
        # isFirst=True
        for h, vals in od.items():
            # if(isFirst):
            #     print("h="+str(h))
            #     print("vals="+str(vals))
            #     isFirst=False
            new_h_list += [h] * len(vals[0])
            new_t_list += list(vals[0])
            new_r_list += list(vals[1])
            new_v_list += list(vals[2])


        assert sum(new_h_list) == sum(all_h_list)
        assert sum(new_t_list) == sum(all_t_list)
        assert sum(new_r_list) == sum(all_r_list)
        # try:
        #     assert sum(new_v_list) == sum(all_v_list)
        # except Exception:
        #     print(sum(new_v_list), '\n')
        #     print(sum(all_v_list), '\n')
        print('\tsort all data done.')

        # print("len(new_h_list)="+str(len(new_h_list)))
        # print("len(new_r_list)="+str(len(new_r_list)))
        # print("len(new_t_list)="+str(len(new_t_list)))
        # print("len(new_v_list)="+str(len(new_v_list)))

        return new_h_list, new_r_list, new_t_list, new_v_list

    def _generate_train_A_batch(self):
        exist_heads = self.all_kg_dict.keys()

        if self.batch_size_kg <= len(exist_heads):
            heads = rd.sample(exist_heads, self.batch_size_kg)
        else:
            heads = [rd.choice(exist_heads) for _ in range(self.batch_size_kg)]

        def sample_pos_triples_for_h(h, num):
            pos_triples = self.all_kg_dict[h]
            n_pos_triples = len(pos_triples)

            pos_rs, pos_ts = [], []
            while True:
                if len(pos_rs) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_triples, size=1)[0]

                t = pos_triples[pos_id][0]
                r = pos_triples[pos_id][1]

                if r not in pos_rs and t not in pos_ts:
                    pos_rs.append(r)
                    pos_ts.append(t)
            return pos_rs, pos_ts

        def sample_neg_triples_for_h(h, r, num):
            neg_ts = []
            while True:
                if len(neg_ts) == num: break

                t = np.random.randint(low=0, high=self.n_users + self.n_entities, size=1)[0]
                if (t, r) not in self.all_kg_dict[h] and t not in neg_ts:
                    neg_ts.append(t)
            return neg_ts

        pos_r_batch, pos_t_batch, neg_t_batch = [], [], []

        for h in heads:
            pos_rs, pos_ts = sample_pos_triples_for_h(h, 1)
            pos_r_batch += pos_rs
            pos_t_batch += pos_ts

            neg_ts = sample_neg_triples_for_h(h, pos_rs[0], 1)
            neg_t_batch += neg_ts

        return heads, pos_r_batch, pos_t_batch, neg_t_batch

    def generate_train_batch(self):
        users, pos_items, neg_items = self._generate_train_cf_batch()

        batch_data = {}
        batch_data['users'] = users
        batch_data['pos_items'] = pos_items
        batch_data['neg_items'] = neg_items

        return batch_data

    def generate_train_feed_dict(self, model, batch_data):
        feed_dict = {
            model.users: batch_data['users'],
            model.pos_items: batch_data['pos_items'],
            model.neg_items: batch_data['neg_items'],

            model.mess_dropout: eval(self.args.mess_dropout),
            model.node_dropout: eval(self.args.node_dropout),
        }

        return feed_dict

    def generate_train_A_batch(self):
        heads, relations, pos_tails, neg_tails = self._generate_train_A_batch()

        batch_data = {}

        batch_data['heads'] = heads
        batch_data['relations'] = relations
        batch_data['pos_tails'] = pos_tails
        batch_data['neg_tails'] = neg_tails
        return batch_data

    def generate_train_A_feed_dict(self, model, batch_data):
        feed_dict = {
            model.h: batch_data['heads'],
            model.r: batch_data['relations'],
            model.pos_t: batch_data['pos_tails'],
            model.neg_t: batch_data['neg_tails'],

        }

        return feed_dict


    def generate_test_feed_dict(self, model, user_batch, item_batch, drop_flag=True):

        feed_dict ={
            model.users: user_batch,
            model.pos_items: item_batch,
            model.mess_dropout: [0.] * len(eval(self.args.layer_size)),
            model.node_dropout: [0.] * len(eval(self.args.layer_size)),

        }

        return feed_dict

