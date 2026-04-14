import numpy as np
from sklearn.neighbors import KDTree
import os
import gc
import json


class LRUKnn:
    """Per-action episodic memory buffer using KD-Tree for nearest-neighbor lookup."""

    def __init__(self, args, saving_agent_logs_path, action_size, capacity, z_dim,
                 env_name, save_qa_path, save_qa_index, save_type, test=False):
        self.env_name = env_name
        self.capacity = capacity
        self.states = np.empty([capacity, z_dim], dtype=np.float32)
        self.states_text = []
        self.q_values_decay = np.zeros(capacity)
        self.cur_env_num = []
        self.lru = np.zeros(capacity)
        self.curr_capacity = 0
        self.tm = 0.0
        self.tree = None
        self.bufpath = saving_agent_logs_path + '/buffer/%s/%s' % (save_type, self.env_name)
        self.build_tree_times = 0
        self.build_tree = False
        self.save_qa_index = save_qa_index
        self.save_type = save_type
        self.save_qa_path = save_qa_path
        self.args = args
        self.alpha = 0.7
        self.action_size = action_size
        self.test = test

        if self.args.save_ec_buffer and not self.test:
            os.makedirs(self.save_qa_path, exist_ok=True)
            self.json_file_path = os.path.join(
                self.save_qa_path,
                'ec_buffer_{}_{}.json'.format(self.save_type, self.save_qa_index),
            )
            if os.path.exists(self.json_file_path):
                os.remove(self.json_file_path)

    def load(self, action):
        try:
            assert os.path.exists(self.bufpath)
            lru = np.load(os.path.join(self.bufpath, 'lru_%d.npy' % action))
            cap = lru.shape[0]
            self.curr_capacity = cap
            self.tm = np.max(lru) + 0.01

            self.states[:cap] = np.load(os.path.join(self.bufpath, 'states_%d.npy' % action))
            self.q_values_decay[:cap] = np.load(os.path.join(self.bufpath, 'q_values_decay_%d.npy' % action))
            self.lru[:cap] = lru
            loaded_texts = np.load(os.path.join(self.bufpath, 'states_text_%d.npy' % action))
            self.states_text = list(loaded_texts)
            self.tree = KDTree(self.states[:self.curr_capacity])
            self.build_tree = True
            print("load %d-th buffer success, cap=%d" % (action, cap))
        except Exception:
            print("load %d-th buffer failed" % action)

    def save(self, action):
        os.makedirs(self.bufpath, exist_ok=True)
        np.save(os.path.join(self.bufpath, 'states_%d' % action), self.states[:self.curr_capacity])
        np.save(os.path.join(self.bufpath, 'q_values_decay_%d' % action), self.q_values_decay[:self.curr_capacity])
        np.save(os.path.join(self.bufpath, 'lru_%d' % action), self.lru[:self.curr_capacity])
        np.save(os.path.join(self.bufpath, 'states_text_%d' % action), self.states_text[:self.curr_capacity])

    def peek(self, key, value_decay, modify, flag=None, get_act=False, save_flag=False):
        if self.curr_capacity == 0 or not self.build_tree:
            return None, None, None, None

        dist, ind = self.tree.query([key], k=1)
        ind = ind[0][0]

        if np.allclose(self.states[ind], key, atol=1e-08):
            self.lru[ind] = self.tm
            self.tm += 0.01
            if modify:
                if value_decay > self.q_values_decay[ind]:
                    self.q_values_decay[ind] = value_decay
                if save_flag and not self.test:
                    self.save_to_json()
            return self.q_values_decay[ind], self.states_text[ind], dist[0][0], ind

        return None, None, None, None

    def knn_value(self, key, knn):
        knn = min(self.curr_capacity, knn)
        if self.curr_capacity == 0 or not self.build_tree:
            return 0.0

        dist, ind = self.tree.query([key], k=knn)

        value_decay = 0.0
        for index in ind[0]:
            value_decay += self.q_values_decay[index]
            self.lru[index] = self.tm
            self.tm += 0.01

        return value_decay / knn

    def add(self, key, value_decay, key_text, env_num):
        if self.curr_capacity >= self.capacity:
            old_index = np.argmin(self.lru)
            self.states[old_index] = key
            self.states_text[old_index] = key_text
            self.cur_env_num[old_index] = env_num
            self.q_values_decay[old_index] = value_decay
            self.lru[old_index] = self.tm
        else:
            self.states[self.curr_capacity] = key
            self.states_text.append(key_text)
            self.cur_env_num.append(env_num)
            self.q_values_decay[self.curr_capacity] = value_decay
            self.lru[self.curr_capacity] = self.tm
            self.curr_capacity += 1

        self.tm += 0.01

    def save_to_json(self):
        data_to_save = []
        for i in range(self.curr_capacity):
            data_to_save.append({
                'env_num': self.cur_env_num[i],
                'state_text': self.states_text[i],
                'q_value_decay': float(self.q_values_decay[i]),
            })
        with open(self.json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)

    def update_kdtree(self):
        if self.build_tree:
            del self.tree
        self.tree = KDTree(self.states[:self.curr_capacity])
        self.build_tree = True
        self.build_tree_times += 1
        if self.build_tree_times == 50:
            self.build_tree_times = 0
            gc.collect()
