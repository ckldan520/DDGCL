# -*-coding:utf-8-*-

class config():
    def __init__(self):
        self._configs = {}
        self._configs['data'] = 'wikipedia'
        self._configs['dims'] = 128
        self._configs['lr'] = 2e-4
        self._configs['epochs'] = 10
        self._configs['max_num_nodes'] = 10000
        self._configs['batchsize'] = 64
        self._configs['node_pro_length'] = 3
        self._configs['n_heads'] = 8
        self._configs['input_feat_dim'] = 172 # wiki and reddit
        self._configs['neigh_num'] = 20 #neighbor num

    @property
    def data(self):
        return self._configs['data']

    @property
    def dims(self):
        return self._configs['dims']

    @property
    def lr(self):
        return self._configs['lr']

    @property
    def epochs(self):
        return self._configs['epochs']

    @property
    def max_num_nodes(self):
        return self._configs['max_num_nodes']

    @property
    def batchsize(self):
        return self._configs['batchsize']

    @property
    def node_pro_length(self):
        return self._configs['node_pro_length']

    @property
    def n_heads(self):
        return self._configs['n_heads']

    @property
    def input_feat_dim(self):
        return self._configs['input_feat_dim']

    @property
    def neigh_num(self):
        return self._configs['neigh_num']

    def update_config(self, key, value):
        if key in self._configs.keys():
            self._configs[key] = value
        else:
            raise RuntimeError('Update_Config_Error')

cfg = config()