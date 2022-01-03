import numpy as np

class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list):
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]


def count_neighbors(neighbor_find, src_idx, dst_idx, ts_idx, neigh_num):
    b = src_idx.shape[0]
    src_ngh1_node_batch, src_ngh1_eidx_batch, src_ngh1_t_batch = neighbor_find.get_temporal_neighbor(src_idx, ts_idx, num_neighbors=neigh_num)
    dst_ngh1_node_batch, dst_ngh1_eidx_batch, dst_ngh1_t_batch = neighbor_find.get_temporal_neighbor(dst_idx, ts_idx,
                                                                                                     num_neighbors=neigh_num)

    src_ngh2_node_batch, src_ngh2_eidx_batch, src_ngh2_t_batch = neighbor_find.get_temporal_neighbor(src_ngh1_node_batch.flatten(), src_ngh1_t_batch.flatten(),
                                                                                                     num_neighbors=neigh_num)
    dst_ngh2_node_batch, dst_ngh2_eidx_batch, dst_ngh2_t_batch = neighbor_find.get_temporal_neighbor(dst_ngh1_node_batch.flatten(),
                                                                                                     dst_ngh1_t_batch.flatten(),
                                                                                                     num_neighbors=neigh_num)

    # Modify time to find a subgraph before the current time
    new_ts_idx = ts_idx * np.random.uniform(0.6, 0.9, len(ts_idx))
    cl_src_ngh1_node_batch, cl_src_ngh1_eidx_batch, cl_src_ngh1_t_batch = neighbor_find.get_temporal_neighbor(src_idx, new_ts_idx,
                                                                                                     num_neighbors=neigh_num)
    cl_src_ngh2_node_batch, cl_src_ngh2_eidx_batch, cl_src_ngh2_t_batch = neighbor_find.get_temporal_neighbor(
        cl_src_ngh1_node_batch.flatten(), cl_src_ngh1_t_batch.flatten(),
        num_neighbors=neigh_num)


    ngh1_node_batch = np.concatenate( (src_ngh1_node_batch.reshape(b, 1, neigh_num), dst_ngh1_node_batch.reshape(b, 1, neigh_num), cl_src_ngh1_node_batch.reshape(b, 1, neigh_num)), axis=1)
    ngh1_t_batch = np.concatenate( (src_ngh1_t_batch.reshape(b, 1, neigh_num), dst_ngh1_t_batch.reshape(b, 1, neigh_num), cl_src_ngh1_t_batch.reshape(b, 1, neigh_num)), axis=1)
    ngh1_edge_batch = np.concatenate((src_ngh1_eidx_batch.reshape(b, 1, neigh_num), dst_ngh1_eidx_batch.reshape(b, 1, neigh_num), cl_src_ngh1_eidx_batch.reshape(b, 1, neigh_num)), axis=1)

    ngh2_node_batch = np.concatenate( (src_ngh2_node_batch.reshape(b, 1, neigh_num*neigh_num), dst_ngh2_node_batch.reshape(b, 1, neigh_num*neigh_num), cl_src_ngh2_node_batch.reshape(b, 1, neigh_num*neigh_num)), axis=1)
    ngh2_t_batch = np.concatenate( (src_ngh2_t_batch.reshape(b, 1, neigh_num*neigh_num), dst_ngh2_t_batch.reshape(b, 1, neigh_num*neigh_num), cl_src_ngh2_t_batch.reshape(b, 1, neigh_num*neigh_num)), axis=1)
    ngh2_edge_batch = np.concatenate((src_ngh2_eidx_batch.reshape(b, 1, neigh_num*neigh_num), dst_ngh2_eidx_batch.reshape(b, 1, neigh_num*neigh_num), cl_src_ngh2_eidx_batch.reshape(b, 1, neigh_num*neigh_num)), axis=1)

    return ngh1_node_batch, ngh1_edge_batch, ngh1_t_batch, ngh2_node_batch, ngh2_edge_batch, ngh2_t_batch, new_ts_idx


class NeighborFinder:
    def __init__(self, adj_list, uniform=False):
        """
        Params
        ------
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """

        node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l

        self.off_set_l = off_set_l

        self.uniform = uniform

    def init_off_set(self, adj_list):
        """
        Params
        ------
        adj_list: List[List[int]]

        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0]
        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: x[1])
            n_idx_l.extend([x[0] for x in curr])
            e_idx_l.extend([x[1] for x in curr])
            n_ts_l.extend([x[2] for x in curr])

            off_set_l.append(len(n_idx_l))
        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        assert (len(n_idx_l) == len(n_ts_l))
        assert (off_set_l[-1] == len(n_ts_l))

        return n_idx_l, n_ts_l, e_idx_l, off_set_l

    def find_before(self, src_idx, cut_time):
        """

        Params
        ------
        src_idx: int
        cut_time: float
        """
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l

        neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]

        if len(neighbors_idx) == 0 or len(neighbors_ts) == 0:
            return neighbors_idx, neighbors_ts, neighbors_e_idx

        left = 0
        right = len(neighbors_idx) - 1

        while left + 1 < right:
            mid = (left + right) // 2
            curr_t = neighbors_ts[mid]
            if curr_t < cut_time:
                left = mid
            else:
                right = mid

        if neighbors_ts[right] < cut_time:
            return neighbors_idx[:right], neighbors_e_idx[:right], neighbors_ts[:right]
        else:
            return neighbors_idx[:left], neighbors_e_idx[:left], neighbors_ts[:left]

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert (len(src_idx_l) == len(cut_time_l))

        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.float32)
        out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            ngh_idx, ngh_eidx, ngh_ts = self.find_before(src_idx, cut_time)

            if len(ngh_idx) > 0:
                if self.uniform:
                    sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)

                    out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]

                    # resort based on time
                    pos = out_ngh_t_batch[i, :].argsort()
                    out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                    out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
                    out_ngh_eidx_batch[i, :] = out_ngh_eidx_batch[i, :][pos]
                else:
                    ngh_ts = ngh_ts[:num_neighbors]
                    ngh_idx = ngh_idx[:num_neighbors]
                    ngh_eidx = ngh_eidx[:num_neighbors]

                    assert (len(ngh_idx) <= num_neighbors)
                    assert (len(ngh_ts) <= num_neighbors)
                    assert (len(ngh_eidx) <= num_neighbors)

                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    out_ngh_eidx_batch[i, num_neighbors - len(ngh_eidx):] = ngh_eidx

        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch

    def find_k_hop(self, k, src_idx_l, cut_time_l, num_neighbors=20):
        """Sampling the k-hop sub graph
        """
        x, y, z = self.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors)
        node_records = [x]
        eidx_records = [y]
        t_records = [z]
        for _ in range(k - 1):
            ngn_node_est, ngh_t_est = node_records[-1], t_records[-1]  # [N, *([num_neighbors] * (k - 1))]
            orig_shape = ngn_node_est.shape
            ngn_node_est = ngn_node_est.flatten()
            ngn_t_est = ngh_t_est.flatten()
            out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = self.get_temporal_neighbor(ngn_node_est,
                                                                                                 ngn_t_est,
                                                                                                 num_neighbors)
            out_ngh_node_batch = out_ngh_node_batch.reshape(*orig_shape, num_neighbors)  # [N, *([num_neighbors] * k)]
            out_ngh_eidx_batch = out_ngh_eidx_batch.reshape(*orig_shape, num_neighbors)
            out_ngh_t_batch = out_ngh_t_batch.reshape(*orig_shape, num_neighbors)

            node_records.append(out_ngh_node_batch)
            eidx_records.append(out_ngh_eidx_batch)
            t_records.append(out_ngh_t_batch)
        return node_records, eidx_records, t_records









