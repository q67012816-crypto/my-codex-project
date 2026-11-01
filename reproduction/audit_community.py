"""Repeat the read-only community/candidate checks without changing training inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
import torch
from torch_geometric.datasets import Planetoid

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-records', action='store_true',
                        help='Compare counts with the saved read-only audit records.')
    args = parser.parse_args()
    torch.set_num_threads(1)
    data = Planetoid(root=str(ROOT / 'data'), name='Cora', split='public')[0]
    labels = KMeans(n_clusters=5, init='k-means++', max_iter=50,
                    n_init=10, random_state=0).fit_predict(data.x.numpy())
    n = data.num_nodes
    rows, cols = data.edge_index.numpy()
    a = sparse.csr_matrix((np.ones(len(rows), dtype=np.int64), (rows, cols)), shape=(n, n))
    adjacency = a.toarray() > 0
    degree = np.asarray(a.sum(axis=1)).ravel()
    key = hashlib.sha256(
        (ROOT / 'vendor/cgt/util.py').read_bytes() + data.x.numpy().tobytes() +
        data.edge_index.numpy().tobytes() + json.dumps(dict(clusters=5, p=3), sort_keys=True).encode()
    ).hexdigest()[:16]
    cache = ROOT / 'cache' / f'cora-paper-{key}.pt'
    cached = torch.load(cache, map_location='cpu')
    b = cached['B'].numpy()
    same = labels[:, None] == labels[None, :]
    upper = np.triu(np.ones((n, n), dtype=bool), 1)
    existing = adjacency & upper
    b_nonzero = b > 0
    community = dict(
        clusters=int(np.unique(labels).size),
        cluster_sizes=np.bincount(labels).tolist(),
        existing_edge_same_community_fraction=float(same[existing].mean()),
        B_mean_existing_same=float(b[existing & same].mean()),
        B_mean_existing_cross=float(b[existing & ~same].mean()),
        B_nonzero_fraction=float(b_nonzero.mean()),
        B_nonzero_existing_edges=int((b_nonzero & adjacency).sum()),
        B_cross_community_nonzeros=int((b_nonzero & ~same).sum()),
        B_within_community_nonzeros=int((b_nonzero & same).sum()),
        candidate_new_edges=int((b_nonzero & upper).sum()))
    a2 = a @ a
    a3 = a2 @ a
    within = (a + a2 + a3).toarray() > 0
    exact3 = a3.toarray() > 0
    candidates = within & same & ~adjacency & upper
    degree_filtered = candidates & (np.abs(degree[:, None] - degree[None, :]) <= 1)
    leaves = degree == 1
    leaf_pairs = same & leaves[:, None] & leaves[None, :] & ~adjacency & upper & (a2.toarray() > 0)
    degree_filtered_symmetric = degree_filtered | degree_filtered.T
    hops = dict(
        paper_within3_same_community_nonadjacent_pairs=int(candidates.sum()),
        after_extra_hard_degree_gap_at_most1=int(degree_filtered.sum()),
        after_exact3_walk_requirement=int((degree_filtered & exact3).sum()),
        actual_B_candidate_pairs=int((b_nonzero & upper).sum()),
        within3_degree_filtered_pairs_missed_by_exact3=int((degree_filtered & ~exact3).sum()),
        degree1_nodes=int(leaves.sum()),
        same_community_degree1_pairs_at_distance2=int(leaf_pairs.sum()),
        such_leaf_pairs_present_in_B=int((leaf_pairs & b_nonzero).sum()),
        degree1_nodes_without_B_candidates=int((leaves & ~b_nonzero.any(axis=1)).sum()),
        degree1_nodes_without_within3_degree_filtered_candidates=int(
            (leaves & ~degree_filtered_symmetric.any(axis=1)).sum()))
    if args.verify_records:
        for name, current in [('community-audit.json', community), ('candidate-hop-audit.json', hops)]:
            saved = json.loads((ROOT / 'results' / name).read_text())
            for field, value in current.items():
                if isinstance(value, float):
                    assert np.isclose(saved[field], value, rtol=0, atol=1e-7), field
                else:
                    assert saved[field] == value, (field, saved[field], value)
    print(json.dumps(dict(cache_key=key, verified_records=args.verify_records,
                          community=community, candidates=hops), indent=2))


if __name__ == '__main__':
    main()
