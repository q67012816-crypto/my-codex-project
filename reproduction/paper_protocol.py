"""Explicit Table 2 settings, Eq. 14–17 losses and test-only r=1 grouping."""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PaperConfig:
    layers: int = 4
    heads: int = 4
    clusters: int = 5
    dropout: float = 0.1
    hidden_dim: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-5
    alpha: float = 1.0
    beta1: float = 0.5
    beta2: float = 0.5
    xi: float = 0.8
    zeta: float = 0.2
    # Working choice from the repository; not specified in Table 2.
    transition_steps: int = 3


CONFIG = PaperConfig()


def build_model(in_features, classes, config=CONFIG):
    from models import Transformer
    # Upstream always appends an output attention layer.
    model = Transformer(in_features, config.hidden_dim, classes,
                        hidden_dim=config.hidden_dim, num_layers=config.layers - 1,
                        num_heads=config.heads, k_transition=config.transition_steps,
                        aug_check=1, sim_check=1, phi_check=1,
                        alfa=config.zeta, beta=config.xi, dropout=config.dropout)
    assert len(model.layers) == config.layers
    assert all(layer.dropout == config.dropout for layer in model.layers)
    return model


def pretraining_loss(h, reconstructed, sampled, features, adjacency, log_transitions,
                     config=CONFIG):
    """Eq.14–16, sums (not means). Feature ||.||_2 is flattened vector L2.

    Standard PyTorch probability BCE clamps log(0) to -100 at hard 0/1 values.
    This endpoint convention and interpretation of the feature matrix norm are
    explicit implementation choices; neither is specified by the paper.
    """
    cosine = F.normalize(h, p=2, dim=1) @ F.normalize(h, p=2, dim=1).T
    transition = sum((cosine - target).square().sum() for target in log_transitions)
    feature = torch.linalg.vector_norm(reconstructed - features) / features.shape[0]
    augmentation = F.binary_cross_entropy(sampled, adjacency, reduction='sum')
    total = config.beta1 * transition + config.beta2 * feature + config.alpha * augmentation
    return total, dict(transition=transition, feature=feature, augmentation=augmentation)


def degree_statistical_parity(predictions, labels, degree, test_mask, fraction=0.2):
    """Eq.17: macro mean of absolute class-frequency gaps, r=1 only.

    Rank *test candidates* by (original-graph degree, global node id); select
    floor(fraction * n_test) at each end. No labels enter group selection.
    """
    pred = np.asarray(predictions)
    truth = np.asarray(labels)
    deg = np.asarray(degree)
    mask = np.asarray(test_mask)
    if any(x.ndim != 1 for x in [pred, truth, deg, mask]):
        raise ValueError('all inputs must be one-dimensional')
    if len({len(x) for x in [pred, truth, deg, mask]}) != 1:
        raise ValueError('input lengths differ')
    if mask.dtype != np.bool_ or not np.isfinite(deg).all():
        raise ValueError('test mask must be boolean and degrees finite')
    if not (0 < fraction <= 0.5):
        raise ValueError('fraction must lie in (0, 0.5]')
    ids = np.flatnonzero(mask)
    count = math.floor(len(ids) * fraction)
    if count < 1:
        raise ValueError('too few test nodes for nonempty groups')
    ranked = ids[np.lexsort((ids, deg[ids]))]
    low, high = ranked[:count], ranked[-count:]
    classes = np.unique(truth[ids])
    p_low = np.array([(pred[low] == c).mean() for c in classes])
    p_high = np.array([(pred[high] == c).mean() for c in classes])
    gaps = np.abs(p_low - p_high)
    value = float(gaps.mean())
    return dict(delta_sp=value, delta_sp_percent=100 * value, r=1,
                formula='paper page 8, Eq. (17)', candidate_scope='test nodes only',
                degree_source='original full undirected graph, no added self loops',
                group_fraction=fraction, group_size=count, test_size=len(ids),
                rounding='floor', tie_break='ascending global node id within ascending degree order',
                low_node_ids=low.tolist(), high_node_ids=high.tolist(),
                low_degree_range=[float(deg[low].min()), float(deg[low].max())],
                high_degree_range=[float(deg[high].min()), float(deg[high].max())],
                classes=classes.tolist(), low_prediction_frequencies=p_low.tolist(),
                high_prediction_frequencies=p_high.tolist(), per_class_absolute_gaps=gaps.tolist())


def config_dict():
    return asdict(CONFIG)
