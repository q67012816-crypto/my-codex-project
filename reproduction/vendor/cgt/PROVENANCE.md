Source: https://github.com/NSLab-CUK/Community-aware-Graph-Transformer
Commit: 948f25477eb0253873ffdf6d38fcf53f258fdd83

`models.py` and `util.py` are copied from that commit, under the included license.
Model changes: `edge_probs.cuda()` becomes `edge_probs.to(device_2)` for CPU
execution; Transformer/GraphTransformerLayer accept optional dropout (default
0.5 for the legacy runner). Old checkpoints without the attribute also use 0.5.
The new runner passes 0.1 and requests 3 hidden layers plus the upstream output
layer, giving exactly 4 attention layers. `util.py` is unchanged; its cluster
count argument is supplied as 5 by the new runner. See ../../alignment.md.

The repository calls this model CGT. The uploaded arXiv:2504.15075v1 paper
names its model DegFairGT on page 1 and links this repository on page 7.
The correspondence is verified; implementation differences remain (see ../../paper-audit.md).
This copy enables a small classification smoke run; it does not establish paper
reproduction. The original entry point requires multiple GPUs for its pretraining
forward call and does not implement degree statistical parity evaluation.
