import torch


def dropedge(adj, p=0.5, training=True):
    if training:
        probs = torch.zeros_like(adj)
        probs[adj != 0] = p
        mask = torch.bernoulli(probs).bool()
        adj[mask] = 0

    return adj