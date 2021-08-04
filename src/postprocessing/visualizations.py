import os
from typing import List, Tuple
from numpy.lib.npyio import save
from numpy.lib.utils import safe_eval
from torch.tensor import Tensor
from tqdm import tqdm
from src.postprocessing.matrix_vis import mat_vis
import torch


def vis_3d_tensor(x, xticks, yticks, zticks, save_path):
    mats = [x[:, :, i] for i in range(x.shape[2])]
    os.makedirs(save_path, exist_ok=True)
    for j, mat in enumerate(mats):
        mat_vis(
            mat,
            x_ticks=xticks,
            y_ticks=yticks,
            save_path=os.path.join(save_path, f"rel_{zticks[j]}.jpeg")
        )


def visualize_local_layers(
    extras,
    tokens2d,
    ents_df,
    p,
    rel_id2label,
    key,
    max_len=100,
):
    ents = ents_df[["start", "end", "step_idx"]].values.tolist()
    ents = [
        " ".join(tokens2d[step_idx][start:end+1])
        for start, end, step_idx in ents
    ] # yapf: disable

    for layer_id, layer_data in tqdm(extras.items(), total=len(extras), desc=f"visualizing {key}"):
        mat = layer_data[key].permute(1, 2, 0).detach().cpu().numpy()
        if mat.shape[0] > max_len:
            continue

        # -> (N x N x r)
        save_dir = f"vis/{p.orig_doc}_{p.part_key}/{layer_id}/{key}"
        save_path = os.path.join(os.getcwd(), save_dir)
        os.makedirs(save_path, exist_ok=True)
        vis_3d_tensor(
            mat,
            xticks=ents,
            yticks=ents,
            zticks=rel_id2label,
            save_path=save_path
        )


def visualize_att_weights(att_weights, tokens2d, ents_df, p, max_count=10):
    ents = ents_df[["start", "end", "step_idx"]].values.tolist()
    ents = [
        " ".join(tokens2d[step_idx][start:end+1])
        for start, end, step_idx in ents
    ] # yapf: disable
    att_weights_per_rel = torch.split(att_weights, 1, dim=0)
    for i, (mat, ent) in tqdm(
        enumerate(zip(att_weights_per_rel, ents)),
        total=len(att_weights_per_rel)
    ):
        if i > max_count:
            break

        mat = mat.cpu().numpy()
        # -> (1 x N)

        save_dir = f"vis_att/{p.orig_doc}_{p.part_key}"
        save_path = os.path.join(os.getcwd(), save_dir)
        os.makedirs(save_path, exist_ok=True)
        mat_vis(
            mat,
            x_ticks=ent.split(" "),
            y_ticks=["0"],
            save_path=os.path.join(save_path, f"ent_{i}.jpeg")
        )


def visualize_gates(
    extras, tokens2d, ents_df, p, rel_id2label, key, max_len=100
):
    ents = ents_df[["start", "end", "step_idx"]].values.tolist()
    ents = [
        " ".join(tokens2d[step_idx][start:end+1])
        for start, end, step_idx in ents
    ] # yapf: disable

    for layer_id, layer_data in tqdm(extras.items(), total=len(extras), desc=f"visualizing {key}"):
        gate = layer_data[key]
        if gate is None:
            continue

        save_dir = f"vis_gates/{p.orig_doc}_{p.part_key}/{layer_id}/{key}"
        save_path = os.path.join(os.getcwd(), save_dir)
        os.makedirs(save_path, exist_ok=True)

        gate = gate.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # -> (N x N x r)
        if gate.shape[0] > max_len:
            continue

        vis_3d_tensor(
            gate,
            xticks=ents,
            yticks=ents,
            zticks=rel_id2label,
            save_path=save_path
        )


def vis_all(extras, tokens2d, ents_df, p, rel_id2label):
    # expected extras structure
    # {
    #   'layer_i': {
    #       'A' : A,
    #       'f_gate': f_gate,
    #       'i_gate': i_gate,
    #       'pred_rels: pred_rels,
    #    }
    #    'att_weights': att_weights
    # }

    att_weights = extras['att_weights']

    # ! blech
    extras.pop('att_weights', None)

    visualize_local_layers(extras, tokens2d, ents_df, p, rel_id2label, key="A")
    visualize_local_layers(
        extras, tokens2d, ents_df, p, rel_id2label, key="pred_rels"
    )
    visualize_gates(extras, tokens2d, ents_df, p, rel_id2label, key="f_gate")
    visualize_gates(extras, tokens2d, ents_df, p, rel_id2label, key="i_gate")
    visualize_att_weights(att_weights, tokens2d, ents_df, p)
