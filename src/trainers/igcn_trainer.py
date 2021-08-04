import logging
import os
from src.pytorch_models.r_gcn_decoder import DecoderIGCN

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from definitions import ROOT_DIR
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from src.corpus.tokenizers import WordPieceTokenizer
from src.corpus.wlp_data_module import WLPDataModule
from src.pytorch_models.bert import Bert
from src.pytorch_models.iterative_graph_framework import \
    IterativeGraphFramework
from src.pytorch_models.mlp import MLP
from src.pytorch_models.span_extractor import SpanExtractor

log = logging.getLogger(__name__)


def trainer(cfg):

    torch.autograd.set_detect_anomaly(True)
    dm = WLPDataModule(
        train_path=os.path.join(ROOT_DIR, cfg.data.train),
        dev_path=os.path.join(ROOT_DIR, cfg.data.dev),
        test_path=os.path.join(ROOT_DIR, cfg.data.test),
        ent_classes=cfg.wlp_props.ents.classes,
        max_pos=cfg.hparams.max_pos,
        max_step=cfg.hparams.max_step,
        max_step_gap=cfg.hparams.max_step_gap,
        max_span_width=cfg.hparams.max_span_width,
        num_workers=cfg.num_workers,
        process_protocols=None,
        rel_classes_set=cfg.wlp_props.rels.classes_set,
        rel_neg_frac=cfg.wlp_props.rels.neg_frac,
        rel_pos_frac=cfg.wlp_props.rels.pos_frac,
        rel_undirected_classes=cfg.wlp_props.rels.undirected_classes,
        tokenizer=WordPieceTokenizer(
            lowercase=cfg.wlp_props.lowercase,
            replace_digits=cfg.wlp_props.replace_digits,
        )
    )

    model = IterativeGraphFramework(
        module_lm=Bert(
            hidden_dim=cfg.hparams.bert_hidden_dim,
            nb_ents=len(dm.ent_label2id)
        ),
        module_span=SpanExtractor(
            input_dim=cfg.hparams.tkn_emb_dim,
            out_dim=cfg.hparams.span_emb_dims,
            max_width=cfg.hparams.max_span_width,
            max_pos=cfg.hparams.max_pos,
            max_step=cfg.hparams.max_step,
            pos_step_emb=cfg.hparams.pos_step_embedding,
            emb_dim=cfg.hparams.span_aux_emb_dims,
            s2t=cfg.hparams.s2t
        ),
        module_span_scorer=MLP(
            input_features=cfg.hparams.span_emb_dims,
            output_features=1,
            arch=cfg.hparams.ent_arch or [],
            act=cfg.hparams.ent_act,
            dropout=cfg.hparams.ent_dropout,
            layernorm=False,
            bias=cfg.hparams.ent_bias
        ),
        module_ent=MLP(
            input_features=cfg.hparams.span_emb_dims,
            output_features=len(dm.ent_label2id),
            arch=cfg.hparams.ent_arch or [],
            act=cfg.hparams.ent_act,
            dropout=cfg.hparams.ent_dropout,
            layernorm=False,
            bias=cfg.hparams.ent_bias
        ),
        module_igcn=DecoderIGCN(
            in_features=cfg.hparams.span_emb_dims,
            nb_rels=dm.nb_rels,
            act=cfg.hparams.gcn_act,
            dropout=cfg.hparams.rel_dropout,
            layernorm=cfg.hparams.layernorm,
            n_layers=cfg.hparams.n_layers,
            rel_conv=cfg.hparams.rel_conv,
            multi_head_r_gcn=cfg.hparams.multi_head_r_gcn,
            transcoder_block=cfg.hparams.transcoder_block
        ),
        ent_classes=dm.ent_classes,
        ent_id2label=dm.ent_id2label,
        ent_label2id=dm.ent_label2id,
        ent_neg_class=dm.ent_neg_class,
        ent_pad_class=dm.special_tokens.pad,
        ent_prune_score_threshold=cfg.hparams.ent_prune_score_threshold,
        hparams_ent_loss_alpha=cfg.hparams.ent_loss_alpha,
        hparams_error_analysis=True,
        hparams_label_smoothing_epsilon=cfg.hparams.label_smoothing_epsilon,
        hparams_lr=cfg.hparams.lr,
        hparams_max_pos=100,
        hparams_max_step_gap=cfg.hparams.max_step_gap,
        hparams_per_spans_to_keep=cfg.hparams.per_spans_to_keep,
        hparams_pos_emb_size=144,
        hparams_rel_loss_alpha=cfg.hparams.rel_loss_alpha,
        hparams_span_emb_dims=cfg.hparams.span_emb_dims,
        hparams_span_loss_alpha=cfg.hparams.span_loss_alpha,
        rel_classes_set=dm.rel_classes_set,
        rel_id2label=dm.rel_id2label,
        rel_label2id=dm.rel_label2id,
        rel_neg_class=dm.rel_neg_class,
        true_doc_path={
            "test": cfg.data.full_test, "val": cfg.data.full_dev
        },
        vis_protocol=cfg.vis_protocol,
        eval_wlp=cfg.eval_wlp,
        cross_sentence=cfg.cross_sentence
    )
    chk = os.path.join(ROOT_DIR, cfg.chk) if cfg.chk else None
    trainer = pl.Trainer(
        gpus=1,
        deterministic=cfg.deterministic,
        #profiler='advanced',
        checkpoint_callback=ModelCheckpoint(
            filepath=cfg.results.model_save.format(1),
            save_top_k=1,
            verbose=True,
            mode='max',
            monitor=cfg.monitor_val
        ),
        #val_check_interval=3000,
        #resume_from_checkpoint=chk,
        max_epochs=40,
        num_sanity_val_steps=5,
        callbacks=[
            EarlyStopping(monitor=cfg.monitor_val, patience=15, mode='max')
        ],
        auto_lr_find=cfg.lr_find,
        progress_bar_refresh_rate=1,
        logger=TensorBoardLogger(
            save_dir=os.getcwd(),
            name=cfg.results.exp_name,
            log_graph=True,
        ),
    )
    if cfg.lr_find:
        lr_finder = trainer.tuner.lr_find(
            model=model,
            min_lr=1e-5,
            max_lr=1e-2,
            num_training=100,
            datamodule=dm
        )

        # Results can be found in
        lr_finder.results

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.savefig("lr_plot.png")

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        model.lr = new_lr
        log.info(f"Learning Rate set to: {new_lr}")

    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm)


@hydra.main(
    config_path=os.path.join(ROOT_DIR, "src/configs"), config_name="igcn"
)
def main(cfg):
    log.info(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    trainer(cfg)


if __name__ == "__main__":
    main()
