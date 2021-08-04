import os
from src.pytorch_models.mlp import MLP
from numpy.core.numeric import cross

from torch.nn.modules.loss import BCELoss
from src.pytorch_models.label_smoothing import LabelSmoothingCrossEntropy
from src.postprocessing.visualizations import vis_all
from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from definitions import ROOT_DIR
from src.evaluator.eval_wlp import MetaEvalWLP
from src.evaluator.ner import EvalNER
from src.evaluator.re import EvalRE
from src.postprocessing.matrix_vis import mat_vis
from src.pytorch_models import utils
from src.pytorch_models.pruner import Pruner
from torch import BoolTensor, LongTensor, nn
from torch.optim.adamw import AdamW
from tqdm import tqdm


class IterativeGraphFramework(pl.LightningModule):
    def __init__(
        self,
        ent_classes: List[str],
        ent_id2label: Dict[int, str],
        ent_label2id: Dict[int, str],
        ent_neg_class: str,
        ent_pad_class: str,
        ent_prune_score_threshold: float,
        hparams_ent_loss_alpha: float,
        hparams_error_analysis: bool,
        hparams_label_smoothing_epsilon: float,
        hparams_lr: float,
        hparams_max_pos: int,
        hparams_max_step_gap: int,
        hparams_per_spans_to_keep: float,
        hparams_pos_emb_size: int,
        hparams_rel_loss_alpha: float,
        hparams_span_emb_dims: float,
        hparams_span_loss_alpha: float,
        module_ent: nn.Module,
        module_igcn: nn.Module,
        module_lm: nn.Module,
        module_span: nn.Module,
        module_span_scorer: nn.Module,
        rel_classes_set: Dict[str, List[str]],
        rel_id2label: Dict[int, str],
        rel_label2id: Dict[str, int],
        rel_neg_class: str,
        true_doc_path: Dict[str, str],
        vis_protocol: str,
        eval_wlp: bool,
        cross_sentence: bool,
    ):
        super().__init__()

        # Modules
        self.token_encoder = module_lm
        self.span_encoder = module_span
        self.span_scorer = module_span_scorer
        self.ent_encoder = module_ent
        self.igcn = module_igcn

        # TODO cleanup passing pad_id and neg
        self.gold_pruner = Pruner(
            None,
            pad_id=ent_label2id[ent_pad_class],
            neg_sample_id=ent_label2id[ent_neg_class],
            gold_beam=True
        )
        self.non_gold_pruner = Pruner(
            None,
            gold_beam=False,
            pad_id=ent_label2id[ent_pad_class],
            neg_sample_id=ent_label2id[ent_neg_class],
            min_score=ent_prune_score_threshold
        )

        # hparams
        self.ent_loss_alpha = hparams_ent_loss_alpha
        self.error_analysis = hparams_error_analysis
        self.lr = hparams_lr
        self.max_pos = hparams_max_pos
        self.max_step_gap = hparams_max_step_gap
        self.nb_rels = len(rel_classes_set['full']) + 1
        self.per_spans_to_keep = hparams_per_spans_to_keep
        self.rel_id2label = rel_id2label
        self.rel_label2id = rel_label2id
        self.rel_loss_alpha = hparams_rel_loss_alpha
        self.rel_neg_class = rel_neg_class
        self.rel_neg_class_id = rel_label2id[rel_neg_class]
        self.span_loss_alpha = hparams_span_loss_alpha

        self.vis_protocol = vis_protocol
        self.do_eval_wlp = eval_wlp
        self.cross_sentence = cross_sentence

        # NER evaluators
        self.ner_eval = {
            dataset_type: EvalNER(
                ent_classes,
                ent_id2label,
                neg_class=ent_neg_class,
                compute_on_step=False,
                prefix=dataset_type
            )
            for dataset_type in
            ["train", "gold_val", "non_gold_val", "gold_test", "non_gold_test"]
        }

        # RE evaluators
        self.re_eval = {
            dataset_type: EvalRE(
                classes_set=rel_classes_set,
                id2label=rel_id2label,
                neg_class=rel_neg_class,
                compute_on_step=False,
                prefix=dataset_type
            )
            for dataset_type in
            ["train", "gold_val", "non_gold_val", "gold_test", "non_gold_test"]
        }

        # full doc eval
        self.doc_eval = {
            dataset_type: MetaEvalWLP(
                ent_classes=ent_classes,
                rel_classes_set=rel_classes_set,
                ent_id2label=ent_id2label,
                rel_id2label=rel_id2label,
                ent_neg_class=ent_neg_class,
                rel_neg_class=rel_neg_class,
                compute_on_step=False,
                prefix=f"{dataset_type}_doc_eval",
                true_doc_path=os.path.join(
                    ROOT_DIR, true_doc_path[dataset_type]
                )
            )
            for dataset_type in ["val", "test"]
        }

        # label smoothing
        self.loss_func = LabelSmoothingCrossEntropy(
            hparams_label_smoothing_epsilon
        )

        # self.label_embeddings = nn.Embedding(
        #     num_embeddings=len(ent_label2id), embedding_dim=50
        # )

        # self.linear = MLP(
        #     input_features=336 + 50,
        #     output_features=336,
        #     act='relu'
        # )

        self.layernorm = nn.LayerNorm(hparams_span_emb_dims)

    def forward(
        self,
        tkn_ids: LongTensor,
        tkn_mask: BoolTensor,
        spans: LongTensor,
        span_mask: BoolTensor,
        gold_span_labels: LongTensor = None,
        get_extras: bool = False,
    ):

        extras = None

        tkn_embs, cls_tkns = self.token_encoder(tkn_ids, tkn_mask)
        # -> (batch_size, max_seq_len, bert_hidden_dims)

        span_embs, att_weights = self.span_encoder(
            sequences=tkn_embs,
            spans=spans,
            span_mask=span_mask,
            sequence_mask=tkn_mask
        )
        # -> (batch_size, max_n_spans, span_emb_dims)

        span_logits = self.span_scorer(span_embs)
        # -> (batch_size, max_n_spans, 1)

        span_scores = F.sigmoid(span_logits).squeeze(2)
        # -> (batch_size, max_n_spans)

        max_spans = torch.ceil(
            span_mask.sum(dim=1) * self.per_spans_to_keep
        ).long() # yapf: disable

        # prune low scoring spans. Generate a mask to keep top scoring spans.
        if gold_span_labels is not None:
            top_span_mask = self.gold_pruner(
                span_embs=span_embs,
                span_mask=span_mask,
                max_spans=max_spans,
                scores=span_scores,
                gold_labels=gold_span_labels
            )
            # encode span label embeddings (only for WNUT)
            #label_embs = self.label_embeddings(gold_span_labels)
            #span_embs = torch.cat([span_embs, label_embs], dim=2)
            #span_embs = self.linear(span_embs)
        else:
            top_span_mask = self.non_gold_pruner(
                span_embs=span_embs,
                span_mask=span_mask,
                max_spans=max_spans,
                scores=span_scores  # skip neg class scores
            )

        (
            top_span_embs,
            rel_logits,
            extras,
        ) = self.igcn(span_embs[top_span_mask], get_extras=get_extras)

        # -> (top_spans, top_spans, nb_rels)

        assert rel_logits.shape[0] == torch.sum(top_span_mask).item()

        # get ent logits for the top scoring spans that went through the igcn
        # network. These spans are now rich with relational information
        # which could be useful in improving entity extraction.

        # add up newly realized span representations to the ent logits. Make
        # sure to only add to those positions to which they belong.
        ent_logits = self.ent_encoder(top_span_embs)

        # rel mask indicates which pair of spans were selected as candidates
        # for relation extraction.

        # only generate a rel mask for validation set. Since during training
        # top_span_mask is always going to be gold_spans. Hence rel_mask will
        # be pre-generated in dataloaders. By skipping this we shave off a
        # few seconds.

        # build a relation mask to mask out filtered spans. When predicting
        # on gold ents, this mask will be true for golden entity pairs.
        # When predicting on predicted ents, this mask will be true for only
        # the top scoring entities, which will be paired up.

        ent_mask = top_span_mask[span_mask]
        if self.cross_sentence:
            rel_mask = utils.make_pairs_by_mask(ent_mask)
        else:
            # mask out cross-sentence relations
            full_rel_mask = utils.make_pairs_by_mask(ent_mask)
            rel_mask = utils.make_pairs_by_sent_mask(ent_mask, span_mask)

            full_rel_logits = torch.zeros(
                (rel_mask.shape[0], rel_mask.shape[1], rel_logits.shape[2])
            ).to(rel_logits.device)

            full_rel_logits[full_rel_mask] = rel_logits.view(-1, self.nb_rels)

            # remove self loops
            rel_mask = rel_mask * (
                1 - torch.eye(rel_mask.shape[0]).to(rel_mask.device)
            )

            rel_mask = rel_mask.bool()

            # -> (all_spans x all_spans x nb_rels)
            rel_logits = full_rel_logits[rel_mask]
            # -> (all_valid_spans x nb_rels)

        #assert rel_logits.shape[0]**2 == torch.sum(rel_mask).item()

        if get_extras:
            extras["att_weights"] = att_weights[top_span_mask]
            return span_logits, ent_logits, rel_logits, top_span_mask, rel_mask, extras
        else:
            return span_logits, ent_logits, rel_logits, top_span_mask, rel_mask

    def training_step(self, batch, batch_idx):
        ents, rels, p = batch
        all_spans = ents.span_mask.shape[0]
        true_rels = torch.sparse.LongTensor(
            rels.indices.long().t(),
            rels.label_ids.long(),
            rels.label_mask.shape
        ).to(self.device)

        # forward pass
        span_logits, ent_logits, rel_logits, ent_mask, rel_mask = self(
            tkn_ids=ents.tkn_ids,
            tkn_mask=ents.tkn_mask,
            spans=ents.spans,
            span_mask=ents.span_mask,
            gold_span_labels=ents.label_ids,
        )

        # loss
        # span loss
        # Span scores weed out negative samples.

        span_loss = F.binary_cross_entropy_with_logits(
            span_logits[ents.span_mask],
            target=ents.is_span[ents.span_mask].unsqueeze(1)
        )

        # entity loss will be trained on only positive samples.
        # ents.label_ids
        ent_loss = self.loss_func(ent_logits, ents.label_ids[ent_mask])
        # rels.label_mask controls which relations should be trained.
        # self loops and relations between ents greater than max_step_gap are
        # not trained. In addition to that we can choose to only train some
        # negative samples. (eg: Hard Negative Mining).

        rel_loss = self.loss_func(
            rel_logits.view(-1, self.nb_rels),
            true_rels.to_dense()[rel_mask].view(-1)
        )
        loss = (
            self.span_loss_alpha * span_loss +
            self.ent_loss_alpha * ent_loss +
            self.rel_loss_alpha * rel_loss
        ) # yapf: disable

        # logging + metrics
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        self.log("train/span_loss", span_loss, on_epoch=True, on_step=False)
        self.log("train/ent_loss", ent_loss, on_epoch=True, on_step=False)
        self.log("train/rel_loss", rel_loss, on_epoch=True, on_step=False)

        ents_df = self.ner_eval["train"].predict(
            ent_logits, ents.spans, ent_mask
        )
        self.ner_eval["train"](ents_df, ents.gold_df)
        pred_df = self.re_eval["train"].predict(
            rel_logits, ents_df, cross_sentence=self.cross_sentence
        )
        self.re_eval["train"](pred_df, rels.gold_df)

        if self.global_step == 0:
            self.logger.log_graph(
                self,
                input_array=[
                    ents.tkn_ids,
                    ents.tkn_mask,
                    ents.spans,
                    ents.span_mask,
                    ents.label_ids
                ]
            )

        return loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        metrics, ent_image = self.ner_eval["train"].compute()
        self.log_dict(metrics)
        self.logger.experiment.add_image(
            tag="train_ents",
            img_tensor=ent_image,
            global_step=self.global_step,
            dataformats="HWC",
        )
        results = self.re_eval["train"].compute()
        for eval_type, (metrics, image) in results.items():
            self.log_dict(metrics)
            self.logger.experiment.add_image(
                tag=f"gold_train_rel_{eval_type}",
                img_tensor=image,
                global_step=self.global_step,
                dataformats="HWC",
            )

        super().training_epoch_end(outputs)

    def log_grads(self, model, tag=""):
        for name, weight in model.named_parameters():
            self.logger.experiment.add_histogram(
                f"{tag}/{name}", weight, self.global_step
            )
            if weight.grad is not None:
                self.logger.experiment.add_histogram(
                    f'{tag}/{name}.grad', weight.grad, self.global_step
                )

    def on_after_backward(self) -> None:
        if self.global_step == 1 or self.global_step % 500 == 0:
            self.log_grads(self.token_encoder, tag="token")
            self.log_grads(self.span_encoder, tag="span")
            self.log_grads(self.ent_encoder, tag="ent")
            self.log_grads(self.span_scorer, tag='span_scorer')
            self.log_grads(self.igcn, tag='igcn')

        return super().on_after_backward()

    def validation(self, batch, batch_idx, use_gold_ents=False):
        set_type = "gold_val" if use_gold_ents else "non_gold_val"
        ents, rels, p = batch
        get_extras = (p.orig_doc == self.vis_protocol and p.part_key == '0')
        extras = None

        true_rels = torch.sparse.LongTensor(
            rels.indices.long().t(),
            rels.label_ids.long(),
            rels.label_mask.shape
        ).to(self.device)

        with torch.no_grad():

            # gold ents forward pass
            out = self(
                ents.tkn_ids,
                ents.tkn_mask,
                spans=ents.spans,
                span_mask=ents.span_mask,
                gold_span_labels=ents.label_ids if use_gold_ents else None,
                get_extras=get_extras,
            )

            if get_extras:
                span_logits, ent_logits, rel_logits, ent_mask, rel_mask, extras = out
            else:
                span_logits, ent_logits, rel_logits, ent_mask, rel_mask = out

            # loss
            span_loss = F.binary_cross_entropy_with_logits(
                span_logits[ents.span_mask],
                target=ents.is_span[ents.span_mask].unsqueeze(1)
            )
            ent_loss = F.cross_entropy(ent_logits, ents.label_ids[ent_mask])
            rel_loss = F.cross_entropy(
                rel_logits.view(-1, self.nb_rels),
                true_rels.to_dense()[rel_mask].view(-1),
            )
            loss = (
                self.span_loss_alpha * span_loss +
                self.ent_loss_alpha * ent_loss +
                self.rel_loss_alpha * rel_loss
            ) # yapf: disable

            # logging + metrics
            self.log(f"{set_type}/loss", loss, on_epoch=True, on_step=False)
            self.log(
                f"{set_type}/span_loss",
                span_loss,
                on_epoch=True,
                on_step=False
            )
            self.log(
                f"{set_type}/ent_loss", ent_loss, on_epoch=True, on_step=False
            )
            self.log(
                f"{set_type}/rel_loss", rel_loss, on_epoch=True, on_step=False
            )

            ents_df = self.ner_eval[set_type].predict(
                ent_logits, ents.spans, ent_mask
            )
            self.ner_eval[set_type](ents_df, ents.gold_df)

            pred_df = self.re_eval[set_type].predict(
                rel_logits, ents_df, cross_sentence=self.cross_sentence
            )

            self.re_eval[set_type](pred_df, rels.gold_df)

            if not use_gold_ents:
                rels_df = pred_df.rename(columns={"pred_label": "label"})
                self.doc_eval["val"](
                    pred_doc=(ents_df, rels_df),
                    tokens2d=ents.tokens2d,
                    orig_key=p.orig_doc,
                    partition_key=p.part_key
                )
                if get_extras:
                    vis_all(
                        extras,
                        tokens2d=ents.tokens2d,
                        ents_df=ents_df,
                        p=p,
                        rel_id2label=self.rel_id2label
                    )

        return loss

    def validation_step(self, batch, batch_idx):
        # TODO: Optimize. Too much duplication.
        gold_loss = self.validation(batch, batch_idx, use_gold_ents=True)
        non_gold_loss = self.validation(batch, batch_idx, use_gold_ents=False)

        return gold_loss

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        print("on validation end...")
        val_sets = ['gold_val', 'non_gold_val']

        for set_type in val_sets:
            metrics, ent_image = self.ner_eval[set_type].compute()
            self.log_dict(metrics)
            self.logger.experiment.add_image(
                tag=f"{set_type}_ents",
                img_tensor=ent_image,
                global_step=self.global_step,
                dataformats="HWC",
            )

            results = self.re_eval[set_type].compute()
            for eval_type, (metrics, image) in results.items():
                self.log_dict(metrics)
                self.logger.experiment.add_image(
                    tag=f"{set_type}_rel_{eval_type}",
                    img_tensor=image,
                    global_step=self.global_step,
                    dataformats="HWC",
                )

        if self.do_eval_wlp:
            metrics, html_tables, images = self.doc_eval["val"].compute()
            self.log_dict(metrics)
            for title, image in images.items():
                self.logger.experiment.add_image(
                    tag=title,
                    img_tensor=image,
                    global_step=self.global_step,
                    dataformats="HWC",
                )
            self.doc_eval["val"].pred_doc_merger.save_prediction("brat")

            for tag, table in html_tables.items():
                self.logger.experiment.add_text(tag, table, self.global_step)

        print("valid end complete")

        super().validation_epoch_end(outputs)

    def test(self, batch, batch_idx, use_gold_ents=False):
        set_type = "gold_test" if use_gold_ents else "non_gold_test"
        ents, rels, p = batch
        get_extras = p.orig_doc == self.vis_protocol
        extras = None
        true_rels = torch.sparse.LongTensor(
            rels.indices.long().t(),
            rels.label_ids.long(),
            rels.label_mask.shape
        ).to(self.device)

        with torch.no_grad():

            # gold ents forward pass
            out = self(
                ents.tkn_ids,
                ents.tkn_mask,
                spans=ents.spans,
                span_mask=ents.span_mask,
                gold_span_labels=ents.label_ids if use_gold_ents else None,
                get_extras=get_extras
            )

            if get_extras:
                span_logits, ent_logits, rel_logits, ent_mask, rel_mask, extras = out
            else:
                span_logits, ent_logits, rel_logits, ent_mask, rel_mask = out

            # loss
            span_loss = F.binary_cross_entropy_with_logits(
                span_logits[ents.span_mask],
                ents.is_span[ents.span_mask].unsqueeze(1)
            )
            ent_loss = F.cross_entropy(ent_logits, ents.label_ids[ent_mask])
            rel_loss = F.cross_entropy(
                rel_logits.view(-1, self.nb_rels),
                true_rels.to_dense()[rel_mask].view(-1),
            )
            loss = (
                self.span_loss_alpha * span_loss +
                self.ent_loss_alpha * ent_loss +
                self.rel_loss_alpha * rel_loss
            ) # yapf: disable

            # logging + metrics
            self.log(f"{set_type}/loss", loss, on_epoch=True, on_step=False)
            self.log(
                f"{set_type}/span_loss",
                span_loss,
                on_epoch=True,
                on_step=False
            )
            self.log(
                f"{set_type}/ent_loss", ent_loss, on_epoch=True, on_step=False
            )
            self.log(
                f"{set_type}/rel_loss", rel_loss, on_epoch=True, on_step=False
            )

            ents_df = self.ner_eval[set_type].predict(
                ent_logits, ents.spans, ent_mask
            )
            self.ner_eval[set_type](ents_df, ents.gold_df)

            rels_df = self.re_eval[set_type].predict(
                rel_logits, ents_df, cross_sentence=self.cross_sentence
            )

            self.re_eval[set_type](rels_df, rels.gold_df)

            if not use_gold_ents:
                rels_df = rels_df.rename(columns={"pred_label": "label"})
                self.doc_eval["test"](
                    pred_doc=(ents_df, rels_df),
                    tokens2d=ents.tokens2d,
                    orig_key=p.orig_doc,
                    partition_key=p.part_key
                )
                if get_extras:
                    vis_all(
                        extras=extras,
                        tokens2d=ents.tokens2d,
                        ents_df=ents_df,
                        p=p,
                        rel_id2label=self.rel_id2label
                    )

        return loss

    def test_step(self, batch, batch_idx):
        gold_loss = self.test(batch, batch_idx, use_gold_ents=True)
        non_gold_loss = self.test(batch, batch_idx, use_gold_ents=False)

        return non_gold_loss

    def test_epoch_end(self, outputs: List[Any]) -> None:

        if self.do_eval_wlp:
            metrics, html_tables, images = self.doc_eval["test"].compute()

            self.log_dict(metrics)
            for title, image in images.items():
                self.logger.experiment.add_image(
                    tag=f"test_{title}",
                    img_tensor=image,
                    global_step=self.global_step,
                    dataformats="HWC",
                )

            self.doc_eval["test"].pred_doc_merger.save_prediction("brat")
            for tag, table in html_tables.items():
                self.logger.experiment.add_text(tag, table, self.global_step)
        else:
            test_sets = ['gold_test', "non_gold_test"]
            for set_type in test_sets:
                metrics, ent_image = self.ner_eval[set_type].compute()
                self.log_dict(metrics)
                self.logger.experiment.add_image(
                    tag=f"{set_type}_ents",
                    img_tensor=ent_image,
                    global_step=self.global_step,
                    dataformats="HWC",
                )

                results = self.re_eval[set_type].compute()
                for eval_type, (metrics, image) in results.items():
                    self.log_dict(metrics)
                    self.logger.experiment.add_image(
                        tag=f"{set_type}_rel_{eval_type}",
                        img_tensor=image,
                        global_step=self.global_step,
                        dataformats="HWC",
                    )

        super().test_epoch_end(outputs)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)
