import html
import itertools as it
import json
import logging
import os
from collections import Counter
from typing import List

import pandas as pd
from src.corpus.brat_file import BratFile, Link
from src.corpus.protocol_objects import Entity, Relation, Tag
from src.corpus.seqeval import get_entities
from src.corpus.tokenizers import WordPieceTokenizer

from tokenizers import Tokenizer

log = logging.getLogger(__name__)


def base_tokenizer(sent):
    return sent.split()


# its a good idea to keep a datastructure like
# list of sentences, where each sentence is a list of words :
# [[word1, word2, word3,...], [word1, word2...]]


class ProtoFile(BratFile):
    def __init__(
        self,
        filename: str,
        pos: str = "Genia",
        gen_features: bool = False,
        rel_undirected_classes: List[str] = list(),
        ent_neg_class: str = "O",
        rel_neg_class: str = "O",
        to_lower: bool = False,
        replace_digits: bool = False,
        tokenizer: Tokenizer = WordPieceTokenizer(),
        start_tkn: str = '[CLS]',
        stop_tkn: str = '[SEP]',
        max_step_gap: int = 5,
        process_titles: bool = False,
    ):
        """
        Protocol object. Supports: 
        - brat protocol file parsing for actions, entities and relations 
            (both within sentence and cross sentence). 
        - tokenization (unigram, WordPiece, or bpe).
        - For entity labels -- BIO-encoding.
        - For relation pairs -- Relation objects.
        - (Optional) Generate Pos and dependency syntactic features.
        - 

        Args:
            filename (str): The filepath to the protocol files (.txt and .ann).
            pos (str, optional): Pos tagger to use. Only works if gen_features 
                is True. Defaults to "Genia".
            gen_features (bool, optional): Whether to pre-generate syntactic 
                features for the protocol. Defaults to False.
            ent_classes (List[str], optional): All the entity class names to 
                generate BIO-encoding. Defaults to None.
            ent_neg_class (str, optional): Negative label for entity labels. 
                Defaults to 'O'.
            to_lower (bool, optional): Whether or not to convert all the 
                words in the protocol to lowercase. Defaults to False.
            replace_digits (bool, optional): Whether or not to convert all 
                digits into 0. Defaults to False.
            truncate (int, optional): The number of lines of the protocol to keep. 
                Useful to reduce the scope of the protocol when training to 
                predict the full protocol graph. Defaults to sys.maxsize.
            tokenizer (str, optional): Choose a tokenizer from the currenly 
                supported tokenizers. [`bpe`, `word_piece`, `split`]. 
                Defaults to `split`.
        """

        self.filename = filename
        self.basename = os.path.basename(filename)
        self.protocol_name = self.basename

        self.text_file = self.filename + '.txt'
        self.ann_file = self.filename + '.ann'

        # initializing BratFile. Mixins dont have a init function.
        super().__init__(
            self.text_file, self.ann_file, process_titles=process_titles
        )

        self.to_lower = to_lower
        self.replace_digits = replace_digits
        self.ent_neg_class = ent_neg_class
        self.rel_neg_class = rel_neg_class
        self.tokenizer = tokenizer
        self.start = start_tkn
        self.stop = stop_tkn
        self.max_step_gap = max_step_gap
        self.rel_undirected_classes = rel_undirected_classes

        self.tokens2d, self.tags = self.gen_tokens(
            self.tags,
            self.sents,
            tokenizer=self.tokenizer,
            process_titles=process_titles
        )

        # make sure that all tags have tkn_start and tkn_end filled correctly
        # assert all(tag.is_valid() for tag in self.tags), [
        #     tag for tag in self.tags if not tag.is_valid()
        # ]

        self.rels = self.gen_relations(self.links)
        self.ents = self.gen_entities(self.tokens2d)

        # sanity checks
        # assert len(self.ents) == len(self.tags), \
        #     (self.ents, self.tags)

        # all pos and neg relations
        # //self.rels_df = self.all_rels_df(self.ents, self.rels)

        self.unique_tags = set([tag.tag_name for tag in self.tags])
        self.word_cnt = sum(len(tokens1d) for tokens1d in self.tokens2d)

        if gen_features:
            self.gen_features(pos)

    def update_links(self, links):
        links = [
            Link(
                id=link.id,
                name=link.name,
                arg1=self.tags[self.tags.index(link.arg1)],
                arg2=self.tags[self.tags.index(link.arg2)],
                sent_idx1=link.sent_idx1,
                sent_idx2=link.sent_idx2
            ) for link in links
        ]

        return links

    def gen_entities(self, tokens2d):

        ret = []
        for line_id, tokens in enumerate(tokens2d):
            ents = get_entities([tkn.label for tkn in tokens])
            ret.extend(
                [
                    Entity(
                        label=tag_name, start=start, end=end, step_idx=line_id
                    ) for tag_name,
                    start,
                    end in ents
                ]
            )

        return ret

    def gen_relations(self, links):
        # make sure that the tags stored in links are updated with
        # token start and end index.
        links = self.update_links(links)

        rels = {
            Relation(
                arg1=Entity(
                    label=link.arg1.tag_name,
                    start=link.arg1.tkn_start,
                    end=link.arg1.tkn_end,
                    step_idx=link.arg1.lineid,
                ),
                arg2=Entity(
                    label=link.arg2.tag_name,
                    start=link.arg2.tkn_start,
                    end=link.arg2.tkn_end,
                    step_idx=link.arg2.lineid,
                ),
                label=link.name
            )
            for link in links
            if all(
                v is not None for v in [
                    link.arg1.tkn_start,
                    link.arg1.tkn_end,
                    link.arg2.tkn_start,
                    link.arg2.tkn_end
                ]
            )
        }

        # generate relations for the inverse if their labels are undirected

        # rels = rels U reversed_rels
        rels = rels | {
            Relation(arg1=r.arg2, arg2=r.arg1, label=r.label)
            for r in rels
            if r.label in self.rel_undirected_classes
        }

        return rels

    def all_rels_df(self, ents, rels):

        args = [
            'arg1_start',
            'arg1_end',
            'arg1_sent_idx',
            'arg1_step_idx',
            'arg2_start',
            'arg2_end',
            'arg2_sent_idx',
            'arg2_step_idx'
        ]

        pos_rels = pd.DataFrame(
            [
                {
                    'arg1_start': r.arg1.start,
                    'arg1_end': r.arg1.end,
                    'arg1_sent_idx': r.arg1.sent_idx,
                    'arg1_step_idx': r.arg1.step_idx,
                    'arg2_start': r.arg2.start,
                    'arg2_end': r.arg2.end,
                    'arg2_sent_idx': r.arg2.sent_idx,
                    'arg2_step_idx': r.arg2.step_idx,
                    'true_label': r.label,
                } for r in rels
            ]
        ).set_index(args)
        # get all entity pairs
        ne_pairs_df = pd.DataFrame(
            [
                {
                    'arg1_start': a1.start,
                    'arg1_end': a1.end,
                    'arg1_sent_idx': a1.sent_idx,
                    'arg1_step_idx': a1.step_idx,
                    'arg2_start': a2.start,
                    'arg2_end': a2.end,
                    'arg2_sent_idx': a2.sent_idx,
                    'arg2_step_idx': a2.step_idx,
                } for a1,
                a2 in it.product(ents, repeat=2)
            ]
        )

        ne_pairs_df = ne_pairs_df.set_index(args)
        # left intersect on all pairs. Negative pairs will have NaNs
        ne_pairs_df = ne_pairs_df.join(pos_rels, how='left')
        ne_pairs_df = ne_pairs_df.reset_index()

        # fill pairs that didnt intersect with positive labels as negative
        # labels.
        ne_pairs_df[['true_label']] = \
            ne_pairs_df[['true_label']].fillna(self.rel_neg_class)

        # # mark is iap or cap
        # ne_pairs_df['is_iap'] = ne_pairs_df.apply(
        #     lambda row: row['arg1_ap'] == row['arg2_ap'], axis=1)

        # # remove 'arg1_ap' and 'arg2_ap' markers
        # ne_pairs_df.drop(columns=['arg1_ap', 'arg2_ap'])

        return ne_pairs_df

    @staticmethod
    def clean_html_tag(token):
        token.word = html.unescape(token.word)
        return token

    def gen_tokens(
        self,
        tags: List[Tag],
        sents: List[List[str]],
        tokenizer: Tokenizer = WordPieceTokenizer(),
        process_titles: bool = False,
    ):
        """
        Generates tokens from sentences in the file and tags found in 
        `BratFile`. The tokens are generated based on the type of tokenizer to
        be used.  

        Args:
            tags (List[Tag]): List of Tag objects as parsed by `BratFile`.
            sents (List[List[str]]): List of sentences of the protocol, where 
                each sentence is a list of words.
            labels_allowed (List[str], optional): Assigns negative class to 
                tokens that are marked by class labels not found in 
                `labels_allowed`. If not specified, only tokens with no class 
                label is marked with negative class.
            lowercase (bool, optional): Convert all ngrams to lowercase. 
                Defaults to False.
            replace_digits (bool, optional): Convert any digit to 0. 
                Defaults to False.

        Returns:
            List[List[Token]]: Returns the following, where each internal list 
                represents a sentence in the protocol.
                [
                    [Token(ngram, label)],
                    [Token(ngram, label), Token(ngram, label), Token(ngram, label)]
                ]
                label is BIO encoded.
        """

        sents = [" ".join(sent) for sent in sents]
        # in addition to making bio-tags for each token, it will also
        # infuse the token position (start, end) inside the respective tags.
        # this is needed later for mapping relations.

        heading_offset = 0 if process_titles else len(" ".join(self.heading))

        tokens2d, tags = tokenizer.tokenize(
            sents, tags, heading_offset=heading_offset
        )

        return tokens2d, tags

    def get_label_counts(self, add_no_ne_label=False):
        """Returns the count of each label in the article/document.
        Count means here: the number of words that have the label.

        Args:
            add_no_ne_label: Whether to count how often unlabeled words appear.
            (Default is False.)
        Returns:
            List of tuples of the form (label as string, count as integer).
        """
        tokens = list(it.chain.from_iterable(self.tokens2d))
        if add_no_ne_label:
            counts = Counter([token.label for token in tokens])
        else:
            counts = Counter(
                [
                    token.label
                    for token in tokens
                    if token.label != self.wlp_props.ents.neg_class
                ]
            )
        return counts.most_common()

    def count_labels(self, add_no_ne_label=False):
        """Returns how many named entity tokens appear in the article/document.

        Args:
            add_no_ne_label: Whether to also count unlabeled words.
            (Default is False.)
        Returns:
            Count of all named entity tokens (integer).
        """
        return sum(
            [
                count[1] for count in
                self.get_label_counts(add_no_ne_label=add_no_ne_label)
            ]
        )

    def to_json(self, s, e=None, prefix=False):
        if e is None:
            e = len(self.sents)

        json_dict = {}
        json_dict["sentences"] = self.sents[s:e]

        tokens = [
            [t.word for t in tokens[:511]] for tokens in self.tokens2d[s:e]
        ]

        json_dict["tokens"] = tokens

        json_dict["bio-tags"] = [
            [t.label for t in tokens[:511]] for tokens in self.tokens2d[s:e]
        ]

        ents = [
            {
                "start": ent.start,
                "end": ent.end,
                "step_idx": ent.step_idx - s,
                "label": ent.label
            } for ent in self.ents if s <= ent.step_idx < e and ent.end < 511
        ]

        ents = sorted(ents, key=lambda x: (x["step_idx"], x["start"], x["end"]))

        max_step = len(json_dict['tokens'])

        json_dict['named_entities'] = [
            [e
             for e in ents
             if e['step_idx'] == idx]
            for idx in range(max_step)
        ]

        json_dict['relations'] = [
            {
                "arg1_start": rel.arg1.start,
                "arg1_end": rel.arg1.end,
                "arg1_step_idx": rel.arg1.step_idx - s,
                "arg2_start": rel.arg2.start,
                "arg2_end": rel.arg2.end,
                "arg2_step_idx": rel.arg2.step_idx - s,
                "label": rel.label
            } for rel in self.rels if (
                s <= rel.arg1.step_idx < e and s <= rel.arg2.step_idx < e and
                rel.arg1.end < 511 and rel.arg2.end < 511
            )
        ]

        json_dict['relations'] = sorted(
            json_dict['relations'],
            key=lambda x: (
                x["arg1_step_idx"],
                x["arg2_step_idx"],
                x["arg1_start"],
                x["arg2_start"]
            )
        )

        if prefix:
            json_dict['doc_key'] = f"{self.protocol_name}_{s}"

        else:
            json_dict['doc_key'] = self.protocol_name

        assert len(json_dict['tokens']) == len(json_dict['named_entities']), \
            (
                len(json_dict['tokens']),
                len(json_dict['named_entities']),
                json_dict['doc_key']
            )
        return json.dumps(json_dict)

    def tile_to_json(self, kernel_size=16, stride=2):

        json_dicts = [
            self.to_json(i, i + kernel_size, prefix=True)
            for i in range(0, len(self.tokens2d), stride)
        ]

        return json_dicts
