import logging
import re
from typing import List

from nltk import tag

from src.corpus.protocol_objects import Tag
from transformers.tokenization_auto import AutoTokenizer

log = logging.getLogger(__name__)


class Tokenizer:
    def tokenize(s):
        pass

    def vocab():
        pass


class Token(object):
    """
    Encapsulates a token/word.
    Members:
        token.word: The string content of the token, without the label.
        token.label: The label of the token.
        token.feature_values: The feature values, after they have been applied.
            (See EntityWindow.apply_features().)
    """
    def __init__(self, word, label="O", lowercase=False, replace_digits=False):
        """
        Initialize a new Token object.
        Args:
            word: The original word as found in the text document, including
            the label,
                e.g. "foo", "John/PER".
        """
        self.word = word

        if lowercase:
            self.word = self.word.lower()

        if replace_digits:
            self.word = re.sub(r'\d', '0', self.word)

        self.label = label
        self.original = word
        # self._word_ascii = None
        self.feature_values = None

    def __repr__(self):
        return "Token({0}, {1})".format(self.word, self.label)


class SplitTokenizer(Tokenizer):
    def __init__(
        self,
        vocab,
        replace_digits=True,
        lowercase=True,
        neg_class='O',
        add_special_tokens=False,
        start_tkn='[CLS]',
        end_tkn='[SEP]',
        unk_tkn='[UNK]'
    ):
        self._vocab = vocab
        self.replace_digits = replace_digits
        self.lowercase = lowercase
        self.add_special_tokens = add_special_tokens
        self.start_tkn = start_tkn
        self.end_tkn = end_tkn
        self.unk_tkn = unk_tkn
        self.neg_class = neg_class

    def tokenize(self, sents: List[str], tags: List[Tag], heading_offset: int):
        if self.lowercase:
            sents = [sent.lower() for sent in sents]

        tokens2d = []
        for i, sent in enumerate(sents):
            tokens, tags = self.make_bio(
                sent=sent,
                line_offset=heading_offset + sum(len(s) for s in sents[:i]),
                tags=tags
            )
            tokens2d.append(tokens)

        # validate that all tags have been filled with their token positions
        # assert all(tag.is_valid() for tag in tags), [
        #     tag for tag in tags if not tag.is_valid()
        # ]

        # make tkn_start and tkn_end start from the begining of the doc.

        return tokens2d, tags

    def make_bio(self, sent: str, line_offset: int, tags: List[Tag]):

        pad_start_tag = 0

        if self.add_special_tokens:
            pad_start_tag = 1

        tokenized_sent = sent.split(" ")

        start = 0
        labels = []
        prev_tag = None

        for i, ngram in enumerate(tokenized_sent):
            ngram = ngram.strip()
            start = sent.find(ngram, start)
            end = start + len(ngram)

            tag_idx = self.find_tag(tags, start, end, line_offset)

            if tag_idx is None:
                # assign negative class
                labels.append(self.neg_class)

            elif prev_tag == tags[tag_idx]:
                labels.append(f"I-{tags[tag_idx].tag_name}")
                tags[tag_idx].tkn_end = i + pad_start_tag

            else:
                labels.append(f"B-{tags[tag_idx].tag_name}")
                prev_tag = tags[tag_idx]
                tags[tag_idx].tkn_start = i + pad_start_tag
                tags[tag_idx].tkn_end = i + pad_start_tag

            start = end

        tokens = [
            Token(sub_word, label) for sub_word,
            label in zip(tokenized_sent, labels)
        ]

        if self.add_special_tokens:
            start_tkn = Token(word=self.start_tkn, label=self.neg_class)
            end_tkn = Token(word=self.end_tkn, label=self.neg_class)
            tokens = [start_tkn] + tokens + [end_tkn]

        return tokens, tags

    def find_tag(self, tags, start, end, line_offset):
        """
        Find the tag by calculating the true start and end positions using
        the `line_id`. The `start` and `end` variables will be relative to the
        current line.

        Args:
            start (int): The start position relative to the `line_id`'th line.
            end (int): The end position relative to the `line_id`'th line.
            line_id (int): The line's idx where we will find the tag.

        Returns:
            Tag: Returns the tag that encompasses the `start` and `end`
                positions in the given `line_id`.
        """

        start = line_offset + start
        end = line_offset + end

        tag_idx = [
            i for i,
            tag in enumerate(tags)
            if self.contains((start, end), (tag.start, tag.end))
        ]

        # assert len(tag_idx) <= 1, (
        #     f"We have overlapping tags {[tags[i] for i in tag_idx]}"
        # )

        return tag_idx[0] if len(tag_idx) == 1 else None

    @staticmethod
    def contains(smaller_tag, bigger_tag):
        """Is smaller_tag contained inside bigger_tag.

        Args:
            smaller_tag (Tuple[int]): (smaller) Tag's start and end positions
                that must contain inside `bigger_tag`
            bigger_tag (Tuple[int]): (bigger) Tag's start and end positions
                that contains `smaller_tag`

        Returns:
            [type]: [description]
        """
        s1, e1 = smaller_tag
        s2, e2 = bigger_tag
        if s2 <= s1 and e1 <= e2:
            return True
        elif not (s2 >= s1 and e2 >= e1 or s2 <= s1 and e2 <= e1):
            log.debug(f"partial overlap: {s1} {e1} {s2} {e2}")
            return False
        return False

    def vocab(self):
        return self._vocab


class WordPieceTokenizer(Tokenizer):
    def __init__(
        self,
        lowercase=True,
        replace_digits=True,
        neg_class='O',
        start_tkn='[CLS]',
        end_tkn='[SEP]',
        unk_tkn='[UNK]'
    ):
        self.word_piece = AutoTokenizer.from_pretrained(
            "dmis-lab/biobert-v1.1",
            do_lower_case=lowercase,
            add_special_tokens=False,
        )
        self.replace_digits = replace_digits
        self.lowercase = lowercase
        self.start_tkn = start_tkn
        self.end_tkn = end_tkn
        self.unk_tkn = unk_tkn
        self.neg_class = neg_class

    def tokenize(self, sents, tags, heading_offset):

        # get bio-tags for each token. Also infuse start and end token position
        # into each tag in `tags`

        if self.lowercase:
            sents = [sent.lower() for sent in sents]

        tokens2d = []
        for i, sent in enumerate(sents):
            tokens, tags = self.make_bio(
                sent=sent,
                line_offset=heading_offset + sum(len(s) for s in sents[:i]),
                tags=tags
            )
            tokens2d.append(tokens)

        # validate that all tags have been filled with their token positions
        # assert all(tag.is_valid() for tag in tags), [
        #     tag for tag in tags if not tag.is_valid()
        # ]

        # make tkn_start and tkn_end start from the begining of the doc.

        return tokens2d, tags

    def vocab(self):
        return self.word_piece.get_vocab()

    def make_bio(self, sent, line_offset, tags: List[Tag]):
        # returns [B-tag, I-tag, O, O, B-tag, O]
        # where the label is encoded with B, I, or O based on its position in
        # the tag

        start_tkn = Token(word=self.start_tkn, label=self.neg_class)
        end_tkn = Token(word=self.end_tkn, label=self.neg_class)

        tokenized_sent = self.word_piece.tokenize(sent)

        start = 0
        labels = []
        prev_tag = None

        for i, ngram in enumerate(tokenized_sent):
            start, end = self.find_start_end(sent, ngram, start)
            tag_idx = self.find_tag(tags, start, end, line_offset)

            if tag_idx is None:
                # assign negative class
                labels.append(self.neg_class)

            elif prev_tag == tags[tag_idx]:
                labels.append(f"I-{tags[tag_idx].tag_name}")
                tags[tag_idx].tkn_end = i + 1

            else:
                labels.append(f"B-{tags[tag_idx].tag_name}")
                prev_tag = tags[tag_idx]
                tags[tag_idx].tkn_start = i + 1
                tags[tag_idx].tkn_end = i + 1

            start = end

        tokens = [start_tkn] + [
            Token(sub_word, label) for sub_word,
            label in zip(tokenized_sent, labels)
        ] + [end_tkn]

        return tokens, tags

    def find_start_end(self, sent, token_ngram, init_start):
        """
        Given an initial starting point in the original sentence, find the
        starting position of the token_ngram


        Args:
            sent ([type]): Original sentence (untokenized)
            token_ngram ([type]): substring to look for (as tokenized
                using word_piece)
            init_start ([type]): initial starting point to search from.
        """

        # handle special cases
        if token_ngram.startswith("##"):
            token_ngram = token_ngram[2:]

        start = sent.find(token_ngram, init_start)

        # handle special cases for length

        end = start + len(token_ngram)

        return start, end

    def find_tag(self, tags, start, end, line_offset):
        """
        Find the tag by calculating the true start and end positions using
        the `line_id`. The `start` and `end` variables will be relative to the
        current line.

        Args:
            start (int): The start position relative to the `line_id`'th line.
            end (int): The end position relative to the `line_id`'th line.
            line_id (int): The line's idx where we will find the tag.

        Returns:
            Tag: Returns the tag that encompasses the `start` and `end`
                positions in the given `line_id`.
        """

        start = line_offset + start
        end = line_offset + end

        tag_idx = [
            i for i,
            tag in enumerate(tags)
            if self.contains((start, end), (tag.start, tag.end))
        ]

        #  one word case.
        if len(tag_idx) == 0:
            tag_idx = [
                i for i,
                tag in enumerate(tags)
                if self.contains((start, start + 1), (tag.start, tag.end))
            ]

        assert len(tag_idx) <= 1, (
            f"We have overlapping tags {[tags[i] for i in tag_idx]}"
        )

        return tag_idx[0] if len(tag_idx) == 1 else None

    @staticmethod
    def contains(smaller_tag, bigger_tag):
        """Is smaller_tag contained inside bigger_tag.

        Args:
            smaller_tag (Tuple[int]): (smaller) Tag's start and end positions
                that must contain inside `bigger_tag`
            bigger_tag (Tuple[int]): (bigger) Tag's start and end positions
                that contains `smaller_tag`

        Returns:
            [type]: [description]
        """
        s1, e1 = smaller_tag
        s2, e2 = bigger_tag
        if s2 <= s1 and e1 <= e2:
            return True
        elif not (s2 >= s1 and e2 >= e1 or s2 <= s1 and e2 <= e1):
            log.debug(f"partial overlap: {s1} {e1} {s2} {e2}")
            return False
        return False
