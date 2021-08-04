import html
import itertools as it
import logging
import os
from collections import namedtuple

from src import utils
from src.corpus.protocol_objects import Tag

log = logging.getLogger(__name__)

Link = namedtuple("Link", "id, name, arg1, arg2, sent_idx1, sent_idx2")


class BratFile:
    def __init__(
        self,
        text_file,
        ann_file,
        max_lines=None,
        max_tokens=None,
        process_titles=False
    ):
        self.heading, self.sents, self.ann = self.read_file(
            text_file, ann_file, max_lines, max_tokens, process_titles
        )

        if not self.is_valid():
            # dont process the rest of the brat file if its empty.
            log.debug(f"Bratfile in {text_file} not valid")
            return

        self.filename = os.path.splitext(text_file)[0]

        # parse all tags
        self.tags = [
            self.parse_tag(t.rstrip()) for t in self.ann if t[0] == 'T'
        ]

        # keep tags sorted by their start index
        self.tags = sorted(self.tags, key=lambda tag: tag.start)

        # parse all links
        e_links = list(
            it.chain.from_iterable(
                [self.parse_e(line) for line in self.ann if line[0] == 'E']
            )
        )

        r_links = [self.parse_r(line) for line in self.ann if line[0] == 'R']

        self.links = e_links + r_links

        # truncate tags and links
        if max_lines is not None:
            self.tags = [tag for tag in self.tags if tag.lineid <= max_lines]
            self.links = [
                link for link in self.links if (
                    link.arg1.lineid <= max_lines or
                    link.arg2.lineid <= max_lines
                )
            ]
        if max_tokens is not None:
            self.tags = [tag for tag in self.tags if tag.tkn_end <= max_tokens]
            self.links = [
                link for link in self.links if (
                    link.arg1.tkn_end <= max_tokens or
                    link.arg2.tkn_end <= max_tokens
                )
            ]

    def read_file(
        self,
        text_file,
        ann_file,
        max_lines=None,
        max_words=None,
        process_titles=False
    ):
        with open(text_file, mode='r', encoding='utf-8', newline='') as t_f:
            lines = [html.unescape(line) for line in t_f.readlines()]

        with open(ann_file, mode='r', encoding='utf-8', newline='') as a_f:
            ann = a_f.readlines()

            # replace e_tags in links with their tag ids.
            # Makes for easy processing later.
            ann = [self.replace_e_tags(line, ann) for line in ann]

        sents = [
            line.split(" ") for line in lines
        ]  # generate list of list of words

        heading = []
        if not process_titles:
            heading = sents[0]
            sents = sents[1:]

        if max_lines is not None:
            sents = sents[:max_lines]

        if max_words is not None:
            sents = [sent[:max_words] for sent in sents]

        # replace unknown unicode characters (eg: tm, (c), etc) with *
        sents = [
            [
                self.replace_unk(
                    word,
                    pairs={
                        '™': '*',
                        '˚': 'o',
                        '\xad': '-',
                        '✝': '*',
                        '℃': 'C',
                        'ü': 'u',
                    }
                ) for word in sent
            ] for sent in sents
        ]

        return heading, sents, ann

    def replace_unk(self, word, pairs):
        for k, v in pairs.items():
            word = word.replace(k, v)

        return word

    def is_valid(self):
        """
        Returns false if annotation file or text file is empty
        :return:
        """
        if len(self.sents) == 0:
            log.warning(f"{self.filename} is empty")
            return False
        if len(self.ann) < 1:
            log.warning(f"{self.filename} has no annotations")
            return False
        return True

    def parse_tag(self, tag):
        tag_split = tag.split('\t')

        tag_id = tag_split[0]
        tag_info = tag_split[1]

        # in case words have tab space in them.
        words = " ".join(tag_split[2:])
        tag_name, start, end = tag_info.split()
        return Tag(
            tag_id=tag_id,
            tag_name=tag_name,
            start=int(start),
            end=int(end),
            words=words,
            lineid=self.line_id(s=int(start), e=int(end))
        )

    def replace_e_tags(self, line, ann):
        """
        modifies the ann text such that:

        Exx Action:Txx Using:Exx -> Exx Action:Txx Using:Tyy

        Makes it easier to resolve tags later since Txx can be independently
        resolved, whereas Exx sometimes have forward and backward dependencies

        Args:
            line (str): a line in .ann file.
            ann (List[str]): raw lines from .ann file

        Returns:
            [type]: [description]
        """
        # For every line `line`:

        # 1. Generate args = [(Action, Txx), (Using, Exx)]
        if line[0] == 'E':
            sp_res = line.split()
            front_half = sp_res[0]
            args = [tuple(sp.split(':')) for sp in sp_res[1:]]

        elif line[0] == 'R':
            sp_res = line.split()
            r_id = sp_res[0]
            r_name = sp_res[1]
            front_half = " ".join([r_id, r_name])
            args = [tuple(sp.split(':')) for sp in sp_res[2:]]

        else:
            # nothing to replace
            return line

        # 2. (Exx -> Txx) replaced_args = [(Action, Txx), (Using, Txx)]
        replaced_args = [
            (rel_name, self.event_tag_id(tid, ann)) for rel_name, tid in args
        ]

        # 3. Stitch it back together. args_str = "Action:Txx Using:Txx"
        args_str = " ".join([":".join(item) for item in replaced_args])

        # return "Exx Action:Txx Using:Txx"
        return "\t".join([front_half, args_str])

    @staticmethod
    def event_tag_id(e_id, ann):
        if e_id[0] != "E":
            return e_id

        line = [line for line in ann if line.split("\t")[0] == e_id]
        if len(line) == 0:
            raise ValueError(f"Event link {e_id} not found")
        elif len(line) > 1:
            raise ValueError(
                f"more than 1 event link {e_id} found. Found {line}"
            )

        line = line[0]

        # action = Action:Txx
        action = line.split()[1]

        # return Txx
        return action.split(":")[1]

    def get_tag(self, tid: str):
        assert tid[0] == 'T', (
            f"can only search for tag ids starting with `T`."
            f"{tid} was passed."
        )

        ret = [tag for tag in self.tags if tag.tag_id == tid]

        assert len(ret) != 0, f"{tid} not found"

        return ret[0]

    def parse_e(self, e):
        links = []
        try:
            e_id, rels_str = e.rstrip().split("\t")
        except:
            print(f"{e} does not split on \\t")

        # rel_pairs = [(Action, Txx), (Using, Txx)]
        rel_pairs = [rel.split(":") for rel in rels_str.split(' ')]

        # arg1_tag = (Action, Txx)
        arg1_tag = self.get_tag(rel_pairs[0][1])

        links = [
            Link(
                id=e_id,
                name=utils.remove_numbers(r_name),
                arg1=arg1_tag,
                arg2=self.get_tag(arg2_id),
                sent_idx1=self.line_id_by_tag(arg1_tag),
                sent_idx2=self.line_id_by_tag(self.get_tag(arg2_id))
            ) for r_name,
            arg2_id in rel_pairs[1:]
        ]

        return links

    def parse_r(self, r):
        r_id, r_name, arg1, arg2 = r.rstrip().split()
        arg1_id = arg1.split(':')[1]
        arg2_id = arg2.split(':')[1]

        link = Link(
            id=r_id,
            name=utils.remove_numbers(r_name),
            arg1=self.get_tag(arg1_id),
            arg2=self.get_tag(arg2_id),
            sent_idx1=self.line_id_by_tag(self.get_tag(arg1_id)),
            sent_idx2=self.line_id_by_tag(self.get_tag(arg2_id))
        )
        return link

    def line_id_by_tag(self, tag):
        # find the sentence number based on tag. If not found returns None.
        s = tag.start
        e = tag.end
        assert s < e, f"poorly organized tag in {self.basename}"

        return self.line_id(s, e)

    def line_id(self, s, e):
        sent_start = len(" ".join(self.heading))
        sent_end = sent_start
        for i, sent in enumerate(self.sents):
            sent_end += len(" ".join(sent))
            if sent_start <= s and e <= sent_end:
                return i

            sent_start = sent_end

        return None

    def tag_idx_in_line(self, start, end, line_id):
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
        line_offset = len(" ".join(self.heading))

        line_offset += sum(len(" ".join(sent)) for sent in self.sents[:line_id])

        start = line_offset + start
        end = line_offset + end

        tag_idx = [
            i for i,
            tag in enumerate(self.tags)
            if self.contains((start, end), (tag.start, tag.end))
        ]

        assert len(tag_idx) <= 1, (
            f"We have overlapping tags in the BRAT file {self.filename}"
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
