import itertools as it
import os
from typing import List, Tuple

import pandas as pd


class Tag:
    def __init__(self, idx, label, start, end, text, lineid=None):
        """
        Class for maintaining BRAT tags (Tx). str(Tag()) will generate a BRAT
        tag string.

        eg:  T16	Action 1006 1013	Extract

        Args:
            idx (int): The index of the tag
            label (str): The tag label
            start (int): The start character idx
            end (int): The end character idx
            text (str): The full span as string
        """
        self.idx = idx
        self.label = label
        self.start = start
        self.end = end
        self.text = text
        self.lineid = lineid

    def __str__(self):
        return (
            f"T{self.idx}\t{self.label} {self.start} {self.end}\t{self.text}"
        )

    def __repr__(self) -> str:
        return self.__str__()


class Elink:
    def __init__(self, idx, event_tag, rel_tag_pairs=None):
        """
        Class for maintaining BRAT event links (Ex). str(Elink()) will generate
        a BRAT event string.

        eg: E16	Action:T16 Using:T66 Using2:T67 Acts-on2:E15 Product:T65

        Args:
            idx (int):
                event link index. eg: 16

            event_tag (Tag):
                the action associated with this event.
                eg: Tag(idx=16, label="Action", ...)

            rel_tag_pairs (optional List[Tuple[str,str]]):
                list of (relation label, Tag id) pairs. For every relation the
                event_tag is the "Arg1" of, we have a pair of that relation's
                label along with the "Arg2" tag.
                eg: [(Using, Tag(66, ...)), (Using, Tag(67, ...), ...)]
        """
        self.idx = idx
        self.event_tag = event_tag

        if rel_tag_pairs:
            # make sure to rename duplicate relations
            self.rel_tag_pairs = self.make_rel_non_duplicate(rel_tag_pairs)
        else:
            self.rel_tag_pairs = []

    def make_rel_non_duplicate(self, rel_tag_pairs):
        """
        Generate non duplicate (relation, tag) pairs. Sometimes duplicate
        relations can exist for a given event. This function transforms
        relation label by appending an index if multiple relations of the same
        exact type exists.

        Args:
            rel_tag_pairs (List[Tuple[str,str]]):
                list of (relation label, str) pairs. For every relation the
                event_tag is the "Arg1" of, we have a pair of that relation's
                label along with the "Arg2" tag.
                eg: [(Using, T66), (Using, E67, ...)]

        Returns:
            rel_tag_pairs (List[Tuple[str,str]]):
                list of (relation label, str) pairs, where the relation tags
                are non duplicates to each other in the list. Multiple copies
                having the same relation are indexed.
                eg: [(Using, T66), (Using2, E67), ...)]
        """
        non_dup_rels = []
        ret = []
        for rel, tag in rel_tag_pairs:
            new_rel = rel
            i = 2

            # generate non-dup relation name
            while new_rel in non_dup_rels:
                new_rel = rel + str(i)
                i += 1

            non_dup_rels.append(new_rel)
            ret.append((new_rel, tag))

        return ret

    def append(self, rel_type, arg2):
        self.rel_tag_pairs = self.make_rel_non_duplicate(
            self.rel_tag_pairs + [(rel_type, arg2)]
        )

    def __str__(self):
        rel_pairs = " ".join(
            [f"{rel}:{tag}" for rel, tag in self.rel_tag_pairs]
        )
        return (
            f"E{self.idx}\t"
            f"{self.event_tag.label}:T{self.event_tag.idx} "
            f"{rel_pairs}"
        )


class Rlink:
    def __init__(self, idx, relation, arg1, arg2):
        """
        Class for maintaining BRAT relation links (Ex). str(Rlink()) will
        generate a BRAT relation string.

        eg: R23	Setting Arg1:E14 Arg2:T62

        Args:
            idx (int):
                relation link index. eg: 23

            relation (str):
                the relation label.
                eg: Setting

            arg1 (str):
                The first argument of the given relation as a tag id.
                eg: E14

            arg2 (str):
                The second argument of the given relation as a tag id.
                eg: T62
        """
        self.idx = idx
        self.relation = relation
        self.arg1 = arg1
        self.arg2 = arg2

    def __str__(self):

        return (
            f"R{self.idx}\t"
            f"{self.relation} Arg1:{self.arg1} Arg2:{self.arg2}"
        )


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """
    check if a chunk ended between the previous and current word
    arguments: previous and current chunk tags, previous and current types
    """

    chunk_end = False

    if prev_tag == 'E':
        chunk_end = True
    if prev_tag == 'S':
        chunk_end = True

    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    # these chunks are assumed to have length 1
    if prev_tag == ']':
        chunk_end = True
    if prev_tag == '[':
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """
    check if a chunk started between the previous and current word
    arguments: previous and current chunk tags, previous and current types
    """

    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'E' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'I':
        chunk_start = True
    if prev_tag == 'S' and tag == 'E':
        chunk_start = True
    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    # these chunks are assumed to have length 1
    if tag == '[':
        chunk_start = True
    if tag == ']':
        chunk_start = True

    return chunk_start


class BratWriter(object):
    def __init__(
        self,
        dir,
        name,
        tokens2d: List[List[str]],
        event_tag_type="Action",
        e_rels: List[str] = None,
        r_rels: List[str] = None,
    ):
        """
        Class for writing into BRAT text and ann files.

        Args:
            dir (str): The directory in which to put the .txt and .ann files.
            name (str): file name/title.
            event_tag_type (str):
                Tag type that emulates BRAT event type. An Elink will be
                generated for each tag of this type in `tags`.
            e_rels (List[str], optional):
                List of relation types that are associated with BRAT event
                type. Everytime we find such relations, we find the event that
                they are connected to, and add them to their Elink.
            r_rels (List[str], optional):
                List of relation types (specified as string) that are
                associated with BRAT relation type (R).
        """
        self.pno = name
        self.txt_fname = os.path.join(dir, name + ".txt")
        self.ann_fname = os.path.join(dir, name + ".ann")
        self.event_tag_type = event_tag_type
        self.e_rels = e_rels
        self.r_rels = r_rels
        self.tokens2d = tokens2d

        self.tags = []
        self.rels = []
        self.elinks = []
        self.text = None

        self.__start = 0
        self.__end = 0
        self.__tag_id = 1

    def clear_files(self):
        open(self.txt_fname, "w").close()
        open(self.ann_fname, "w").close()

        return self

    def isbio(self, label):
        return (
            isinstance(label, str) and
            (label[:1] == "B" or label[:1] == "I" or label[:1] == "O")
        )

    def save(self):
        with open(self.txt_fname, 'a', encoding='utf-8') as txt, \
                open(self.ann_fname, 'a', encoding='utf-8') as ann_file:
            # write the text file
            txt.write("".join(self.text))

            # write the entity tags and the relations into the .ann file.
            ann_file.write("\n".join(str(tag) for tag in self.tags))
            ann_file.write("\n")
            ann_file.write("\n".join(str(rel) for rel in self.rels))

    def get_tag_id(self, tag: Tag):
        if tag.label == "Action":
            elink = self.get_elink(tag)
            if elink:
                return f"E{elink.idx}"
        else:
            return f"T{tag.idx}"

    def get_elink(self, tag: Tag):
        for elink in self.elinks:
            if elink.event_tag == tag:
                return elink

        return None

    def generate_rels(self, relations: List[Tuple[Tag, Tag, str]]):
        """
        Generate relations in the form of Rlinks and Elink objects.
        Populate those objects appropriately, and returns a list of Rlink and
        Elink objects.

        Args:
            relations (List[Tuple[Tag, Tag, str]]):
                List composted of a tuple containing arg1 and arg2 Tags along
                with the relation type as string.

                eg: [
                        (Tag(... start=0, end=5, ...), "Action"),
                        (Tag(... start=15, end=20, ...), "Reagent),
                        ...
                    ]
        """
        rlinks = []

        # first establish all the elinks,
        # cos they will be needed to make rlinks.
        for arg1, arg2, rel_type in relations:
            if arg1 is None or arg2 is None:
                continue

            if rel_type in self.e_rels and arg1.label == self.event_tag_type:
                # update e_links by
                # either finding the appropiate event to add the relation to,
                # or create a new entry in e_links.
                self.update_elinks(rel_type, arg1, arg2)

        # Then establish rlinks.
        for arg1, arg2, rel_type in relations:
            if arg1 is None or arg2 is None:
                continue
            if rel_type in self.r_rels:
                rlinks.append(
                    Rlink(
                        idx=len(rlinks) + 1,
                        relation=rel_type,
                        arg1=self.get_tag_id(arg1),
                        arg2=self.get_tag_id(arg2)
                    )
                )

        self.rels = self.elinks + rlinks

        # enables daisy chaining function calls
        return self

    def update_elinks(self, rel_type, arg1: Tag, arg2: Tag):
        """
        Update the appropriate link in `elinks` with a new relation if
        event string (Ex) has already been created
        or create a new Event tag (Ex) and append it to elinks.

        Args:
            rel_type (str): The relation type of the new relation being added
            arg1 (Tag): Tag object for the first argument
            arg2 (Tag): Tag object for the second argument
            elinks (List[Elink]): List of elink objects generated so far.
                This object is where the relation will be appended.

        Returns:
            (List[Elink]): The updated list of Elink objects, now containing
                the new relation, either as part of an existing elink object
                or as a new elink object (if it was never created for the
                event in arg1)
        """

        elink = self.get_elink(tag=arg1)

        if elink:
            elink.append(rel_type, self.get_tag_id(arg2))
        else:
            # link not found, the relationship is newly added
            self.elinks.append(
                Elink(
                    idx=len(self.elinks) + 1,
                    event_tag=arg1,
                    rel_tag_pairs=[(rel_type, self.get_tag_id(arg2))]
                )
            )

    def get_tag(self, tkn_start, tkn_end, line_id):
        """
        Using the start and end char index, find the tag.
        Returns none if not found.

        Args:
            start (int): The start character index
            end (int): The end character index

        Returns:
            tag (Tag):
        """
        start = sum(len(" ".join(tokens)) for tokens in self.tokens2d[:line_id])
        start += line_id  # one for each newline char
        tokens = self.tokens2d[line_id]
        start += len(" ".join(tokens[:tkn_start]))

        # if its the first token in the sentence, dont add
        # otherwise add 1 to account for the space.
        start += 0 if tkn_start == 0 else 1
        end = start + len(" ".join(tokens[tkn_start:tkn_end + 1]))

        for tag in self.tags:
            if start == tag.start and end == tag.end:
                return tag

        # do +-1 jugad
        for tag in self.tags:
            if (
                (start - 1 == tag.start and end - 1 == tag.end) or
                (start + 1 == tag.start and end + 1 == tag.end)
            ):
                return tag

        return None

    def add_title(self, title):
        self.title = " ".join(title) + "\n"
        self.__start = len(self.title)
        self.__end = len(self.title)

        return self

    def generate_tags_bio(self, bio_labels: List[str]):
        """
        Generate a list of Tag objects using list of words in `words` and a
        list of BIO- tagged labels in `bio_labels`.

        eg: T7	Location 386 392	bottle

        Args:
            sents (List[List[str]]): List of sentences.
                Each sentence is a list of words (no newline).
            bio_labels (List[str]): List of bio tagged labels.

        Returns:
            tags (List[Tag]):
                Returns a list of Tag objects, ordered by their index.
        """
        word_span = []
        tags = []
        start = self.__start
        end = self.__end
        tag_id = self.__tag_id
        tag_span = []

        words = it.chain.from_iterable(self.tokens2d)

        for i, word in enumerate(words):
            prev_tag, prev_label = self.__split_tag_label(
                bio_labels[i - 1] if i - 1 >= 0 else 'O'
            )
            tag, label = self.__split_tag_label(bio_labels[i])
            next_tag, next_label = self.__split_tag_label(
                bio_labels[i + 1] if i + 1 < len(bio_labels) else 'O'
            )
            if start_of_chunk(prev_tag, tag, prev_label, label):
                word_span = []
                tag_span = []
                start = end

            word_span.append(word)
            tag_span.append(tag)
            end = end + len(word)

            if (
                end_of_chunk(tag, next_tag, label, next_label) and label != 'O'
            ):

                tags.append(
                    Tag(
                        idx=tag_id,
                        label=label,
                        start=start,
                        end=end,
                        text=" ".join(word_span),
                    )
                )
                if label == "Action":
                    # create elink
                    self.elinks.append(
                        Elink(len(self.elinks) + 1, event_tag=tags[-1])
                    )
                tag_id += 1
                start = end + 1  # account for space char
                word_span = []
                tag_span = []

            end += 1  # space char

        self.__start = start
        self.__end = end
        self.__tag_id = tag_id

        self.tags = tags
        self.text = [" ".join(sent) + "\n" for sent in self.tokens2d]

        # enables daisy chaining functions calls
        return self

    def generate_tags(
        self, spans: List[Tuple[int, int, int]], labels: List[str]
    ):
        tags = []
        for (start, end, step_idx), label in zip(spans, labels):
            start, end, step_idx = int(start), int(end), int(step_idx)
            offset = sum(
                len(" ".join(tokens1d)) + 1  # sentence length + `\n`
                for tokens1d in self.tokens2d[:step_idx]
            )
            sent = self.tokens2d[step_idx]

            ch_start = offset + len(" ".join(sent[:start])) + (
                1 if start > 0 else 0  # add a space if its not the first token
            )

            ch_end = offset + len(" ".join(sent[:end + 1]))
            word_span = " ".join(sent[start:end + 1])
            tags.append(
                Tag(
                    idx=self.__tag_id,
                    label=label,
                    start=ch_start,
                    end=ch_end,
                    text=word_span,
                )
            )
            if label == "Action":
                # create elink
                self.elinks.append(
                    Elink(len(self.elinks) + 1, event_tag=tags[-1])
                )
            self.__tag_id += 1

        self.tags = tags
        self.text = [" ".join(sent) + "\n" for sent in self.tokens2d]

        return self

    @staticmethod
    def __split_tag_label(label):
        if len(label) < 2:
            return label, label

        return label[:1], label[2:]

    @staticmethod
    def __partition(a_list, lengths):
        i = 0
        ret = []
        for le in lengths:
            ret.append(a_list[i:i + le])
            i = i + le
        return ret


def write_file(
    out_dir: str,
    name: str,
    tokens: List[List[str]],
    ents: pd.DataFrame,
    rels: pd.DataFrame
):

    os.makedirs(out_dir, exist_ok=True)
    # ent spans to bio tags
    spans = list(ents[['start', 'end', 'step_idx']].to_records(index=False))
    labels = ents['label'].values.tolist()

    bfile = BratWriter(
        dir=out_dir,
        name=name,
        tokens2d=tokens,
        e_rels=[
            "Acts-on", "Site", "Product", "Using", "Count", "Measure-Type-Link"
        ],
        r_rels=[
            "Coreference-Link",
            "Mod-Link",
            "Setting",
            "Measure",
            "Meronym",
            "Or",
            "Of-Type",
            "Enables",
            "Prevents",
            "Overlaps"
        ]
    ).generate_tags(spans=spans, labels=labels)

    bfile.generate_rels(
        relations=[
            (bfile.get_tag(*rel.span1), bfile.get_tag(*rel.span2), rel.label)
            for _,
            rel in rels.iterrows()
        ]
    ).clear_files().save()
