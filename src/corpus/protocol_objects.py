from collections import namedtuple

import nltk

Entity = namedtuple('Entity', ['label', 'start', 'end', 'step_idx'])


class Tag(object):
    def __init__(self, tag_id, tag_name, start, end, words, lineid):
        self._tag_id = tag_id
        self._tag_name = tag_name
        self._start = start
        self._end = end
        self._words = words
        self.tkn_start = None
        self.tkn_end = None
        self.sent_start = 0
        self._lineid = lineid

    def is_valid(self):
        return self.tkn_start is not None and self.tkn_end is not None

    @property
    def tag_id(self):
        return self._tag_id

    @property
    def tag_name(self):
        return self._tag_name

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def words(self):
        return self._words

    @property
    def lineid(self):
        return self._lineid

    def __equal(self, other):
        return isinstance(other, Tag) and \
            self.tag_name == other.tag_name and \
            self.start == other.start and \
            self.end == other.end and \
            " ".join(self.words) == " ".join(other.words) and \
            self.lineid == self.lineid

    def __eq__(self, other):
        return self.__equal(other)

    def __ne__(self, other):
        return not self.__equal(other)

    def __hash__(self):
        return hash(
            (
                self.tag_name,
                self.start,
                self.end,
                " ".join(self.words),
                self.lineid
            )
        )

    def __repr__(self):
        return (
            f"Tag ("
            f"tag_id={self.tag_id}, "
            f"tag_name={self.tag_name}, "
            f"start={self.start}, "
            f"end={self.end}, "
            f"words={self.words}, "
            f"lineid={self.lineid}, "
            f"tkn_start={self.tkn_start}, "
            f"tkn_end={self.tkn_end})"
        )


class Relation:
    def __init__(self, arg1: Entity, arg2: Entity, label):
        self._arg1 = arg1
        self._arg2 = arg2
        self._label = label
        self.is_iap = not self.iscap()

    def iscap(self):
        if self.label == "Acts-on":
            return (
                self.arg1.step_idx != self.arg2.step_idx or
                self.arg2.label == "Action"
            )
        elif self.label == "Site":
            return (
                self.arg1.step_idx != self.arg2.step_idx or
                self.arg2.label == "Action"
            )
        elif self.label == "Or":
            return (
                self.arg1.step_idx != self.arg2.step_idx or
                self.arg1.label == "Action" or self.arg2.label == "Action"
            )
        elif self.label in [
            "Enables", "Prevents", "Overlaps", "Coreference-Link", "Product"
        ]:
            return True

        return False

    @property
    def arg1(self):
        return self._arg1

    @property
    def arg2(self):
        return self._arg2

    @property
    def label(self):
        return self._label

    def __eq__(self, other):
        return self.arg1 == other.arg1 and self.arg2 == other.arg2

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.arg1, self.arg2))

    def __repr__(self):
        return "Relation({arg1}, {arg2}, '{label}')".format(
            arg1=self.arg1, arg2=self.arg2, label=self.label
        )


class BratRelation:
    def __init__(self, arg1, arg2, label=None, p_name=None, parent_doc=None):
        """
        Relation class is used to generate features for a given relation.
        Each relation acts as a datapoint for the logistic/maxent model.

        :param arg1: Tag representing argument 1 of the relation.
            Typically the predicate in a traditional SRL sense

        :param arg2: Tag representing argument 2 of the relation.
        :param label: String of the relation label or tag. eg: Arg0, or Acts-on
        """
        self._arg1 = arg1
        self._arg2 = arg2
        self._label = label
        self._p_name = p_name
        self._p_doc = parent_doc

        if parent_doc:
            self.tokens2d = parent_doc.tokens2d
            self.pos_tags = parent_doc.pos_tags
            self.conll_deps = parent_doc.conll_deps

        self.feature_dict = None

        # type checks
        assert isinstance(self._arg1, Tag)
        assert isinstance(self._arg2, Tag)
        assert isinstance(self._label, str)

    @property
    def arg1(self):
        return self._arg1

    @property
    def arg2(self):
        return self._arg2

    @property
    def label(self):
        return self._label

    @property
    def p_name(self):
        return self._p_name

    @property
    def p_doc(self):
        return self._p_doc

    def __equal(self, other):
        return isinstance(other, BratRelation) and \
            self.arg1 == other.arg1 and \
            self.arg2 == other.arg2 and \
            self.label == other.label and \
            self.p_name == other.p_name

    def __eq__(self, other):
        return self.__equal(other)

    def __ne__(self, other):
        return not self.__equal(other)

    def __hash__(self):
        return hash((self.arg1, self.arg2, self.label, self.p_name))

    def __str__(self):
        return (
            "Relation: \n "
            "0. In {p_name}\n"
            "1. Between (Tag1: {arg1}, Tag2: {arg2}) \n "
            "2. Label {label} \n "
            "3. sents: (arg1_sent: {lineid1}, arg2_sent: {lineid2})".format(
                p_name=self.p_name,
                arg1=self.arg1,
                arg2=self.arg2,
                label=self.label,
                lineid1=self.arg1.lineid,
                lineid2=self.arg2.lineid
            )
        )

    def __repr__(self):
        return self.__str__()

    def get_isr_gold(self):
        return [
            link
            for link in self.p_doc.links if link.arg1.lineid == link.arg2.lineid
        ]

    def _get_all_isr(self, arg):
        isr_gold = self.get_isr_gold()

        return [
            link for link in isr_gold if link.arg1 == arg or link.arg2 == arg
        ]

    def get_nearest_isr(self, n, arg):
        def __dist(link):
            if self.__is_1_before_2(link.arg1, link.arg2):
                return link.arg2.tkn_start - link.arg1.tkn_end
            else:
                return link.arg1.tkn_start - link.arg2.tkn_end

        near_isr_links = self._get_all_isr(arg)

        sorted_isr_links = sorted(near_isr_links, key=__dist, reverse=True)
        if n - 1 < len(sorted_isr_links):
            return sorted_isr_links[n - 1]
        else:
            return None

    def sameNP(self):
        if self.arg1.lineid == self.arg2.lineid:
            c_type = self.__is_same_chunk()
            return c_type == "NP"

        return False

    def sameVP(self):
        if self.arg1.lineid == self.arg2.lineid:
            c_type = self.__is_same_chunk()
            return c_type == "VP"

        return False

    def samePP(self):
        if self.arg1.lineid == self.arg2.lineid:
            c_type = self.__is_same_chunk()
            return c_type == "PP"

        return False

    def arg1_deps(self):
        return self.__arg_deps(self.arg1)

    def arg2_deps(self):
        return self.__arg_deps(self.arg2)

    def get_arg1_tokens(self):
        return self.__get_tokens(self.arg1)

    def get_arg2_tokens(self):
        return self.__get_tokens(self.arg2)

    def is_1_before_2(self):
        return self.__is_1_before_2(self.arg1, self.arg2)

    def __is_1_before_2(self, arg1, arg2):
        if arg1.lineid < arg2.lineid:
            return True
        elif self.arg1.lineid == self.arg2.lineid:
            if self.arg1.tkn_end < self.arg2.tkn_start:
                return True

        return False

    def count_sent_bet(self):
        return abs(self.arg1.lineid - self.arg2.lineid)

    def get_tokens_bet(self):
        if self.is_1_before_2():
            return self.__get_bet(self.tokens2d, self.arg1, self.arg2)
        else:
            return self.__get_bet(self.tokens2d, self.arg2, self.arg1)

    def get_b_tokens(self, no):
        return self.__get_b(self.tokens2d, no)

    def get_a_tokens(self, no):
        return self.__get_a(self.tokens2d, no)

    def get_bet_chunks(self):
        if self.is_1_before_2():
            pos = self.__get_bet(self.pos_tags, self.arg1, self.arg2)
        else:
            pos = self.__get_bet(self.pos_tags, self.arg2, self.arg1)

        return [p[2] for p in pos]

    def get_b_chunks(self, no):
        pos = self.__get_b(self.pos_tags, no)
        return [p[2] for p in pos]

    def get_a_chunks(self, no):
        pos = self.__get_a(self.pos_tags, no)
        return [p[2] for p in pos]

    @staticmethod
    def __get_bet(list2d, arg1, arg2):
        ret = []
        sent_idx1 = arg1.lineid
        sent_idx2 = arg2.lineid

        if sent_idx1 != sent_idx2:
            ret.extend(list2d[sent_idx1][arg1.tkn_end:])
            for idx in range(sent_idx1 + 1, sent_idx2 - 1):
                ret.extend(list2d[idx][:])

            ret.extend(list2d[sent_idx2][:arg2.tkn_start])
        else:
            ret.extend(list2d[sent_idx1][arg1.tkn_end:arg2.tkn_start])

        return ret

    def __get_b(self, list2d, no):

        if self.is_1_before_2():
            return list2d[self.arg1.lineid][self.arg1.tkn_start -
                                            no:self.arg1.tkn_start]
        else:
            return list2d[self.arg2.lineid][self.arg2.tkn_start -
                                            no:self.arg2.tkn_start]

    def __get_a(self, list2d, no):

        if self.is_1_before_2():
            return list2d[self.arg2.lineid
                         ][self.arg2.tkn_end:self.arg2.tkn_end + no]
        else:
            return list2d[self.arg1.lineid
                         ][self.arg1.tkn_end:self.arg1.tkn_end + no]

    def __arg_deps(self, arg, rel=None):
        deps = [
            nltk.DependencyGraph(conll_dep, top_relation_label='root')
            for conll_dep in self.conll_deps
        ]
        ret = None

        def get_deps(triples, word):
            dep = (0, 0)
            for g, r, d in triples:
                if g[0] == word and rel == r:
                    return d

            return dep

        try:
            dep_graph = deps[arg.lineid]
            t = dep_graph.triples()
            tokens = self.__get_tokens(arg)
            ret = [get_deps(t, token.word) for token in tokens]
        except IndexError as e:
            print(e)
            print(self.conll_deps)

        return ret

    @staticmethod
    def get_dep(word, dep_graph, rel):
        triples = dep_graph.triples()
        ret = None
        for g, r, d in triples:
            # print(g, r, d)
            if g[0] == word and rel == r:
                return d[0]

        return ret

    def __get_tokens(self, arg):
        return self.tokens2d[arg.lineid][arg.tkn_start:arg.tkn_end + 1]

    def is_copular(self, subj=None, obj=None):
        # we dont want to save this for every relation
        dep_graphs = [
            nltk.DependencyGraph(conll_dep, top_relation_label='root')
            for conll_dep in self.conll_deps
        ]
        if subj is None and obj is None:
            raise AttributeError("cannot have both subj and obj be None")

        if obj is not None:
            # cop_objs is a list of words from `obj`
            # that are objects of a copular
            dep_graph = dep_graphs[obj.lineid]
            obj_tokens = self.__get_tokens(obj)
            if " ".join([token.word
                         for token in obj_tokens]) != " ".join(obj.words):
                print(self, obj_tokens, self.tokens2d[obj.lineid], obj)

                obj_tokens = obj_tokens[:-1]

            cop_objs = [
                token for token in obj_tokens
                if self.get_dep(token.word, dep_graph, rel='cop')
            ]

        else:
            # cop_objs is a list of all words in the subj's sent.
            # that are objects of copular.
            # This list will be used to identify all subjects of these cop_objs
            dep_graph = dep_graphs[subj.lineid]
            cop_objs = [
                token for token in self.tokens2d[subj.lineid]
                if self.get_dep(token.word, dep_graph, rel='cop')
            ]

        subjs = [self.get_dep(o.word, dep_graph, rel='nsubj') for o in cop_objs]

        if subj is not None:
            return len(set(subjs) & set(subj.words)) > 0
        else:
            return len(cop_objs) != 0

    def __is_same_chunk(self):
        # checks if arg1 and arg2 are in the same chunk.
        # If so, it will return the chunk type.

        if self.is_1_before_2():
            pos = self.pos_tags[self.arg1.lineid
                               ][self.arg1.tkn_start:self.arg2.tkn_end]
        else:
            pos = self.pos_tags[self.arg1.lineid
                               ][self.arg2.tkn_start:self.arg1.tkn_end]

        if not pos:
            return False

        start_chunk = pos[0][2]
        if start_chunk != 'O':
            start_chunk = start_chunk[2:]
        for p in pos:
            if p[2][0] == 'O' or p[2][0] == 'B':
                return False

        return start_chunk
