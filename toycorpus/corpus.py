from typing import List
from cached_property import cached_property
import random
import numpy as np


class ToyCorpusTypes:
    """
    create a collection of sentences,
     each consisting of a string where artificial nouns are followed by a non-noun (other).
    example document: "n1 o5 n34 o82 n93 o3 n45 o11".
    the documents are sorted using either or both:
     1. the population from which nouns are sampled is gradually increased
     2. the population from which non-nouns are sampled is gradually increased

    constraint 1. result in an ordered collection of sentences,
    where the conditional entropy of nouns given the probability distribution over non-nouns,
    and joint entropy of nouns and non-nouns both are maintained.
    this happens because in the case where fewer noun types exist in a partition,
    ONLY THOSE existing noun types are used to compute entropy measures,
    and for those noun types, the next-word distribution never changes.

    to systematically decrease the conditional entropy, while maintaining constant joint entropy,
    constraint 2 must be used.
    """

    def __init__(self,
                 num_sentences: int = 100_000,
                 num_types: int = 4096,
                 increase_noun_types: bool = False,  # whether to gradually introduce new nouns
                 increase_other_types: bool = True,  # whether to gradually introduce new others
                 ) -> None:
        self.num_sentences = num_sentences
        self.num_types = num_types
        self.increase_noun_types = increase_noun_types
        self.increase_other_types = increase_other_types

        self.num_nouns = num_types // 2
        self.num_others = num_types // 2
        self.min_nouns = self.num_nouns // 2
        self.min_others = self.num_others // 2
        assert self.num_nouns + self.num_others == self.num_types

        self.nouns = [f'n{i:0>6}' for i in range(self.num_nouns)]
        self.others = [f'o{i:0>6}' for i in range(self.num_others)]

    @cached_property
    def sentences(self) -> List[str]:
        res = []
        for i in range(self.num_sentences):
            res.append(self.make_sentence(i))
        return res

    def make_sentence(self,
                      sentence_id: int,
                      ) -> str:

        # gradually increase noun population across consecutive sentences from which to sample
        if self.increase_noun_types:
            limit = self.num_nouns * (sentence_id // self.num_sentences)
            nouns = self.nouns[:int(max(self.min_nouns, limit + 1))]
        else:
            nouns = self.nouns

        # gradually increase non-noun/other population across consecutive documents from which to sample
        if self.increase_other_types:
            limit = self.num_others * (sentence_id // self.num_sentences)
            others = self.others[:int(max(self.min_others, limit + 1))]
        else:
            others = self.others

        # sample
        noun = random.choice(nouns)
        other = random.choice(others)
        res = f'{noun} {other} '  # whitespace after each

        return res


class ToyCorpusEntropic:
    """
    same as corpus above but instead of changing the number of types across sentences,
     the choice of whether a low or high entropy context type is chosen varies across sentences.
    this enables testing the hypothesis that CHILDES starts with nouns in high-entropy contexts,
     and gradually, nouns occur more frequently with low-entropy context-types.

     note: there are 2 kinds of context-types: high and low-entropy.
     a probability called "le_prob" (low-entropy probability) gradually increases with sentence_id,
     resulting in gradually more low-entropy context-types being sampled.

     note: for simplicity, a low-entropy context can only occur with one noun.
    """

    def __init__(self,
                 num_sentences: int = 100_000,
                 num_types: int = 4096,
                 le_p_start: float = 0.0,
                 le_p_end: float = 0.1,
                 ) -> None:
        self.num_sentences = num_sentences
        self.num_types = num_types

        self.num_nouns = num_types // 2
        self.num_others = num_types // 2
        self.min_nouns = self.num_nouns // 2
        self.min_others = self.num_others // 2
        assert self.num_nouns + self.num_others == self.num_types

        self.nouns = [f'n{i:0>6}' for i in range(self.num_nouns)]
        self.others = [f'o{i:0>6}' for i in range(self.num_others)]

        # divide context-types into low and high entropy
        self.others_he = self.others[:self.num_nouns // 2]
        self.others_le = self.others[self.num_nouns // 2:]

        # probabilities of choosing a low-entropy context-type, increasing with doc_id
        self.le_probabilities = np.linspace(le_p_start, le_p_end, num=self.num_sentences)

        self.noun2other_le = {n: np.random.choice(self.others_le) for n in self.nouns}

    @cached_property
    def sentences(self) -> List[str]:
        res = []
        for i in range(self.num_sentences):
            res.append(self.make_sentence(i))
        return res

    def make_sentence(self,
                      sentence_id,
                      ) -> str:

        # get probability of sampling low-entropy context-types
        le_prob = self.le_probabilities[sentence_id]

        # sample
        noun = random.choice(self.nouns)

        # sample low-entropy context
        if random.random() < le_prob:
            other = self.noun2other_le[noun]

        # sample high-entropy context
        else:
            other = random.choice(self.others_he)

        res = f'{noun} {other} '  # whitespace after each

        return res
