#!/usr/bin/env python3

import argparse
import collections
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from typing import List, Optional


# use argparse to specify input file path and output dir


parser = argparse.ArgumentParser(description='Create equity vs random plots')
parser.add_argument('--input-file', '-i', type=str, help='path to equity file')
parser.add_argument('--output-dir', '-o', type=str, default='.', help='output directory')
args = parser.parse_args()

input_file = args.input_file
output_dir = args.output_dir


if not input_file:
    print(f'Required argument --input-file/-i is missing')
    sys.exit(1)

if not output_dir:
    print(f'Required argument --output-dir/-o is missing')
    sys.exit(1)

os.makedirs(output_dir, exist_ok=True)

with open(input_file, 'r') as f:
    lines = f.readlines()


Rank = int  # 0-12, deuce=0, three=1, ..., king=11, ace=12
Suit = int  # 0-3, cdhs
Card = int  # 4*rank + suit
SuitFlavor = str  # ssss, sss, ss, ds, 'r'


RANKS = '23456789TJQKA'
REVERSE_RANKS = {s:i for (i, s) in enumerate(RANKS)}


SUITS = 'cdhs'
REVERSE_SUITS = {s:i for (i, s) in enumerate(SUITS)}


SUIT_FLAVORS = ('ds', 'ss', 'sss', 'ssss', 'r')
GOOD_SUIT_FLAVORS = ('ds', 'ss', 'sss', 'r')


def rank_repr(rank: Rank) -> str:
    return RANKS[rank]


def parse_rank(s: str) -> Rank:
    return REVERSE_RANKS[s]


def suit_repr(suit: Suit) -> str:
    return SUITS[suit]


def parse_suit(s: str) -> Suit:
    return REVERSE_SUITS[s]


def extract_rank(card: Card) -> Rank:
    return card // 4


def extract_suit(card: Card) -> Suit:
    return card % 4


def to_card(rank: Rank, suit: Suit) -> Card:
    return 4 * rank + suit


def card_repr(card: Card) -> str:
    return rank_repr(extract_rank(card)) + suit_repr(extract_suit(card))


def parse_card(s: str) -> Card:
    return to_card(parse_rank(s[0]), parse_suit(s[1]))


def permute_suit(card: Card, permutation) -> Card:
    return to_card(extract_rank(card), permutation[extract_suit(card)])


class Hand:
    def __init__(self, cards: List[Card], equity_vs_random: float):
        self.cards = tuple(reversed(sorted(cards)))  # high-to-low
        self.rank_counts = np.zeros(13, dtype=int)
        self.suit_counts = np.zeros(4, dtype=int)
        for c in cards:
            self.rank_counts[extract_rank(c)] += 1
            self.suit_counts[extract_suit(c)] += 1

        self.top_paired_rank = self._top_paired_rank()
        self.best_non_top_paired_rank = self._best_non_top_paired_rank()
        self.hand_width = self._hand_width()
        self.width_tightest_3_cluster = self._width_tightest_3_cluster()
        self.top_gap = self._top_gap()
        self.max_gap = self._max_gap()
        self.suit_flavor = self._suit_flavor()
        self.multiplicity = self._multiplicity()
        self.canonical_repr = self._canonical_repr()
        self.equity_vs_random = equity_vs_random

        assert self.suit_flavor in SUIT_FLAVORS, self
        assert 0 < self.equity_vs_random < 1, (self, self.equity_vs_random)

    @staticmethod
    def parse(s: str):
        # 'AcAdAhQs 0.56615\n'
        tokens = s.split()
        cards = [parse_card(tokens[0][i:i+2]) for i in (0, 2, 4, 6)]
        equity_vs_random = float(tokens[1])
        return Hand(cards, equity_vs_random)

    def __str__(self):
        return ''.join(card_repr(c) for c in self.cards)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.cards == other.cards

    def _top_paired_rank(self) -> Optional[Rank]:
        for i, k in reversed(list(enumerate(self.rank_counts))):
            if k >= 2:
                return i
        return None

    def _best_non_top_paired_rank(self) -> Optional[Rank]:
        for i, k in reversed(list(enumerate(self.rank_counts))):
            if k and i != self.top_paired_rank:
                return i
        return None

    def _hand_width(self) -> int:
        # Rank-gap between high and low cards
        return extract_rank(self.cards[0]) - extract_rank(self.cards[3])

    def _width_tightest_3_cluster(self) -> int:
        # Tightest rank-gap achievable by removing a card
        width1 = extract_rank(self.cards[0]) - extract_rank(self.cards[2])
        width2 = extract_rank(self.cards[1]) - extract_rank(self.cards[3])
        return min(width1, width2)

    def _top_gap(self) -> Optional[int]:
        # Rank-gap between 2 highest distinct ranks
        ranks = [i for (i, k) in enumerate(self.rank_counts) if k]
        if len(ranks) == 1:
            return None
        return ranks[-1] - ranks[-2]

    def _max_gap(self) -> Optional[int]:
        # Largest rank-gap between 2 successive distinct ranks
        ranks = [i for (i, k) in enumerate(self.rank_counts) if k]
        if len(ranks) == 1:
            return None
        gaps = []
        for r in range(len(ranks) - 1):
            gaps.append(ranks[r+1] - ranks[r])
        return max(gaps)

    def _suit_flavor(self) -> SuitFlavor:
        counts = collections.defaultdict(int)
        multiplicities = [k for k in self.suit_counts if k]
        for m in multiplicities:
            counts[m] += 1

        assert sum(k*v for k, v in counts.items()) == 4, self

        c4 = counts[4]
        c3 = counts[3]
        c2 = counts[2]
        c1 = counts[1]
        if c4 == 1:
            return 'ssss'
        if c3 == 1:
            return 'sss'
        if c2 == 2:
            return 'ds'
        if c2 == 1:
            return 'ss'
#             # if the top rank is suited, ss+. Else, ss-
#             top_rank = max(r for r in (self.top_paired_rank, self.best_non_top_paired_rank) if r is not None)
#             for c in self.cards:  # cards are ordered from best rank to worst
#                 if extract_rank(c) != top_rank:
#                     continue
#                 s = extract_suit(c)
#                 if self.suit_counts[s] >= 2:
#                     return 'ss+'
#             return 'ss-'
        assert c1 == 4, self
        return 'r'

    def _multiplicity(self) -> int:
        s = set()
        for p in itertools.permutations([0,1,2,3]):
            cards = tuple(sorted(permute_suit(c, p) for c in self.cards))
            s.add(cards)
        return len(s)

    def _canonical_repr(self) -> str:
        strs = []
        for c in self.cards:
            cr = card_repr(c)
            s = extract_suit(c)
            if self.suit_counts[s] == 1:
                strs.append(cr[0])
            else:
                strs.append(cr)
        return ''.join(strs)
#         return ''.join(rank_repr(extract_rank(c)) for c in self.cards) + self.suit_flavor


ALL_HANDS = [Hand.parse(s) for s in lines]


def plot_equity_curve(ax, hands: List[Hand], ylabel: Optional[str], title: Optional[str]):
    if not hands:
        ax.axis('off')
        ax.set_frame_on(False)
        return

    hands.sort(key=lambda h: h.equity_vs_random)
    m = np.array([h.multiplicity for h in hands], dtype=float)
    m_sum = sum(m)
    cm = np.cumsum(m)
    cm /= m_sum

    eq = np.array([h.equity_vs_random for h in hands])

    x = np.concatenate(([0], cm))
    y = 100 * np.concatenate((eq[:1], eq))

    ax.step(x, y, color='red', linewidth=0.75)
    ax.axhline(40, color='gray', linewidth=0.5, linestyle=':')
    ax.axhline(50, color='green', linewidth=0.5, linestyle='--')
    ax.axhline(60, color='gray', linewidth=0.5, linestyle=':')

    if y[0] < 50 and y[-1] > 50:
        # Find the intersection point and draw vertical dashed line
        idx = np.searchsorted(y, 50) - 1
        ax.axvline(x[idx], color='red', linewidth=0.5, linestyle='--')

        horizontal_alignment = 'right' if x[idx] > 0.5 else 'left'
        text = hands[idx].canonical_repr + f'\n{x[idx]*100:.1f}%'
        ax.text(x[idx], 0.05, text, ha=horizontal_alignment, va='bottom')

        ax.axvspan(0, x[idx], facecolor='pink', alpha=0.5)
        ax.axvspan(x[idx], 1, facecolor='lightblue', alpha=0.5)
    elif y[-1] <= 50:
        ax.axvspan(0, 1, facecolor='pink', alpha=0.5)
    else:
        ax.axvspan(0, 1, facecolor='lightblue', alpha=0.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)

    ax.set_xticks([])
    if ylabel:
        ax.set_ylabel(ylabel)
        ax.set_yticks([40, 50, 60])
    else:
        ax.set_yticks([])

    if title:
        ax.set_title(title)


def plot_paired_hands(top_paired_rank: Rank):
    pair_str = rank_repr(top_paired_rank) * 2
    hands = [h for h in ALL_HANDS if h.top_paired_rank == top_paired_rank]
    if not hands:
        return
    non_quad_hands = [h for h in hands if h.best_non_top_paired_rank is not None]
    hands_by_flavor = collections.defaultdict(list)
    for h in hands:
        hands_by_flavor[h.suit_flavor].append(h)

    NUM_ROWS = 5
    fig, axes = plt.subplots(NUM_ROWS, len(GOOD_SUIT_FLAVORS), figsize=(8, 10))
    for f, flavor in enumerate(GOOD_SUIT_FLAVORS):
        ax = axes[0, f]
        ylabel = None if f else ('all ' + pair_str)
        plot_equity_curve(ax, hands_by_flavor[flavor], ylabel, flavor)

    row = 0
    for rank in reversed(range(13)):
        if rank == top_paired_rank:
            continue

        row += 1
        last = row + 1 == NUM_ROWS
        if last:
            subhands = [h for h in non_quad_hands if h.best_non_top_paired_rank <= rank]
            descr = 'other ' + pair_str
        else:
            subhands = [h for h in non_quad_hands if h.best_non_top_paired_rank == rank]

            kicker_str = rank_repr(rank)
            if rank > top_paired_rank:
                descr = kicker_str + pair_str + 'x'
            else:
                descr = pair_str + kicker_str + 'x'

        subhands_by_flavor = collections.defaultdict(list)
        for h in subhands:
            subhands_by_flavor[h.suit_flavor].append(h)

        for f, flavor in enumerate(GOOD_SUIT_FLAVORS):
            ax = axes[row, f]
            ylabel = None if f else descr
            plot_equity_curve(ax, subhands_by_flavor[flavor], ylabel, None)

        if last:
            break

    fig.suptitle(f'{pair_str} hands')
    plt.tight_layout()
    output_filename = os.path.join(output_dir, f'paired-{top_paired_rank+2:02d}-{pair_str}xx.png')
    fig.savefig(output_filename, dpi=300)
    plt.close(fig)
    print(f'Wrote {output_filename}')


def plot_unpaired_hands(top_rank: Rank):
    rank_str = rank_repr(top_rank)
    hands = [h for h in ALL_HANDS if h.top_paired_rank is None and h.best_non_top_paired_rank == top_rank]
    if not hands:
        return
    hands_by_flavor = collections.defaultdict(list)
    for h in hands:
        hands_by_flavor[h.suit_flavor].append(h)

    NUM_ROWS = 5
    fig, axes = plt.subplots(NUM_ROWS, len(GOOD_SUIT_FLAVORS), figsize=(8, 10))
    for f, flavor in enumerate(GOOD_SUIT_FLAVORS):
        ax = axes[0, f]
        ylabel = None if f else (f'all {rank_str}-high')
        plot_equity_curve(ax, hands_by_flavor[flavor], ylabel, flavor)

    def helper(row, descr, subhands):
        subhands_by_flavor = collections.defaultdict(list)
        for h in subhands:
            subhands_by_flavor[h.suit_flavor].append(h)

        for f, flavor in enumerate(GOOD_SUIT_FLAVORS):
            ax = axes[row, f]
            ylabel = None if f else descr
            plot_equity_curve(ax, subhands_by_flavor[flavor], ylabel, None)

    other_hands = hands

    subhands = [h for h in other_hands if h.hand_width <= 4]
    other_hands = [h for h in other_hands if h.hand_width > 4]
    helper(1, 'w<=4', subhands)

    subhands = [h for h in other_hands if h.hand_width <= 5]
    other_hands = [h for h in other_hands if h.hand_width > 5]
    helper(2, 'w==5', subhands)

    subhands = [h for h in other_hands if h.width_tightest_3_cluster <= 3]
    other_hands = [h for h in other_hands if h.width_tightest_3_cluster > 3]
    helper(3, 'w>=6 & sw<=3', subhands)

    helper(4, 'w>=6 & sw>=4', other_hands)

    fig.suptitle(f'{rank_str}-high hands')
    plt.tight_layout()
    output_filename = os.path.join(output_dir, f'unpaired-{top_rank+2:02d}-{rank_str}-high.png')
    fig.savefig(output_filename, dpi=300)
    plt.close(fig)
    print(f'Wrote {output_filename}')


for rank in range(13):
    plot_paired_hands(rank)
    plot_unpaired_hands(rank)
