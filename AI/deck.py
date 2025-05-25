import random
from cards import cards

class deck:
    def __init__(self):
        suits = ["♥", "♦", "♣", "♠"]
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10","11", "12", "13", "14"]
        self.cards = [cards(suit, rank) for rank in ranks for suit in suits]

    def shuffle(self,seed):
        random.Random(seed).shuffle(self.cards)
        #random.shuffle(self.cards)

    def draw(self):
        if not self.cards:
            return None
        return self.cards.pop()