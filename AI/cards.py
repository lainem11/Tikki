

class cards:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return f"{self.rank}{self.suit}"
    
    def __hash__(self):
        return hash((self.suit,self.rank))



