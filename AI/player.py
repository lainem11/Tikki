import random
from deck import deck
import torch
from collections import deque
#from torch.distributions import Categorical
from fast_categorical import Categorical

class player:
    def __init__(self, name):
        self.name = name
        self.hand = []
        #self.bets = ["win","2-win","pass"]

    def draw_card(self, deck,n):
        for i in range(n):
            card = deck.draw()
            if card:
                self.hand.append(card)
            else:
                return None
            if len(self.hand) > 5:
                raise Exception("Illegal draw!")
            
            
    def shuffle_hand(self):
        random.shuffle(self.hand)

    def play_bet(self,Tikki):
        while 1:
            index = input("Place bet (pass, win, 2-win): ")
            match index:
                case "pass":
                    return ("pass",)
                case "win":
                    return ("win",)
                case "2-win":
                    return ("2-win",)
                case _:
                    print(f"Invalid card.")
            
    def play_card(self,Tikki):
        while 1:
            index = int(input("Enter the index of the card to play: "))-1
            if index < len(self.hand) and self.hand[index] in self.valid_cards(Tikki):
                return (self.hand.pop(index),)
            else:
                print(f"Invalid card.")
            
    def show_hand(self):
        print(f"{self.name}'s hand:")
        str = ""
        for card in self.hand:
            str = str + " " + card.suit + card.rank
        print(str)

    def valid_cards(self,Tikki):
        forced_flag = any([card.suit == Tikki.round_suit for card in self.hand])
        if not forced_flag:
            return self.hand
        return [card for card in self.hand if card.suit == Tikki.round_suit]

class random_AI(player):
    def play_bet(self, Tikki):
        return ["pass"]
    def play_card(self,Tikki):
        valids = self.valid_cards(Tikki)
        card_to_play = valids[random.randint(0,len(valids)-1)]
        return [self.hand.pop(self.hand.index(card_to_play))]
    
class AI(player):
    def __init__(self,name,net,device):
        super(AI, self).__init__(name)
        self.net = net.to(device)
        self.Deck = deck()
        self.n_outputs = 52+3
        self.device = device

        self.card_to_net_dict = dict(zip([hash(c) for c in self.Deck.cards],list(range(0,52)))) # maps card to index on the one-hot input vector
        self.net_to_card_dict = dict(zip(list(range(0,52)),[hash(c) for c in self.Deck.cards])) # maps card to index on the one-hot input vector

        # Initialize categorical distribution with sizes
        self.Cat_bets = Categorical((3,))
        self.Cat_acts = Categorical((self.n_outputs,))
        self.Cat_cards = [Categorical((1,)),Categorical((2,)),Categorical((3,)),Categorical((4,)),Categorical((5,))]

    def play_bet(self,Tikki):
        state = self.get_state(Tikki).to(self.device)
        torch.cuda.synchronize()
        action, action_index = self.choose_action(state,Tikki,"bet")
        return action, action_index, state
        # this should be only public
    
    def play_card(self,Tikki):
        state = self.get_state(Tikki).to(self.device)
        torch.cuda.synchronize()
        action = self.choose_action(state,Tikki,"card")
        card_hash = self.net_to_card_dict[action]
        index = [hash(c) for c in self.hand].index(card_hash)
        return self.hand.pop(index), action, state
    
    def choose_action(self,state,Tikki,action):
        if action == "card":
            indices = self.get_valid_net_indices(Tikki)
            with torch.no_grad():  
                probs = torch.nn.functional.softmax(self.net(state)[0,indices],dim=0)
                dist = self.Cat_cards[len(indices)-1]
                dist.set_probs(probs)
                action = dist.sample()
                # Action is the index of card to play
                return indices[action]
        else:
            # action == bet
            bets = ["pass","win","2-win"]
            with torch.no_grad():
                probs = torch.nn.functional.softmax(self.net(state)[0,-3:],dim=0)
                self.Cat_bets.set_probs(probs)
                action = self.Cat_bets.sample()
                action_index = action.item() + self.n_outputs - 3
                # Action is the index of card to play
                return bets[action.item()], action_index
    
    def get_valid_net_indices(self,Tikki):
        cards = dict(zip([hash(c) for c in self.Deck.cards],list(range(0,52)))) # maps card to index on the one-hot input vector
        valids = self.valid_cards(Tikki)
        indices = []
        for i in range(len(valids)):
            indices.append(cards[hash(valids[i])])        # list with indices corresponding to valid cards in order
        return indices
    
    def get_state(self,Tikki):
        # State as list of my cards, played moves, and scores. Each card is represented by 1,2,3,..,52. 
        # Bets as 53,54,55. No card is represented by 0. Scores from -10 to 5 by 56,..71
        cards = dict(zip([hash(c) for c in self.Deck.cards],list(range(1,53)))) # maps card to index on the one-hot input vector
        i_cards = torch.zeros((1,5),dtype=torch.long)
        for i,card in enumerate(self.hand):
            i_cards[0,i] = cards[hash(card)]

        n_p = len(Tikki.players)
        # Remove player ids
        i_hist = torch.zeros(1,n_p*(1+5),dtype=torch.long)
        input_counter = 0

        hist = Tikki.play_history
        # For each round, add history for each turn
        for r in range(len(hist)):
            for player,move in hist[r]:
                if r != 0:
                    i_hist[0,input_counter] = cards[hash(move)]
                    input_counter += 1
                else:
                    #bet = torch.zeros(1,dtype=torch.long)
                    match move:
                        case "pass":
                            bet = 53
                        case "win": 
                            bet = 54
                        case "2-win":
                            bet = 55
                    i_hist[0,input_counter] = bet
                    input_counter += 1
        score_vocab = dict(zip(list(range(-10,6,1)),list(range(56,56+16,1))))
        i_score = torch.zeros((1,n_p),dtype=torch.long)
        game_score = Tikki.player_scores
        Tikki_players = deque(Tikki.players)
        Tikki_players.rotate(-Tikki.players.index(self))
        Tikki_players = list(Tikki_players)

        for ind,plr in enumerate(Tikki_players):
            i_score[0,ind] = score_vocab[game_score[plr]]
        return torch.cat((torch.cat((i_cards,i_hist),1),i_score),1)
   