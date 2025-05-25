from deck import deck
from collections import defaultdict
import random
import time
from player import AI
import copy

class Tikki:
    # Example usage
    def __init__(self,players):
        self.players = players
        self.max_rounds = 5     # in practice +1 because of betting round
        self.max_score = 5
        self.min_score = -5
        self.n_players = len(self.players)
        self.max_game_length = 6*20
        self.game_length = 0
        # round = each player takes one turn
        # match = each player plays all moves, scores come at end of every match
        # game = players play matches until someone gets 5 points

        # Decide decks at start for replayability
        self.shuffle_seeds = random.sample(range(1, 2**60), 1000)

    def play_game(self):
        self.new_game()
        complete = 0
        while not complete:
            _,_,_,done,_ = self.next_turn(1)
            time.sleep(0)
            if done:
                complete = 1
        return 1
    
    def set_players(self,players):
        self.players = players
    
    def new_game(self):
        self.round_winner = None
        self.game_winner = None
        self.player_scores = dict.fromkeys(self.players, 0)
        self.current_player_index = random.randrange(self.n_players)
        self.match_starter = self.players[self.current_player_index]
        self.turn_count = 0
        self.round_suit = ""
        self.play_history = defaultdict(list)
        self.match_started = 0
        self.shuffle_ind = 0
        self.game_length = 0
        for player in self.players:
            player.hand = []
        self.new_match()
        

    def new_match(self):
        self.match_started = 1
        self.round_n = 0
        self.round_suit = ""
        self.play_history = defaultdict(list)
        self.deck = deck()
        self.deck.shuffle(self.shuffle_seeds[self.shuffle_ind])
        self.shuffle_ind = (self.shuffle_ind + 1) % len(self.shuffle_seeds)  # Increment index

        # Deal cards in random player order
        randomized_players = random.sample(self.players, self.n_players)
        for player in randomized_players:
            player.draw_card(self.deck, self.max_rounds)

    def next_turn(self,annot):
        done = 0    # game end
        terminated = 0  # match end
        current_player = self.players[self.current_player_index]
        # Match not started if none is played yet or last ended without game end
        if not self.match_started:
            self.new_match()
            self.match_starter = current_player

        if annot:# and not isinstance(current_player,AI):
            current_player.show_hand()

        # Play move
        if self.round_n == 0:   # Play bets on first round
            play_feedback = current_player.play_bet(self)
            played_move = play_feedback[0]
            self.play_history[self.round_n].append((current_player,played_move))
            if annot:
                print(f"{current_player.name} played: {played_move}\n")
            if self.turn_count == (self.n_players - 1):
                # Go to next round after bets
                self.round_n += 1
            self.current_player_index = (self.current_player_index + 1) % self.n_players
        else:      # is not first round             
            play_feedback = current_player.play_card(self)
            played_move = play_feedback[0]
            self.play_history[self.round_n].append((current_player,played_move))
            if annot:
                print(f"{current_player.name} played: {played_move}\n")
            # is round end?
            if self.turn_count == (self.n_players - 1):
                # Check who won the round
                max_rank = 0
                for player,card in self.play_history[self.round_n]:
                    if (card.suit == self.round_suit) and (int(card.rank) > max_rank):
                        max_rank = int(card.rank)
                        self.round_winner = player
                if annot:
                    print(f"\nRound winner: {self.round_winner.name}!")
                # If last round, give score and prepare for next match or end game
                if self.round_n == self.max_rounds:
                    terminated = 1
                    self.match_started = 0
                    if annot:
                        print("Round over!")
                    self.give_score()
                    if annot:
                        print("\nScores:")
                        for i in range(self.n_players):
                                print(f"{self.players[i].name}: {self.player_scores[self.players[i]]}")
                    if (self.player_scores[self.round_winner] >= self.max_score) or (any(value <= self.min_score for value in self.player_scores.values())):
                        done = 1
                        self.game_winner = self.round_winner
                        if annot:
                            print("Game over!")
                            print(f"{self.round_winner.name}'s won")
                    else:
                        self.current_player_index = (self.players.index(self.match_starter) + 1) % self.n_players
                else:
                    self.current_player_index = self.players.index(self.round_winner)
                self.round_suit = ""
                self.round_n += 1
            else:   # turns left in round
                self.current_player_index = (self.current_player_index + 1) % self.n_players
                if self.turn_count == 0:
                    self.round_suit = played_move.suit

        self.turn_count = (self.turn_count + 1) % self.n_players
        if len(play_feedback) > 1: 
            action = play_feedback[1]
            state = play_feedback[2]
            valid_act_mask = play_feedback[3]
        else:
            action = None
            state = None
            valid_act_mask = None

        self.game_length += 1
        return action,state,terminated,done,valid_act_mask

    def give_score(self):
        for player_firstround,bet in self.play_history[0]:
            for player_lastround,card_lastround in self.play_history[self.round_n]:
                if player_firstround == player_lastround:
                    last_card_rank = card_lastround.rank
            
                    is_winner = (self.round_winner == player_firstround)
                    won_with_2 = last_card_rank == "2"
                    if is_winner:
                        self.player_scores[player_firstround] += 1
                        if won_with_2:
                                    self.player_scores[player_firstround] += 1
                    match bet:
                        case "win":
                            if is_winner:
                                self.player_scores[player_firstround] += 1  # Award 1 extra points for winning with call
                            else:
                                self.player_scores[player_firstround] -= 1  # Penalty of 1 point for losing with call
                        case "2-win":
                            if is_winner and last_card_rank == "2":
                                self.player_scores[player_firstround] += 2  # Award 2 extra points for winning with a rank 2 call
                            else:
                                self.player_scores[player_firstround] -= 2  # Penalty of 2 point for not winning with a rank 2 call
                        case "pass":
                            pass
                        case _:
                            print("Error in bet")
                    # cap score to set limited vocabulary for embedding input
                    if self.player_scores[player_firstround] < self.min_score:
                        self.player_scores[player_firstround] = self.min_score

                    if self.player_scores[player_firstround] > self.max_score:
                        self.player_scores[player_firstround] = self.max_score

    def step(self,player):
        terminated = 0
        done = 0
        old_scores = list(self.player_scores.values())
        player_played = 0
        reward = 0
        # Play action and advance game state just until next player action
        # terminated = round over
        # done = game over
        while 1:
            if self.current_player_index == self.players.index(player):
                if player_played == 0:
                    my_action, my_state, terminated, done, valid_act_mask = self.next_turn(0)
                    player_played = 1
                    # Stop game after player move if game goes too long
                    if self.game_length >= self.max_game_length:
                        done = 1
                else:
                    break
            else:
                _,_,terminated,done,_ = self.next_turn(0)

            if terminated:
                # Score for each point
                new_scores = list(self.player_scores.values())
                for i,i_player in enumerate(self.players):
                    if i_player == player:
                        reward += (new_scores[i] - old_scores[i])*10
                # Score for passing and losing
                player_bet = None
                for first_player,bet in self.play_history[0]:
                    if first_player == player:
                        player_bet = bet
                if player != self.round_winner and player_bet == "pass":
                    reward += 0#5
            #else:
                #if player == self.round_winner:
                #    reward += 1
            if done: break

        if done:
            if self.game_winner == player:
                reward = 100
            else:
                reward = -100

        return my_action, my_state, reward, done, valid_act_mask