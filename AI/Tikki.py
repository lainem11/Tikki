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

        # Decide decks at start for replayability
        self.shuffle_seeds = random.sample(range(1, 1001), 1000)

    def play_game(self):
        self.new_game()
        complete = 0
        while not complete:
            _,_,_,done = self.next_turn(1)
            time.sleep(0)
            if done:
                complete = 1
        return 1
    
    def new_game(self):
        self.round_winner = None
        self.game_winner = None
        self.player_scores = dict.fromkeys(self.players, 0)
        self.current_player_index = random.randrange(len(self.players))
        self.match_starter = self.players[self.current_player_index]
        self.turn_count = 0
        self.round_suit = ""
        self.play_history = defaultdict(list)
        self.match_started = 0
        self.shuffle_ind = 0
        for player in self.players:
            player.hand = []
        

    def new_match(self):
        self.match_started = 1
        self.round_n = 0
        self.round_suit = ""
        self.play_history = defaultdict(list)
        self.deck = deck()
        self.deck.shuffle(self.shuffle_seeds[self.shuffle_ind])
        self.shuffle_ind += 1
        for player in self.players:
            player.draw_card(self.deck,self.max_rounds)

    def next_turn(self,annot):
        done = 0
        terminated = 0
        current_player = self.players[self.current_player_index]
        # Match not started if none is played yet or last ended without game end
        if not self.match_started:
            self.new_match()
            self.match_starter = current_player

        if annot and not isinstance(current_player,AI):
            current_player.show_hand()

        # Play move
        if self.round_n == 0:   # Play bets on first round
            #played_move, log_prob, value = current_player.play_bet(self)
            play_feedback = current_player.play_bet(self)
            played_move = play_feedback[0]
            self.play_history[self.round_n].append((current_player,played_move))
            if annot:
                print(f"{current_player.name} played: {played_move}\n")
            if self.turn_count == (len(self.players) - 1):
                # Go to next round after bets
                self.round_n += 1
            self.current_player_index = (self.current_player_index + 1) % len(self.players)
        else:      # is not first round             
            #played_move, log_prob, value = current_player.play_card(self)
            play_feedback = current_player.play_card(self)
            played_move = play_feedback[0]
            self.play_history[self.round_n].append((current_player,played_move))
            if annot:
                print(f"{current_player.name} played: {played_move}\n")
            # is round end?
            if self.turn_count == (len(self.players) - 1):
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
                        for i in range(len(self.players)):
                                print(f"{self.players[i].name}: {self.player_scores[self.players[i]]}")
                    if self.player_scores[self.round_winner] >= self.max_score:
                        done = 1
                        self.game_winner = self.round_winner
                        if annot:
                            print("Game over!")
                            print(f"{self.round_winner.name}'s won")
                    else:
                        self.current_player_index = (self.players.index(self.match_starter) + 1) % len(self.players)
                else:
                    self.current_player_index = self.players.index(self.round_winner)

                self.round_suit = ""
                self.round_n += 1
            else:   # turns left in round
                self.current_player_index = (self.current_player_index + 1) % len(self.players)
                if self.turn_count == 0:
                    self.round_suit = played_move.suit

        self.turn_count = (self.turn_count + 1) % len(self.players)
        if len(play_feedback) > 1: 
            action = play_feedback[1]
            state = play_feedback[2]
        else:
            action = None
            state = None
        return action,state,terminated,done

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
                    if self.player_scores[player_firstround] < -10:
                        self.player_scores[player_firstround] = -10

                    if self.player_scores[player_firstround] > 5:
                        self.player_scores[player_firstround] = 5

    def step(self,player):
        step_end = 0
        terminated = 0
        done = 0
        old_scores = list(self.player_scores.values())
        player_played = 0
        # Play action and advance game state just until next player action
        # terminated = round over
        # done = game over
        while 1:

            if self.current_player_index == self.players.index(player):
                if player_played == 0:
                    my_action, my_state, terminated, done = self.next_turn(0)
                    player_played = 1
                else:
                    break
            else:
                _,_,terminated,done = self.next_turn(0)
            if done: break

        if done:
            if self.round_winner == player:
                reward = 100
            else:
                reward = -100
        else:
            new_scores = list(self.player_scores.values())
            reward = 0
            for i,i_player in enumerate(self.players):
                if i_player == player:
                    reward += (new_scores[i] - old_scores[i])*10
                else:
                    reward -= (new_scores[i] - old_scores[i])*10
        return my_action, my_state, reward, done