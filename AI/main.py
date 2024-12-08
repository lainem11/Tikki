from Tikki import Tikki
from player import player,AI, random_AI
import model5 as model
import torch
import os
import model_enemy


# Example usage
n_players = 2
round_n = 5
card_n = 5
n_bets = 3
n_inputs = card_n + n_players*(1+round_n) + n_players
n_outputs = 52+3
hidden_size = 30
Actor = model.Actor(n_inputs,hidden_size,n_outputs)
#Critic = model.Critic(n_inputs,hidden_size)

model_folder_path = './model'
#model_name = "actor_PPO_bestagaincont_hid90_rolls5_nups1_actlr8e-05_critlr6e-05.pth"
model_name = "actor_V1-2_lr6.08e-04,r4,lrbd0.90,g0.978591241041834,rbd0.97,hid30.pt"
file_name = os.path.join(model_folder_path,model_name)
thing = torch.load(file_name)
Actor.load_state_dict(thing["model_state"])

players = [player("Elli"),AI("XXX_SLAYAH_XXX",Actor,device='cpu')]

game = Tikki(players)
game.play_game()
