import numpy as np
import torch
import random
from collections import deque
from Tikki import Tikki
from player import player,random_AI,AI
import model as model
import model5 as myModel
import torch.optim as optim
import torch.nn as nn
from itertools import count
from deck import deck
import time
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import pandas
import os
import pickle
import copy
import itertools
import torch.nn.functional as F
import tempfile
#from torch.distributions import Categorical
#from torchrl.objectives import ClipPPOLoss
#from torchrl.objectives.value import GAE
from scipy.ndimage import uniform_filter1d
#import multiprocessing
from fast_categorical import Categorical
#import torch.multiprocessing as mp
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint

from collections import defaultdict
times = defaultdict(list)

# Set the random seed for Python's random number generator
random_seed = 42
random.seed(random_seed)

# Set the random seed for NumPy
np.random.seed(random_seed)

# Set the random seed for PyTorch
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Disable CuDNN benchmarking for reproducibility

class Agent(player):

    def __init__(self,name,EPS_N,Actor,Critic,device,play_device,actor_optim,critic_optim,scheduler_actor,scheduler_critic,GAMMA,reward_lambda,enemy,n_rollouts=1,n_updates_per_iter=5):
        super(Agent, self).__init__(name)
        self.EPS_N = EPS_N
        self.Actor = Actor
        self.Critic = Critic
        self.device = device
        self.play_device = play_device
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_scheduler = scheduler_actor
        self.critic_scheduler = scheduler_critic
        self.identifier = name

        self.GAMMA = GAMMA
        self.eps_clip = 0.2
        self.lmbda = reward_lambda
        self.n_updates_per_iter = n_updates_per_iter
        self.n_rollouts = n_rollouts
        self.enemy = enemy

        self.losses = []
        self.n_steps = 0
        self.Deck = deck()
        self.n_outputs = 52+3
        self.winners = []

        self.card_to_net_dict = dict(zip([hash(c) for c in self.Deck.cards],list(range(0,52)))) # maps card to index on the one-hot input vector
        self.net_to_card_dict = dict(zip(list(range(0,52)),[hash(c) for c in self.Deck.cards])) # maps card to index on the one-hot input vector

        # Initialize categorical distribution with sizes
        self.Cat_bets = Categorical((3,))
        self.Cat_acts = Categorical((self.n_outputs,))
        self.Cat_cards = [Categorical((1,)),Categorical((2,)),Categorical((3,)),Categorical((4,)),Categorical((5,))]
        '''
        # Loss plotting
        plt.ion()
        n_plots = 2
        fig,ax = plt.subplots(1,n_plots)
        line0, = ax[0].plot(0, 0)
        line1, = ax[1].plot(0, 0)
        self.loss_fig = fig
        self.loss_lines = [line0,line1]
        ax[0].set_xlabel('Optimization steps')
        ax[0].set_ylabel('Actor Loss')
        ax[1].set_xlabel('Optimization steps')
        ax[1].set_ylabel('Critic Loss')
        self.loss_axes = ax
        '''

    def play_bet(self,Tikki):
        state = self.get_state(Tikki).to(self.play_device)
        torch.cuda.synchronize()
        action, action_index = self.choose_action(state,Tikki,"bet")
        return action, action_index, state
        # this should be only public
    
    def play_card(self,Tikki):
        state = self.get_state(Tikki).to(self.play_device)
        torch.cuda.synchronize()
        action = self.choose_action(state,Tikki,"card")
        card_hash = self.net_to_card_dict[action]
        index = [hash(c) for c in self.hand].index(card_hash)
        return self.hand.pop(index), action, state
    
    def choose_action(self,state,Tikki,action):
        if action == "card":
            indices = self.get_valid_net_indices(Tikki)
            with torch.no_grad():  
                probs = nn.functional.softmax(self.Actor(state)[0,indices],dim=0)
                dist = self.Cat_cards[len(indices)-1]
                dist.set_probs(probs)
                action = dist.sample()
                # Action is the index of card to play
                return indices[action]
        else:
            # action == bet
            bets = ["pass","win","2-win"]
            with torch.no_grad():
                probs = nn.functional.softmax(self.Actor(state)[0,-3:],dim=0)
                self.Cat_bets.set_probs(probs)
                action = self.Cat_bets.sample()
                action_index = action.item() + self.n_outputs - 3
                # Action is the index of card to play
                return bets[action.item()], action_index

    def get_state(self,Tikki):
        # State as list of my cards, played moves, and scores. Each card is represented by 1,2,3,..,52. 
        # Bets as 53,54,55. No card is represented by 0. Scores from -10 to 5 by 56,..71
        # Players go from 53,...
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
            for _,move in hist[r]:
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
        # Always my score in the same spot
        Tikki_players = deque(Tikki.players)
        Tikki_players.rotate(-Tikki.players.index(self))
        Tikki_players = list(Tikki_players)

        for ind,plr in enumerate(Tikki_players):
            i_score[0,ind] = score_vocab[game_score[plr]]

        return torch.cat((torch.cat((i_cards,i_hist),1),i_score),1)
   
    def get_valid_net_indices(self,Tikki):
        cards = self.card_to_net_dict
        valids = self.valid_cards(Tikki)
        indices = []
        for i in range(len(valids)):
            indices.append(cards[hash(valids[i])])        # list with indices corresponding to valid cards in order
        return indices
    
    def rollout(self,number_of_rollouts):
        #global winners,times
        game = Tikki(players = [self,self.enemy])
        actions = []
        states  = []
        rewards = []
        dones   = []
        for _ in range(number_of_rollouts):
            game.new_game()
            while 1:
                # Or compute logprob, value directly with pass to net
                action, state, reward, done = game.step(self)

                actions.append(action)
                states.append(state)
                rewards.append(reward)
                dones.append(done)

                if done:
                    self.winners.append(game.game_winner==self)
                    break
        return actions, states, rewards, dones
    
    def compute_rtgs(self, rewards, dones):
        """
        Compute the rewards-to-go (rtg) for each timestep in the batch.
        
        Args:
        - returns: List of lists containing rewards from each episode in the batch
        - dones: List of lists containing boolean values indicating the end of each episode
        
        Returns:
        - List of tensors, each containing rewards-to-go for each timestep in the corresponding episode
        """
        batch_rtgs = []  # List to store rewards-to-go for this episode
        discounted_reward = 0  # Initialize discounted reward for this episode

        # Iterate through each timestep in the episode backwards
        for reward, done in zip(reversed(rewards), reversed(dones)):
            # If the episode ended, reset the discounted reward
            if done:
                discounted_reward = 0
            # Update discounted reward
            discounted_reward = reward + discounted_reward * self.GAMMA
            # Insert the discounted reward at the beginning of the list
            batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float,device=self.device)
        return batch_rtgs
    
    
    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0

            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = ep_rews[t] + self.GAMMA * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]

                advantage = delta + self.GAMMA * self.lmbda * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage)

            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float, device=self.device)
    
    def learn(self):
        quantiles = [int(self.EPS_N * i / 10) for i in range(1, 11)] 
        #global winners, times
        start = 0
        checkpoint = train.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint_dict_act = torch.load(os.path.join(checkpoint_dir, f"actor_{self.identifier}.pt"))
                checkpoint_dict_cri = torch.load(os.path.join(checkpoint_dir, f"critic_{self.identifier}.pt"))
                self.Actor.load_state_dict(checkpoint_dict_act["model_state"])
                self.Critic.load_state_dict(checkpoint_dict_cri["model_state"])
                start = checkpoint_dict_act["epoch"] + 1
                
        for episode in range(start,self.EPS_N):
            # Perform rollouts
            actions, states, rewards, dones = self.rollout(self.n_rollouts)
            # Convert to tensors and move to the appropriate device
            states = torch.stack(states).squeeze().to(self.device)
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
            
            self.Actor = self.Actor.to(self.device)
            self.Critic = self.Critic.to(self.device)

            # Compute old log probabilities and value estimates
            with torch.no_grad():
                self.Cat_acts.set_probs(nn.functional.softmax(self.Actor(states), dim=1))
                old_log_probs = self.Cat_acts.log_prob(actions)
                values = self.Critic(states).squeeze()
                batch_rtgs = self.compute_rtgs(rewards, dones)

                # Calculate and normalize advantages
                A = batch_rtgs - values
                A = (A - A.mean()) / (A.std() + 1e-10)

            # Perform multiple updates per iteration
            for _ in range(self.n_updates_per_iter):
                # Compute new log probabilities and values
                self.Cat_acts.set_probs(nn.functional.softmax(self.Actor(states), dim=1))
                new_log_probs = self.Cat_acts.log_prob(actions)
                new_values = self.Critic(states).squeeze()

                # Calculate the ratio of probabilities
                ratios = torch.exp(new_log_probs - old_log_probs)

                # Surrogate loss components
                surr1 = ratios * A
                surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * A

                # Losses
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.SmoothL1Loss()(new_values, batch_rtgs)

                # Update Actor network
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                # Update Critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                self.losses.append([actor_loss.item(), critic_loss.item()])
            self.Actor = self.Actor.to(self.play_device)
            self.Critic = self.Critic.to(self.play_device)

            # Step the schedulers
            self.actor_scheduler.step()
            self.critic_scheduler.step()

            # Print and save progress every 10% progress
            
            if (episode+1) in quantiles:
                print(f"Progress: {((episode + 1)/ self.EPS_N * 100):.0f}%")
                # Total wins of last 1000 fresh hands
                win_count = 1000*self.n_rollouts
                recent_winners = self.winners[-win_count:]
                winrate = np.array(recent_winners,dtype=float).mean() if len(recent_winners) > 0 else 0.0
                print(f"Winrate (last 1000 games): {winrate:.2f}")

                df = pandas.DataFrame({'winners': self.winners})
                self.save_agent_results(df, self.identifier)
                # Log to Ray
                with tempfile.TemporaryDirectory() as tempdir:
                    torch.save(
                        {"episode": episode, "model_state": self.Actor.state_dict()},
                        os.path.join(tempdir, f"actor_{self.identifier}.pt"),
                    )
                    torch.save(
                        {"episode": episode, "model_state": self.Critic.state_dict()},
                        os.path.join(tempdir, f"critic_{self.identifier}.pt"),
                    )
                    train.report({"winrate":winrate},checkpoint=Checkpoint.from_directory(tempdir))
                # Save models and optimizer states
                model_dir = './model'
                os.makedirs(model_dir, exist_ok=True)
                torch.save(self.Actor.state_dict(), os.path.join(model_dir, f"actor_{self.identifier}.pth"))
                torch.save(self.Critic.state_dict(), os.path.join(model_dir, f"critic_{self.identifier}.pth"))
                torch.save({
                    'Actor_opt_state_dict': self.actor_optim.state_dict(),
                    'Critic_opt_state_dict': self.critic_optim.state_dict(),
                }, os.path.join(model_dir, f"optimizers_{self.identifier}.pth"))

        return self.Actor

    def plot_losses(self,show_result=False):
        x = list(range(len(self.losses)))
        y = np.array(self.losses)
        for i in range(len(self.loss_lines)):
            self.loss_lines[i].set_xdata(x)  
            self.loss_lines[i].set_ydata(y[:,i])
            y_padding = 0.1 * (np.max(y[:,i]) - np.min(y[:,i]))
            self.loss_axes[i].set_ylim(min(y[:,i]) - y_padding, max(y[:,i]) + y_padding)
            self.loss_axes[i].set_xlim(0,len(self.losses))
        # drawing updated values
        self.loss_fig.canvas.draw()
    
        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        self.loss_fig.canvas.flush_events()
 
        time.sleep(0.5)

    def save_agent_results(self,df,identifier):
        folder_path = './results'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_name = os.path.join(folder_path,identifier + ".csv")
        df.to_csv(file_name,index=False,sep=',')

def run_agent_simulation(config

):  

    round_n=5
    card_n=5
    bet_n=3
    player_n=2
    GAMMA=config["g"]
    hidden_size=config["hid"]
    lr_actor = config["lr"]
    lr_critic = config["lr"]
    n_rollouts = config["r"]
    n_updates_per_iter = 1
    EPS_N=80000
    agent_name=None
    opt_state_dict=None
    scheduler_lmbd=config["lrbd"]
    reward_lambda = config["rbd"]
    identifier=f"V1-2_lr{config['lr']:.2e},r{config['r']},lrbd{config['lrbd']:.2f},g{config['g']},rbd{config['rbd']:.2f},hid{config['hid']}"
    EPS_N_normed = int(EPS_N/n_rollouts)

    #global winners, times
    #winners = []
    start1 = time.time()
    n_inputs = card_n + player_n*(1+round_n) + player_n 
    n_outputs = 52+bet_n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print(f"Device: {device}")
    play_device = "cpu"
    # Load net
    Actor = myModel.Actor(n_inputs,hidden_size,n_outputs).to(play_device)
    Critic = myModel.Critic(n_inputs,hidden_size).to(play_device)
    if agent_name:
        Actor.load_state_dict(torch.load(os.path.join('./model',f"Actor_{agent_name}.pth")))
        Critic.load_state_dict(torch.load(os.path.join('./model',f"Critic_{agent_name}.pth")))
    
    # Uncomment if using scheduler
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99999)

    # Load enemy
    enemy_net = myModel.Actor(n_inputs, 120, n_outputs)
    model_folder_path = "D:/Python/Tikki/model/"
    #model_name = "PPO_tests_model2_6.pth"
    model_name = "actor_V1-2_lr4.32e-04,r4,lrbd0.74,rbd0.98,hid120.pt"
    file_name = os.path.join(model_folder_path, model_name)
    load_dict = torch.load(file_name)
    enemy_net.load_state_dict(load_dict["model_state"])
    enemy = AI("XXX_SLAYAH_XXX",enemy_net,play_device)
    # Set optimizers
    actor_optim = optim.Adam(Actor.parameters(),lr=lr_actor)
    critic_optim = optim.Adam(Critic.parameters(),lr=lr_critic)
    # Define learning rate schedulers
    # Define the schedulers
    actor_scheduler = lr_scheduler.StepLR(actor_optim,step_size=EPS_N_normed//10,gamma=scheduler_lmbd)
    critic_scheduler = lr_scheduler.StepLR(critic_optim,step_size=EPS_N_normed//10,gamma=scheduler_lmbd)

    #opt_file_path = os.path.join(model_folder_path,f"optimizers_{identifier}")
    if opt_state_dict:
        load_opt_state = torch.load(os.path.join(model_folder_path,opt_state_dict))
        actor_optim.load_state_dict(load_opt_state['Actor_opt_state_dict'])
        critic_optim.load_state_dict(load_opt_state['Critic_opt_state_dict'])
        for act_param_group,crit_param_group in zip(actor_optim.param_groups,critic_optim.param_groups):
            act_param_group['lr'] = lr_actor
            crit_param_group['lr'] = lr_critic

    start = time.time()
    ME = Agent(identifier,EPS_N_normed,Actor,Critic,device,play_device,actor_optim,critic_optim,actor_scheduler,critic_scheduler,GAMMA,reward_lambda,enemy,n_rollouts,n_updates_per_iter)
    policy_net = ME.learn()
    print(f"Duration: {time.time()-start1}\n\n")
'''
def run_pool(all_combinations):
    mp.set_start_method('spawn', force=True)  # Ensures compatibility with CUDA
    pool = mp.Pool(mp.cpu_count() - 1)

    for combo in all_combinations:
        identifier = f"PPO_model4_test1_hid{combo[0]}_rolls{combo[1]}_nups{combo[2]}_actlr{combo[3]}_critlr{combo[4]}"
        filename = os.path.join("results", identifier + ".csv")
        if not os.path.isfile(filename):
            print(f"Running: {identifier}")
            task = {
                "identifier": identifier,
                "hidden_size": combo[0],
                "n_rollouts": combo[1],
                "n_updates_per_iter": combo[2],
                "lr_actor": combo[3],
                "lr_critic": combo[4],
            }
            pool.apply_async(run_agent_simulation, kwds=task)
        else:
            print(f"File exists, skipping...")

    pool.close()
    pool.join()
'''
if __name__ == '__main__':

    search_space = {
        "r": tune.choice([4,8]),
        "lr": tune.uniform(1e-4,5e-3),
        "lrbd": tune.uniform(0.8,1.0),
        'g': tune.uniform(0.95,1),
        "rbd": tune.uniform(0.95,1),
        "hid": tune.choice([30,90,150])
    }

    #agent_name = ["PPO_gridTest2_hid90_rolls5_nups1_actlr0.0001_critlr6e-05_again_cont"]
    #opt_state_dict= ["optimizers_PPO_gridTest2_hid90_rolls5_nups1_actlr0.0001_critlr6e-05_again_cont"]
    agent_name = None
    opt_state_dict = None
    os.environ["RAY_TMPDIR"] = r"D:\Python\Tikki\ray_tmp"
    trainable_with_resources = tune.with_resources(run_agent_simulation, {"cpu": 1, "gpu":1.0/11.1})
    run_config = train.RunConfig(
        storage_path="D:/Python/Tikki/ray_tuner", 
        name="tuning4",
        )
    scheduler = ASHAScheduler(grace_period=5)
    tuner = tune.Tuner(
        trainable_with_resources,
        run_config=run_config,
        tune_config=tune.TuneConfig(
            search_alg=OptunaSearch(),
            metric="winrate",
            mode="max",
            num_samples=200,
            max_concurrent_trials=11,
            scheduler = scheduler,
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)
    # Compute cartesian product of parameter grids
    #all_combinations = list(itertools.product(
    #    hidden_size, n_rollouts, n_updates_per_iter, actor_lr, critic_lr, #agent_name, opt_state_dict
    #))
    #random.shuffle(all_combinations)

    #run_pool(all_combinations)
    #run_agent_simulation(identifier="PPO_gridTest2_hid90_rolls5_nups1_actlr0.0001_critlr6e-05_again_cont_1",hidden_size=90,n_rollouts=5,n_updates_per_iter=1,lr_actor=0.0001,lr_critic=6e-5,agent_name="PPO_gridTest2_hid90_rolls5_nups1_actlr0.0001_critlr6e-05_again_cont",opt_state_dict=f"optimizers_PPO_gridTest2_hid90_rolls5_nups1_actlr0.0001_critlr6e-05_again_cont")


   

