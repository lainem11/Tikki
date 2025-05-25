import numpy as np
import torch
import random
from Tikki import Tikki
from player import random_AI,AI,random_player
import model_relu_deep_skip as model

import torch.optim as optim
import torch.nn as nn
from deck import deck
import time
import torch.optim.lr_scheduler as lr_scheduler
import os
import tempfile
from fast_categorical import Categorical
import ray
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
from collections import defaultdict
import gc
import inspect

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

### Measure execution times
times = defaultdict(list)

script_dir = os.path.dirname(os.path.abspath(__file__))

class Agent(AI):

    def __init__(self,name,EPS_N,Actor,Critic,play_device,train_device,actor_optim,critic_optim,scheduler_actor,scheduler_critic,GAMMA,LAMBDA_,eps_clip,enemy,n_rollouts,n_batches,n_updates_per_iter,entropy_scale,entropy_steepness):
        super(Agent, self).__init__(name,Actor,play_device)
        self.EPS_N = EPS_N  # Number of training episodes
        self.Actor = Actor  # Actor net
        self.Critic = Critic    # Critic net
        self.play_device = play_device  # Game is executed on play_device
        self.train_device = train_device    # Training happens on train device
        self.actor_optim = actor_optim  # Actor and Critic optimizer
        self.critic_optim = critic_optim 
        self.actor_scheduler = scheduler_actor  # Optimizer scheduler for Actor and critic
        self.critic_scheduler = scheduler_critic    
        self.identifier = name  # Save filename

        # Hyperparameters
        self.GAMMA = GAMMA  # Learning rate decay
        self.LAMBDA_ = LAMBDA_  # GAE discount factor
        self.eps_clip = eps_clip    # PPO loss clip
        self.n_updates_per_iter = n_updates_per_iter    # Number of train steps per batch
        self.n_rollouts = n_rollouts    # Number of repeated games played per training batch
        self.n_batches = n_batches # Number of unique games batches per iteration
        self.enemy = enemy  # Enemy player net
        self.initial_entropy_scale = torch.tensor(entropy_scale,device=train_device) # Encourages exploration
        self.entropy_steepness = entropy_steepness

        self.losses = []    # Save training loss
        self.Deck = deck()  # Load deck
        self.n_outputs = 52+3   # Number of net outputs for Actor (Critic always 1)
        self.checkpoint_count = 40
        self.winner_count = 10000
        self.winners = []   # Save list of winners

        # Initialize categorical distribution with sizes. These are used to sample action from a distribution. Using my own fast implementation. 
        self.Cat_bets = Categorical((3,))
        self.Cat_acts = Categorical((self.n_outputs,))
        self.Cat_cards = [Categorical((1,)),Categorical((2,)),Categorical((3,)),Categorical((4,)),Categorical((5,))]

        # Store old state dicts to use as enemy
        self.lagging_state_dict = self.Actor.state_dict()
    
    def rollout(self,number_of_rollouts,number_of_batches,episode):
        '''
        Initializes fresh Tikki game with deterministic deck. Executes game steps until winner is found, and repeats number_of_rollouts times. Each repetition
        has the same cards dealt, but actions are re-thought. This makes it easier to learn from different possible score outcomes with the same hand.
        '''
        
        actions = []
        states  = []
        rewards = []
        dones   = []
        valid_act_masks = []

        for b in range(number_of_batches):
            game = Tikki(players = [self,self.enemy])
            for r in range(number_of_rollouts):
                game.new_game()
                while 1:
                    # Or compute logprob, value directly with pass to net
                    action, state, reward, done, valid_act_mask = game.step(self)

                    actions.append(action)
                    states.append(state[0])
                    rewards.append(reward)
                    dones.append(done)
                    valid_act_masks.append(valid_act_mask[0])

                    if done:
                        self.winners.append(game.game_winner==self)
                        break
        return actions, states, rewards, dones, valid_act_masks

    def find_gpu_tensors_detailed(self):
        '''
        Example usage:
        Print GPU tensors and their details
        gpu_tensors = self.find_gpu_tensors_detailed()
        for name, size, shape, dtype in gpu_tensors:
            print(f"Variable '{name}': {size / (1024**2):.2f} MB, Shape: {shape}, Dtype: {dtype}")
        '''
        gpu_tensors = []
        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.Tensor) and obj.is_cuda:
                    size = obj.element_size() * obj.nelement()
                    shape = obj.size()
                    dtype = obj.dtype
                    for frame in inspect.stack():
                        local_vars = frame.frame.f_locals
                        for var_name, var_val in local_vars.items():
                            if var_val is obj:
                                gpu_tensors.append((var_name, size, shape, dtype))
            except Exception:
                pass
        return gpu_tensors
        
    def check_and_clear_cache(self,threshold_gb, device='cuda:0'):
        # Convert threshold from GB to bytes
        threshold_bytes = threshold_gb * 1024 ** 3

        # Get current memory usage
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        
        # Clear cache if reserved memory exceeds threshold
        if reserved > threshold_bytes:
            print(f"Reserved memory exceeds {threshold_gb} GB. Clearing cache...")
            torch.cuda.empty_cache()
            print("Cache cleared.")
            print(f"Allocated Memory: {allocated / 1024**3:.2f} GB")
            print(f"Reserved Memory: {reserved / 1024**3:.2f} GB")

    def compute_rtgs(self, rewards, dones):
        """
        Args:
        - rewards: Flat list or tensor of rewards across all timesteps in the batch
        - dones: Flat list or tensor of done flags indicating episode ends

        Returns:
        - Tensor containing rewards-to-go for each timestep in the batch
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
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float,device=self.train_device)
        return batch_rtgs
    
    def compute_gae(self, rewards, values, dones, gamma, lambda_):
        advantages = torch.zeros_like(rewards)

        end_inds = dones.nonzero()
        start_ind = 0
        for e in range(len(end_inds)):
            last_advantage = 0
            for t in reversed(range(start_ind,end_inds[e]+1)):
                if t == end_inds[e]:
                    delta = rewards[t] - values[t]
                else:
                    delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
                advantages[t] = delta + gamma * lambda_ * (1 - dones[t]) * last_advantage
                last_advantage = advantages[t]
            start_ind = end_inds[e]+1
        return advantages
    
    def learn(self):
        '''
        Train agent.
        '''
        num_breaks = self.checkpoint_count # Determines number of checkpoints and learning rate decay steps
        quantiles = [int(self.EPS_N * i / num_breaks) for i in range(1, num_breaks+1)] 

        start = 0
        #self.entropy_scale = self.initial_entropy_scale
        k = -np.log(self.entropy_steepness) / 0.5  # Decay to ~1% of the initial value at 50% progress

        # Load from checkpoint if exists
        checkpoint = train.get_checkpoint()
        if checkpoint:
            try:
                with checkpoint.as_directory() as checkpoint_dir:  
                    print(f"Loading previous trial: {self.identifier}")
                    checkpoint_dict_act = torch.load(os.path.join(checkpoint_dir, f"actor_{self.identifier}.pt"))
                    checkpoint_dict_cri = torch.load(os.path.join(checkpoint_dir, f"critic_{self.identifier}.pt"))
                    self.Actor.load_state_dict(checkpoint_dict_act["model_state"])
                    self.Critic.load_state_dict(checkpoint_dict_cri["model_state"])
                    start = checkpoint_dict_act["episode"] + 1

                    self.Actor = self.Actor.to(self.train_device)
                    self.Critic = self.Critic.to(self.train_device)
                    self.actor_optim.load_state_dict(checkpoint_dict_act["optimizer_state"])
                    self.critic_optim.load_state_dict(checkpoint_dict_cri["optimizer_state"])

                    if "scheduler_state" in checkpoint_dict_act and "scheduler_state" in checkpoint_dict_cri:
                        self.actor_scheduler.load_state_dict(checkpoint_dict_act["scheduler_state"])
                        self.critic_scheduler.load_state_dict(checkpoint_dict_cri["scheduler_state"])
                    else:
                        self.actor_scheduler.last_epoch = start - 1
                        self.critic_scheduler.last_epoch = start - 1

            except (FileNotFoundError, KeyError, RuntimeError) as e:
                print(f"Failed to load checkpoint: {e}. Starting from scratch.")
                start = 0

        # Make sure correct devices are used
        self.Actor = self.Actor.to(self.play_device)
        self.Critic = self.Critic.to(self.play_device)

        for episode in range(start,self.EPS_N):
            actions, states, rewards, dones, valid_act_masks = self.rollout(self.n_rollouts,self.n_batches,episode)

            self.Actor = self.Actor.to(self.train_device)
            self.Critic = self.Critic.to(self.train_device)

            # Convert to tensors and move to the appropriate device
            states = torch.stack(states).squeeze().to(self.train_device)
            actions = torch.tensor(actions, dtype=torch.long, device=self.train_device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.train_device)
            dones = torch.tensor(dones, dtype=torch.float32, device=self.train_device)
            valid_act_masks = torch.stack(valid_act_masks).squeeze().to(self.train_device)

            with torch.no_grad():
                # Compute policy and value outputs
                old_net_output = self.Actor(states)
                masked_logits = old_net_output.masked_fill(~valid_act_masks, float('-inf'))
                self.Cat_acts.set_probs(nn.functional.softmax(masked_logits, dim=-1))
                old_log_probs = self.Cat_acts.log_prob(actions)
                values = self.Critic(states).squeeze()

                # Calculate advantages
                advantages = self.compute_gae(rewards,values,dones,self.GAMMA,self.LAMBDA_)
                rtgs = advantages + values
                
            # Apply exponential decay to entropy scale
            self.entropy_scale = self.initial_entropy_scale * np.exp(-k * episode/self.EPS_N)

            # Perform multiple updates per iteration
            batch_size = actions.shape[0] // self.n_updates_per_iter
            for nup in range(self.n_updates_per_iter):
                idx = torch.randperm(states.shape[0])[:batch_size]

                batch_states = states[idx,:]
                batch_actions = actions[idx]
                batch_A = advantages[idx]
                batch_rtgs = rtgs[idx]
                batch_valid_act_masks = valid_act_masks[idx,:]
                batch_old_log_probs = old_log_probs[idx]

                # Compute new log probabilities and values
                new_net_output = self.Actor(batch_states)
                new_masked_logits = new_net_output.masked_fill(~batch_valid_act_masks, float('-inf'))
                self.Cat_acts.set_probs(nn.functional.softmax(new_masked_logits, dim=-1))
                new_log_probs = self.Cat_acts.log_prob(batch_actions)
                new_values = self.Critic(batch_states).squeeze()

                # Calculate the ratio of probabilities
                ratios = torch.exp(new_log_probs - batch_old_log_probs)

                # Surrogate loss components
                batch_A = (batch_A - batch_A.mean()) / (batch_A.std() + 1e-10)

                surr1 = ratios * batch_A
                surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * batch_A

                # Regularize loss by maximizing entropy
                entropy = self.Cat_acts.entropy()
                # Losses
                actor_loss = -torch.min(surr1, surr2).mean() -(entropy.mean() * self.entropy_scale)
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

            # Checkpoint
            if (episode+1) in quantiles:
                #self.check_and_clear_cache(8)   # Potential for bugs when continueing from checkpoint?

                # Total wins of last N 'fresh' hands
                win_count = self.winner_count*self.n_rollouts
                recent_winners = self.winners[-win_count:]
                winrate = np.array(recent_winners,dtype=float).mean() if len(recent_winners) > 0 else 0.0

                # Clear win rate to save memory
                self.winners = []

                # Create checkpoint of trial progress
                with tempfile.TemporaryDirectory() as tempdir:
                    torch.save(
                        {
                            "episode": episode,
                            "model_state": self.Actor.state_dict(),
                            "optimizer_state": self.actor_optim.state_dict(),
                            "scheduler_state": self.actor_scheduler.state_dict(),
                        },
                        os.path.join(tempdir, f"actor_{self.identifier}.pt"),
                    )
                    torch.save(
                        {
                            "episode": episode,
                            "model_state": self.Critic.state_dict(),
                            "optimizer_state": self.critic_optim.state_dict(),
                            "scheduler_state": self.critic_scheduler.state_dict(),
                        },
                        os.path.join(tempdir, f"critic_{self.identifier}.pt"),
                    )
                    # Create metric report and move checkpoint to persistent storage
                    train.report({"winrate":winrate},checkpoint=Checkpoint.from_directory(tempdir))
        return self.Actor

    def save_agent_results(self,df,identifier):
        folder_path = './results'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_name = os.path.join(folder_path,identifier + ".csv")
        df.to_csv(file_name,index=False,sep=',')

def run(config):  
    '''
    Training function. Gets dictionary of tunable hyperparameters as an input.
    '''
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print(f"Device: {train_device}")
    play_device = "cpu" # This is much faster on cpu, even if training with GPU, as it minimizes overhead

    # Tunable hyperparameters
    GAMMA=0.9
    LAMBDA_=0.9
    hidden_size = 80
    lr_actor = config["alr"]
    lr_critic = config["clr"]
    n_rollouts = 3
    n_batches = 10
    n_updates_per_iter = 10
    emb_size = 60
    EPS_N = 2_000_000
    EPS_N_normed = int(EPS_N/n_rollouts/n_batches)
    scheduler_lmbd= config["sched_lmbd"]
    eps_clip = 0.13
    entropy_scale = 7.5e-3
    entropy_steepness = config["e_st"] 

    id_string = ""
    
    for key, value in config.items():
        try:
            sci_value = "{:.1e}".format(float(value))
            id_string += f"{key}{sci_value}_"
        except (ValueError, TypeError):
            # If value can't be converted to float, use original value
            id_string += f"{key}{value}_"
    # Remove the trailing comma and space if the string isn't empty
    id_string = id_string[:-2]

    identifier="V1-0_"+id_string
    
    # Load training net
    round_n=5
    card_n=5
    bet_n=3
    player_n=2
    n_inputs = card_n + player_n*(1+round_n) + player_n 
    n_outputs = 52+bet_n
    Actor = model.Actor(n_inputs,hidden_size,n_outputs,emb_size).to(train_device)
    Critic = model.Critic(n_inputs,hidden_size,emb_size).to(train_device)

    # Load enemy
    enemy_net = model.Actor(n_inputs,hidden_size,n_outputs,emb_size)
    load_dict = torch.load(os.path.join(script_dir,r"model/trained_actor_V1-0.pt"))
    enemy_net.load_state_dict(load_dict["model_state"])
    enemy = AI("XXX_SLAYAH_XXX",enemy_net,play_device)

    # Set optimizers
    actor_optim = optim.Adam(Actor.parameters(),lr=lr_actor,foreach=True)
    critic_optim = optim.Adam(Critic.parameters(),lr=lr_critic,foreach=True)
    actor_scheduler = lr_scheduler.StepLR(actor_optim,step_size=max(1,EPS_N_normed//40),gamma=scheduler_lmbd)
    critic_scheduler = lr_scheduler.StepLR(critic_optim,step_size=max(1,EPS_N_normed//40),gamma=scheduler_lmbd)

    ME = Agent(identifier,EPS_N_normed,Actor,Critic,play_device,train_device,actor_optim,critic_optim,actor_scheduler,critic_scheduler,GAMMA,LAMBDA_,eps_clip,enemy,n_rollouts,n_batches,n_updates_per_iter,entropy_scale,entropy_steepness)
    policy_net = ME.learn()

    [print(f"{key}: {np.array(value).mean()}") for key,value in times.items()]

if __name__ == '__main__':
    '''
    Use Ray Tuner to search the best hyperparameters. Several models can be trained in parallel using CPU and GPU resources.
    On my PC, CPU and GPU are utilized 100% and 70%, respectively, during training.
    '''
    run_ray = 1

    if not run_ray:
        global exec_times
        exec_times = defaultdict(list)
        config = {
            "alr":2e-4,
            "clr": 3.5e-3,
            "e_st": 0.6,
            "sched_lmbd": 0.8
        }
        #print(f"Episodes = {config['EPS_N']} / {config['n_b']} batch / {config['r']} rollout = {config['EPS_N']/config['n_b']/config['r']}")
        run(config)
        for key,value in exec_times.items():
            print(f"{key}, mean time: {sum(value)/len(value)}")
    else:
        ray.init()
        #os.environ["RAY_DEDUP_LOGS"] = "0"
        # Set search space for all hyperparameters you want to optimize
        search_space = {
            "alr": tune.uniform(9e-5,4e-4),
            "clr": tune.uniform(2.5e-3,6e-3),
            "e_st": tune.uniform(0.2,0.7),
            "sched_lmbd": tune.uniform(0.8,1.0)
        }

        # Set tempdir. May be necessary to avoid lengthy filepaths that prevent saving.
        os.environ["RAY_TMPDIR"] = os.path.join(script_dir, "ray_tmp")
        trainable_with_resources = tune.with_resources(run, {"cpu": 1, "gpu":1.0/11.1})

        # Define persistent OptunaSearch
        optuna_search = OptunaSearch(seed=42)
        search_alg_dir = os.path.join(script_dir, "ray_tuner")

        #scheduler = ASHAScheduler(grace_period=12)   # Schedules trials so that after grace_period number of checkpoints, prunes bad trials. By default, 75% of trials get pruned.
        
        root_path = search_alg_dir
        log_name = "run1_0"
        log_path = os.path.join(root_path,log_name)

        run_config = train.RunConfig(
            storage_path=root_path, 
            name=log_name,    # Creates folder with this name
        )

        def custom_trial_dirname(trial):
            # Create a short, unique name for the trial
            # trial.trial_id is a unique identifier (e.g., "ef3b37f8_2")
            # You can include a subset of config params if needed
            short_name = f"trial_{trial.trial_id}"
            return short_name

        # Check if there's an existing experiment to restore
        if os.path.exists(log_path):
            # Restore previous experiment
            print(f"Restoring experiment from {log_path}")
            tuner = tune.Tuner.restore(log_path,trainable_with_resources,resume_errored=True)
        else:
            tuner = tune.Tuner(
                trainable_with_resources,
                run_config=run_config,
                tune_config=tune.TuneConfig(
                    search_alg=optuna_search,
                    metric="winrate",
                    mode="max",
                    num_samples=100,             # Number of trials to run. Each proper trials takes at least a few hours, up to a day.
                    max_concurrent_trials=11,   # Set this and leave one CPU core available. Makes working with the PC while training ok.
                    trial_dirname_creator=custom_trial_dirname
                ),
                param_space=search_space,
            )
        results = tuner.fit()
        print("Best config is:", results.get_best_result().config)

