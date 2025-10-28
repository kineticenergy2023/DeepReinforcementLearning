# Use multiprocess rather than multiprocessing if running in a Jupyter notebook
import multiprocessing as mp
import numpy as np

from typing import List
from enum import Enum, auto
import random

import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
# from IPython.display import clear_output

class Percept():
    time_step: int
    bump: bool
    breeze: bool
    stench: bool
    scream: bool
    glitter: bool
    reward: int
    done: bool

    def __init__(self, time_step: int, bump: bool, breeze: bool, stench: bool, scream: bool, glitter: bool, reward: int, done: bool):
        self.time_step = time_step
        self.bump = bump
        self.breeze = breeze
        self.stench = stench
        self.scream = scream
        self.glitter = glitter
        self.reward = reward
        self.done = done
        
    def __str__(self):
        return f'time:{self.time_step}: bump:{self.bump}, breeze:{self.breeze}, stench:{self.stench}, scream:{self.scream}, glitter:{self.glitter}, reward:{self.reward}, done:{self.done}'

class Action(Enum):
    LEFT = 0
    RIGHT = 1
    FORWARD = 2
    GRAB = 3
    SHOOT = 4
    CLIMB = 5
    
    
    @staticmethod
    def random() -> 'Action':
        return random.choice(list(Action))
    
    @staticmethod
    def from_int(n: int) -> 'Action':
        return Action(n)

class Orientation(Enum):
    E = 0
    S = 1
    W = 2
    N = 3

    def symbol(self) -> str:
        match self:
            case Orientation.E:
                return '>'
            case Orientation.S:
                return 'v'
            case Orientation.W:
                return '<'
            case Orientation.N:
                return '^'

    def turn_right(self) -> 'Orientation':
        match self:
            case Orientation.E:
                return Orientation.S
            case Orientation.S:
                return Orientation.W
            case Orientation.W:
                return Orientation.N
            case Orientation.N:
                return Orientation.E

    def turn_left(self) -> 'Orientation':
        match self:
            case Orientation.E:
                return Orientation.N
            case Orientation.N:
                return Orientation.W
            case Orientation.W:
                return Orientation.S
            case Orientation.S:
                return Orientation.E

class Location:
    x: int
    y: int
        
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        
    def __str__(self):
        return f'({self.x}, {self.y})'
    
    def is_left_of(self, location: 'Location')->bool:
        return self.x < location.x and self.y == location.y
        
    def is_right_of(self, location: 'Location')->bool:
        return self.x > location.x and self.y == location.y
        
    def is_above(self, location: 'Location')->bool:
        return self.y > location.y and self.x == location.x
        
    def is_below(self, location: 'Location')->bool:
        return self.y < location.y and self.x == location.x
        
    def neighbours(self)->List['Location']:
        neighbourList = []
        if self.x > 0: neighbourList.append(Location(self.x - 1, self.y))
        if self.x < 3: neighbourList.append(Location(self.x + 1, self.y))
        if self.y > 0: neighbourList.append(Location(self.x, self.y - 1))
        if self.y < 3: neighbourList.append(Location(self.x, self.y + 1))
        return neighbourList
    
    def is_location(self, location: 'Location')->bool:
        return self.x == location.x and self.y == location.y
    
    def at_left_edge(self) -> bool:
        return self.x == 0
    
    def at_right_edge(self) -> bool:
        return self.x == 3
    
    def at_top_edge(self) -> bool:
        return self.y == 3
    
    def at_bottom_edge(self) -> bool:
        return self.y == 0
    
    def forward(self, orientation) -> bool:
        bump = False
        match orientation:
            case Orientation.W:
                if self.at_left_edge():
                    bump = True
                else: self.x = self.x - 1
            case Orientation.E:
                if self.at_right_edge():
                    bump = True
                else: self.x = self.x + 1
            case Orientation.N:
                if self.at_top_edge():
                    bump = True
                else: self.y = self.y + 1
            case Orientation.S:
                if self.at_bottom_edge():
                    bump = True
                else: self.y = self.y - 1
        return bump
    
    def set_to(self, location: 'Location'):
        self.x = location.x
        self.y = location.y
        
    @staticmethod
    def from_linear(n: int) -> 'Location':
        return Location(n % 4, n // 4)
    
    def to_linear(self)->int:
        return self.y * 4 + self.x
    
    @staticmethod
    def random() -> 'Location':
        return Location.from_linear(random.randint(1, 15))

class Environment:
    wumpus_location: Location
    wumpus_alive: bool
    has_wumpus: bool
    agent_location: Location
    agent_orientation: Orientation
    agent_has_arrow: bool
    agent_has_gold: bool
    game_over: bool
    gold_location: Location
    pit_locations: List[Location]
    time_step: int
    
    def init(self, pit_prob: float, allow_climb_without_gold: bool, has_wumpus: bool):
        self.agent_location = Location(0, 0)
        self.agent_orientation = Orientation.E
        self.agent_has_arrow = True
        self.agent_has_gold = False
        self.pit_prob = pit_prob
        self.allow_climb_without_gold = allow_climb_without_gold
        self.has_wumpus = has_wumpus
        self.make_wumpus(has_wumpus)
        self.make_gold()
        self.make_pits(pit_prob)
        self.game_over = False
        self.time_step = 0
        return Percept(self.time_step, False, self.is_breeze(), self.is_stench(), False, False, 0, False)
    
    def make_wumpus(self, has_wumpus: bool):
        self.wumpus_location = Location.random()
        self.wumpus_alive = has_wumpus
        
    def make_gold(self):
        self.gold_location = Location.random()
        
    def make_pits(self, pit_prob: float):
        self.pit_locations = []
        for i in range(1, 16):
            if random.random() < pit_prob: self.pit_locations.append(Location.from_linear(i))
    
    def is_pit_at(self, location: Location) -> bool:
        return any(pit_location.is_location(location) for pit_location in self.pit_locations)
    
    def is_pit_adjacent_to_agent(self) -> bool:
        for agent_neighbour in self.agent_location.neighbours():
            for pit_location in self.pit_locations:
                if agent_neighbour.is_location(pit_location):
                    return True
        return False
    
    def is_wumpus_adjacent_to_agent(self) -> bool:
        return self.has_wumpus and any(self.wumpus_location.is_location(neighbour) for neighbour in self.agent_location.neighbours())
    
    def is_agent_at_hazard(self)->bool:
        return self.is_pit_at(self.agent_location) or (self.is_wumpus_at(self.agent_location) and self.wumpus_alive)
    
    def is_wumpus_at(self, location: Location) -> bool:
        return self.has_wumpus and self.wumpus_location.is_location(location)
    
    def is_agent_at(self, location: Location) -> bool:
        return self.agent_location.is_location(location)
    
    def is_gold_at(self, location: Location) -> bool:
        return self.gold_location.is_location(location)
    
    def is_glitter(self) -> bool:
        return self.is_gold_at(self.agent_location)
    
    def is_breeze(self) -> bool:
        return self.is_pit_adjacent_to_agent() or self.is_pit_at(self.agent_location)
    
    def is_stench(self) -> bool:
        return self.is_wumpus_adjacent_to_agent() or self.is_wumpus_at(self.agent_location)
    
    def wumpus_in_line_of_fire(self) -> bool:
        match self.agent_orientation:
            case Orientation.E: return self.has_wumpus and self.agent_location.is_left_of(self.wumpus_location)
            case Orientation.S: return self.has_wumpus and self.agent_location.is_above(self.wumpus_location)
            case Orientation.W: return self.has_wumpus and self.agent_location.is_right_of(self.wumpus_location)
            case Orientation.N: return self.has_wumpus and self.agent_location.is_below(self.wumpus_location) 
    
    def kill_attempt(self) -> bool:
        if not (self.has_wumpus and self.wumpus_alive): return False
        scream = self.wumpus_in_line_of_fire()
        self.wumpus_alive = not scream
        return scream
    
    def step(self, action: Action) -> Percept:
        special_reward = 0
        bump = False
        scream = False
        #if self.time_step == 999:
          #  self.game_over = True
        if self.game_over:
            reward = 0
        else:
            match action:
                case Action.LEFT:
                    self.agent_orientation = self.agent_orientation.turn_left()
                case Action.RIGHT: 
                    self.agent_orientation = self.agent_orientation.turn_right()
                case Action.FORWARD:
                    bump = self.agent_location.forward(self.agent_orientation)
                    if self.agent_has_gold: self.gold_location.set_to(self.agent_location)
                    if self.is_agent_at_hazard():
                        special_reward = -1000
                        self.game_over = True
                case Action.GRAB:
                    if self.agent_location.is_location(self.gold_location):
                        self.agent_has_gold = True
                case Action.SHOOT:
                    if self.agent_has_arrow:
                        scream = self.kill_attempt()
                        special_reward = -10
                        self.agent_has_arrow = False
                case Action.CLIMB:
                    if self.agent_location.is_location(Location(0, 0)):
                        if self.agent_has_gold:
                           special_reward = 1000
                           print("************ WIN ************")
                        if self.allow_climb_without_gold or self.agent_has_gold:
                            self.game_over = True
            reward = -1 + special_reward
        
        breeze = self.is_breeze()
        stench = self.is_stench()
        glitter = self.is_glitter()
        self.time_step = self.time_step + 1
        return Percept(self.time_step, bump, breeze, stench, scream, glitter, reward, self.game_over)
                   
    def visualize(self):
        for y in range(3, -1, -1):
            line = '|'
            for x in range(0, 4):
                loc = Location(x, y)
                cell_symbols = [' ', ' ', ' ', ' ']
                if self.is_agent_at(loc): cell_symbols[0] = self.agent_orientation.symbol()
                if self.is_pit_at(loc): cell_symbols[1] = 'P'
                if self.has_wumpus and self.is_wumpus_at(loc):
                    if self.wumpus_alive:
                        cell_symbols[2] = 'W'
                    else:
                        cell_symbols[2] = 'w'
                if self.is_gold_at(loc): cell_symbols[3] = 'G'
                for char in cell_symbols: line += char
                line += '|'
            print(line)

class StateAwareAgent:
    location: Location
    location_ = np.ndarray
    orientation: Orientation
    orientation_: np.ndarray
    visited_locations_: np.ndarray
    stench_locations_: np.ndarray
    breeze_locations_: np.ndarray
    heard_scream_: np.ndarray
    has_gold_: np.ndarray
    has_arrow_: np.ndarray
    is_glitter_: np.ndarray
        
    def init(self, percept: Percept):
        self.visited_locations_ = np.zeros(16)
        self.stench_locations_ = np.zeros(16)
        self.breeze_locations_ = np.zeros(16)
        self.set_orientation(Orientation.E)
        self.set_location(Location(0, 0))
        self.heard_scream_ = np.array([0.])
        self.has_gold_ = np.array([0.])
        self.has_arrow_ = np.array([1.])
        self.is_glitter_ = np.array([0.])
        if percept.breeze: self.set_breeze_at(Location(0, 0))
        if percept.stench: self.set_stench_at(Location(0, 0))
    
    def set_orientation_(self):
        match self.orientation:
            case Orientation.E: self.orientation_ = np.array([1., 0., 0., 0.])
            case Orientation.S: self.orientation_ = np.array([0., 1., 0., 0.])
            case Orientation.W: self.orientation_ = np.array([0., 0., 1., 0.])
            case Orientation.N: self.orientation_ = np.array([0., 0., 0., 1.])
                
    def set_orientation(self, orientation: Orientation):
        self.orientation = orientation
        self.set_orientation_()
    
    def set_location_(self):
        self.location_ = np.zeros(16)
        self.location_[self.location.to_linear()] = 1.0
        self.visited_locations_[self.location.to_linear()] = 1.0
        
    def set_location(self, location):
        self.location = location
        self.set_location_()
    
    def set_stench_at(self, location: Location):
        self.stench_locations_[location.to_linear()] = 1.0
        
    def set_breeze_at(self, location: Location):
        self.breeze_locations_[location.to_linear()] = 1.0
        
    def set_heard_scream(self):
        self.heard_scream_ = np.array([1.])
        
    def turn_right(self):
        self.orientation = self.orientation.turn_right()
        self.set_orientation_()
        
    def turn_left(self):
        self.orientation = self.orientation.turn_left()
        self.set_orientation_()
        
    def forward(self, percept: Percept):
        self.location.forward(self.orientation)
        self.set_location_()
        if percept.breeze: self.set_breeze_at(self.location)
        if percept.stench: self.set_stench_at(self.location)
        if percept.glitter:
            self.is_glitter_ = np.array([1.])
        else:
            self.is_glitter_ = np.array([0.])
        
    def set_has_gold(self):
        self.has_gold_ = np.array([1.])
        
    def set_used_arrow(self):
        self.has_arrow_ = np.array([0.])
        
    def render_np(self) -> np.array:
        
        return np.concatenate((
            self.location_,
            self.orientation_,
            self.visited_locations_,
            self.stench_locations_,
            self.breeze_locations_,
            self.heard_scream_,
            self.has_gold_,
            self.has_arrow_,
            self.is_glitter_
            ))
    
    def step(self, action: Action, percept: Percept):
        match action:
            case Action.FORWARD:
                self.forward(percept)
            case Action.RIGHT: self.turn_right()
            case Action.LEFT: self.turn_left()
            case Action.GRAB:
                if percept.glitter:
                    if self.has_gold_ == np.array([0.]):
                        print('Snagged gold at step ', percept.time_step)
                    self.set_has_gold()
            case Action.SHOOT:
                self.set_used_arrow()
                if percept.scream:
                    self.set_heard_scream()
                    print('Killed Wumpus at step ', percept.time_step)

    def print_belief_state(self):
        print('Loc:', self.location_, 'Orient:', self.orientation_, 'Visited:', self.visited_locations_)
        print('Stenches:', self.stench_locations_, 'Breezes:', self.breeze_locations_)
        print('Scream:', self.heard_scream_, 'Gold:', self.has_gold_, 'Arrow:', self.has_arrow_)

    def run_naive(self):
        env = Environment()
        percept = env.init(0.2, True, True)
        self.init(percept)
        env.visualize()
        self.print_belief_state()
        while not percept.done:
            print()
            print('Percept:', percept)
            action = self.choose_naive_action()
            print('Action:', action)
            percept = env.step(action)
            env.visualize()
            self.step(action, percept)
            self.print_belief_state()
        print('Percept:', percept)

    def choose_naive_action(self):
        return Action.random()
    
        print('Percept:', percept)
        print('Cumulative reward:', cumulative_reward)

def update_params(worker_opt,values,logprobs,rewards,clc=0.1,gamma=0.95):
        rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1) #A
        logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
        values = torch.stack(values).flip(dims=(0,)).view(-1)
        Returns = []
        ret_ = torch.Tensor([0])
        for r in range(rewards.shape[0]): #B
            ret_ = rewards[r] + gamma * ret_
            Returns.append(ret_)
        Returns = torch.stack(Returns).view(-1)
        Returns = F.normalize(Returns,dim=0)
        actor_loss = -1*logprobs * (Returns - values.detach()) #C
        critic_loss = torch.pow(values - Returns,2) #D
        loss = actor_loss.sum() + clc*critic_loss.sum() #E
        loss.backward()
        worker_opt.step()
        return actor_loss, critic_loss, len(rewards)

def run_episode(agent: StateAwareAgent, environment: Environment, worker_model, max_steps: int):
    state = torch.from_numpy(agent.render_np()).float() #A
    values, logprobs, rewards = [],[],[] #B
    done = False
    j=0
    while (done == False and j < max_steps): #C
        j+=1
        policy, value = worker_model(state) #D
        values.append(value)
        logits = policy.view(-1)
       # print(f'Logits: {logits}')
        action_dist = torch.distributions.Categorical(logits=logits)
       # print(f'Action distribution: {action_dist}')
        action = action_dist.sample() #E
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        act = Action(action.detach().item())
        percept = environment.step(act)
        # print(f'Action: {act} Percept: {percept}')
        agent.step(act, percept)
        state_ = agent.render_np()
        done = percept.done
        state = torch.from_numpy(state_).float()
        rewards.append(percept.reward)
    return values, logprobs, rewards

def worker(t, worker_model, counter, params):
    print(f'Worker {t} starting.')
    worker_opt = optim.Adam(lr=1e-4, params=worker_model.parameters())
    worker_opt.zero_grad()
    for i in range(params['epochs']):
        print(f'Worker {t}: epoch:{i}')
        worker_opt.zero_grad()
        worker_env = Environment()
        initial_percept = worker_env.init(0.0, False, True)
        agent = StateAwareAgent()
        agent.init(initial_percept)
        values, logprobs, rewards = run_episode(agent, worker_env, worker_model, params['max_steps'])
        print(f'Rewards total: {sum(rewards)}')
        actor_loss, critic_loss,eplen = update_params(worker_opt, values, logprobs, rewards)
        counter.value = counter.value + 1 #D

from torch import nn
from torch import optim
from torch.nn import functional as F

class ActorCritic(nn.Module): #B
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(72,25)
        self.l2 = nn.Linear(25,50)
        self.actor_lin1 = nn.Linear(50,6) # 6 possible actions
        self.l3 = nn.Linear(50,25)
        self.critic_lin1 = nn.Linear(25,1)
    def forward(self,x):
        x = F.normalize(x,dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = F.log_softmax(self.actor_lin1(y),dim=0) #C
        c = F.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c)) #D
        return actor, critic #E
    
    def save(self, path):
        torch.save(self.state_dict(), path)

if __name__ == '__main__':
    processors = mp.cpu_count()

    print(processors, ' processors found.')

    MasterNode = ActorCritic()

    MasterNode.share_memory()

    processes = []

    params  = {
        'epochs': 20000,
        'n_workers': processors - 4,
        'max_steps': 50000
    }

    counter = mp.Value('i', 0)

    for i in range(params['n_workers']):
        p = mp.Process(target=worker, args=(i, MasterNode, counter, params))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for p in processes:
        p.terminate()

    path = f'wumpusworld_ac3 without pits or exit {params.epochs} episodes max {params.max_steps} steps {params.n_workers} workers.pth'
    MasterNode.save(path)
    print('Model saved to ', path)