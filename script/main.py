import argparse
import pickle
from collections import namedtuple
from itertools import count
import random
import os, time
import numpy as np
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from time import sleep


from grid import Agent, Task, GridWorld


height=6
width=8

class Actor(nn.Module):

    def __init__(self,observation_space,action_space):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(observation_space, 128)
        self.action_head = nn.Linear(128, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x=self.action_head(x)
        # x = [a1:[t1,t2,t3],a2:[t1,t2,t3]]
        x = x.reshape(len(x), 2, 3)
        action_prob = F.softmax(x, dim=-1)
        return action_prob  # (N, A, T) (batch_size,agent_num,task_num)


class Critic(nn.Module):
    def __init__(self,observation_space):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(observation_space, 128)
        self.state_value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO():


    def __init__(self,observation_space,action_space,args):
        super(PPO, self).__init__()
        self.clip_param = args.clip_param
        self.max_grad_norm = args.max_grad_norm
        self.ppo_update_time = args.ppo_update_time
        self.buffer_capacity = args.buffer_capacity
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.eps = args.eps
        self.entropy_coef = args.entropy_coef
        self.agent_num=args.agent_number

        self.actor_net = Actor(observation_space,action_space)
        self.critic_net = Critic(observation_space)
        self.buffer = []
        self.counter = 0
        self.training_step = 0

        if not os.path.exists('runs_result/ppo'):
            os.makedirs('runs_result/ppo')
        self.writer = SummaryWriter('runs_result/ppo')

        self.lossvalue_norm=True
        self.loss_coeff_value=0.5

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)


    def select_action(self, state,train):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
            if train==True:
                if random.random() > self.eps:
                    action_prob = action_prob
                else:
                    N, A, T = action_prob.shape
                    uniform_prob = np.zeros((N, A, T))
                    for n in range(N):
                        for a in range(A):
                            uniform_prob[n][a] = np.random.uniform(low=0, high=1, size=(T,))
                    action_prob = torch.tensor(uniform_prob)
                self.eps*=0.99999
                self.eps=max(self.eps,0.01)
            else:
                action_prob = action_prob
            action,action_log_prob=self.encode(action_prob)
            return action,action_log_prob

    def encode(self,action_prob):
        dist=Categorical(action_prob)
        agent_action = dist.sample()  # (N, A)
        log_prob= dist.log_prob(agent_action)
        action = agent_action.detach()
        action_log_prob = log_prob.detach()
        return action, action_log_prob  # (N, A)

    def save_param(self):
        torch.save(self.actor_net.state_dict(), 'runs_result/actor_net.pth')
        torch.save(self.critic_net.state_dict(), 'runs_result/critic_net.pth')
        print('save completely')

    def load(self):
        self.actor_net.load_state_dict(torch.load( 'runs_result/actor_net.pth'))
        self.critic_net.load_state_dict(torch.load('runs_result/critic_net.pth'))
        print('load completely')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action= torch.cat([t.action for t in self.buffer]).long()
        reward = [t.reward for t in self.buffer]
        old_action_log_prob=torch.cat([t.a_log_prob for t in self.buffer]).float()
        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)


        for i in range(self.ppo_update_time):

            self.entropy_coef*=0.9999

            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):

                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])

                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]) #N(batch_size)*A(agent with assigned task)
                old_action=action[index]
                act_prob=[]
                for i in range(action_prob.shape[0]):
                    #todo: multi task
                    prob_0=action_prob[i][0][old_action[0][0]]
                    prob_1 =action_prob[i][1][old_action[0][1]]
                    act_prob.append([prob_0,prob_1])

                act_prob = torch.tensor(act_prob, dtype=torch.float) # N*A

                a=torch.tensor(1e-5,requires_grad=True)
                ratio = act_prob / torch.maximum(old_action_log_prob[index],a.repeat(act_prob.shape,1))
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                loss_surr=-(torch.min(surr1, surr2)).mean()
                if self.lossvalue_norm:
                    return_std=Gt[index].std()
                    loss_value=torch.mean((self.critic_net(state[index])-Gt[index]).pow(2))/return_std

                else:
                    loss_value = torch.mean((self.critic_net(state[index]) - Gt[index]).pow(2))

                # update actor network
                action_loss = loss_surr+self.loss_coeff_value*loss_value
                self.writer.add_scalar('ppo-train_loss/action_loss', action_loss, global_step=self.training_step)

                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.writer.add_scalar('ppo-train_loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:]


def main(args):
    # Parameters
    agents = [Agent([1,0]),
              # Agent([1,0]),
              Agent([0,1])]
    tasks = [Task([1,0], 100), Task([0,1], 100),Task([1,1], 100)]
    global env
    env=GridWorld(
        (10, 10),
        agents,
        tasks,
        timeout=100
    )

    gamma =args.gamma

    seed = args.seed

    # wrapped

    observation_space = env.obs_dim
    #print('observation_space',observation_space)

    action_space = env.act_dim
    #print('action_space',action_space)
    torch.manual_seed(seed)

    model=PPO(observation_space,action_space,args)
    Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward'])

    if args.mode=='train':
        for i_epoch in range(args.max_episode):
            observation,info = env.reset()
            total_reward=0
            while not env._get_done():
                action, action_prob = model.select_action(observation,train=True)
                goal = []
                for i in range(len(agents)):
                    goal.append(tasks[action[0][i].item()].position)
                next_observation, reward, done, info  = env.step(goal)
                if args.render:
                     env.render()
                     sleep(1)
                trans = Transition(observation, action, action_prob, reward)
                #print('reward',reward)
                model.store_transition(trans)
                observation=next_observation
                total_reward+=reward
                if done:
                    if len(model.buffer) >= model.batch_size:
                        model.update(i_epoch)
               # agent.writer.add_scalar('episode/return', sum(r * gamma ** t for t, r in enumerate(rewards)), i_epoch)
                    model.writer.add_scalar('ppo-train/return', total_reward, i_epoch)
                    break

        model.save_param()

    else:
        model.load()
        for i_epoch in range(args.eval_times):
            observation,info=env.reset()
            total_reward=0
            action, action_prob = model.select_action(observation, train=False)
			#todo:encode goal
            goal=[]
            for i in range(len(agents)):
                goal.append(tasks[action[i]].position)
            next_observation, reward, done, _, info = env.step(goal)
            total_reward+=reward
            observation=next_observation
            if done:
                model.writer.add_scalar('ppo-test/return', total_reward, i_epoch)
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--scenario', default="classic_CartPole-v0", type=str)

    parser.add_argument('--max_episode', default=5000,type=int)
    parser.add_argument('--algo', default="PPO", type=str)
    parser.add_argument('--buffer_capacity', default=int(1e5), type=int)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--clip_param', default=0.2, type=int)
    parser.add_argument('--max_grad_norm', default=0.5, type=int)
    parser.add_argument('--ppo_update_time', default=10, type=int)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--entropy_coef', default=0.5)
    parser.add_argument('--eps', default=1)
    parser.add_argument('--opponent_policy', default='random',help='random/greedy')
    parser.add_argument('--mode', default="train", type=str, help="train/evaluate")
    parser.add_argument('--eval_times', default=1000, type=int)
    parser.add_argument('--render', default=False, type=str)
    parser.add_argument('--agent_number', default=3, type=int)

    args = parser.parse_args()
    main(args)
    print("end")
