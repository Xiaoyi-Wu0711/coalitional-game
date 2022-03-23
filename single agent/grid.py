import random
from time import sleep
import numpy as np
import sys
from torch.utils.tensorboard import SummaryWriter
import os
import copy
import datetime


REWARD = 1
PUNISH = -.1


class Agent:
    def __init__(
        self,
        skill,  # {skill_type: capacity}
    ):
        self.skill = skill
        self.position = None
        self.goal = None  # (x, y)

    def update_goal(self, goal):
        self.goal = goal

    def set_world(self, world):
        self.world = world

    def init(self, position):
        if not hasattr(self, "world"):
            raise Error("world not set.")
        
        self.position = position

    def act(self):
        x, y = self.position
        x_, y_ = self.goal
        xn = np.sign(x_ - x)
        if xn != 0:
            x += xn
        else:
            y += np.sign(y_ - y)

        self.position = (x, y)



class Task:
    def __init__(
        self, 
        require,  # {skill_type: capacity}
        timeout
    ):
        self.require = require
        self.time = timeout
        self.position = None

    def set_world(self, world):
        self.world = world

    def init(self, position):
        if not hasattr(self, "world"):
            raise Error("world not set.")
        
        self.position = position
        self.resource = self.require.copy()

    def update(self):
        agents=self.world.agents
        for i in range(len(self.resource)):
            for agent in agents:
                if agent.position==self.position and self.resource[i]!=0 and agent.skill[i]!=0:
                    self.resource[i]-=1

    def is_done(self):
        # check if fulfilled
        # fulfill from timeout, 0 reward
        # otherwise, 1 reward
        if self.world.time > self.time:
            return (True, 0)
        if sum(self.resource) == 0:
            return (True, 1)
        return (False,0)

# todo: agent constant skill capacity
# A: 50
# agent: A:1

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class GridWorld:

    def __init__(
        self, 
        grid_size,  # (w, h)
        agents,
        tasks,
        timeout
    ):
        self.agents = agents
        self.tasks = tasks
        for task in tasks:
            task.set_world(self)
            self.require_num=len(task.require)
        for agent in agents:
            agent.set_world(self)
            self.skill_num=len(agent.skill)
            # self.goal_num=len(agent.goal)

        self.grid_size = grid_size
        # set required vectorized gym env property
        self.agent_num = len(agents)
        self.task_num = len(tasks)

        self.timeout = timeout
        self.w = grid_size[0]
        self.h=grid_size[1]
        self.time = 0
        self.reward = 0
        self.info=None
        self.state=None
        #todo: observation dim
        self.obs_dim=20*self.agent_num+self.skill_num*self.agent_num+self.task_num*20+self.task_num*self.require_num+200

        #todo: action dim
        self.act_dim=len(agents)*len(tasks)

        # rendering
        self.mesh = "*" * (self.w+2) + "\n"
        for _ in range(self.h):
            self.mesh += "*" + "-" *self.w + "*" "\n"

        self.mesh += "*" * (self.w+2)



    def step(self, goals):

        # set action for each agent
        for i, agent in enumerate(self.agents):
            agent.update_goal(goals[i])
            agent.act()

        for task in self.tasks:
            task.update()

        next_state=self.update_state()

        new_reward = 0

        for task in self.tasks:
            (done, r) = task.is_done()
            if done:
                new_reward += r
                task.init((np.random.randint(self.w),np.random.randint(self.h)))

        self.time += 1

        # record observation for each agent
        info = self._get_info()
        obs = self._get_obs(next_state)
        reward = new_reward
        done = self._get_done()

        return obs,reward, done, info


    def reset(self):
        # reset world
        self.time = 0
        self.reward = 0
        self.r=copy.copy(self.reward)
        # np.random.seed(args.seed)

        # reset tasks and agents
        for task in self.tasks:
            task.init(
                (np.random.randint(self.w),
                np.random.randint(self.h),)
            )

        for agent in self.agents:
            agent.init(
                (np.random.randint(self.w),
                np.random.randint(self.h),)
            )

        current_state=self.update_state()
        #agent update their goal

        info=self._get_info()
        observation= self._get_obs(current_state)


        return observation,info

    def update_state(self):
        self.state=np.zeros((self.w,self.h,2))
        for i in self.agents:
            pos=i.position
            self.state[pos[0]][pos[1]][0] = 1
        for i in self.tasks:
            pos=i.position
            self.state[pos[0]][pos[1]][1] = 1
        return self.state

    # get info used for benchmarking
    def _get_info(self):
        agent_position=[]
        # agent_task = []
        agent_skill = []
        # agent goal
        for agent in self.agents:
            agent_position.append(agent.position)
            agent_skill.append(agent.skill)
            # todo: agent goal
            #  agent_task.append(agent.goal)
        task_position=[]
        task_resource =[]
        for task in self.tasks:
            task_position.append(task.position)
            task_resource.append(task.resource)
        #self.info
        self.info={
            'agent_position':agent_position,
            'agent_skill':agent_skill,
            # 'agent_task':agent_task,
            'task_position':task_position,
            'task_resource':task_resource
        }

        return self.info

    # get observation for a particular agent
    def _get_obs(self,state):
        observations = np.zeros(self.obs_dim)
        state =state.flatten()# 10*10*2
        agent_pos=np.zeros((self.agent_num,2,10))
        for i in range(len(self.info['agent_position'])):
            pos=self.info['agent_position'][i]
            agent_pos[i][0][pos[0]]=1
            agent_pos[i][1][pos[1]]=1
        agent_position=agent_pos.flatten()
        agent_skill=np.array(self.info['agent_skill']).flatten()
        # agent_task=np.array(self.info['agent_skill']).flatten()
        task_pos=np.zeros((self.task_num,2,10))
        for i in range(len(self.info['task_position'])):
            pos=self.info['task_position'][i]
            task_pos[i][0][pos[0]]=1
            task_pos[i][1][pos[1]]=1

        task_position=task_pos.flatten()
        task_resource=np.array(self.info['task_resource']).flatten()

        offset = 0

        # observation: agent position
        observations[offset:offset+self.agent_num*20] =agent_position
        offset += 20*self.agent_num

        # observation: agent skill
        observations[offset:offset+self.skill_num*self.agent_num]=agent_skill
        offset += self.skill_num*self.agent_num

        # observation: task position
        observations[offset:offset+self.task_num*20]=task_position
        offset += self.task_num*20

        # observation:task_require
        observations[offset:offset+self.task_num*self.require_num]=task_resource
        offset += self.task_num*self.require_num

        # observation: task skill
        # observations[24:30]=agent_task
        # observation: global map
        observations[offset:]=state

        return observations

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self):
        return self.time >= self.timeout

    # get reward for a particular agent
    def _get_reward(self):
        return self.r

    # reset rendering assets
    def _reset_render(self):
        pass

    # render environment
    def render(self):
        mesh = list(self.mesh)
        w, h = self.grid_size
        agent_ls=['A','B','C']
        task_ls=['a','b','c']
        for i in range(len(self.agents)):
            x, y = self.agents[i].position
            mesh[(x+1) + (w+3)*(y+1)] =agent_ls[i]

        for i in range(len(self.tasks)):
            x, y = self.tasks[i].position
            mesh[(x+1) + (w+3)*(y+1)] =task_ls[i]

        print(' '+' '.join(mesh))
        print()
        pass



def assign_goal(policy,agents,tasks):
    goal_list=[]
    if policy=='random':
        for agent in agents:
            i=np.random.randint(len(tasks))
            goal_list.append(tasks[i].position)

    if policy=='greedy':

        for agent in agents:
            #{task:distance}
            qualify_task={}
            for task in tasks:
                for skill in range(len(task.require)):
                    if agent.skill[skill]!=0 and task.resource[skill]!=0:
                        qualify_task[task]=abs(agent.position[0]-task.position[0])+abs(agent.position[1]-task.position[1])
                        break
            #todo: check min func
            # print('qualify_task',qualify_task)

            sub_goal=min(qualify_task,key=qualify_task.get)
            goal_list.append(sub_goal.position)

    return goal_list


if __name__ == "__main__":
    agents = [Agent([1,0]),]
              # Agent([1,0]),
              # Agent([0,1])]
    #todo: 一次性生成足够量的任务， 任务做完后消失

    tasks = [Task([1,0], 100), Task([0,1], 100),Task([1,1], 100)]
    policy='greedy'
    gw = GridWorld(
        (10, 10),
        agents,
        tasks,
        timeout=100     
    )
    # np.random.seed(666)
    # np.random.seed(10)
    # np.random.seed(100)

    # # np.random.seed(233)
    # np.random.seed(7)
    # # np.random.seed(11)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir='./result/'+policy+'_'+current_time
    if not os.path.exists(dir):
        os.makedirs(dir)
    writer = SummaryWriter(dir)
    for i in range(5000):
        total_reward=0
        observation,info=gw.reset()
        while not gw._get_done():
            goal=assign_goal(policy,agents,tasks)
            obs, reward, done, info = gw.step(goal)
            # print('reward',reward)
            total_reward += reward
            # print('reward',total_reward)

            if done:
                label=str(policy)+'/return'
                writer.add_scalar(label, total_reward, i)
                # print("total_reward:", total_reward)
                print()

                break

        # if i == 4: break

    print('finish')
            # gw.render()
            # sleep(1)





