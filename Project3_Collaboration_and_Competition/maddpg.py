import numpy as np
import ddpg
import replay_buffer
import torch

class MADDPG:
    
    def __init__(self, hp):
        self.hp = hp
        self.memory = replay_buffer.ReplayBuffer(hp)
        self.agents = [ddpg.Agent(self.hp) for _ in range(self.hp.num_agents)]
        self.losses = (0., 0.)
        
    def reset(self):
        for agent in self.agents: agent.reset()
            
    def act(self, states, add_noise = True):
        return [agent.act(state, add_noise) for agent, state in zip(self.agents, states)]
    
    # The replay buffer and the method step were deleted from the ddpg module. 
    # Now both concepts are centralized in this class.
    def step(self, states, actions, rewards, next_states, dones):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) < self.hp.batch_size: return
        critic_losses = []
        actor_losses = []
        for agent in self.agents:
            experiences = self.memory.sample()
            critic_loss, actor_loss = agent.learn(experiences)
            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)
        self.losses = (np.mean(critic_losses), np.mean(actor_losses))


    def save_weights(self):
        """Saves all networks."""
        for i in range(self.hp.num_agents):
            torch.save(self.agents[i].actor_local.state_dict(), 'checkpoint_actor_{}.pth'.format(i))
            torch.save(self.agents[i].critic_local.state_dict(), 'checkpoint_critic_{}.pth'.format(i))
 
    def load_weights(self):
        """Restores all agents."""
        for i in range(self.hp.num_agents):  
            self.agents[i].actor_local.load_state_dict(torch.load('checkpoint_actor_{}.pth'.format(i), map_location='cpu'))
            self.agents[i].critic_local.load_state_dict(torch.load('checkpoint_critic_{}.pth'.format(i), map_location='cpu'))
      