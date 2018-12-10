# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:12:05 2018

@author: Admin
"""

class ExperienceReplay:
    
    def __init__(self, max_experience,min_experience, flags):
        
        self.max_experience=max_experience
        self.min_experience=min_experience
        self.FLAGS=flags
        
        self.experience={'state':[], 'action':[],'reward':[], 'next_state':[], 'done':[]}
        
    
    def addExperience(self, state, action, reward, next_state,done):
        
        if len(self.experience)>self.max_experience:
            self.experience['state'].pop(0)
            self.experience['action'].pop(0)
            self.experience['reward'].pop(0)
            self.experience['next_state'].pop(0)
            self.experience['done'].pop(0)

        self.experience['state'].append(state)
        self.experience['action'].append(action)
        self.experience['reward'].append(reward)
        self.experience['next_state'].append(next_state)
        self.experience['done'].append(done)
    

    def get_sample(self):
        
        idx = np.random.choice(len(self.experience['state']), size=self.FLAGS.batch_size, replace=False)
        
        state=np.array([self.experience['state'][i] for i in idx]).reshape(self.FLAGS.batch_size,self.state_size)
        action=[self.experience['action'][i] for i in idx]
        reward=[self.experience['reward'][i] for i in idx]
        next_state=np.array([self.experience['next_state'][i] for i in idx]).reshape(self.FLAGS.batch_size,self.state_size)
        dones=[self.experience['done'][i] for i in idx]
        
        return state, action, reward, next_state, dones