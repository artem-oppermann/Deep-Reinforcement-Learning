# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:22:48 2018

@author: Admin
"""

class ExperienceReplay:
    
    def __init__(self):
        
        self.__max_experience=10000
        self.__min_experience=100
        self.__experience={'state':[], 'action':[],'reward':[], 'next_state':[], 'done':[]}
    
    def get_experience(self):
        return self.__experience
    
    def get_min_experience_count(self):
        return self.__min_experience
    
    def get_max_experience_count(self):
        return self.__max_experience
    
    '''Add experience to the memory. '''
    def addExperience(self, state, action, reward, next_state,done):
        
        if len(self.__experience)>self.__max_experience:
            self.__experience['state'].pop(0)
            self.__experience['action'].pop(0)
            self.__experience['reward'].pop(0)
            self.__experience['next_state'].pop(0)
            self.__experience['done'].pop(0)

        self.__experience['state'].append(state)
        self.__experience['action'].append(action)
        self.__experience['reward'].append(reward)
        self.__experience['next_state'].append(next_state)
        self.__experience['done'].append(done)