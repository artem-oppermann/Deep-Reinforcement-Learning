# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:13:32 2018

@author: Admin
"""

import numpy as np

class OrnsteinUhlenbeckActionNoise:
    
    def __init__(self, env,mu= 0.0, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        
        self.theta =  theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.initial_noise_scale = 0.1	
        self.noise_decay = 0.99		
        self.noise_process = np.zeros(1)
        
        self.env=env
        
    def get_noise(self, ep):   
        
        noise_scale = (self.initial_noise_scale * self.noise_decay**ep) * (self.env.action_space.high - self.env.action_space.low)
        self.noise_process = self.theta*(self.mu - self.noise_process) + self.sigma*np.random.randn(1)
        
        return noise_scale*self.noise_process