#Uncomment these when running on Google Colab
#!pip3 install box2d-py
#!pip3 install pyglet==1.5.11


# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 16:41:54 2021

@author: poujo
"""
import os 
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import gym
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# %%
""" Defining Environment """
class Environnement:
    
    def __init__(self, E):
        self.E = E;
        self.env = gym.make(E);
        
    def play_with_agent(self,agent, episode):
        S = self.env.reset();
        R_total = 0;
        done = False;
        nb_steps = 0;
        while not done and nb_steps < self.step_limit:
            nb_steps += 1;
            
            # if episode%50==0:
            #     self.env.render()
            A =  agent.take_action(S);
            #print("Ceci est mon action :",A)
            S_dot, R, done, info = self.env.step(A);
            
            #if nb_steps > 0.9*self.step_limit:#This is ad hoc
                #R -= 100;
                
            if done:
                S_dot = None;
            
            agent.new_experience([S, A, R, S_dot]);
            #if nb_steps%10==0:
            agent.learn()
            
            S = S_dot;
            R_total += R;
            
        
        return R_total
    
    def play_with_agentAtRandom(self,agent, episode):
        S = self.env.reset();
        done = False;
        nb_steps = 0;
        while not done and nb_steps < self.step_limit:
            nb_steps += 1;
            A =  self.env.action_space.sample();
            #print("Ceci est mon action :",A)
            S_dot, R, done, info = self.env.step(A);
            
            if done:
                S_dot = None;
            
            agent.new_experienceRandom([S, A, R, S_dot]);
        
        agent.epsilon = agent.epsMax;
        agent.pas = 0;
    
    def play_with_trainedAgent(self, agent, episode):
        """In this function, we only use a given agent and do not train"""
        S = self.env.reset();
        R_total = 0;
        done = False;
        nb_steps = 0;
        while not done and nb_steps < self.step_limit:
            nb_steps += 1;
            
            if episode%100==0:
                self.env.render()
            A =  agent.take_actionNoRandom(S);
            #print("Ceci est mon action :",A)
            S_dot, R, done, info = self.env.step(A);
            
            if done:
                S_dot = None;
            
            #agent.new_experience([S, A, R, S_dot]);
            #agent.learn()
            
            S = S_dot;
            R_total += R;
        return R_total

# %%
""" Defining Neural Network """
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
#run_eagerly = False

#tensorflow.keras (may allow the use of predict_on_batch)
"""from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as bck
"""

#keras alone

from keras.models import Sequential, load_model
from keras.layers import *
from keras.optimizers import *
from keras import backend as bck


class Associated_DQN:
    #BatchSize = 32;
    def __init__(self, TEE, TEA, trained=False,l=0.001,s1 = 164, s2=128):
        self.stateSpace = TEE;
        self.actionSpace = TEA;
        self.l = l;
        self.size1 = s1;
        self.size2  =s2;
        
        if not trained:
            #Building Neural Network 
            Q_NN =  Sequential();
            #Q_NN.add(experimental.preprocessing.Normalization(axis=-1, dtype=None, mean=None, variance=None))
            
            Q_NN.add(Dense(units = s1, activation = 'relu', input_dim=self.stateSpace)) #units = output_space_dimension
            Q_NN.add(Dense(units = s2, activation = 'relu')) #units = output_space_dimension
            
            Q_NN.add(Dense(units =self.actionSpace, activation = 'linear'))
            #opt = SGD(learning_rate=0.005);
            opt = Adam(lr=l);
            Q_NN.compile(loss='mse' , optimizer=opt)
            #End of building Neural Network
            
            self.Q_NN = Q_NN;
            
            
            #Building Target Neural Network - same as before 
            targetQ_NN =  Sequential();
            #targetQ_NN.add(experimental.preprocessing.Normalization(axis=-1, dtype=None, mean=None, variance=None))
            
            targetQ_NN.add(Dense(units = s1, activation = 'relu', input_dim=self.stateSpace)) #units = output_space_dimension
            targetQ_NN.add(Dense(units = s2, activation = 'relu')) #units = output_space_dimension
            
            targetQ_NN.add(Dense(units =self.actionSpace, activation = 'linear'))
            #opt = SGD(learning_rate=0.005);
            targetQ_NN.compile(loss='mse', optimizer=opt)#Using absolute value loss function
            #End of building Neural Network
            self.targetQ_NN = targetQ_NN;
        else:
            self.Q_NN = load_model('LunarLander_NN.h5');
        
    #Functions for the Q_NN
    def training(self, x, y, epoch=1, verbose=0):
        return self.Q_NN.fit(x, y, epochs = epoch, verbose=verbose)
    
    def prediction(self, s):
        #print(s)
        return self.Q_NN.predict_on_batch(s)
    
    def predictionOneState(self, S):
        #print(S)
        s = S.reshape(1,self.stateSpace);
        #print(s)
        return self.Q_NN.predict_on_batch(s).flatten();
    
    #Functions for the TargetQ_NN
    def predictionTarget(self, s):
        #print(s)
        return self.targetQ_NN.predict_on_batch(s)
    
    def predictionTargetOneState(self, S):
        #print(S)
        s = S.reshape(1,self.stateSpace);
        #print(s)
        return self.targetQ_NN.predict_on_batch(s).flatten();
    
    def update_targetQ_NN(self):
        self.targetQ_NN.set_weights(self.Q_NN.get_weights())

# %%
""" Implementing Experience Storage """
class ExperienceReplay:
    stock = [];
    def __init__(self, SizeOfStock):
        self.SizeOfStock = SizeOfStock;
        self.stock = [];
    
    def remmeber(self, quadruplet):
        if len(self.stock) >= self.SizeOfStock:
            self.stock.pop(0)
        self.stock.append(quadruplet)
        
    def call4experience(self, sizeOfCall):
        n = min(sizeOfCall, len(self.stock));
        return random.sample(self.stock,n)

# %%
""" Implementing the Agent """
import math 
class Agent:
    epsMax = 0.99;
    epsMin = 0.01;
    gamma = 0.99;
    sizeOfCall = 64;#Size of batch to call
    epsilon = epsMax;
    pas = 0;
    LAMBDA = 0.005; #decay kinetic
    C = 300; #Update targetQ_NN every C steps 
    
    def __init__(self, TEE, TEA, trained=False, l=0.001, sizeOfStock = 100000,s1=164,s2=128):
        self.TEE = TEE;
        self.TEA = TEA;
        self.sizeOfStock = sizeOfStock;
        self.associated_DQN = Associated_DQN(TEE, TEA,trained,l,s1,s2);
        self.experienceReplay = ExperienceReplay(self.sizeOfStock)
        self.epsilon = self.epsMax
        
        
    
    def take_action(self, s):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, TEA)
        else:
            return np.argmax(self.associated_DQN.predictionOneState(s))
        
         
    def take_actionNoRandom(self, s):
        return np.argmax(self.associated_DQN.predictionOneState(s))
    
    def new_experience(self, quadruplet):
        self.experienceReplay.remmeber(quadruplet)
        
        self.epsilon = self.epsMin+(self.epsMax-self.epsMin)*math.exp(-self.LAMBDA*self.pas);
        self.pas += 1;
        
        if self.pas % self.C == 0:
            self.associated_DQN.update_targetQ_NN()
            
    def new_experienceRandom(self, quadruplet):
        self.experienceReplay.remmeber(quadruplet)
               
        
    def learn(self):
        #Pick some previous experiences that are still stored
        some_experiences = self.experienceReplay.call4experience(self.sizeOfCall);
        len_exp = len(some_experiences);
        
        All_S = np.array([quadruplet[0] for quadruplet in some_experiences]);
        All_S_dot = np.array([(np.zeros(self.TEE) if quadruplet[3] is None else quadruplet[3]) for quadruplet in some_experiences])
        
        prediction = self.associated_DQN.prediction(All_S);
        prediction_dot = self.associated_DQN.predictionTarget(All_S_dot);#Prediction S_dot for TargetQ_NN
        
        x = np.zeros((len_exp, self.TEE));
        y = np.zeros((len_exp,self.TEA));
        
        for i in range(len_exp):
            quadruplet = some_experiences[i];
            
            #unfold quadruplet
            S = quadruplet[0];
            A = quadruplet[1];
            r = quadruplet[2];
            S_dot = quadruplet[3];
            
            target = prediction[i];
            if S_dot is None:
                target[A] = r;
            else:
                target[A] = r + self.gamma*np.amax(prediction_dot[i]);
                
            x[i] = S;
            y[i] = target;
            # if i%100==0:
            #     print("\n\n### Here is X ###", x, "\n")
            
        # axes=[0, 1]
        # eps=1e-8
        # mean, variance = tf.nn.moments(x, axes=axes)
        # x = (x - mean) / tf.sqrt(variance + eps) # epsilon to avoid dividing by zero
        self.associated_DQN.training(x, y)
        
        #if self.pas % self.C == 0:
            #self.associated_DQN.update_targetQ_NN()
        
# %%
numberOfEpisodes = 800;
numberOfRandomRuns = 0;
#E = 'CartPole-v0';
E = 'LunarLander-v2';
gym_environnement = Environnement(E);

#TEE means State Space Size (Taille de l'Espace d'Etat, in french)
#TEA means Action Space Size (Taille de l'Espace d'Action, in french)
TEE = gym_environnement.env.observation_space.shape[0]
TEA = gym_environnement.env.action_space.n;

def train(E,gym_environnement,TEE,TEA,s1,s2):
    """In the train function, there are many optional parameters that can be entered
    and that allow to reproduce the figures presented in the report"""
    agent = Agent(TEE,TEA,False,l=0.001,s1=s1,s2=s2);
    #Population the replay store randomly
    for i in range(numberOfRandomRuns):
        print("--> iteration :",i,"of",numberOfRandomRuns," <--" )
        gym_environnement.play_with_agentAtRandom(agent, 3)
    
    R = []
    STEPS = []
    previous_pas = 0;
    for i in range(numberOfEpisodes):
        R += [gym_environnement.play_with_agent(agent, i)];
        STEPS += [agent.pas-previous_pas];
        previous_pas = agent.pas;
        if i%100 == 0:
            print("--> iteration :",i,"of",numberOfEpisodes," <--" )
            print("      epsilon :",agent.epsilon)
            print("      memoire :",len(agent.experienceReplay.stock))
            print("      reward  -|moyen :" ,np.mean(R[len(R)-100:]),"\n              -|last  :",R[i])
        if i>100 and np.mean(R[len(R)-100:])>200:
            break;
    agent.associated_DQN.Q_NN.save('LunarLander_NN.h5')
    T = [R, STEPS]
    return T


S = [1]
R_all = []
for s in S:
  R_training = train(E,gym_environnement,TEE,TEA,64,128)

  plt.plot(R_training[0], label = ["Run number ",s])
  plt.plot(R_training[1], label = ["Steps per episode"])
  
  R_all += [R_training]
plt.xlabel('Episodes')
plt.ylabel('Total reward per episode')
plt.title("Rewards on Training")
plt.legend()
plt.show()

#This load the agent saved at the end of the train function
agent = Agent(TEE,TEA, True);

R = []

for i in range(500):
    R += [gym_environnement.play_with_trainedAgent(agent, i)];
    if i%10 == 0:
        if i%100 == 0:
            print("\n--> iteration :",i,"of",500," <--" )
        print("°", end='')
        
    
    

gym_environnement.env.close()
plt.plot(R,'+',label = 'Reward per episode')
plt.plot([i for i in range(99,500)],np.convolve(np.array(R), np.ones(100), 'valid') / 100, label = 'Average Reward on 100 episodes')
plt.xlabel('Episodes')
plt.ylabel('Total reward per episode')
plt.title("Rewards for a Trained Agent")
plt.legend()
plt.show()

#Saving R in a separate file for later uses
with open('BestRunStep.npy', 'wb') as f:
  np.save(f,R_all)
  print(R_all)

#May (or may not) raise an error depending on what you change in the above...
i = 0;
for R in R_all:
  plt.plot(np.convolve(np.array(R), np.ones(100), 'valid') / 100, label = ["Layer sizes =",S[i]])
  i = i+1
  
plt.xlabel('Episodes')
plt.ylabel('Total reward per episode')
plt.title("Rewards on Training")
plt.legend()
plt.show()
                          