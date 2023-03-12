# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 17:32:13 2021

@author: poujon
"""
import numpy as np
import cvxopt as co
import cvxpy as cp

#Playing around with solver parameter trying to faster the execution !
co.solvers.options['show_progress'] = False
co.solvers.options['maxiters']: 100
#co.solvers.options['glpk'] = {'LPX_K_MSGLEV': 0, 'msg_lev': "GLP_MSG_OFF"}


import math as math
import matplotlib.pyplot as plt

class Players:
    
    def __init__(self, p1, p2, ball):
        self.p1 = p1;#Positions are from 0 to 8
        self.p2 = p2;
        self.ball = ball;#Ball is 1 or 2 depending on who has the ball
        
    def reset(self):
        self.p1 = 1;
        self.p2 = 2;
        self.ball = 1;#At start player 1 has the ball
        return
        
    def globalMAJ(self, i, p, ball):
        self.p = p;
        self.ball = ball;
    
    def pMAJ(self, i, p):
        if i==1:
            self.p1 = p;
        else:
            self.p2 = p;
   
        
    def ballMAJ(self, ball):
        self.ball = ball;
        
    #In the call of the methods players are referenced to by 1 or 2
    def getPos(self, i):
        if i==1:
            return self.p1;
        if i==2:
            return self.p2
    
    def collision(self):
        if self.p1 == self.p2:
            return True;
        else:
            return False;
        
    def goal(self,i):
       """returns +1 if player i goals in right side
       returns -1 if player i goals in wrong side
       returns 0 else"""
       #print(i)
       if self.ball != i:
           return 0;
       if i == 1:
           if self.p1 == 7 or self.p1 == 3:
               return 1;
           elif self.p1 == 0 or self.p1 == 4:
               return -1;
           return 0;
       else:#i==2
           if self.p2 == 7 or self.p2 == 3:
               return -1;
           elif self.p2 == 0 or self.p2 == 4:
               return 1;
           return 0;
       
    def state(self):
        return (self.p1,self.p2,self.ball)
       
        
class Soccer:
    p1 = 1;
    p2 = 2;
    ball = 1;
    
    
    def __init__(self):
        self.players = Players(self.p1, self.p2, self.ball);
        self.states = self.createStateDict();
        self.steps = 0;
        
    def reset(self):
        self.players.reset()
        self.steps = 0;
        return
    
    def state(self):
        return self.players.state()
        
    def createStateDict(self):
        states = dict()
        id = 0
        for has_ball in range(2):
            for p1 in range(8):
                for p2 in range(8):
                    if p1 != p2:
                        s = (p1, p2,has_ball+1)
                        states[s] = id
                        id += 1
        return states
                
    def possible(self,i,a):
        pos = self.players.getPos(i);
        if a == 0:
            if ((pos+1)%4)!=0:
                self.players.pMAJ(i,pos+1);
                #print("   ->droite")
                
                return 1;
        elif a == 1:
            if (pos-4)>=0:
                self.players.pMAJ(i,pos-4)
                #print("   ->haut")
                
                return 1;
        elif a == 2:
            if ((pos-1)%4)!=3:
                self.players.pMAJ(i,pos-1)
                #print("   ->gauche")
                
                return 1;
        elif a ==3: #a == 3:
            if (pos+4)<=7:
                self.players.pMAJ(i,pos+4)
                #print("   ->bas")
                
                return 1;
        return -1;
        
    def action(self, A):
        self.steps += 1;
        """A = [a1, a2]"""
        R = [0,0];
        #Choose an action at random:
        i = np.random.choice(2);
        #print("Joue en premier : ",chr(65+i));
        a = A[i];
        
        #Do the action of random first player
        if self.possible(i+1,a)!=-1:
            g = self.players.goal(i+1)
            if self.players.collision():
                #print("Collision",chr(65+i))
                if (self.players.ball == (i+1)):
                    self.players.ballMAJ((i+1)%2+1); #Vol de la balle
                self.possible(i+1,(a-2)%4)
                S = self.players.state();
                return R,S;
            if g!=0:
                #print("Goal",chr(65+i))
                R = g*np.array([100,-100]);
                if i+1!=1:
                    R *= -1;
                S = self.players.state();
                return R,S;
            
        
        #Do the action of random second player
        i = (i+1)%2;
        #print("Joue en second : ",chr(65+i));
        a = A[i];
        if self.possible(i+1,a)!=-1:
            g = self.players.goal(i+1)
            if self.players.collision():
                #print("Collision",chr(65+i))
                if (self.players.ball == (i+1)):
                    self.players.ballMAJ((i+1)%2+1); #Vol de la balle
                self.possible(i+1,(a-2)%4)
                S = self.players.state();
                return R,S;
                
            if g!=0:
                #print("Goal",chr(65+i))
                R = g*np.array([100,-100]);
                if i+1!=1:
                    R *= -1;
                S = self.players.state();
                return R,S;
            
        S = self.players.state();
        return R,S;
    
    def render(self):
        grid = ["#" for i in range(8)];
        p1 = self.players.getPos(1);
        p2 = self.players.getPos(2);
        ball = self.players.ball;
        if ball == 1:
            if p1!=p2:
                grid[p1]="A";
                grid[p2]="b";
            else:
                grid[p1]="Ab"
        else:
            if p1!=p2:
                grid[p1]="a";
                grid[p2]="B";   
            else:
                grid[p1]="aB"
        print(grid[:4],"\n",grid[4:])
        return;
        
class Main():
    def __init__(self):
        pass
    
    def chooseAction(self, Q1, Q2, s):
        a1 = np.argmax(Q1[s]);
        a2 = np.argmax(Q2[s]);
        A = [a1,a2]
        return A
    
    #self methods for simple solvings
    def solveManual(self):
        env = Soccer();
        
        done = False;
        while not done:
            env.render();
            #choix actions
            a1 = int(input('action de A :'));
            a2 = int(input('action de B :'));
            A = [a1,a2];
            
            R = env.action(A);
            
            if R[0]!=0:
                done = True;
        env.render();
        print(R)
        return
    
    def solveAtRandom(self):
        env = Soccer();
        R,S = env.action([4,4]);
        print("State : ",S)
        done = False;
        while not done:
            env.render();
            print()
            #choix actions
            a1 = np.random.randint(5);
            a2 = np.random.randint(5);
            A = [a1,a2];
            
            R,S = env.action(A);
            print("State : ",S)
            if R[0]!=0:
                done = True;
        env.render();
        print(R)
        return


###### SOLVING FOR FIGURE IN ARTICLES ######
#%%
### LEARNER -- GENERAL CLASS ###
class SolveLearning():
    STATE_SPACE = 7*8*2;#Player 1 can be in one of the 8 cells, then 7 cells remain for player 2. And eihter player 1 or 2 can possess the ball
    ACTION_SPACE = 5;# Actions are indexed "0,1,2,3,4" for "E,N,W,S,Do nothing" respectively
    limitStep = 100;
    
    
    def __init__(self, env, gamma=.9, alpha=.1, decay=.00001, alpha_min=.001):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.decay = decay

        self.episodes = 0

        env.reset()

        self.Q1 = np.random.normal(
                loc=0., scale=3.,
                size=(self.STATE_SPACE, self.ACTION_SPACE, self.ACTION_SPACE))
        
        
        self.Q2 = np.random.normal(
                loc=0., scale=3.,
                size=(self.STATE_SPACE, self.ACTION_SPACE, self.ACTION_SPACE))
        
        

    def Alpha(self):
        return self.alpha * math.exp(-self.decay*self.episodes) + self.alpha_min

    def chooseActions(self, state=None):
        #Off-Policy        
        a1 = np.random.randint(5);
        a2 = np.random.randint(5);
        
        A = [a1, a2]
        
        return A

    def V(self, state, player):
        return 0

    def updateQ(self, S, A, S_dot, R):
            s = self.env.states[S];
            
            
            q = self.Alpha() * ((1-self.gamma)*R[0] + self.gamma * self.V(S_dot, 1))
            q += (1-self.Alpha())*self.Q1[s, A[0], A[1]]
            self.Q1[s, A[0], A[1]] = q
            
            q = self.Alpha() * ((1-self.gamma)*R[1] + self.gamma * self.V(S_dot, 2))
            q += (1-self.Alpha())*self.Q2[s, A[0], A[1]]
            self.Q2[s, A[0], A[1]] = q

    def train(self):
        self.episodes += 1

        S = self.env.state()
        A = self.chooseActions()
        R, S_dot = self.env.action(A)
       
        self.updateQ(S, A, S_dot, R)

        if self.env.steps > self.limitStep:
            self.env.reset()
        if R[0]!=0:
            self.env.reset()
            # print(rewards)
        
        return self.Q1[8,3,4].copy()
            
#%%
### FRIEND-Q LEARNING ###
class solveFriendQ(SolveLearning):

    def __init__(self, env, gamma=.9, alpha=1, decay=.000005, alpha_min=.001):
        super(solveFriendQ, self).__init__(env, gamma, alpha, decay, alpha_min)
        
    def V(self, s, player):
        if player == 1:
            v = np.max(self.Q1[self.env.states[s]])
        else:
            v = np.max(self.Q2[self.env.states[s]])
        return v
    
    def chooseActions(self, state=None):
        s = state if state is not None else self.env.state()
        A = []
        
        a = np.random.choice(range(self.ACTION_SPACE))
        A.append(a)
        if np.random.random() < 0.4:
            a = np.random.choice(range(self.ACTION_SPACE))
        else:
            a = np.argmax(self.Q1[self.env.states[s],a,:])
        A.append(a)

        #print(A)
        return A
    
    def updateQ(self, S, A, S_dot, R):
            s = self.env.states[S];
            
            
            q = self.Alpha() * ((1-self.gamma)*R[0] + self.gamma * self.V(S_dot, 1))
            q += (1-self.Alpha())*self.Q1[s, A[0], A[1]]
            self.Q1[s, A[0], A[1]] = q
            
            

### CORRELATED-Q LEARNING ###
class solveCorrelatedQ(SolveLearning):

    def __init__(self, env, gamma=.9, alpha=1, decay=.000005, alpha_min=.001):
        super(solveCorrelatedQ, self).__init__(env, gamma, alpha, decay, alpha_min)
        
    def updateQ(self, S, A, S_dot, R):
        Vs = self.V(S_dot)
        if Vs is None:
            return
        
        
        q = self.Alpha() * ((1-self.gamma)*R[0] + self.gamma * Vs[0])
        q += (1-self.Alpha())*self.Q1[self.env.states[S], A[0], A[1]]
        self.Q1[self.env.states[S], A[0], A[1]] = q
        
        q = self.Alpha() * ((1-self.gamma)*R[1] + self.gamma * Vs[1])
        q += (1-self.Alpha())*self.Q2[self.env.states[S], A[0], A[1]]
        self.Q2[self.env.states[S], A[0], A[1]] = q
        
    def corrEq(self, s_index):
        #G of size (40,25) --> 40 = 20 + 20
        G = np.zeros((2 * self.ACTION_SPACE * (self.ACTION_SPACE - 1), (self.ACTION_SPACE * self.ACTION_SPACE)))
        
        #Get the Q-arrays from s_index
        Mat1 = np.array(self.Q1[s_index])
        Mat2 = np.array(self.Q2[s_index]).T
        FlatMat1 = Mat1.flatten()
        FlatMat2 = Mat2.flatten()
        i = 0
    
        #Inequations for Correlated Q-Learning (eq 10 in 2007 paper)
        for a1 in range(self.ACTION_SPACE):
            for a2 in range(self.ACTION_SPACE):
                if a1 != a2:
                    G[i, a1 * self.ACTION_SPACE:(a1 + 1) * self.ACTION_SPACE] = Mat1[a1] - Mat1[a2]
                    G[i + self.ACTION_SPACE * (self.ACTION_SPACE - 1), a1:(self.ACTION_SPACE * self.ACTION_SPACE):self.ACTION_SPACE] = Mat2[a1] - Mat2[a2]
                    i += 1
    
        G = co.matrix(G)
        
    
        G = np.hstack((np.ones((G.size[0], 1)), G)) #A = [block of 1; block of correlated equations] 
        
        #Implementing pi > 0, padding by 0 to fit A dimensions
        pi_is_positive = np.hstack((np.zeros((self.ACTION_SPACE*self.ACTION_SPACE, 1)), -np.eye(self.ACTION_SPACE*self.ACTION_SPACE)))   
        G = co.matrix(np.vstack((G, pi_is_positive)))
        
        # Constraint: Sum(P) == 1 (We say pi => 1 and pi <= 1, I could have defined A and b such that A*x = b for this purpose)
        # But the solver takes much more time (and sometimes freezes when using the additionnal matrix system !)
        G = co.matrix(np.vstack((G, np.hstack((0,np.ones(self.ACTION_SPACE*self.ACTION_SPACE))), np.hstack((0,-np.ones(self.ACTION_SPACE*self.ACTION_SPACE))))))
    
        #h is a vector such that G*x <= h
        h = co.matrix(np.hstack((np.zeros(G.size[0] - 2), [1, -1])))
    
        #c is a vector such that the solver minimizes c.T*x (1 for first 'additional' term)
        #We do time -1 because we seek in fact maximisation !!!
        c = co.matrix(np.hstack(([-1.], -(Mat1+Mat2).flatten())))
        
        #Solving 
        sol = co.solvers.lp(c,G,h, solver='glpk')
    
    
        #Get optimized solution pi (could be None)
        if sol['x'] is None:
            return None, None
        pi = sol['x'][1:]
        
    
        #Compute pi*Q to find expected value
        v1 = np.matmul(FlatMat1, pi)[0]
        v2 = np.matmul(FlatMat2, pi)[0]
        
    
        return v1, v2
    
    def V(self, state):
        s_index = self.env.states[state];
        v = self.corrEq(s_index);
        if v == (None,None):
            return None
        return v



### FOE-Q LEARNING ###
class solveFoeQ(SolveLearning):
    StockV1 = []
    StockV2 = []
    def __init__(self, env, gamma=.9, alpha=1, decay=.000005, alpha_min=.001):
        super(solveFoeQ, self).__init__(env, gamma, alpha, decay, alpha_min)
        
        
    def updateQ(self, S, A, S_dot, R):
        s = self.env.states[S];
        
        v1 = self.V(S_dot, 1)
        v2 = self.V(S_dot, 2)
        self.StockV1.append(v1)
        self.StockV2.append(v2)
        
        if v1 != None:
            q = self.Alpha() * ((1-self.gamma)*R[0] + self.gamma * v1)
            q += (1-self.Alpha())*self.Q1[s, A[0], A[1]]
            self.Q1[s, A[0], A[1]] = q
        
        if v2 != None:
            q = self.Alpha() * ((1-self.gamma)*R[1] + self.gamma * v2)
            q += (1-self.Alpha())*self.Q2[s, A[0], A[1]]
            self.Q2[s, A[0], A[1]] = q

    def maximin(self, values):
        Mat_v = values.copy()

        # We need pi(a) >= 0 for all a
        # Which is G*pi <= h, thus G = -I 
        G = np.identity(self.ACTION_SPACE + 1)[1:, :]
        G = G * -1

        # values constraints
        G = np.append(G, np.insert(Mat_v.T, 0, -1, axis=1) * -1, axis=0)
        """
        G becomes [G       with arr being filled with -1 on first column (and then time -1 to fit the inequation)
                   arr]
        """

        G = G.T #Need to transpose G to fit cvxopt inequalities
        h = np.zeros(G.shape[1])
        
        # sum of action probabilities must equal 1, A[0,0] excluded as it corresponds to the constraints on the values 
        A = np.ones((G.shape[0], 1))
        A[0, 0] = 0
        b = [[1.]]

        

        # c.T * x to minimize
        c = np.zeros(G.shape[0])
        c[0] = -1

        # Formatting for solver
        G = co.matrix(G.tolist())
        h = co.matrix(h.tolist())
        c = co.matrix(c.tolist())
        A = co.matrix(A.tolist())
        b = co.matrix(b)

        """
        Minimize c.T*x such that Gx + s = h (where s > 0, so it is equivalent to Gx < h)
        And Ax = b                
        """
        
        solution = co.solvers.lp(c, G, h, A, b, solver='glpk')
        
        if solution['x'] is None:
            return None
        
        solution = list(solution['x'])
        
        return solution[0]  # return maximin value


    def V(self, state, player):
        if player == 1:
            Mat = self.Q1[self.env.states[state]]
        else:
            Mat = -self.Q1[self.env.states[state]]
            
        
        
            
        v = self.maximin(Mat)
        return v

### CLASSICAL Q-LEARNING ###
class solveClassicQ(SolveLearning):

    def __init__(self, env, gamma=.9, alpha=1, decay=.000005, alpha_min=.001, eps=1., epsDecay=.000005):
        super(solveClassicQ, self).__init__(env, gamma, alpha, decay, alpha_min)

        self.init_eps = eps
        self.eps_decay = epsDecay
        

        self.Q1 = np.random.normal(size=(self.STATE_SPACE, self.ACTION_SPACE))
        self.Q2 = np.random.normal(size=(self.STATE_SPACE, self.ACTION_SPACE))

        

    def Eps(self):
        return self.init_eps * math.exp(-self.eps_decay*self.episodes) + 0.01

    def updateQ(self, S, A, S_dot, R):
        
        q = (1-self.Alpha())*self.Q1[self.env.states[S], A[0]]
        q += self.Alpha() * ((1-self.gamma)*R[0] + self.gamma*self.V(S_dot, 1))
        self.Q1[self.env.states[S], A[0]] = q

    def chooseActions(self, state=None):
        s = state if state is not None else self.env.state()
        A = []
        
        if np.random.random() < self.Eps():
            a = np.random.choice(range(self.ACTION_SPACE))
        else:
            a = np.argmax(self.Q1[self.env.states[s]])
        A.append(a)
            
        if np.random.random() < self.Eps():
            a = np.random.choice(range(self.ACTION_SPACE))
        else:
            a = np.argmax(self.Q2[self.env.states[s]])
        A.append(a)

        return A

    def V(self, state, player):
        if player == 1:
            v = np.max(self.Q1[self.env.states[state],:])
        else:
            v = np.max(self.Q2[self.env.states[state],:])
        
        return v
    
    def train(self):
        self.episodes += 1

        S = self.env.state()
        A = self.chooseActions()
        R, S_dot = self.env.action(A)
       
        self.updateQ(S, A, S_dot, R)

        if self.env.steps > self.limitStep:
            self.env.reset()
        if R[0]!=0:
            self.env.reset()
            # print(rewards)
        
        return self.Q1[8,3].copy()



#%%
#THE ENVIRONMENT

m = Main();

#Some tests for the environment
#m.solveManual();
#m.solveAtRandom();



#1000000
q_values = [];
env = Soccer();
print("### STARTING Friend-Q ###")
T = 1000000;
Solv = solveFriendQ(env);
#for t in range(T):
while Solv.episodes < T:
    t = Solv.episodes
    if t%100000 == 0:
        print("      --> Iteration ",t," of ",T," <--")
        Q_values = np.array(q_values)
        errs = np.abs(Q_values[1:] - Q_values[:-1])
                    
        plt.figure(1)
        plt.plot(errs)
        plt.ylim([0, 0.5])
        plt.show()
    q = Solv.train()
    q_values.append(q)

Q_values = np.array(q_values)
errs = np.abs(Q_values[1:] - Q_values[:-1])
            
plt.figure(1)
plt.plot(errs)
plt.ylim([0, 0.5])
plt.title("Friend-Q")
plt.show()



#23.33081525056417
q_values = [];
env.reset();
print("### STARTING Correlated-Q ###")
T = 1000000;
Solv = solveCorrelatedQ(env);
#for t in range(T):
while Solv.episodes < T:
    t = Solv.episodes
    if t%1000 == 0:
        
        if t%100000 == 0:
            print("\n      --> Iteration ",t," of ",T," <--")
            Q_values = np.array(q_values)
            errs = np.abs(Q_values[1:] - Q_values[:-1])
            
            plt.figure(1)
            plt.plot(errs)
            plt.ylim([0, 0.5])
            plt.show()
        print("°", end='')
        
    q = Solv.train()
    q_values.append(q)

Q_valuesC = np.array(q_values)
errs = np.abs(Q_valuesC[1:] - Q_valuesC[:-1])
            
plt.figure(1)
plt.plot(errs)
plt.ylim([0, 0.5])
plt.title("Correlated-Q")
plt.show()    
    
    
q_values = [];
env.reset();
print("### STARTING Foe-Q ###")
T = 1000000;
Solv = solveFoeQ(env);
#for t in range(T):
while Solv.episodes < T:
    t = Solv.episodes
    if t%1000 == 0:
        if t%100000 == 0:
            print("\n      --> Iteration ",t," of ",T," <--")
            Q_values = np.array(q_values)
            errs = np.abs(Q_values[1:] - Q_values[:-1])
            
            plt.figure(1)
            plt.plot(errs)
            plt.ylim([0, 0.5])
            plt.show()
        print("°", end='')
    q = Solv.train()
    q_values.append(q)

Q_valuesF = np.array(q_values)
errs = np.abs(Q_valuesF[1:] - Q_valuesF[:-1])
            
plt.figure(1)
plt.plot(errs)
plt.ylim([0, 0.5])
plt.title("Foe-Q")
plt.show()


q_values = [];
env.reset();
print("### STARTING Classical-Q ###")
T = 1000000;
Solv = solveClassicQ(env);
#for t in range(T):
while Solv.episodes < T:
    t = Solv.episodes
    if t%100000 == 0:
        print("      --> Iteration ",t," of ",T," <--")
        Q_values = np.array(q_values)
        errs = np.abs(Q_values[1:] - Q_values[:-1])  
        
        plt.figure(1)
        plt.plot(errs)
        plt.ylim([0, 0.5])
        plt.show()
    q = Solv.train()
    q_values.append(q)

Q_values = np.array(q_values)
errs = np.abs(Q_values[1:] - Q_values[:-1])
            
plt.figure(1)
plt.plot(errs)
plt.ylim([0, 0.5])
plt.title("Q-Learning")
plt.show()

    



        