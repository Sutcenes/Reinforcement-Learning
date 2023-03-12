## Imports
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

###### Functions for Figure 3 #####
def genereSequence():
    s = 2;
    sequence = [s];
    go = True;
    while go: #looping until final state

                #This is a step
                A = next_state(s);
                s = A[0];
                sequence += [s];

                reward = A[1];#This is not used
                if s == -1 or s == 5:
                    go = False;
    return sequence

def compute_delta_w(l,alpha,sequence,w):#Used for Figure 3
    e = np.array([0, 0, 1, 0, 0]);#Always start at position D
    delta_w = np.zeros(5);
    for i in range(len(sequence)-1):

                #States
                s_current = sequence[i];
                s_next = sequence[i+1];

                #Update e = nabla_w(P)+lambda*e
                e = l*e;#l=0 --> e = [0 0 0 0 0]
                e[s_current] += 1;# l=0 --> e = [0 1 0 0 0] (le 1 à l'indice de l'état courant)

                if s_next == -1:#Final state A
                    #Compute delta_w
                    delta_w += alpha*(0-w[s_current])*e;
                    """if l==0:
                        print("    >",delta_w)"""

                elif s_next == 5:#Final state G
                    #Compute delta_w
                    delta_w += alpha*(1-w[s_current])*e;#l=0 --> e = [0 0 0 0 1] --> delta_w = [0 0 0 0 alpha]
                    """if l==0:
                        print("    >",delta_w)"""

                else:#Any transient state B,C,D,E,F
                    #Compute delta_w
                    delta_w += alpha*(w[s_next]-w[s_current])*e;
    return delta_w

def train_over_10(w,l,alpha,SEQUENCES):

        S = 0;#I accumulate delta_w in S

        #A training set = 10 episodes
        for SQ in SEQUENCES:
            sequence = SQ;#This is a sequence
            delta_w = compute_delta_w(l,alpha,sequence,w)
            #Sequence is ended
            #Accumulate delta_w
            S = S + delta_w;
        return S


def TD(l,alpha,SEQUENCES,Accumulation):#l = lambda rate, alpha step size

    w = 0.5*np.array([1, 1, 1, 1, 1]);#weights : 5 states B,C,D,E,F
    converge_w = False;
    count = 0;
    while not(converge_w):#Training set presented until convergence
        #Save current w in w_previous
        w_previous = w.copy();
        #Train over 10 sequences without updating w
        DELTA_w = train_over_10(w,l,alpha,SEQUENCES)#Cumulated delta_w over 10 sequences
        #Updating w
        w = w + DELTA_w*Accumulation;#You can divide by 10 when takin alpha rate greater than 0.02

        count = count + 1;
        difference = np.max(np.abs(w-w_previous));
        """if l == 0:#count == 0:
            print("Et donc -> w = ",w)
            print("    D = ",difference)"""
        if difference < 1e-2:
            converge_w = True;

    #print("Nb d'itérations :",count)
    return w

def next_state(s):
    """Choose next state from state s, at random"""
    next_state = s + random.choice([-1,1])
    reward = 0;
    if next_state == 0:
        reward = 0;
    elif next_state == 6:
        reward = +1;
    R = [next_state,reward]
    return R

def exp1(L,ALPHA,Accumulation):
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    numberOfTrainingSets = 100;
    numberOfEpisode = 10;

    errors = np.zeros([len(L),len(ALPHA)]);
    #True solution of random walk is :
    #[A  B   C   D   E   F  G]
    # ↓  ↓   ↓   ↓   ↓   ↓  ↓
    #[0 1/6 2/3 3/6 4/6 5/6 1]

    w_ref=np.arange(1/6,1-1/6,1/6);

    Ll = 0;
    La = 0;
    for l in L:
        La = 0;
        for alpha in ALPHA:
            for i in range(numberOfTrainingSets):

                w = train(l,alpha,numberOfEpisode,Accumulation)
                #if l == 0:
                    #print("End of training",i,"(",l,alpha,")\nw = ",w,"\n---oOo---\n")
                errors[Ll,La] = errors[Ll,La] + np.sqrt(np.mean(np.power(w-w_ref,2)))
            La += 1;
        Ll += 1;

    #Cmpute errors
    FinalErrors = errors/(numberOfTrainingSets);
    return FinalErrors


def train(l,alpha,numberOfEpisode,Accumulation):
    """ --- Sequences Generation --- """
    SEQUENCES = [];
    for k in range(numberOfEpisode):
        SEQUENCES += [genereSequence()];

    """ Compute w over the 'numberOfEpisode' SEQUENCES"""
    w = TD(l,alpha,SEQUENCES,Accumulation);

    return w




##### Function for figures 4 and 5 #####
def compute_delta_w2(l,alpha,sequence,w):#Used for Figure 4 and 5
    e = np.array([0, 0, 1, 0, 0]);#Always start at position D
    delta_w = np.zeros(5);
    for i in range(len(sequence)-1):

                #States
                s_current = sequence[i];
                s_next = sequence[i+1];

                #Update e = nabla_w(P)+lambda*e
                e = l*e;#l=0 --> e = [0 0 0 0 0]
                e[s_current] += 1;# l=0 --> e = [0 1 0 0 0] (le 1 à l'indice de l'état courant)

                if s_next == -1:#Final state A
                    #Compute delta_w
                    delta_w += alpha*(0-w[s_current])*e;
                    """if l==0:
                        print("    >",delta_w)"""

                elif s_next == 5:#Final state G
                    #Compute delta_w
                    delta_w += alpha*(1-w[s_current])*e;#l=0 --> e = [0 0 0 0 1] --> delta_w = [0 0 0 0 alpha]
                    """if l==0:
                        print("    >",delta_w)"""

                else:#Any transient state B,C,D,E,F
                    #Compute delta_w
                    delta_w += alpha*(w[s_next]-w[s_current])*e;
    return delta_w






def TD2(l,alpha,SEQUENCES):#l = lambda rate, alpha step size

    w = 0.5*np.array([1, 1, 1, 1, 1]);#weights : 5 states B,C,D,E,F
    #Train over 10 sequences with updates of w


    #A training set = 10 episodes
    for SQ in SEQUENCES:
        sequence = SQ;#This is a sequence
        delta_w = compute_delta_w2(l,alpha,sequence,w)
        #Sequence is ended
        #Update w
        w = w + delta_w;
    return w

def next_state(s):
    """Choose next state from state s, at random"""
    next_state = s + random.choice([-1,1])
    reward = 0;
    if next_state == 0:
        reward = 0;
    elif next_state == 6:
        reward = +1;
    R = [next_state,reward]
    return R

def exp2(L,ALPHA):
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    numberOfTrainingSets = 100;
    numberOfEpisode = 10;

    errors = [np.zeros(len(i)) for i in ALPHA];
    #True solution of random walk is :
    #[A  B   C   D   E   F  G]
    # ↓  ↓   ↓   ↓   ↓   ↓  ↓
    #[0 1/6 2/3 3/6 4/6 5/6 1]

    w_ref=np.arange(1/6,1-1/6,1/6);

    Ll = 0;
    La = 0;
    for i in range(numberOfTrainingSets):
        SEQUENCES = createCommonTrainingSet(numberOfEpisode);
        Ll = 0;
        for l in L:
            La = 0;
            for alpha in ALPHA[Ll]:
                """Compute w over the common training set"""
                w = TD2(l,alpha,SEQUENCES)
                """if l == 0:
                    print("End of training",i,"(",l,alpha,")\nw = ",w,"\n---oOo---\n")"""
                errors[Ll][La] = errors[Ll][La] + np.sqrt(np.mean(np.power(w-w_ref,2)))
                La += 1;
            Ll += 1;

    #Cmpute errors
    FinalErrors = [];
    for e in errors:
        FinalErrors += [e/(numberOfTrainingSets)];
    return FinalErrors

def createCommonTrainingSet(numberOfEpisode):
    """ --- Sequences Generation --- """
    SEQUENCES = [];
    for k in range(numberOfEpisode):
        SEQUENCES += [genereSequence()];
    return SEQUENCES



##### PLOTTING #####

def figure_3(NumberOfExperiment,Accumulation,alpha):
    print("### COMPUTING FIGURE 3 ###")
    print("Averaging over",NumberOfExperiment,"experiments")
    ALPHA = [alpha];#Try all these alphas !
    L = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1];#Try all these lambdas !

    FinalErrors = np.zeros([len(L),len(ALPHA)]);
    for i in range(NumberOfExperiment):
        FinalErrors += exp1(L,ALPHA,Accumulation);
        print("   --->",i+1,"of",NumberOfExperiment,"<---")
    print("### END OF COMPUTING FIGURE 3 ###\n   ---oOo---\n")
    FinalErrors = FinalErrors/NumberOfExperiment
    #Plotting
    plt.figure()
    plt.plot(L,FinalErrors,'+:',label =[''+str(i) for i in L])
    plt.xlabel('lambdas')
    plt.ylabel('RMS error')
    plt.legend()
    plt.title("Figure 3 - Replica")
    return


def figure_4():
    print("   ### COMPUTING FIGURE 4 ###")
    ALPHA = [np.arange(0,0.6,0.05),
             np.arange(0,0.6,0.05),
             np.arange(0,0.6,0.05),
             np.arange(0,0.6,0.05)
            ];#Try all these alphas !
    """np.arange(0,0.6,0.05),
    np.arange(0,0.6,0.05),
    np.arange(0,0.6,0.05),
    np.arange(0,0.6,0.05),
    np.arange(0,0.6,0.05),
    np.arange(0,0.6,0.05),"""

    L = [0.,0.3,0.8,1];#Try all these lambdas !

    FinalErrors = exp2(L,ALPHA);
    #Plotting

    plt.figure()
    for i in range(len(ALPHA)):
        plt.plot(ALPHA[i],FinalErrors[i],'+:',label =['lambda = '+str(L[i])] )
        print("      --->",i+1,"of",len(ALPHA),"<---")
    print("   ### END OF COMPUTING FIGURE 4 ###\n   ---oOo---")
    plt.xlabel('alphas')
    plt.ylabel('RMS error')
    plt.legend()
    plt.title("Figure 4 - Replica")


    return


def figure_5():
    print("   ### COMPUTING FIGURE 5 ###")
    ALPHA = [np.arange(0,0.6,0.05),
             np.arange(0,0.6,0.05),
             np.arange(0,0.6,0.05),
             np.arange(0,0.6,0.05),
             np.arange(0,0.6,0.05),
             np.arange(0,0.6,0.05),
             np.arange(0,0.6,0.05),
             np.arange(0,0.6,0.05),
             np.arange(0,0.6,0.05),
             np.arange(0,0.6,0.05),
             np.arange(0,0.6,0.05)
            ];#Try all these alphas !"""


    L = np.arange(0,1.1,0.1);#Try all these lambdas !

    FinalErrors = exp2(L,ALPHA);

    plt.figure()
    #Find lower lambda-alpha-error
    Min_error=[];
    for i in range(len(L)):
        minimum = minimum = (FinalErrors[i][0]);
        for j in range(len(ALPHA[i])):
            current = (FinalErrors[i][j]);
            if current < minimum:
                minimum = current;
        Min_error += [minimum];

    print("   ### END OF COMPUTING FIGURE 3 ###\n---oOo---")

    plt.plot(L,Min_error,'+:')
    plt.xlabel('lambdas')
    plt.ylabel('RMS error using best alpha')
    plt.title("Figure 5 - Replica")

    return


## PRINTING FIGURES :
figure_3(1,0.01,1);#(number of experiments, alpha rate, mode of accumulation (delta_w += modeOfAccumulation * delta_w)

numberOfFigures = int(input("How many graphs do you want to plot for each figure :"));
print("YOU WILL PLOT",numberOfFigures,"graphs !")
print("---ooOO Starting Now OOoo---")
for i in range(numberOfFigures):#Choose how many figures you want to plot
    print("---ooOO",i+1,"of",numberOfFigures,"OOoo---")
    figure_4()
    figure_5()
    print("\n\n")
    plt.show()
    
#######################################    
##      Additional  experiment       ##
#######################################
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def figure_4_stats():
    print("   ### COMPUTING FIGURE 4 with Statistical Analysis ###")
    ALPHA = [np.array([0.3,0.55]),
             np.array([0.3,0.55]),
             np.array([0.3,0.55]),
             np.array([0.3,0.55])
            ];#Try all these alphas !
            
    L = [0.,0.3,0.8,1];#Try all these lambdas !
    
    
    StatsAlpha3 = [[] for i in range(4)]
    StatsAlpha6 = [[] for i in range(4)]
    for i in range(50):
        print("      --->",i+1,"of 50 <---")
        FinalErrors = exp2(L,ALPHA);
        for j in range(len(L)):
            StatsAlpha3[j].append([FinalErrors[j][0]])
            StatsAlpha6[j].append([FinalErrors[j][1]])
    
    
    
    moyennes3 = np.mean(StatsAlpha3,axis=1)
    variance3 = np.std(StatsAlpha3,axis=1)
    moyennes6 = np.mean(StatsAlpha6,axis=1)
    variance6 = np.std(StatsAlpha6,axis=1)
    """print(moyennes3)
    print(variances3)
    print(moyennes6)
    print(variances6)"""
    
    """Compute gaussian distributions for lambdas, alphas"""
    x = np.linspace(0, 20, 1000)
    x1 = np.linspace(0, 2, 1000)
    Gauss0 = [gaussian(x1,moyennes3[0],variance3[0]),gaussian(x,moyennes6[0],variance6[0])]
    Gauss03 = [gaussian(x1,moyennes3[1],variance3[1]),gaussian(x,moyennes6[1],variance6[1])]
    Gauss08 = [gaussian(x1,moyennes3[2],variance3[2]),gaussian(x,moyennes6[2],variance6[2])]
    Gauss1 = [gaussian(x1,moyennes3[3],variance3[3]),gaussian(x,moyennes6[3],variance6[3])]
    
    Gauss = [Gauss0,Gauss03,Gauss08,Gauss1]
    
    plt.figure()
    for i in range(len(L)):
        plt.plot(x,Gauss[i][0],label = ['lambda = '+str(L[i])])
    plt.title("Gaussian distribution of errors for alpha = 0.3")
    plt.xlabel('RMS error distribution')
    plt.ylabel('RMS error using best alpha')
    plt.legend()
    
    plt.figure()
    for i in range(len(L)):
        plt.plot(x,Gauss[i][1],label = ['lambda = '+str(L[i])])
    plt.title("Gaussian distribution of errors for alpha = 0.55")
    plt.legend()

    print("   ### END OF COMPUTING FIGURE 4 with Statistical Analysis ###\n---oOo---")
    return
    

figure_4_stats()