PROJECT 1

The file provided can be launched without any modifications. It will plot a replicata of figure 3, and ask you for the number of differents graphs you want for figures 4 and 5, you may answer 1. It will also plot two figures for statistical analysis of figure 4.

At line 356, you find the call to the plot function of figure 3 of the form : figure_3(n,alpha,mode) with n = number of experiments, alpha = alpha rate and mode = mode of accumulation. The mode of accumulation refers to the way of accumulating the delta_w over the 10 sequences : results from my paper are for combinations (alpha =0.01, mode = 1) and (alpha=0.1, mode=0.1). 

n refers to the numbers of experiments. With n=1, the plot will provide the classical figure 3 obtained from the methodology described in Sutton's 88 paper. For n>1, the full experiment will be launched n times and the results will then be averaged, one figure is plotted and it's name will still be "Figure 3 - replicata" even if it does not follow Sutton's methodology anymore.

Lines 356, 363, 364 are those which launch the figures computation and plotting : you may selectively comment these lines if you want to avoid plotting respective figures. Line 358 calls an "input" function, you may also feed the variable directly from the script with the value of your choice by editing this line.

The end of the file is dedicated to a simple statistical analysis of figure 4, you may comment the last line of the file to avoid launching this analysis.

Thank you for taking the time to read these information. 
Regards,
Adrien Poujon.
