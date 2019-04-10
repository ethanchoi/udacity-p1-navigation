# Report on project1-Navigation

In this project, the DQN learning algorithm has been used to solve the Navigation problem.
Other learning algorithms like Dobule DQN, Prioritized Experience Replay DQN, Dueling DQN will be added later.

The report will describe the learning algorithm with used hyper parameters, the arcitectures for neural netwoorks.


### Learning Algorithm
#### Deep Q-Network

**Q-learning** is a value-based Reinforcement Learning algorithm that is used to find the optimal action-selection policy using a q function, *`Q(s,a)`*

It's goal is to maximize the value function Q

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathit{Q}^{*}(s,&space;a)&space;=&space;\underset{\pi}{\mathrm{max}}\left&space;\{&space;r_{t}&space;&plus;&space;\gamma&space;r_{t&plus;1}&space;&plus;&space;\gamma&space;^{2}&space;r_{t&plus;2}&plus;...|s_{t}=s,&space;a_{t}=a,&space;\pi&space;\right&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathit{Q}^{*}(s,&space;a)&space;=&space;\underset{\pi}{\mathrm{max}}\left&space;\{&space;r_{t}&space;&plus;&space;\gamma&space;r_{t&plus;1}&space;&plus;&space;\gamma&space;^{2}&space;r_{t&plus;2}&plus;...|s_{t}=s,&space;a_{t}=a,&space;\pi&space;\right&space;\}" title="\mathit{Q}^{*}(s, a) = \underset{\pi}{\mathrm{max}}\left \{ r_{t} + \gamma r_{t+1} + \gamma ^{2} r_{t+2}+...|s_{t}=s, a_{t}=a, \pi \right \}" /></a>

which is the maximum sum of rewards r<sub>t</sub> discounted by &gamma; at each timestep t, achievable by a behaviour policy *&pi;=P(a|s)*, after making an
observation (s) and taking an action (a)

The follwoing is Q learning algorithm pseudo-code
1. Initialze Q-values *Q(s,a)* arbitrarily for all state-action pairs.
2. For i=1 to # num_episodes <br/>
  Choose an action A<sub>t</sub> int eht current state (s) based on current Q-value estimates (e,g &epsilon;-greedy) </br>
  Take action A<sub>t</sub> amd observe reward and state, R<sub>t+1</sub>, S<sub>t+1</sub>
  Update *Q(s|a)* <br/>
  
    <a href="https://www.codecogs.com/eqnedit.php?latex=\mathit{Q(s_{t}|a_{t})}&space;=&space;\mathit{Q(s_{t}|a_{t})}&space;&plus;&space;\alpha(\mathitt{R_{t&plus;1}&plus;\gamma&space;\mathrm{max_{a}\mathit{Q(S_{t&plus;1},&space;a)-\mathit{Q(s_{t}|a_{t})}&space;}}})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathit{Q(s_{t}|a_{t})}&space;=&space;\mathit{Q(s_{t}|a_{t})}&space;&plus;&space;\alpha(\mathitt{R_{t&plus;1}&plus;\gamma&space;\mathrm{max_{a}\mathit{Q(S_{t&plus;1},&space;a)-\mathit{Q(s_{t}|a_{t})}&space;}}})" title="\mathit{Q(s_{t}|a_{t})} = \mathit{Q(s_{t}|a_{t})} + \alpha(\mathitt{R_{t+1}+\gamma \mathrm{max_{a}\mathit{Q(S_{t+1}, a)-\mathit{Q(s_{t}|a_{t})} }}})" /></a>

**Q-networks** approximate the Q-function as a neural network given a state, Q-values for each action<br/>
*Q(s, a, θ)* is a neural network that define obejctive function by mean-squared error in Q-values
  <a href="https://www.codecogs.com/eqnedit.php?latex=\mathfrak{L}{(\theta)&space;=&space;\mathrm{E}\left&space;[&space;\left&space;(&space;\underbrace{r&space;&plus;&space;\gamma&space;\underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'},\theta)}&space;-&space;Q(s,a,\theta)\right&space;)^{2}&space;\right&space;]&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathfrak{L}{(\theta)&space;=&space;\mathrm{E}\left&space;[&space;\left&space;(&space;\underbrace{r&space;&plus;&space;\gamma&space;\underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'},\theta)}&space;-&space;Q(s,a,\theta)\right&space;)^{2}&space;\right&space;]&space;}" title="\mathfrak{L}{(\theta) = \mathrm{E}\left [ \left ( \underbrace{r + \gamma \underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'},\theta)} - Q(s,a,\theta)\right )^{2} \right ] }" /></a>
  <br/>

To find optimum parameters &theta;, optimise by SGD, using &delta;*L(&theta;)*/&delta;*&theta;* <br/>
This algorithm diverges because stages are correlated and targets are non-stationary. 

**DQN-Experience replay**<br/>
In order to deal with the correlated states, the agent build a dataset of experience and then makes random samples from
the dataset.<br/>

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathfrak{L}{(\theta)&space;=&space;\mathrm{E_{\mathit{s,a,r,s^{'}&space;D}}}\left&space;[&space;\left&space;(r&space;&plus;&space;\gamma&space;\underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'},\theta)&space;-&space;Q(s,a,\theta)\right&space;)^{2}&space;\right&space;]&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathfrak{L}{(\theta)&space;=&space;\mathrm{E_{\mathit{s,a,r,s^{'}&space;D}}}\left&space;[&space;\left&space;(r&space;&plus;&space;\gamma&space;\underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'},\theta)&space;-&space;Q(s,a,\theta)\right&space;)^{2}&space;\right&space;]&space;}" title="\mathfrak{L}{(\theta) = \mathrm{E_{\mathit{s,a,r,s^{'} D}}}\left [ \left (r + \gamma \underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'},\theta) - Q(s,a,\theta)\right )^{2} \right ] }" /></a>

**DQN-Fixed Target** <br/>
Also, the agent fixes the parameter &theta;<sup>-</sup> and then with some frequency updates them<br/>

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathfrak{L}{(\theta)&space;=&space;\mathrm{E_{\mathit{s,a,r,s^{'}&space;D}}}\left&space;[&space;\left&space;(r&space;&plus;&space;\gamma&space;\underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'},\theta^-)&space;-&space;Q(s,a,\theta)\right&space;)^{2}&space;\right&space;]&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathfrak{L}{(\theta)&space;=&space;\mathrm{E_{\mathit{s,a,r,s^{'}&space;D}}}\left&space;[&space;\left&space;(r&space;&plus;&space;\gamma&space;\underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'},\theta^-)&space;-&space;Q(s,a,\theta)\right&space;)^{2}&space;\right&space;]&space;}" title="\mathfrak{L}{(\theta) = \mathrm{E_{\mathit{s,a,r,s^{'} D}}}\left [ \left (r + \gamma \underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'},\theta^-) - Q(s,a,\theta)\right )^{2} \right ] }" /></a>

Reference
1. [Q-Learning](https://medium.freecodecamp.org/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8)
2. [Deep Reinforcement Learning - David Silver](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Talks_files/deep_rl.pdf)
3. [Deep Reinforcement Learning](http://mi.eng.cam.ac.uk/~mg436/LectureSlides/MLSALT7/L6.pdf)

### Plot oof Rewards



### Ideas for Futrue Work
