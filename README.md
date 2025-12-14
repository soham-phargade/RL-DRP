# Reinforcement Learning Experiments

`mab.py`: a simple multi armed bandit with different exploration exploitation policies ($\epsilon$-greedy)
![eval](/assets/mab3.png)

the greedy method performed significantly worse in the long run because it often got stuck performinf suboptimal actions. $\epsilon$ = 0.01 method improved more slowly, but eventually would perform better than the $\epsilon$ = 0.1 method.

O(n) time complexity with n iterations 
O(k) space complexity with k arms of bandit

I kept track of total reward per action and number of times action has taken place to compute the RL agent's current estimation of action rewards. However, running average can be simply computed with this formula - 
$$
\tag{2.3}
Q_{n+1} = Q_n + \frac{1}{n}\left[R_n - Q_n \right]
$$

new estimate = old estimate + step size (target - old estimate)
target - old estimate is the error in the estimate reduced by taking a step towards the target. this simple bandit problem has stationary reward probabilities so the step size incrementally decreases. However, in non stationary problems, we can take step size to be a constant making the estimated reward a weighted average of rewards encountered. Also called exponential recency-weighted average. 