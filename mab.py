# util for graphing multi armed bandit problems 
import numpy as np
import random
import matplotlib.pyplot as plt

def normal_reward(mu, sigma):
    return np.random.normal(mu, sigma)

def compute(arms, VAR, epsilons, ITERS):
    res = []
    for e in epsilons:
        rewards = []
        Q = [0.5]*len(arms)
        total_rewards = [0]*len(arms)
        counts = [0]*len(arms)

        for i in range(ITERS):
            if random.uniform(0.0,1.0) > e: # exploit
                optimal = Q.index(max(Q))
                choice_reward = normal_reward( means[optimal], VAR )
                total_rewards[optimal] += choice_reward
                counts[optimal] += 1
                Q[optimal] = total_rewards[optimal] / counts[optimal]

                if rewards:
                    rewards.append( rewards[-1] + choice_reward )
                else:
                    rewards.append( choice_reward )
            else: # explore
                rand = np.random.randint(0, len(arms))
                choice_reward = normal_reward( means[rand], VAR )
                total_rewards[rand] += choice_reward
                counts[rand] += 1
                Q[rand] = total_rewards[rand] / counts[rand]
                if rewards:
                    rewards.append( rewards[-1] + choice_reward )
                else:
                    rewards.append( choice_reward )

        res.append(rewards.copy())
    return res

def visualize(rewards, epsilon):
    for i, r in enumerate(rewards):
        plt.plot(r, label=f"ε={epsilon[i]}")
    plt.xlabel("step")
    plt.ylabel("average reward")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    means = [0.6,0.5,0.1,0.8]
    VAR = 0.05
    epsilon = [0,0.1,0.2, 0.01]
    ITERS = 1000
    rewards = compute(means, VAR, epsilon, ITERS)
    avg_rewards = []
    for i in range(len(rewards)):
        arr = [0]
        for j in range(len(rewards[i])):
            arr.append(rewards[i][j] / (j + 1))
        avg_rewards.append(arr.copy())
    visualize(avg_rewards, epsilon)