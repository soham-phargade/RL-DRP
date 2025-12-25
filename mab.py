# util for graphing multi armed bandit problems 
import numpy as np
import random
import matplotlib.pyplot as plt

def normal_reward(mu, sigma):
    return np.random.normal(mu, sigma)

def compute_e(arms, VAR, epsilons, ITERS):
    res = []
    for e in epsilons:
        rewards = []
        Q = [0.5]*len(arms)
        total_rewards = [0]*len(arms)
        counts = [0]*len(arms)

        for i in range(ITERS):
            if random.uniform(0.0,1.0) > e: # exploit
                optimal = Q.index(max(Q))
                choice_reward = normal_reward( arms[optimal], VAR )
                total_rewards[optimal] += choice_reward
                counts[optimal] += 1
                Q[optimal] = total_rewards[optimal] / counts[optimal]

                if rewards:
                    rewards.append( rewards[-1] + choice_reward )
                else:
                    rewards.append( choice_reward )
            else: # explore
                rand = np.random.randint(0, len(arms))
                choice_reward = normal_reward( arms[rand], VAR )
                total_rewards[rand] += choice_reward
                counts[rand] += 1
                Q[rand] = total_rewards[rand] / counts[rand]
                if rewards:
                    rewards.append( rewards[-1] + choice_reward )
                else:
                    rewards.append( choice_reward )

        res.append(rewards.copy())
    return res

def compute_ucb(arms, VAR, c, ITERS):
    rewards = []
    Q = [0.0]*len(arms)
    total_rewards = [0.0]*len(arms)
    counts = [0]*len(arms)

    # pull each arm once
    for a in range(len(arms)):
        r = normal_reward(arms[a], VAR)
        total_rewards[a] += r
        counts[a] += 1
        Q[a] = total_rewards[a] / counts[a]
        rewards.append(r)  # per-step reward

    for t in range(len(arms), ITERS):
        ucb_values = [
            Q[a] + c * np.sqrt(np.log(t + 1) / counts[a])
            for a in range(len(arms))
        ]
        choice = np.argmax(ucb_values)

        r = normal_reward(arms[choice], VAR)
        total_rewards[choice] += r
        counts[choice] += 1
        Q[choice] = total_rewards[choice] / counts[choice]

        rewards.append(r)  # per-step reward

    return rewards

def visualize(rewards, epsilon):
    for i, r in enumerate(rewards):
        plt.plot(r, label=f"ε={epsilon[i]}")
    plt.xlabel("step")
    plt.ylabel("average reward")
    plt.legend()
    plt.show()

def visualize_compare(e_rewards, ucb_rewards):
    plt.plot(e_rewards, label="ε-greedy (ε=0.1)")
    plt.plot(ucb_rewards, label="UCB (c=2)")
    plt.xlabel("step")
    plt.ylabel("average reward")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # means = [0.6,0.5,0.1,0.8]
    # VAR = 0.05
    # epsilon = [0,0.1,0.2, 0.01]
    # ITERS = 1000
    # rewards = compute_e(means, VAR, epsilon, ITERS)
    # avg_rewards = []
    # for i in range(len(rewards)):
    #     arr = [0]
    #     for j in range(len(rewards[i])):
    #         arr.append(rewards[i][j] / (j + 1))
    #     avg_rewards.append(arr.copy())
    # visualize(avg_rewards, epsilon)
    means = [0.6, 0.58, 0.55, 0.62]
    VAR = 0.05
    ITERS = 1000
    RUNS = 500

    # --- epsilon-greedy ---
    eps_rewards = np.zeros(ITERS)
    for _ in range(RUNS):
        r = compute_e(means, VAR, [0.1], ITERS)[0]

        # convert cumulative -> per-step rewards
        per_step = np.diff([0] + r)

        eps_rewards += per_step

    eps_rewards /= RUNS


    # --- UCB ---
    ucb_rewards = np.zeros(ITERS)
    for _ in range(RUNS):
        r = compute_ucb(means, VAR, c=2, ITERS=ITERS)
        ucb_rewards += r
    ucb_rewards /= RUNS

    visualize_compare(eps_rewards, ucb_rewards)
