from environment import SafarGoEnv
from agent import QLearningAgent
from baselines import baseline_action


def train(episodes=80, seed=7):
    env = SafarGoEnv(seed=seed)
    agent = QLearningAgent(actions=env.zones + 1)
    reward_history = []
    for episode in range(episodes):
        state = env.reset(seed=seed + episode)
        total_reward = 0.0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        agent.end_episode()
        reward_history.append(total_reward)
    return agent, reward_history


def evaluate(policy="rl", agent=None, episodes=5, seed=500):
    results = []
    for episode in range(episodes):
        env = SafarGoEnv(seed=seed + episode)
        state = env.reset(seed=seed + episode)
        done = False
        while not done:
            action = agent.act(state, explore=False) if policy == "rl" and agent is not None else baseline_action(env, policy)
            state, _, done, _ = env.step(action)
        results.append(env.metrics())
    return {metric: round(sum(row[metric] for row in results) / len(results), 2)
            for metric in results[0]}
