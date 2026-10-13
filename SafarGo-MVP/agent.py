import random
from collections import defaultdict

class QLearningAgent:
    """Tabular Q-learning baseline over a compact, discretized global state."""
    def __init__(self, actions=41, alpha=.15, gamma=.90, epsilon=1.0, epsilon_min=.05, decay=.995, seed=11):
        self.q = defaultdict(lambda: [0.0] * actions)
        self.actions, self.alpha, self.gamma = actions, alpha, gamma
        self.epsilon, self.epsilon_min, self.decay = epsilon, epsilon_min, decay
        self.rng = random.Random(seed)

    def act(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.actions)
        values = self.q[state]
        best = max(values)
        return self.rng.choice([i for i, v in enumerate(values) if v == best])

    def learn(self, state, action, reward, next_state, done):
        target = reward if done else reward + self.gamma * max(self.q[next_state])
        self.q[state][action] += self.alpha * (target - self.q[state][action])

    def end_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)
