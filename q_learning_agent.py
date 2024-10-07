# q_learning_agent.py
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.q_table = defaultdict(float)
        self.alpha = alpha        # Learning rate
        self.gamma = gamma        # Discount factor
        self.epsilon = epsilon    # Exploration rate

    def choose_action(self, state, available_actions):
        """Choose an action based on epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        q_values = [self.q_table[(state, action)] for action in available_actions]
        max_q = max(q_values)
        max_actions = [action for action, q in zip(available_actions, q_values) if q == max_q]
        return random.choice(max_actions)

    def learn(self, state, action, reward, next_state, done, available_actions):
        """Update the Q-value based on the reward received."""
        current_q = self.q_table[(state, action)]
        if done:
            target = reward
        else:
            next_q_values = [self.q_table[(next_state, a)] for a in available_actions]
            target = reward + self.gamma * max(next_q_values)
        self.q_table[(state, action)] = current_q + self.alpha * (target - current_q)
