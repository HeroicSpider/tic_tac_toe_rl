# train_agent.py
from tic_tac_toe import TicTacToe
from q_learning_agent import QLearningAgent
import pickle
import random  # Make sure to import random

def train_agent(episodes=100000000):
    env = TicTacToe()
    agent = QLearningAgent()

    for episode in range(episodes):
        state = tuple(env.reset())  # Ensure state is a tuple
        done = False
        player = 1  # Start with player 1 (agent)
        while not done:
            available_actions = env.available_moves()
            action = agent.choose_action(state, available_actions)
            env.make_move(action, player)

            next_state = env.get_state()  # Already returns a tuple
            reward = 0
            done = env.game_over()

            if done:
                if env.current_winner == player:
                    reward = 1  # Agent wins
                elif env.is_draw():
                    reward = 0.5  # Draw
                else:
                    reward = -1  # Agent loses
                agent.learn(state, action, reward, next_state, done, [])
                break

            # Opponent's turn (random moves)
            opponent_actions = env.available_moves()
            opponent_action = random.choice(opponent_actions)
            env.make_move(opponent_action, -player)

            if env.game_over():
                if env.current_winner == -player:
                    reward = -1  # Agent loses
                elif env.is_draw():
                    reward = 0.5  # Draw
                agent.learn(state, action, reward, next_state, True, [])
                break
            else:
                agent.learn(state, action, reward, next_state, False, env.available_moves())

            state = next_state

        if (episode + 1) % 10000 == 0:
            print(f"Episode {episode + 1}/{episodes} completed.")

    # Save the trained Q-table
    with open('q_table.pkl', 'wb') as f:
        pickle.dump(agent.q_table, f)

if __name__ == "__main__":
    train_agent()
