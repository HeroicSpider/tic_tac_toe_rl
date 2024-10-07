# app.py
from flask import Flask, render_template, jsonify, request
from tic_tac_toe import TicTacToe
from q_learning_agent import QLearningAgent
import pickle

app = Flask(__name__)
env = TicTacToe()

# Load the trained Q-table
with open('q_table.pkl', 'rb') as f:
    q_table = pickle.load(f)

agent = QLearningAgent()
agent.q_table = q_table

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reset', methods=['POST'])
def reset():
    state = env.reset()
    return jsonify({'board': env.board.tolist(), 'message': 'Game reset!'})

@app.route('/move', methods=['POST'])
def move():
    data = request.get_json()
    position = tuple(data['position'])
    player = data['player']

    if env.board[position] != 0:
        return jsonify({'board': env.board.tolist(), 'message': 'Invalid move!', 'status': 'invalid'})

    env.make_move(position, player)

    if env.game_over():
        if env.current_winner == player:
            return jsonify({'board': env.board.tolist(), 'message': 'You win!', 'status': 'win'})
        elif env.is_draw():
            return jsonify({'board': env.board.tolist(), 'message': 'It\'s a draw!', 'status': 'draw'})

    # Agent's turn
    state = env.get_state()
    available_actions = env.available_moves()
    action = agent.choose_action(state, available_actions)
    env.make_move(action, -player)

    if env.game_over():
        if env.current_winner == -player:
            return jsonify({'board': env.board.tolist(), 'message': 'AI wins!', 'status': 'lose'})
        elif env.is_draw():
            return jsonify({'board': env.board.tolist(), 'message': 'It\'s a draw!', 'status': 'draw'})

    return jsonify({'board': env.board.tolist(), 'message': 'Your turn!', 'status': 'continue'})

if __name__ == '__main__':
    app.run(debug=True)
