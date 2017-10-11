from trees import MonteCarloTS
from board import *

import sys, random
argv = sys.argv

boards = {
    'ttt' : lambda args : TicTacToe() if len(args) is 0 else TicTacToe(int(args[0])),
    'c4' : lambda args : ConnectFour() if len(args) is 0 else ConnectFour(int(args[0]), int(args[1]), int(args[2]))
}

if len(argv) == 1 or argv[1] == 'help':
    print('Argument ordering: play.py mode depth time <BOARD_ARGS>')
    print('Board args (must give all or none):')
    print('\tTic Tac Toe: \'ttt\' size')
    print('\tConnect Four: \'c4\' width height length')
    exit()


# Game parameters
mode = -1 if argv[1] == 'CPU' else int(2*random.random()) if argv[1] == 'PLAYER' else None

if mode == None:
    print(sys.argv[0] + ': cannot use', mode, 'as a mode')

search_depth = int(argv[2])
search_time = float(argv[3])
game_board = boards[argv[4]](argv[5:])

mcts = MonteCarloTS(game_board, max_depth = search_depth, decision_time=search_time)

searcher = mcts

def player_move():
    if argv[4] == 'ttt':
        a = int(input('Please give row: '))
        b = int(input('Please give col: '))
        if (a, b) not in game_board.legal_moves(searcher.states):
            print('error: cannot make that move; try again.')
            return player_move()
        else:
            return (a, b)
    elif argv[4] == 'c4':
        a = int(input('Please give a col: '))
        return a

move_makers = [
    searcher.choose_move if mode is not i else player_move
    for i in range(2)
]

player = 1

# Choose a move
winner = 0
while not winner:

    game_board.draw_board(searcher.states[-1])
    
    # Choose a move
    move = move_makers[player-1]()

    print('\nPlayer', player, 'goes at', move)
    searcher.update(move)
    winner = game_board.winner(searcher.states)

    player = 1 + (player % 2)

game_board.draw_board(searcher.states[-1])
print()

if winner > 0:
    print('Player', winner, 'wins!')
else:
    print('The match is a tie')


