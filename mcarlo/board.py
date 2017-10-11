from trees import Board

class TicTacToe(Board):
    
    def __init__(self, size=3):
        self.radius = size

    def start(self):
        return tuple(tuple(0 for i in range(self.radius)) for j in range(self.radius))
    
    def legal_moves(self, hist):
        last_move = hist[-1] if len(hist) > 0 else self.start()
        
        moves = []
        for i in range(self.radius):
            for j in range(self.radius):
                if last_move[i][j] == 0:
                    moves.append((i, j))

        return moves
    
    def current_player(self, state):
        player = 0
        for i in range(self.radius):
            for j in range(self.radius):
                if state[i][j] != 0:
                    player = player + 1
        return 1 + (player % 2)

    def make_move(self, state, move):
        player = self.current_player(state)
        x, y = move

        # Get the player number
        return tuple(tuple(state[i][j] if (move[0] != i or move[1] != j) else player for j in range(self.radius)) for i in range(self.radius))
    
    def winner(self, history):
        state = history[-1]

        for i in range(self.radius):
            # Check rows and columns
            if state[i][0] and all([state[i][0] == state[i][j] for j in range(1, self.radius)]):
                return state[i][0]
            if state[0][i] and all([state[0][i] == state[j][i] for j in range(1, self.radius)]):
                return state[0][i]
        
        # Check diagonals
        if state[0][0] and all([state[0][0] == state[i][i] for i in range(1, self.radius)]):
            return state[0][0]
        elif state[self.radius - 1][0] and all([state[self.radius - 1][0] == state[2-i][i] for i in range(1, self.radius)]):
            return state[self.radius - 1][0]
        
        # None of the states were wins, so check for a tie
        if len(history) >= (self.radius ** 2) + 1:
            return -1
        else:
            return 0

    def loser(self, history):
        winner = self.winner(history)

        if winner > 0:
            return 1 if winner is 2 else 2
        else:
            return winner


    def draw_board(self, state):
        res = ""
        for i in range(self.radius):
            if i > 0:
                res += (self.radius-1) * '-+' + '-\n'

            for j in range(self.radius):
                if j > 0:
                    res += '|'
                if state[i][j] == 0:
                    res += ' '
                else:
                    res += '\033[1m\033[32m' + ('X' if state[i][j] == 1 else 'O') + '\033[0m'

            res += '\n'

        print(res)


class ConnectFour(Board):
    
    def __init__(self, width=7, height=6, length=4):
        self.width = width
        self.height = height
        self.chainlen = length

    def start(self):
        return tuple(() for i in range(self.width))

    def legal_moves(self, hist):
        state = hist[-1] if len(hist) > 0 else self.start()
        
        # Allow unfilled columns to be filled
        moves = []
        for i in range(self.width):
            if len(state[i]) < self.height:
                moves.append(i)

        return moves

    def current_player(self, state):
        net = 0
        for col in state:
            net += len(col)
        return (net%2) + 1

    def make_move(self, state, move):
        current_player = self.current_player(state)

        return tuple(state[i] if move is not i else (state[i] + tuple([current_player])) for i in range(self.width))

    def winner(self, history):
        state = history[-1]
        array = [[0 for i in range(self.height)] for j in range(self.width)]
        for i in range(self.width):
            for j in range(len(state[i])):
                array[i][j] = state[i][j]

        for i in range(self.width):
            for j in range(self.height):
                test = []

                # Check vertical
                if j + self.chainlen <= self.height:
                    test.append([array[i][j+k] for k in range(self.chainlen)])
                
                # Check horizontal
                if i + self.chainlen <= self.width:
                    test.append([array[i+k][j] for k in range(self.chainlen)])
                
                # Check diagonal up-right
                if i + self.chainlen <= self.width and j + self.chainlen <= self.height:
                    test.append([array[i+k][j+k] for k in range(self.chainlen)])
                
                # Check diagonal up-left
                if i >= self.chainlen - 1 < self.width and j + self.chainlen <= self.height:
                    test.append([array[i-k][j+k] for k in range(self.chainlen)])

                for item in test:
                    if all([item[i] == item[0] for i in range(len(item))]) and item[0]:
                        return item[0]

        if len(self.legal_moves(history)) == 0:
            return -1
        else:
            return 0

    def loser(self, history):
        winner = self.winner(history)

        if winner > 0:
            return 1 if winner is 2 else 2
        else:
            return winner
    
    def draw_board(self, state):
        res = ""
        for i in range(self.height):
            row = "|"

            for j in range(self.width):
                if i >= len(state[j]):
                    row += " "
                elif state[j][i] == 0:
                    row += " "
                elif state[j][i] == 1:
                    row += '\033[1m\033[31mX\033[0m'
                else:
                    row += '\033[1m\033[33mO\033[0m'

            row += '|'

            res = row + '\n' + res

        print(res)


