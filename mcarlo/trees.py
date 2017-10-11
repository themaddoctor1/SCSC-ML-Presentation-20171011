import random
import math

import datetime

class Board:
    
    def __init__(self):
        pass
    
    def start(self):
        """Creates the initial game state
        """
        pass

    def legal_moves(self, history):
        """Provides the set of legal moves for this state, given
        the moves previously played (including the current move)
        """
        pass

    def current_player(self, state):
        """Returns the current player in the game
        """
        pass

    def make_move(self, state, move):
        """Given a state, make the next move.
        state:   a sequence of legal game states
        return:  a new state.
        """
        pass

    def winner(self, history):
        """Returns the winner in the given state, if one exists.
        """
        pass
    
    def loser(self, history):
        pass

    def draw_board(self, state):
        pass

class DecisionTreeSearch:
    
    def __init__(self, board):
        pass

    def choose_move(self):
        """Choose a move to play.
        return: A value representing a move, or None if one cannot be made
        """
        pass

    def update(self, move):
        """Update the searcher with the latest move so that the search algorithm
        can make a decision afterwards.
        """
        new_state = self.board.make_move(self.states[-1], move)
        self.states.append(new_state)


   

class MonteCarloTS(DecisionTreeSearch):
    
    def __init__(self, board, decision_time=5, max_depth=100):
        """Initializes a Monte Carlo tree search object.
        board:  A game board to play with
        """

        # Hold the board in use.
        self.board = board

        # Each node will have markers s/n, where
        # s is the difference between the number of wins and losses
        # n is the number of visits
        self.scores = {}
        self.visits = {}

        # The chain of game states visited so far
        self.states = [board.start()]

        # Search parameters
        self.max_depth = max_depth
        self.time_limit = datetime.timedelta(seconds=decision_time)

    def choose_move(self):

        # Get the current state
        state = self.states[-1]
        
        # Gat the current player
        player = self.board.current_player(state)

        rounds = 0
        
        # Perform simulations for the given amount of time
        start_time = datetime.datetime.now()
        while datetime.datetime.now() - start_time < self.time_limit:
            self.simulate()
            rounds += 1

        print('ran', rounds, 'simulations')
        print('seen', len(self.visits), 'states')
        
        # Gather the set of possible moves
        legal_moves = self.board.legal_moves(self.states)
        
        # Retrieve a set of move-state tuples
        options = [(m, self.board.make_move(state, m)) for m in legal_moves]

        if len(options) == 0:
            return None
        
        """
        for m, s in options:
            key = (player, s)
            print(m, ':', self.scores[key], '/', self.visits[key], '(%.3f)' % (self.scores[key] / self.visits[key]))
        """

        # Choose a move. It will be the one with the highest probability of victory
        _, move = max([(self.scores[(player, s)] / self.visits[(player, s)], m) for m, s in options])

        return move

    def simulate(self):
        """Evaluates one possible game in the complete game tree
        """

        # Hold onto the states visited so far
        visited = []

        # Retrieve the check values
        scores = self.scores
        visits = self.visits

        # Gat a local copy of the board
        board = self.board

        # Get current state parameters
        states = self.states[:]
        state = states[-1]
        player = board.current_player(state)

        # The max depth to explore
        max_depth = self.max_depth
        
        """
        Initial conditions:
        * states[-1] is the root of the tree to explore.
        """

        # Initially, the search will parse through explored regions of the tree
        explored = True

        for i in range(max_depth):
            # Pass through 100 moves in the tree
            
            # Get the set of legal moves
            legal_moves = board.legal_moves(states)

            if len(legal_moves) == 0:
                break

            # Choose a random move to explore
            tuples = [(m, board.make_move(state, m)) for m in legal_moves]
            if all([(player, s) in visits for m, s in tuples]):
                # All of the moves have been checked at least once.
                # So, choose a move that is either likely to win, or
                # has not been explored much.

                # Choose maximum confidence, defined by:
                # w_i / n_i + c * sqrt(log(sum(n_i)) / n_i)
                C = 2 ** 0.5
                total = math.log(sum([visits[player, s] for m, s in tuples]))
                
                # Gather the move and follow-up state with the largest confidence level
                _, move, state = max(
                    [scores[player, s] / visits[player, s] + C * math.sqrt(total / visits[player, s]), m, s]
                    for m, s in tuples
                )

            else:
                # Select a random move
                move, state = random.choice(tuples)
            
            # Save the new state 
            states.append(state)
            
            """
            print('From the following tuples:')
            for t in tuples:
                print('   ', t)

            print('Choose:', move, state)
            print()
            """

            # Create a player-state pair. From any state, such a tuple that
            # represents a move can be made by calling make_move
            move_tuple = (player, state)

            if explored and move_tuple not in visits:
                # The state-move combo has never occured, so add check values
                scores[move_tuple] = 0
                visits[move_tuple] = 0

                # No longer parsing explored nodes
                explored = False
            
            # Add to the set of visited states
            if move_tuple not in visited:
                visited.append(move_tuple)

            # Iterate to next player
            player = board.current_player(state)

            # Check for a winner or loser
            winner, loser = board.winner(states), board.loser(states)
            if winner or loser:
                # The game is over, so exit
                break
        
        # Now that a route has been formed, evaluate the outcome.
        for p, s in visited:
            move_tuple = (p, s)
            
            # Only update the checks if there's a check to update
            if move_tuple in visits:
                # Update the visit count
                visits[move_tuple] += 1
                
                # If the current player won entering said state,
                # record a win.
                if p is winner:
                    # The player won
                    scores[move_tuple] += 1
                elif p is not loser:
                    # One of the other players won
                    scores[move_tuple] += 0.5


