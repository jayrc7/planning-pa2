import numpy as np
from collections import defaultdict
from queue import PriorityQueue

class BoardState:
    """
    Represents a state in the game
    """

    def __init__(self):
        """
        Initializes a fresh game state
        """
        self.N_ROWS = 8
        self.N_COLS = 7

        self.state = np.array([1,2,3,4,5,3,50,51,52,53,54,52])
        self.decode_state = [self.decode_single_pos(d) for d in self.state]

    def update(self, idx, val):
        """
        Updates both the encoded and decoded states
        """
        self.state[idx] = val
        self.decode_state[idx] = self.decode_single_pos(self.state[idx])

    def make_state(self):
        """
        Creates a new decoded state list from the existing state array
        """
        return [self.decode_single_pos(d) for d in self.state]

    def encode_single_pos(self, cr: tuple):
        """
        Encodes a single coordinate (col, row) -> Z

        Input: a tuple (col, row)
        Output: an integer in the interval [0, 55] inclusive

        """
        # get row and col
        row = cr[1]
        col = cr[0]

        # compute int representation of coordinate
        return row * self.N_COLS + col

    def decode_single_pos(self, n: int):
        """
        Decodes a single integer into a coordinate on the board: Z -> (col, row)

        Input: an integer in the interval [0, 55] inclusive
        Output: a tuple (col, row)

        """
        col = n % self.N_COLS
        row = n // self.N_COLS
        return (col, row)

    def is_termination_state(self):
        """
        Checks if the current state is a termination state. Termination occurs when
        one of the player's move their ball to the opposite side of the board.

        You can assume that `self.state` contains the current state of the board, so
        check whether self.state represents a termainal board state, and return True or False.
        
        TODO: You need to implement this.
        """
        # first check if the board is valid
        if not self.is_valid():
            return False

        # get white's ball location
        whites_ball = self.state[5]

        # get black's ball location
        blacks_ball = self.state[11]

        # check if black's ball is in white's endzone
        if (blacks_ball >= 0 and blacks_ball <= 6) ^ (whites_ball >= 49 and whites_ball <= 55):
            return True

        # return false if we reach this point since no ball is on the endzone of the other player
        return False

    def is_valid(self):
        """
        Checks if a board configuration is valid. This function checks whether the current
        value self.state represents a valid board configuration or not. This encodes and checks
        the various constrainsts that must always be satisfied in any valid board state during a game.

        If we give you a self.state array of 12 arbitrary integers, this function should indicate whether
        it represents a valid board configuration.

        Output: return True (if valid) or False (if not valid)

        VALID CRITERIA as of now:
            neither player's pieces can be out of bounds
            each player's ball needs to reside in one of their pieces
            two pieces cannot reside in the same location
        """
        # first check that all pieces are within bounds
        whites_pieces = self.state[:5] 
        blacks_pieces  = self.state[6:11]
        all_pieces = np.concatenate((whites_pieces, blacks_pieces))
        for piece in all_pieces: 
            # if there is a piece that's out of bounds, return invalid
            if piece < 0 or piece > 55:
                return False

        # now make sure that each player's ball is in one of their pieces
        whites_ball = self.state[5]
        blacks_ball = self.state[11]
        if whites_ball not in whites_pieces or blacks_ball not in blacks_pieces:
            return False
        
        # finally, make sure there are no pieces that are overlapping
        occupied_locations = defaultdict(bool)
        for location in all_pieces:
            # return false if current pieces location was already occupied by a previous piece
            if occupied_locations[location]:
                return False

            # set location to true to denote it as occupied
            occupied_locations[location] = True

        # return valid if none of the criteria for invalid was met
        return True

class Rules:

    @staticmethod
    def single_piece_actions(board_state : BoardState, piece_idx):
        """
        Returns the set of possible actions for the given piece, assumed to be a valid piece located
        at piece_idx in the board_state.state.

        Inputs:
            - board_state, assumed to be a BoardState
            - piece_idx, assumed to be an index into board_state, identfying which piece we wish to
              enumerate the actions for.

        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that piece_idx can move to during this turn.
        
        TODO: You need to implement this., remove BoardState type checking before turning in
        """
        # first get the encoded location of the piece 
        encoded_piece_location = board_state.state[piece_idx]

        # if this piece contains a ball, return no valid moves
        if encoded_piece_location == board_state.state[5] or encoded_piece_location == board_state.state[11]:
            return []

        # decode it back into it's coordintate version
        col, row = board_state.decode_single_pos(encoded_piece_location)

        # list moves that the piece can make from here
        moves = [
            (col - 1, row + 2), # up     2, left   1
            (col + 1, row + 2), # up     2, right  1
            (col - 1, row - 2), # down   2, left   1
            (col + 1, row - 2), # down   2, right  1
            (col + 2, row + 1), # right  2, up     1
            (col + 2, row - 1), # right  2, down   1
            (col - 2, row + 1), # left   2, up     1
            (col - 2, row - 1)  # left   2, down   1
        ]
        # getting location of all pieces
        whites_pieces = board_state.state[:5] 
        blacks_pieces  = board_state.state[6:11]
        all_pieces = np.concatenate((whites_pieces, blacks_pieces))

        # filter out invalid moves due to out of bounds or location being occupied already
        final_moves = []
        for move in moves: 
            # get x, y of move
            x = move[0]
            y = move[1]

            # skip if x is out of bounds
            if x < 0 or x > board_state.N_COLS - 1:
                continue

            # skip if y is out of bounds
            if y < 0 or y > board_state.N_ROWS - 1:
                continue

            # skip if location is already taken
            current_move_encoded = board_state.encode_single_pos(move)
            if current_move_encoded in all_pieces:
                continue

            # append encoded position to list of final moves if no criteria for skipping was met
            final_moves.append(current_move_encoded)

        
        # return list of final moves
        return final_moves
    
    @staticmethod
    def clear_path(p1, p2, opponent_pieces, board_state : BoardState):
        # get coordinates of first piece
        p1_x = p1[0]
        p1_y = p1[1]

        # get coordinates of second piece
        p2_x = p2[0]
        p2_y = p2[1]

        # capture leftmost and rightmost point
        leftmost_x  = p2_x if p2_x < p1_x        else p1_x
        rightmost_x = p2_x if leftmost_x == p1_x else p1_x
        bottom_y    = p2_y if p2_y < p1_y        else p1_y
        top_y       = p2_y if bottom_y == p1_x   else p1_y

        # check if they're on the same row
        if p1_y == p2_y:
            # make sure there's no opponent pieces in between
            pieces_in_between = [op for op in opponent_pieces if op[0] >= leftmost_x and op[0] <= rightmost_x and op[1] == p1_y]

            # no pieces in between, return True
            if len(pieces_in_between) == 0: return True

        # check if they're in the same column
        elif p1_x == p2_x: 
            # make sure there's no opponent pieces in between
            pieces_in_between = [op for op in opponent_pieces if op[1] >= bottom_y and op[1] <= top_y and op[0] == p1_x]

            # no pieces in between, return True
            if len(pieces_in_between) == 0: return True

        else: # else check if we can reach it via a diagonal
            # check to see if the two pieces are diagonal (right diagonal) from each other
            if abs(p1_x - p2_x) == abs(p1_y - p2_y):
                # try to find an opponent that blocks the diagonal
                for i in range(len(opponent_pieces)):
                    # current opp
                    current_opp = opponent_pieces[i]

                    # first check to see if opponent piece is within x, y range
                    if current_opp[0] >= leftmost_x and current_opp[0] <= rightmost_x and current_opp[1] <= top_y and current_opp[1] >= bottom_y:
                        # now check if it's on the same diagonal, return false if it is
                        if  abs(p1_x - current_opp[0]) == abs(p1_y - current_opp[1]): 
                            return False

                # return true at this point since no opp was in the way
                return True

        # no clear path, return False
        return False

    @staticmethod
    def single_ball_actions(board_state : BoardState, player_idx):
        """
        Returns the set of possible actions for moving the specified ball, assumed to be the
        valid ball for plater_idx  in the board_state

        Inputs:
            - board_state, assumed to be a BoardState
            - player_idx, either 0 or 1, to indicate which player's ball we are enumerating over
        
        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that player_idx's ball can move to during this turn.
        
        TODO: You need to implement this.
        """
        # define variables for capturing curent player's pieces and ball locations
        current_pieces = []
        current_ball = -1
        opponent_pieces = []

        # check to see if we're dealing with white player (even idx) or black player (odd idx)
        if player_idx % 2 == 0:  # they're the white player
            current_pieces  = board_state.decode_state[:5]
            current_ball    = board_state.decode_state[5]
            opponent_pieces = board_state.decode_state[6:11]
        else: # we're dealing with the black player
            current_pieces  = board_state.decode_state[6:11]
            current_ball    = board_state.decode_state[11]
            opponent_pieces = board_state.decode_state[:5]

        # first remove the piece that contains the player's ball
        current_pieces.remove(current_ball)
        
        # now check to see which pieces we can go to
        final_moves = set()
        pieces_found = 0
        pq= PriorityQueue()
        pq.put((0, current_ball))
        visited = defaultdict(bool)
        visited[board_state.encode_single_pos(current_ball)] = True
        cost = defaultdict(int)
        cost[current_ball] = 0

        sorted_current_pieces = sorted(current_pieces, key=board_state.encode_single_pos)

        # iterate until we either have no more neighbors or we've visited all pieces already
        # while pq and pieces_found < 4:
        for i in range(0, 4): 
            for current_piece in sorted_current_pieces:
                # current_tup = pq.get()
                # current_piece = current_tup[1]
                
                # first check to see if there's a direct, unblocked path from the ball to the current piece
                if Rules.clear_path(current_ball, current_piece, opponent_pieces, board_state):
                    final_moves.add(board_state.encode_single_pos(current_piece)) # if so, append piece to list of moves it can make 

                # if not, check to see if there's a clear path between the current piece and a valid move
                # keep updating just in case we have a new valid path that enables others
                for valid_move in final_moves:
                    # if there is a clear path, that means we can hop to this piece
                    if Rules.clear_path(board_state.decode_single_pos(valid_move), current_piece, opponent_pieces, board_state):
                        final_moves.add(board_state.encode_single_pos(current_piece))
                        break
              
        # return valid moves that can be made by the player during this turn
        return final_moves

class GameSimulator:
    """
    Responsible for handling the game simulation
    """

    def __init__(self, players):
        self.game_state = BoardState()
        self.current_round = -1 ## The game starts on round 0; white's move on EVEN rounds; black's move on ODD rounds
        self.players = players

    def run(self):
        """
        Runs a game simulation
        """
        while not self.game_state.is_termination_state():
            ## Determine the round number, and the player who needs to move
            self.current_round += 1
            player_idx = self.current_round % 2
            ## For the player who needs to move, provide them with the current game state
            ## and then ask them to choose an action according to their policy
            action, value = self.players[player_idx].policy( self.game_state.make_state() )
            print(f"Round: {self.current_round} Player: {player_idx} State: {tuple(self.game_state.state)} Action: {action} Value: {value}")

            if not self.validate_action(action, player_idx):
                ## If an invalid action is provided, then the other player will be declared the winner
                if player_idx == 0:
                    return self.current_round, "BLACK", "White provided an invalid action"
                else:
                    return self.current_round, "WHITE", "Black probided an invalid action"

            ## Updates the game state
            self.update(action, player_idx)

        ## Player who moved last is the winner
        if player_idx == 0:
            return self.current_round, "WHITE", "No issues"
        else:
            return self.current_round, "BLACK", "No issues"

    def generate_valid_actions(self, player_idx: int):
        """
        Given a valid state, and a player's turn, generate the set of possible actions that player can take

        player_idx is either 0 or 1

        Input:
            - player_idx, which indicates the player that is moving this turn. This will help index into the
              current BoardState which is self.game_state
        Outputs:
            - a set of tuples (relative_idx, encoded position), each of which encodes an action. The set should include
              all possible actions that the player can take during this turn. relative_idx must be an
              integer on the interval [0, 5] inclusive. Given relative_idx and player_idx, the index for any
              piece in the boardstate can be obtained, so relative_idx is the index relative to current player's
              pieces. Pieces with relative index 0,1,2,3,4 are block pieces that like knights in chess, and
              relative index 5 is the player's ball piece.
            
        TODO: You need to implement this.
        """
        current_pieces = []

        # first determine which player we currently are
        if player_idx % 2 == 0:  # they're the white player
            current_pieces  = self.game_state.state[:6]
        else: # we're dealing with the black player
            current_pieces  = self.game_state.state[6:12]

        # generate valid moves for each piece, INCLUDING the ball
        moves_set = []
        for i in range(len(current_pieces)):
            # get current piece
            current_piece = current_pieces[i]

            # generate valid moves for piece or ball
            valid_moves = Rules.single_piece_actions(self.game_state, player_idx * 6 + i) if i < 5 else Rules.single_ball_actions(self.game_state, player_idx)

            # create a tuple for each move
            for move in valid_moves:
                moves_set.append((i, move))

        # return final set of moves
        return moves_set


    def validate_action(self, action: tuple, player_idx: int):
        """
        Checks whether or not the specified action can be taken from this state by the specified player

        Inputs:
            - action is a tuple (relative_idx, encoded position)
            - player_idx is an integer 0 or 1 representing the player that is moving this turn
            - self.game_state represents the current BoardState

        Output:
            - if the action is valid, return True
            - if the action is not valid, raise ValueError
        
        TODO: You need to implement this.
        """
        # create temp board so that we can test updating the state with the new action
        temp_board = BoardState()
        temp_board.state = [piece for piece in self.game_state.state]

        # decode action tuple
        relative_idx = action[0]
        new_position_location = action[1]

        # generate moves the current player can create
        moves = Rules.single_piece_actions(temp_board, (6 * player_idx) + relative_idx) if relative_idx < 5 else Rules.single_ball_actions(temp_board, player_idx)

        # check to see if new position is in list of valid moves the piece can make
        if new_position_location in moves: 
            return True
        else: 
            raise ValueError("move not allowed!")

    def update(self, action: tuple, player_idx: int):
        """
        Uses a validated action and updates the game board state
        """
        offset_idx = player_idx * 6 ## Either 0 or 6
        idx, pos = action
        self.game_state.update(offset_idx + idx, pos)
