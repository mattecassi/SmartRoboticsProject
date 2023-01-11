import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming

class TSP_solver():
    """
    This classes solves the open TSP problem over the boxes destinations.
    """

    def __init__(self):
        pass


    def generate_dist_mtx(self, pos0, goal, blocks):
        DIM = 1 + blocks.shape[0]
        matrix = np.zeros([DIM, DIM])
        dist_block_goal = np.linalg.norm(blocks - goal, axis=1)
        dist_pos0 = np.linalg.norm(blocks - pos0, axis=1)
        
        matrix[0,1:] = dist_pos0 
        
        for row in range(1,DIM):
            for col in range(1,DIM):
                matrix[row, col] = dist_block_goal[:]

    def __call__(self, pos0, pos_final, blocks):
        """
        This functions return the ordere list of blocs to be visited.
        """       
        pass

if __name__ == "__main__":
    distance_matrix = np.array([
        [0,  5, 4, 10],
        [5,  0, 8,  5],
        [4,  8, 0,  3],
        [10, 5, 3,  0]
    ])
    distance_matrix[:,0] = 0
    permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
    print(permutation)
    print(distance)