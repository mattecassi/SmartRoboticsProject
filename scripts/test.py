import numpy as np

class Orderer():

    def __init__(self):
        pass

    def __call__(self, pos0, blocks, pos_final):
        pos0 = np.array(pos0)
        blocks = np.array(blocks)
        first_elem = np.argmin(np.linalg.norm(pos0 - blocks, axis=1))
        order = np.arange(blocks.shape[0])
        order[[0, first_elem]] = order[[first_elem,0]]
        return [list(b) for b in blocks[order]]

if __name__ == "__main__":
    object_poses = [
        [0.65, 0.23, .3],
        [0.3, -0.2,.3],
        [0.5, 0., .3],
    ]
    pos0 = [0,0,.3]
    orderer = Orderer()
    print(orderer(pos0, object_poses, None))