import numpy as np
cimport numpy as np


def gen(indexes, obs):
    cdef list retX1 = []
    cdef list retX2 = []
    cdef list retX3 = []
    cdef list retY = []
    # cdef str replay, temp
    cdef int wanted_size
    for replay in indexes:
        data = np.load(replay)
        # print(self.replays_path+ replay)
        x = [data['name1'], data['name2'], data['name3']]
        # if data['name1'].shape[0] !=64 or data['name2'].shape[0] !=64 or data['name3'].shape[0] !=64:
        #     print(self.all_replays[idx])
        #     print("error")
        y = data['name4']
        wanted_size = -obs
        retX1.append(x[0][wanted_size:])
        retX2.append(x[1][wanted_size:])
        X3 = [z.flatten() for z in x[2][wanted_size:]]
        # X3 = np.reshape(X3, (wanted_size, 522))
        retX3.append(X3)
        retY.append(y[wanted_size:][0])
    # np.savez("testmdata{0}.npz".format(str(idx)), name1=x[0], name2=x[1], name3=[2], name4=y)
    # X, Y = np.array(retX), np.array(retY)
    # print(X.shape, Y.shape)
    # print("Data obtained.")
    return [np.asarray(retX1), np.asarray(retX2), np.asarray(retX3)], retY
