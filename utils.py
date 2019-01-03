# Importation
import numpy as np
import random as rd


def rd_argmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    if vector.size == 0:
        raise IndexError("argmax of an empty array")
    else:
        m = np.amax(vector)
        indices = np.nonzero(vector == m)[0]
        return rd.choice(indices)

