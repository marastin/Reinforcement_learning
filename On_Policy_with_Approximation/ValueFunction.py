class ValueFunctionTiling:
    def __init__(self, n_tiling:int, alpha=0.1, mem_size=2048) -> None:
        self.iht = IHT(mem_size)
        self.n_tiling = n_tiling
        self.alpha = alpha / float(n_tiling) # step size
        self.weights = np.zeros(mem_size)
        self.scales = None
    
    def define_scales(self, floats_min, floats_max):
        self.scales = [self.n_tiling / (floats_max[i] - floats_min[i]) for i in range(len(floats_max))]

    def value(self, state_floats, state_ints):
        if self.scales is not None:
            state_floats = np.multiply(self.scales, state_floats)
        active_tiles = tiles(self.iht, self.n_tiling, state_floats, state_ints)
        return np.sum(self.weights[active_tiles])
    
    def update(self, delta, state_floats, state_ints):
        active_tiles = tiles(self.iht, self.n_tiling, state_floats, state_ints)
        for tile in active_tiles:
            self.weights[tile] += self.alpha * delta   
        



"""
Tile Coding Software version 3.0beta
by Rich Sutton
Link: http://incompleteideas.net/tiles/tiles3.html
"""
basehash = hash

class IHT:
    "Structure to handle collisions"
    def __init__(self, sizeval):
        self.size = sizeval                        
        self.overfullCount = 0
        self.dictionary = {}

    def __str__(self):
        "Prepares a string for printing whenever this object is printed"
        return "Collision table:" + \
               " size:" + str(self.size) + \
               " overfullCount:" + str(self.overfullCount) + \
               " dictionary:" + str(len(self.dictionary)) + " items"

    def count (self):
        return len(self.dictionary)
    
    def fullp (self):
        return len(self.dictionary) >= self.size
    
    def getindex (self, obj, readonly=False):
        d = self.dictionary
        if obj in d: return d[obj]
        elif readonly: return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfullCount==0: print('IHT full, starting to allow collisions')
            self.overfullCount += 1
            return basehash(obj) % self.size
        else:
            d[obj] = count
            return count

def hashcoords(coordinates, m, readonly=False):
    if type(m)==IHT: return m.getindex(tuple(coordinates), readonly)
    if type(m)==int: return basehash(tuple(coordinates)) % m
    if m==None: return coordinates

from math import floor, log
from itertools import zip_longest

def tiles (ihtORsize, numtilings, floats, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    qfloats = [floor(f*numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling*2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append( (q + b) // numtilings )
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles

def tileswrap (ihtORsize, numtilings, floats, wrapwidths, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints, wrapping some floats"""
    qfloats = [floor(f*numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling*2
        coords = [tiling]
        b = tiling
        for q, width in zip_longest(qfloats, wrapwidths):
            c = (q + b%numtilings) // numtilings
            coords.append(c%width if width else c)
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles