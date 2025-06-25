'''
This is code that generates synthetic theoretical microstructures that have
identical textures but different misorientation distributions.

It creates a cubic lattice of N^3 grains. The crystal orientations of the grains
are from an initial random texture sampling.  A simulated annealing algorithm
will be applied which will iteratively swap the orientations of pairs of grains
in order to optimize for maximum and minimum misorientation configurations.  A
cubic lattice of grains is chosen so that grain orientations can be swapped
arbitrarily without changing the overall volume weighted crystallographic
texture.


I will need the following functions: - Generate texture from uniform SO(3)
sampling - Get the coords of the neighbours to a given cell (6 immediate facing
neighbours or ALL 26 neighbours including edges and corners...?) - Calculate the
misorientation between two orientations (given the hcp symmetry) - Calculate the
summed misorientation of a grain with its neighbours

Main loop: - randomly choose pair - calculate the change in total misorientation
after swapping them dM: - first calculate summed misorientation of both grains,
then the summed misorientation in the swapped configuration - what if they are
on the edge? what if they are already neighbours?  - perform the swap with a
probability (depends on if swap is favourable and the current temperature) -
decrease temperature


It might be most efficient to store the grain information in arrays. E.g. x =
[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], orix = [0. ...], oriy = [0. ...], oriz =
[0. ...]

Orientations should be stored as the three unit vectors which make up the
columns of the rotation matrix (sample to local).
'''

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

N = 5
offsets = [[0,0,1],[0,0,-1],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]]

idx = np.arange(N**3).reshape((N,N,N))

nbrs = [[] for _ in range(N**3)]
for x,y,z in np.ndindex(N,N,N):
    here = idx[x,y,z]
    for dx,dy,dz in offsets:
        i,j,k = x+dx,y+dy,z+dz
        if 0<=i<N and 0<=j<N and 0<=k<N:
            nbrs[here].append(idx[i,j,k])

nbrs = [np.array(n, dtype=np.int32) for n in nbrs]

id = 0
for i in nbrs:
   print(f'{id}:{i}')
   id += 1

for dx,dy,dz in offsets:
    print(f'{dx},{dy},{dz}')
    


def misori_batch(q,qn,sym_ops):
    return 0


def local_energy(qs, node, nbrs, sym_ops):
    q = qs[node]
    qn = qs[nbrs[node]]
    return misori_batch(q,qn,sym_ops).sum()

# E_before = 