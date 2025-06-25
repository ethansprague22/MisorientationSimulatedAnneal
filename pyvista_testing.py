import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# N = 10

# values = np.arange(0,N**3).reshape((N,N,N))
# values.shape

# grid = pv.ImageData()

# grid.dimensions = np.array(values.shape) + 1

# grid.origin = (0,0,0)
# grid.spacing = (1,1,1)

# grid.cell_data['values'] = values.flatten(order='F')

# grid.plot(show_edges=True)

micro = pv.read('micro.vtk')
print(micro.cell_data.keys())
# micro.plot(show_edges=True)