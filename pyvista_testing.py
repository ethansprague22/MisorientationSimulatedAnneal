import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

N = 8

values = np.arange(0,N**3).reshape((N,N,N), order="C")
values.shape

grid = pv.ImageData()

grid.dimensions = np.array(values.shape) + 1
print(grid.dimensions)

grid.origin = (0,0,0)
grid.spacing = (1,1,1)

grid.cell_data['values'] = values.flatten(order='F')

nbrs_values = [138, 136, 201, 73, 145, 129]
nbrs_ids = np.where(np.isin(grid.cell_data['values'],nbrs_values))[0]

# grid.plot(show_edges=True)

threshed_nbrs = grid.extract_cells(nbrs_ids)
threshed_nbrs.plot()
# grid.save('pyvista_micro.vtk')

# micro = pv.read('micro.vtk')
# print(micro.cell_data.keys())
# # micro.plot()

# threshed = micro.threshold(value=(10,15))
# threshed.plot()