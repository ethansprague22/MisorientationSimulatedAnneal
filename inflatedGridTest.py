import numpy as np
import pyvista as pv

# ----------------------------------------------------------------------
# 1. parameters
# ----------------------------------------------------------------------
N_GR        = 8           # grains along each axis
VOX_PER_GR  = 10          # voxels per grain edge
N_VOX       = N_GR * VOX_PER_GR   # 80
SPACING     = (1.0, 1.0, 1.0)     # physical size of one voxel
ORIGIN      = (0.0, 0.0, 0.0)

# ----------------------------------------------------------------------
# 2. build the SMALL 8×8×8 cube of grain IDs (Fortran order ⇢ x-fastest)
# ----------------------------------------------------------------------
grain_ids_small = np.arange(N_GR**3).reshape((N_GR,)*3, order='F')

# ----------------------------------------------------------------------
# 3. inflate each grain to 10×10×10 voxels ⇢ 80×80×80
# ----------------------------------------------------------------------
grain_ids_big = np.repeat(
                    np.repeat(
                        np.repeat(grain_ids_small, VOX_PER_GR, axis=0),
                                 VOX_PER_GR, axis=1),
                                 VOX_PER_GR, axis=2)

# ----------------------------------------------------------------------
# 4. create ImageData (cell-centred) and attach the field
# ----------------------------------------------------------------------
img = pv.ImageData()
img.dimensions = np.array([N_VOX, N_VOX, N_VOX]) + 1   # +1 → point dimensions
img.origin     = ORIGIN
img.spacing    = SPACING

# flatten in Fortran order so VTK cell numbering matches
img.cell_data['grain_id'] = grain_ids_big.flatten(order='F').astype(np.int32)

# ----------------------------------------------------------------------
# 5. plot
# ----------------------------------------------------------------------
p = pv.Plotter()
p.add_mesh(img,
           scalars='grain_id',
           cmap='turbo',
           opacity=1.0,
           show_scalar_bar=True,   # turn off if you’ll add text labels later
           show_edges=True)
p.background_color = 'white'
p.show()
