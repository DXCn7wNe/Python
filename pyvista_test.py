import numpy as np
import pandas as pd
import pyvista as pv
from pyvista import examples

# shirts_path = './models/shirts/shirts_simple.obj'
# img_file = './models/shirts/up_dog.jpg'
shirts_path = './models/Artec3D/Pasha_guard_head.obj'
img_file = './models/Artec3D/Pasha_guard_head_0.png'

tex = pv.read_texture(img_file)

# Manually parse the OBJ file because meshio complains
raw_data = pd.read_csv(shirts_path, header=None, comment="#",
    delim_whitespace=True, names=["type", "a", "b", "c"])
groups = raw_data.groupby("type")
# from IPython.core.debugger import Pdb; Pdb().set_trace()
v = groups.get_group("v")
f = groups.get_group("f")
vt = groups.get_group("vt")[["a", "b"]].values.astype(float)
vertices = v[["a", "b", "c"]].astype(float).values
# fa = np.array([(int(x[0]), int(x[1]), int(x[2])) for x in f["a"].str.split("/")])
# fb = np.array([(int(x[0]), int(x[1]), int(x[2])) for x in f["b"].str.split("/")])
# fc = np.array([(int(x[0]), int(x[1]), int(x[2])) for x in f["c"].str.split("/")])
fa = np.array([(int(x[0]), int(x[1])) for x in f["a"].str.split("/")])
fb = np.array([(int(x[0]), int(x[1])) for x in f["b"].str.split("/")])
fc = np.array([(int(x[0]), int(x[1])) for x in f["c"].str.split("/")])
faces = np.c_[fa[:,0], fb[:,0], fc[:,0]] - 1 # subtract 1
#### End manual parsing

# Create the mesh
cells = np.c_[np.full(len(faces), 3), faces]
mesh = pv.PolyData(vertices, cells)

# Generate the tcoords on the faces
ctcoords = np.c_[fa[:,1], fb[:,1], fc[:,1]] - 1 # subtract 1
ui, vi = ctcoords[:,0], ctcoords[:,1]
cuv = np.c_[vt[:,0][ui], vt[:,1][vi]]
mesh.cell_arrays["Texture Coordinates"] = cuv

# Interpolate the cell-based tcoords to the points
remesh = mesh.cell_data_to_point_data()
# Register the array as texture coords
remesh.t_coords = remesh.point_arrays["Texture Coordinates"]

# Plot it up, yo!
remesh.plot(texture=tex)

# model_path = './models/Artec3D/Pasha_guard_head.obj'
# texture_path = './models/Artec3D/Pasha_guard_head_0.png'
# plt = pv.Plotter()
# mesh = pv.read(model_path)
# tex = pv.read_texture(texture_path)
# # plt.add_mesh(mesh, texture=tex)
# plt.add_mesh(mesh)
# plt.set_background((19/255, 19/255, 36/255), (130/255, 134/255, 243/255))
# plt.show(cpos="xy")
