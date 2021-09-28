# from IPython.core.debugger import Pdb; Pdb().set_trace()
#
import numpy as np
from scipy.spatial.transform import Rotation
import pyvista as pv


def unit_vec(vec):
    vec = np.array(vec)
    return vec / np.linalg.norm(vec)

def _from_normal(old, new):
    old = np.array(old)
    new = np.array(new)
    vec = np.cross(old, new)
    if (np.linalg.norm(vec) == 0):
        rot = Rotation.from_quat((0,0,0,1))
    else:
        angle = np.arccos(np.inner(old, new)/np.linalg.norm(old)/np.linalg.norm(new))
        rotvec = angle * vec/np.linalg.norm(vec)
        rot = Rotation.from_rotvec(rotvec)
    return rot
setattr(Rotation, 'from_normal', _from_normal)

def _add_axis_vec(plt, pos=(0,0,0), length=1, x_vec=(1,0,0), z_vec=(0,0,1)):
    origin_pos = np.array((0,0,0))
    axis_x_vec = np.array((1,0,0))
    axis_y_vec = np.array((0,1,0))
    axis_z_vec = np.array((0,0,1))
    rot_z = Rotation.from_normal(axis_z_vec, z_vec)
    if (np.all(np.cross(axis_x_vec, x_vec) == 0)):
        rot_x = Rotation.from_quat((0,0,0,1))
    else:
        rot_x = Rotation.from_normal(rot_z.apply(axis_x_vec), x_vec)
    rot = rot_x * rot_z
    axis_x = pv.Arrow(pos, rot.apply(axis_x_vec))
    axis_y = pv.Arrow(pos, rot.apply(axis_y_vec))
    axis_z = pv.Arrow(pos, rot.apply(axis_z_vec))
    plt.add_mesh(axis_x, color='red')
    plt.add_mesh(axis_y, color='green')
    plt.add_mesh(axis_z, color='blue')
setattr(pv.Plotter, 'add_axis_vec', _add_axis_vec)

##
plt = pv.Plotter()
plt.add_axes_at_origin()
#
face_pos = np.array((2,0,0))
face_mesh = pv.read('male_head_reduce.stl')
face_mesh.scale(0.02)
face_rot = Rotation.from_euler('ZYX',(0, 45, -30),degrees=True)
face_mesh.rotate_vector(face_rot.as_rotvec(), np.rad2deg(np.linalg.norm(face_rot.as_rotvec())))
face_x_vec = face_rot.apply(np.array((1,0,0)))
face_z_vec = face_rot.apply(np.array((0,0,1)))
face_mesh.translate(face_pos)
plt.add_mesh(face_mesh)
plt.add_axis_vec(face_pos, x_vec=face_x_vec, z_vec=face_z_vec)
plt.show()


## # face
#face_y_vec = (0,1,0)
## ax.arrow3D((0,0,0), unit_vec(face_y_vec)*2, color=cmap[0])
#face_y_vec = Rotation.from_euler('ZYX',(0, 45, -30),degrees=True).apply(face_y_vec)
#ax.arrow3D((0,0,0), unit_vec(face_y_vec)*2, color=cmap[1])
##
#face_nose_vec = (0,0,1)
## ax.arrow3D((0,0,0), unit_vec(face_nose_vec)*2, color=cmap[2])
#face_nose_vec = Rotation.from_euler('ZYX',(0, 45, -30),degrees=True).apply(face_nose_vec)
#ax.arrow3D((0,0,0), unit_vec(face_nose_vec)*2, color=cmap[3])
#ax.origin3D((0,0,0), length=3, z_vec=face_nose_vec, x_vec=np.cross(face_y_vec, face_nose_vec))
## ax.circle3D(face_pos, 1, face_nose_vec, color='gray', alpha=0.2)

# # camera
# camera_pos = (5,5,5)
# camera_z_vec = (1, 0, 0)
# camera_z_vec /= np.linalg.norm(camera_z_vec)
# ax.origin3D(camera_pos, camera_z_vec)

# # vec = np.array([1,0,0])
# # ax.arrow3D((0,0,0), vec)
# # euler = np.array([ 0, 45, 0])
# # rot = Rotation.from_euler('ZYX', euler, degrees=True)
# # ax.arrow3D((0,0,0), rot.apply(vec))


# mesh = mesh_scale(mesh_location_zero(mesh.Mesh.from_file('humanheadBlender_reduce.stl')),10,10,10)
# mesh = mesh_scale(mesh_location_zero(mesh.Mesh.from_file('male_head_reduce.stl')),0.05,0.05,0.05)
# mesh.rotate([0.0, 0.0, 1.0], math.radians(-90))
# mesh.rotate([1.0, 0.0, 0.0], math.radians(90))
# ax.add_collection3d(art3d.Poly3DCollection(mesh.vectors))

