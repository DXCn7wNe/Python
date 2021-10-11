# from IPython.core.debugger import Pdb; Pdb().set_trace()
#
import numpy as np
from scipy.spatial.transform import Rotation
import pyvista as pv
import head_pose_wi_mediapipe as hp

model_path = './models/Pasha_guard_head/Pasha_guard_head.obj'
texture_path = './models/Pasha_guard_head/Pasha_guard_head_0.png'

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

## canvas
# plt = pv.Plotter(off_screen=True)
plt = pv.Plotter(off_screen=False)
plt.add_axes_at_origin()
# axes = pv.Axes(show_actor=True, actor_scale=2.0, line_width=5)
# axes.origin = (3.0, 3.0, 3.0)

## face
face_pos = np.array((0,0,0))
# face_rot = Rotation.from_euler('ZYX',(0, 45, -30),degrees=True)
face_rot = Rotation.from_euler('ZYX',(0, 0, -90),degrees=True)
face_x_vec = face_rot.apply(np.array((1,0,0)))
face_z_vec = face_rot.apply(np.array((0,0,1)))
#
face_mesh = pv.read(model_path)
face_tex = pv.read_texture(texture_path)
face_mesh.scale(0.004)
face_mesh.rotate_vector(face_rot.as_rotvec(), np.rad2deg(np.linalg.norm(face_rot.as_rotvec())))
face_mesh.translate(face_pos)
plt.add_mesh(face_mesh, texture=face_tex)
# plt.add_axis_vec(face_pos, x_vec=face_x_vec, z_vec=face_z_vec)

## camera
plt.camera.position = (3,0,3)
plt.camera.focal_point = (0,0,0)
# plt.camera.roll = 0
plt.camera.up=(0,1,0)
# plt.camera.view_angle=60

# ## plot
# plt.screenshot('a.jpg')
plt.show()

# ## meidapipe
# result = hp.proc_image('a.jpg')
