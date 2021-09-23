# %matplotlib notebook
# from IPython.core.debugger import Pdb; Pdb().set_trace()
#
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import art3d, Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
from scipy.spatial.transform import Rotation
from stl import mesh
import math

def unit_vec(vec):
    vec = np.array(vec)
    return vec / np.linalg.norm(vec)

def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta

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

def pathpatch_2d_to_3d(pathpatch, z = 0, normal = (0,0,1)):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.
    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()
    path = trans.transform_path(path) #Apply the transform
    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color    
    verts = path.vertices #Get the vertices in 2D
    rot = Rotation.from_normal((0,0,1), normal)
    pathpatch._segment3d = np.array([rot.apply((x, y, 0)) + (0, 0, z) for x, y in verts])

class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)
    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

def _arrow3D(ax, pos, normal, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''
    arrow = Arrow3D(pos[0], pos[1], pos[2], normal[0], normal[1], normal[2], mutation_scale=20, arrowstyle="-|>", *args, **kwargs)
    ax.add_artist(arrow)
setattr(Axes3D, 'arrow3D', _arrow3D)

def _origin3D(ax, pos=(0,0,0), length=1, x_vec=(1,0,0), z_vec=(0,0,1)):
    x_axis = (length,0,0)
    y_axis = (0,length,0)
    z_axis = (0,0,length)
    rot_z = Rotation.from_normal(z_axis, z_vec)
    if (np.all(np.cross(x_axis, x_vec) == 0)):
        rot_x = Rotation.from_quat((0,0,0,1))
    else:
        rot_x = Rotation.from_normal(rot_z.apply(x_axis), x_vec)
    rot = rot_x * rot_z
    ax.arrow3D(pos, rot.apply(x_axis), color='r')
    ax.arrow3D(pos, rot.apply(y_axis), color='g')
    ax.arrow3D(pos, rot.apply(z_axis), color='b')
setattr(Axes3D, 'origin3D', _origin3D)

def _circle3D(ax, pos, radius, normal, *args, **kwargs):
    p = Circle((0,0),radius, *args, **kwargs)
    ax.add_patch(p)
    pathpatch_2d_to_3d(p, z=0, normal=normal)
    pathpatch_translate(p, pos)
setattr(Axes3D, 'circle3D', _circle3D)

##
cmap = plt.cm.Set1.colors
fig = plt.figure()
# proj_type = ['ortho', 'persp']
ax = fig.add_subplot(projection='3d', proj_type='ortho', xlabel='x', ylabel='y', zlabel='z')
ax.set_xlim(0,3)
ax.set_ylim(0,3)
ax.set_zlim(0,3)

# global
# ax.origin3D(pos=(0,0,0), z_vec=Rotation.from_euler('ZYX',(0, 45, 45),degrees=True).apply((0,0,1)))
ax.origin3D(length=1)
# ax.origin3D(length=2, z_vec=(1,1,1))
# ax.origin3D(length=3, z_vec=(1,1,1), x_vec=np.cross((1,1,1), (1,1,0)))
# ax.arrow3D((0,0,0), (1,-1,1))
# ax.origin3D(length=3, z_vec=(1,1,1), x_vec=(1,-1,1))

# # face
face_y_vec = (0,1,0)
# ax.arrow3D((0,0,0), unit_vec(face_y_vec)*2, color=cmap[0])
face_y_vec = Rotation.from_euler('ZYX',(0, 45, -30),degrees=True).apply(face_y_vec)
ax.arrow3D((0,0,0), unit_vec(face_y_vec)*2, color=cmap[1])
#
face_nose_vec = (0,0,1)
# ax.arrow3D((0,0,0), unit_vec(face_nose_vec)*2, color=cmap[2])
face_nose_vec = Rotation.from_euler('ZYX',(0, 45, -30),degrees=True).apply(face_nose_vec)
ax.arrow3D((0,0,0), unit_vec(face_nose_vec)*2, color=cmap[3])
ax.origin3D((0,0,0), length=3, z_vec=face_nose_vec, x_vec=np.cross(face_y_vec, face_nose_vec))
# ax.circle3D(face_pos, 1, face_nose_vec, color='gray', alpha=0.2)

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

def mesh_update(my_mesh):
    my_mesh.update_areas()
    my_mesh.update_max()
    my_mesh.update_min()
    my_mesh.update_units()
    return my_mesh

def mesh_location_zero(my_mesh):
    midPosRel = (my_mesh.max_ - my_mesh.min_)/2
    my_mesh.x = my_mesh.x - (midPosRel[0] + my_mesh.min_[0])
    my_mesh.y = my_mesh.y - (midPosRel[1] + my_mesh.min_[1])
    my_mesh.z = my_mesh.z - (midPosRel[2] + my_mesh.min_[2])
    mesh_update(my_mesh)
    return my_mesh

def mesh_scale(my_mesh, scale_x, scale_y, scale_z):
    my_mesh.x = my_mesh.x * scale_x
    my_mesh.y = my_mesh.y * scale_y
    my_mesh.z = my_mesh.z * scale_z 
    mesh_update(my_mesh)
    return my_mesh

# mesh = mesh_scale(mesh_location_zero(mesh.Mesh.from_file('humanheadBlender_reduce.stl')),10,10,10)
mesh = mesh_scale(mesh_location_zero(mesh.Mesh.from_file('male_head_reduce.stl')),0.05,0.05,0.05)
# mesh.rotate([0.0, 0.0, 1.0], math.radians(-90))
# mesh.rotate([1.0, 0.0, 0.0], math.radians(90))
ax.add_collection3d(art3d.Poly3DCollection(mesh.vectors))

plt.draw()
plt.show()

