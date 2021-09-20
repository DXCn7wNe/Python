#! /usr/bin/python3

import sys
import re
import os.path
from pprint import pprint
from itertools import takewhile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits import mplot3d
from pyevtk.hl import pointsToVTK
import joblib

## === Functions
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

## === Main
if len(sys.argv) < 2:
    print("Usage: " + sys.argv[0] + " file")
    sys.exit(0)
spm_file_name = sys.argv[1]
# xyz_file_name = os.path.splitext(spm_file_name)[0] + ".csv"
base_file_name = os.path.splitext(spm_file_name)[0]

with open(spm_file_name, newline='', errors='ignore') as fp:
    lines = fp.readlines()
    
    header = {}
    data_names = []
    # header_idx = [(idx, m.group(0)) for (idx, line) in enumerate(lines) if (m:=re.match(r'^\"(\?\*File list)|(^\"\\\*File list end)\"', line))]
    header_idxs = [(idx, m.group(0).strip('\"')) for (idx, line) in enumerate(lines) if (m:=re.match(r'^\"([\?\\]\*.* list(?: end)?)\"', line))]
    # pprint(header_idxs)
    for idx in range(len(header_idxs)-1):
        if re.match(r'\\\*Ciao image list', header_idxs[idx][1]):
            data_name = [m.group(1) for line in lines[header_idxs[idx][0]+1:header_idxs[idx+1][0]-1] if (m:=re.match(r'\"\\@\d+:Image Data: S \[.*\] \"(.*)\"\"', line))][0]
            header_idxs[idx] = (header_idxs[idx][0], f"{header_idxs[idx][1]} <{data_name}>")
            data_names.append(data_name)
    # pprint(header_idxs)
    # pprint(data_names)
    for idx in range(len(header_idxs) -1):
        name = header_idxs[idx][1].strip('\*?')
        header[name] = [line.strip('\"\\\r\n').split(': ',1) for line in lines[header_idxs[idx][0]+1:header_idxs[idx+1][0]-1]]
        header[name] = {key.strip(): value.strip() for (key,value) in header[name]}
    # pprint(header)
    # pprint(header.keys())
    
    data={}
    markers = [(idx, m.groups()[0].strip()) for (idx, line) in enumerate(lines) if (m:=re.match(r'^\\Exported image units: (.*)', line))]
    markers.append((len(lines), ""))
    # print(markers)
    for idx in range(len(markers)-1):
        skiprows = markers[idx][0] + 1
        max_rows = markers[idx+1][0] - skiprows
        name = data_names[idx]
        header[f"Ciao image list <{name}>"]['Unit'] = markers[idx][1]
        # print(name, skiprows, max_rows)
        fp.seek(0)
        data[name] = np.round(np.loadtxt(fp, delimiter=',', skiprows=skiprows, max_rows=max_rows),3)
    # pprint([(key, value.shape, value) for (key, value) in data.items()])

ret = {}
ret['header'] = header
ret['data'] = data
# pprint(ret)

(lx, ly) = [float(x) for x in ret['header']['Ciao image list <Height>']['Scan size'].split()[0:2]]
(ny, nx) = ret["data"]["Height"].shape
# print(width, height)

## === left(x)-top(y) basis
y, x = np.mgrid[0:ly:1j*ny, 0:lx:1j*nx]
(ret['data']['x'], ret['data']['y']) = (x, y)
# pprint(ret["data"])

z = ret['data']['Height'] / 1000
c = ret['data']['Potential']

if '--no-plot' not in sys.argv:
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    m = plt.cm.ScalarMappable(cmap='jet')
    facecolors = m.to_rgba(c)
    surf = ax.plot_surface(x, y, z, facecolors=facecolors)
    set_axes_equal(ax)
    plt.show()

## === Export xyz file
# np.savetxt(xyz_file_name, np.vstack([x.ravel(), y.ravel(), z.ravel(), c.ravel()]).T, delimiter=',', header=r'"x","y","z","c"') 
## === Export vtk file
pointsToVTK(base_file_name, x.ravel(), y.ravel(), z.ravel(), data={"Potential": c.ravel()})
## === Export with joblib
joblib.dump(ret, base_file_name + ".jb")
