import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize, ListedColormap, LinearSegmentedColormap

jet_512 = cm.get_cmap('jet', 512)
jet_half = ListedColormap(jet_512(np.linspace(0.4, 1.0, 256)))

zrange = (0.0, 1.0)
# zrange = (-1, 1)

# fig, ax = plt.subplots(figsize=(7,10))
fig = plt.figure(figsize=(7,10))
ax = fig.add_subplot(111)
data = np.random.normal(0, 0.2, size=(100,100))
data = np.where(data < zrange[0], zrange[0]-1, data)
data = np.where(data > zrange[1], zrange[1]+1, data)
print(f"{data.shape = }")
cax = ax.imshow(data, interpolation='nearest', cmap=jet_half)
divider = make_axes_locatable(plt.gca())

# colorbar
ax_cb = divider.append_axes("right", '5%', pad='15%')
cb = fig.colorbar(cax, cax=ax_cb, orientation='vertical')
ax_cb.yaxis.set_ticks_position('left')

# hist
ax_hist = divider.append_axes("right", '15%', pad='2%')
# dbin = (zrange[1]-zrange[0])/50
# bins = np.arange(zrange[0], zrange[1]+dbin, dbin)
# print(f"{bins = }")
N, bins, patches = ax_hist.hist(np.ndarray.flatten(data), range=zrange, bins=50, orientation='horizontal')
print(f"{np.sum(N) = }")
norm = Normalize(bins.min(), bins.max())
for bin, patch in zip(bins, patches):
    color = jet_half(norm(bin))
    patch.set_facecolor(color)
ax_hist.set_axis_off()
# ax_hist.axis('off')
# ax_hist.get_yaxis().set_visible(False)

plt.show()

