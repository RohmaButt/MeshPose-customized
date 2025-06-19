import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for script use

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

from meshpose.utils.meshpose_inference import MeshPoseInference
from meshpose.utils import imread

CMAP = mcolors.LinearSegmentedColormap.from_list('RedGreen', ['red', 'green'])

def showverts(image, verts, size, color, save_path):
    plt.figure()
    plt.imshow(image)
    for axis in [0, 2]:
        plt.scatter(verts[:, axis], verts[:, 1], s=size, color=color)
    plt.axis('off')  # Optional: remove axes for cleaner output
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the plot to free memory

if __name__ == '__main__':

    image_path = 'assets/pexels-olly-3799115.jpg'
    bbox = [584, 32, 556, 704]  # COCO bbox definition (x,y,w,h)

    image = imread(image_path)

    meshpose = MeshPoseInference()
    outputs = meshpose(image, bbox)

    # Ensure assets/ directory exists
    os.makedirs('assets', exist_ok=True)

    # Save both visualization plots
    showverts(image, outputs['xyz_hp'], size=1, color=None, save_path='assets/output_hp.png')
    showverts(image, outputs['xyz_lp'], size=5, color=CMAP(outputs['vertex_vis_lp']), save_path='assets/output_lp.png')
