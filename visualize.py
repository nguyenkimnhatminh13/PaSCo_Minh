import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import numpy as np


# Path to the .pkl file
file_path = "/home/anda/minh/PaSCo/gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti/instance_labels_v2/00/000000_1_1.pkl"

# Read the .pkl file
with open(file_path, "rb") as file:
    data = pickle.load(file)

# Print the content of the file
# print(data)

for key, value in data.items():
    print(key)
    print(value.shape)
    # Create a figure and 3D axes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Define a threshold to visualize the 3D array
    threshold = 0.5

    # Create a boolean array where the value is above the threshold
    voxels = value > threshold

    # Create a color array based on the data values
    colors_array = np.empty(voxels.shape, dtype=object)
    colors_array[voxels] = plt.cm.viridis(data[voxels])

    # Plot the 3D array
    ax.voxels(voxels, facecolors=colors_array, edgecolor="k")

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # # Display the plot
    # plt.show()

    output_path = f"./{key}.jpg"  # Change this to your desired output path
    plt.savefig(output_path)
    # print(value)
    print()
