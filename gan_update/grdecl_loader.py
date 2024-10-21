import re
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from mpl_toolkits.mplot3d import Axes3D


def read_grdecl(filename):
    # Dictionary to store the data for PORO and FACIES
    data_dict = {}

    # Regular expressions to match the start of PORO and FACIES blocks
    poro_pattern = re.compile(r'\bPORO\b', re.IGNORECASE)
    facies_pattern = re.compile(r'\bFACIES\b', re.IGNORECASE)

    # Regular expression to match the end of a block ('/' symbol)
    end_pattern = re.compile(r'\/')

    current_key = None
    data_array = []

    # Open the file and process each line
    with open(filename, 'r') as file:
        for line in file:
            # Check if the line starts with PORO or FACIES
            if poro_pattern.search(line):
                current_key = 'PORO'
                data_array = []
                continue
            elif facies_pattern.search(line):
                current_key = 'FACIES'
                data_array = []
                continue

            # Check for the end of the block
            if current_key and end_pattern.search(line):
                # Store the data array for the current key and reset
                data_dict[current_key] = data_array
                current_key = None
                continue

            # If we are inside a PORO or FACIES block, collect the data
            if current_key:
                # Split the line into numbers and add them to the array
                current_data = [float(x) for x in line.split()]
                data_array.extend(current_data)

    porosity_data = np.reshape(data_dict['PORO'], (400, 100, 400), order='F')
    facies_data = np.reshape(data_dict['FACIES'], (400, 100, 400), order='F')
    # vis_3d_data(porosity_data)
    return porosity_data, facies_data

    # return data_dict


def vis_3d_data(data):
    x, y, z = np.meshgrid(np.arange(data.shape[0]),
                          np.arange(data.shape[1]),
                          np.arange(data.shape[2]))

    # Flatten the arrays for plotting
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    values = data.flatten()

    # Create 3D scatter plot to visualize the 3D array values
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize the color map according to values
    norm = plt.Normalize(vmin=0.018, vmax=0.4)
    colors = plt.cm.viridis(norm(values))

    # Scatter plot with color map representing the values
    sc = ax.scatter(x, y, z, c=values, cmap='viridis', marker='o')

    # Adding color bar to show the scale
    cb = fig.colorbar(sc, ax=ax, shrink=0.6)
    cb.set_label('Value')

    # Set labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Array Visualization')

    plt.savefig('poro_new1.png')


def view_porosity(data):
    # Create a structured grid from the 3D array
    # Define the dimensions of the structured grid
    dims = np.array(data.shape) + 1  # The grid dimensions should be one more than the array shape

    # Create the coordinates for the structured grid
    x = np.arange(0, dims[0], 1)
    y = np.arange(0, dims[1], 1)
    z = np.arange(0, dims[2], 1)

    # Create a meshgrid of coordinates for the structured grid
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    # Create the structured grid
    grid = pv.StructuredGrid(x, y, z)

    # Add the 3D array as point data to the grid
    grid["values"] = data.flatten(order="F")

    # Create a PyVista plotter object
    plotter = pv.Plotter()

    # Add the structured grid to the plotter
    plotter.add_mesh(grid, scalars="values", cmap="viridis", show_edges=True)

    # Add labels, legend, and other visualization settings
    plotter.add_scalar_bar(title="Values", vertical=True)
    plotter.show_axes()
    plotter.set_background("white")

    # Display the visualization
    plotter.show()


if __name__ == '__main__':
    filename = r'T:\600\60010\FakeImageDataset\Distinguish\NOFAULT_MODEL_R1.grdecl'  # Replace with your GRDECL file path
    porosity_data, facies_data = read_grdecl(filename)
    # vis_3d_data(porosity_data)
    view_porosity(porosity_data)
