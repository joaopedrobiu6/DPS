import numpy as np
import matplotlib.pyplot as plt
import desc.io
from desc.grid import Grid

def data_input_from_file(file_name):
    # Initialize empty lists to store the separated columns
    psi = []
    theta = []
    zeta = []
    vpar = []

    # Open the file for reading
    with open(file_name, 'r') as file:
        # Read the first line (column headers) and discard it
        file.readline()

        # Read the data row by row
        for line in file:
            # Split the line into columns using tab as the delimiter
            columns = line.strip().split('\t')

            # Convert each column to the appropriate data type and append to the respective lists
            psi.append(float(columns[0]))
            theta.append(float(columns[1]))
            zeta.append(float(columns[2]))
            vpar.append(float(columns[3]))

    psi = np.array(psi)
    theta = np.array(theta)
    zeta = np.array(zeta)
    vpar = np.array(vpar)
    
    return np.array([psi, theta, zeta, vpar]).T

solution = data_input_from_file("solutioninput.LandremanPaul2021_QA_scaled_output.h5.txt")
# print(solution[:, 0])

#Assumindo que isto está bem feito (não sei se o Psi está certo, dá valores muito mais altos do que eu esperava)
def boozer_to_cylindrical(eq_filename, solution=None):
    eq = desc.io.load(eq_filename)[-1]
    eq._iota = eq.get_profile("iota").to_powerseries(order=eq.L, sym=True)
    eq._current = None

    grid = Grid([1, 0, 0], sort=False)
    Psi = eq.compute("Psi", grid = grid)
    Psi = Psi["Psi"][0]
    
    psi = solution[:, 0]*(Psi/2*np.pi)
    theta = solution[:, 1]
    zeta = solution[:, 2]

    return [psi, theta, zeta]




