import math
import numpy as np

def BodyToWindAxisConversion(alpha, beta, phi, cx, cy, cz):
    #this method converts the body axis forces into the wind axis forces
    #cx, cy, cz to CL CD and Cy'
    #Inputs:
    # alpha - angle of attack (radians)
    # beta  - sideslip angle (radians)
    # phi   - roll angle (radians)
    # cx    - force in body x axis
    # cy    - force in body y axis
    # cz    - force in body z axis
    #Output:
    # CL - Coefficient of Lift
    # CD - Coefficient of Drag
    # CyPrime - Side force in wind axis

    ca = math.cos(alpha)
    sa = math.sin(alpha)
    cb = math.cos(beta)
    sb = math.sin(beta)
    cr = math.cos(phi)
    sr = math.sin(phi)

    CD = cx * ca * cb - cy * sr * sa * cb - cy * cr * sb + cz * cr * sa * cb - cz * sr * sb
    CL = -cx * sa - cy * sr * ca + cz * cr * ca
    cyPrime = cx * ca * sb - cy * sr * sa * sb + cy * cr * cb + cz * cr * sa * sb + cz * sr * cb

    return [CL, CD, cyPrime]


def GetDelta(normal, uinf):
    #This method gets the angle between the normal and the wind vector
    #normal is the normal unit vector [i, j, k] for each panel
    #uinf is the airflow unit vector [i, j, k]

    cosdel = - np.dot(normal, uinf)
    return np.pi / 2 - np.arccos(cosdel)


def GetAirflowVector(alpha, beta):
    #returns the normal vector of the airflow direction
    #alpha is the angle of attack (rad)
    #beta is the sideslip angle (rad)
    return np.array([math.cos(alpha)*math.cos(beta), -math.sin(beta), math.sin(alpha)*math.cos(beta)])


def ComputeNetworkForcesAndMoments(normal, centroid, area, cp, xref, yref, zref):
    #compute the force magnitude
    #Inputs:
    # normal is the normal vector for the panel array [i, j, k]
    # centroid is the coordinate of the panel centroid [x, y, z] (m)
    # area is the area of the panel (m^2)
    # cp is the coefficient of pressure for the panel
    # xref is the center of gravity of the vehicle in the body z axis (m)
    # yref is the center of gravity of the vehicle in the body z axis (m)
    # zref is the center of gravity of the vehicle in the body z axis (m)
    fmag = -1 * np.multiply(area, cp.reshape(-1, 1))

    #compute the forces on all the individuals elements
    dFx = np.multiply(fmag, normal[:, 0].reshape(-1, 1))
    dFy = np.multiply(fmag, normal[:, 1].reshape(-1, 1))
    dFz = np.multiply(fmag, normal[:, 2].reshape(-1, 1))

    #compute all the moment contributions in the X directions
    dMx = -1 * dFy * (centroid[:, 2].reshape(-1, 1) - zref)
    dMx = dMx + dFz*(centroid[:, 1].reshape(-1, 1) - yref)

    #compute all the moment contributions in the y directions
    dMy = dFx * (centroid[:, 2].reshape(-1, 1) - zref)
    dMy = dMy - dFz*(centroid[:, 0].reshape(-1, 1) - xref)
    
    #compute all the moment contributions in the z directions
    dMz = -1 * dFx * (centroid[:, 1].reshape(-1, 1) - yref)
    dMz = dMz + dFy * (centroid[:, 0].reshape(-1, 1) - xref)

    #sum up the network forces and moments
    Fx = np.sum(dFx)
    Fy = np.sum(dFy)
    Fz = np.sum(dFz)
    Mx = np.sum(dMx)
    My = np.sum(dMy)
    Mz = np.sum(dMz)

    return [Fx, Fy, Fz, Mx, My, Mz]


def RunSHABPy(alpha, beta, vehicle):
    #Returns the forces and moments acting on a mesh. based on panel methods
    # Inputs:
    # alpha is the angle of attack (rad)
    # beta is the sideslip angle (rad)
    # vehicle is a vehicle object defined in Vehicle.py
    # Outputs:
    # [cx, cy, cz] in the body axis, these are the force coefficients in the x y and z axes
    # [cmx, cmy, cmz] in the body axis, these are the moment coefficients in the x, y and z axes
    # [cl, cd, cyPrime] in the wind axis, these are the force coefficients for lift, drag and side force respectively.

    #get the unit normals, centroids and areas of each panel in the mesh.
    normals     = vehicle.mesh.get_unit_normals()
    centroids   = vehicle.mesh.centroids
    areas       = vehicle.mesh.areas
    cp          = np.zeros(len(areas))

    #get airflow angle
    uinf = GetAirflowVector(alpha, beta)

    # compute the vector of inclination angles
    deltaVec = GetDelta(normals, uinf)

    #Compression (where the angle between the flow vector and the panel is > 0)
    cp[(deltaVec > 0)] = vehicle.compression.calculatecompression(deltaVec[(deltaVec > 0)])

    # expansion (where the angle between the flow vector and the panel is <= 0)
    cp[deltaVec <= 0] = vehicle.expansion.calculateexpansion(deltaVec[deltaVec <= 0])

    #get the total force and moments acting on the body
    [Fx, Fy, Fz, Mx, My, Mz]  = ComputeNetworkForcesAndMoments(normals, centroids, areas, cp, vehicle.xref, vehicle.yref, vehicle.zref)

    #Find the wind axis values
    [cl, cd, cyPrime]         = BodyToWindAxisConversion(alpha, beta, 0, Fx, Fy, Fz)

    #scale these values to coefficients to be used with the current atmospheric conditions to find the total force and moments acting on the body
    cx       = Fx/vehicle.sref
    cy       = Fy/vehicle.sref
    cz       = Fz/vehicle.sref
    cmx      = Mx/vehicle.span
    cmy      = My/vehicle.cbar
    cmz      = Mz/vehicle.span
    cl       = cl/vehicle.sref
    cd       = cd/vehicle.sref
    cyPrime  = cyPrime/vehicle.sref

    return [cp, cx, cy, cz, cmx, cmy, cmz, cl, cd, cyPrime]
