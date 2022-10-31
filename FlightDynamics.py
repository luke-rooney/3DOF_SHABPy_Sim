import numpy as np
import SHABPy
import matmos
import math


def step3DOF(X, C, vehicle, flaperon_l, flaperon_r, dt, g, tgt):
    #X is the state of the vehicle [u, w, q, e0, e1, e2, e3, x, z] where:
    #   u - is speed in body x axis (m/s)
    #   w - is speed in body z axis (m/s)
    #   q - is rotational rate of body (rad/s)
    #   e0 - is first quaternion value
    #   e1 - is second quaternion value
    #   e2 - is third quaternion value
    #   e3 - is fourth quaternion value
    #   x - x coodinate value (m)
    #   z - z coordinate value (m)
    #C - c is the control flap deflection in radians
    #vehicle - vehicle object from LoadVehicle() - specifically unsw
    #flaperon_l - vehicle object from LoadVehicle() - specifically flap_l
    #flaperon_r - vehicle object from LoadVehicle() - specifically flap_r
    #dt  - is our timestep value (s) expected to be around 0.001 or 0.01
    #g   - gravity value (m/s) have been using 9.81m/s
    #tgt - coordinate values (m) - expected array: [x, z]
    #
    # Note State is [u, w, q, e0, e1, e2, e3, x, z]
    # indexes       [0, 1, 2, 3,   4,  5,  6, 7, 8]
    # C is the control position (in 3DOF its just dr)

    V       = np.sqrt(X[0]**2 + X[1]**2)
    eul     = q2e(X[3:7])
    alpha   = np.arctan2(-X[1], X[0])
    gamma   = eul[1] - alpha
    atmos   = matmos.ISA(X[8]/1000)
    T = atmos.t
    rho = atmos.d

    R = 286
    a = np.sqrt(R * T * vehicle.gamma)

    # update vehicle mach number
    vehicle.M = V / a
    flaperon_l.M = V / a
    flaperon_r.M = V / a

    #Get Desired Flaperon Position:
    [dr, dl] = ControlFlaperon3DOF(tgt, X)
    dC       = np.sign(dr) * vehicle.actuation_rate

    [cp, cx, cy, cz, cmx, cmy, cmz, cl, cd, cyPrime] = SHABPy.RunSHABPy(alpha, 0, vehicle)
    [cp_l, cx_l, cy_l, cz_l, cmx_l, cmy_l, cmz_l, cl_l, cd_l, cyPrime_l] = SHABPy.RunSHABPy(alpha + C, 0, flaperon_l)
    [cp_r, cx_r, cy_r, cz_r, cmx_r, cmy_r, cmz_r, cl_r, cd_r, cyPrime_r] = SHABPy.RunSHABPy(alpha + C, 0, flaperon_r)

    q  = 0.5 * rho * V ** 2
    Fx = q * vehicle.sref * (cx + cx_l + cx_r)
    Fz = q * vehicle.sref * (cz + cz_l + cz_r)

    My = q * vehicle.cbar * (cmy + cmy_l + cmy_r)

    correction = 1 - (X[3] ** 2 + X[4] ** 2 + X[5] ** 2 + X[6] ** 2)

    dX = np.array([- X[2] * X[1] - g * np.sin(eul[1]) - Fx / vehicle.m,
                   X[2] * X[0]  - g * np.cos(eul[1]) + Fz / vehicle.m,
                   vehicle.C[5] * My,
                   -0.5 * (X[5] * X[2]) + 0.5 * correction * X[3],
                   -0.5 * (X[6] * X[2]) + 0.5 * correction * X[4],
                    0.5 * (X[3] * X[2]) + 0.5 * correction * X[5],
                    0.5 * (X[4] * X[2]) + 0.5 * correction * X[6],
                   V * np.cos(gamma),
                   V * np.sin(gamma)])

    C = C + dC * dt
    C = min([max([vehicle.mindef, C]), vehicle.maxdef])

    return [X + dX * dt, C]

#Inertia Coefficients to reduce complexity of the state change
def getInertiaCoeffs(Ixx, Iyy, Izz, Ixz):
    #Ixx mass-moment of inertia Ixx - (m^4)
    #Iyy mass-moment of inertia Iyy - (m^4)
    #Izz mass-moment of inertia Izz - (m^4)
    #Ixz mass-moment of inertia Ixz - (m^4)

    C       = np.zeros(10)
    C[0]    = Ixx*Izz - Ixz**2
    C[1]    = Izz/C[0]
    C[2]    = Ixz/C[0]
    C[3]    = C[2]*(Ixx - Iyy + Izz)
    C[4]    = C[1]*(Iyy - Izz) - C[2]*Ixz
    C[5]    = 1/Iyy
    C[6]    = C[5]*Ixz
    C[7]    = C[5]*(Izz - Ixx)
    C[8]    = Ixx/C[0]
    C[9]    = C[8]*(Ixx - Iyy) + C[2]*Ixz
    return C

#Quaternion angles to Euler Angles
def q2e(quat):
    #quat is a quaternion angle array - [e0, e1, e2, e3]
    #returns eul which is euler angles - [roll, pitch, yaw]
    eul     = np.zeros(3)
    eul[0]  = np.arctan2(2*(quat[0]*quat[1] + quat[2]*quat[3]), 1 - 2*(quat[1]**2 + quat[2]**2))
    eul[1]  = np.arcsin(2*(quat[0]*quat[2]-quat[3]*quat[1]))
    eul[2]  = np.arctan2(2*(quat[1]*quat[2] + quat[0]*quat[3]), 1 - 2*(quat[2]**2 + quat[3]**2))
    return eul

def e2q(eul):
    # eul which is euler angles - [roll, pitch, yaw]
    # returns quat is a quaternion angle array - [e0, e1, e2, e3]

    quat    = np.zeros(4)

    cph     = np.cos(eul[0]/2)
    sph     = np.sin(eul[0]/2)
    cth     = np.cos(eul[1]/2)
    sth     = np.sin(eul[1]/2)
    cps     = np.cos(eul[2]/2)
    sps     = np.sin(eul[2]/2)

    quat[0] = cph*cth*cps + sph*sth*sps
    quat[1] = sph*cth*cps - cph*sth*sps
    quat[2] = cph*sth*cps + sph*cth*sps
    quat[3] = cph*cth*sps - sph*sth*cps

    return quat

def ControlFlaperon3DOF(tgt, X):
    # X is the state of the vehicle [u, w, q, e0, e1, e2, e3, x, z] where:
    #   u - is speed in body x axis (m/s)
    #   w - is speed in body z axis (m/s)
    #   q - is rotational rate of body (rad/s)
    #   e0 - is first quaternion value
    #   e1 - is second quaternion value
    #   e2 - is third quaternion value
    #   e3 - is fourth quaternion value
    #   x - x coodinate value (m)
    #   z - z coordinate value (m)
    # Note State is [u, w, q, e0, e1, e2, e3, x, z]
    #               [0, 1, 2, 3,   4,  5,  6, 7, 8]
    # tgt is the coordinate value of the target = [x, z]

    # returns [dr, dl] which are the left and right flap deflections in radians
    [roll, pitch, yaw] = q2e(X[3:7])

    dx = tgt[0] - X[7]
    dz = tgt[1] - X[8]

    tgt_pitch = math.atan2(dz, dx)
    dpitch = tgt_pitch - pitch

    k_pitch = 100
    k_pitchrate = 100

    pitch_dr = -k_pitch * dpitch + k_pitchrate * X[2]

    dr = pitch_dr

    maxdef = 12 * np.pi / 180
    dr = np.min([np.max([dr, -maxdef]), maxdef])

    return [dr, dr]
