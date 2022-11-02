
import FlightDynamics
import numpy as np
import matmos
from matplotlib import pyplot
import LoadVehicle

def LonSim(tgt, start, eul, dt, t):
    #tgt   - Target locations (m) x,z coordinates - Expected array: [x, z]
    #start - Vehicle start location (m) x,z coordinates - Expected array [x, z]
    #eul   - euler orientation angles roll pitch yaw (rad) - Expected array [roll, pitch, yaw]
    #dt    - timestep value (s) expected to be small float value eg. 0.001
    #t     - total run time of the simulation (s) expected to be a single value in seconds eg. 50.0

    #Load the vehicle 3D Mesh (unsw), the left flap (flap_l) and the right flap (flap_r)
    [unsw, flap_l, flap_r] = LoadVehicle.LoadUNSW5_Control()

    #quaternions from euler values given in eul ([roll, pitch, yaw])
    quat = FlightDynamics.e2q(eul)

    #atmosphere values calculated from matmos package.
    #speed of sound calculated using atmosphere values
    atmos = matmos.ISA(start[1]/1000)
    R = 286
    a = np.sqrt(R * atmos.t * unsw.gamma)

    # STATE X = [u, w, q, e0, e1, e2, e3, x, z]
    # State X = [speed body x axis, speed body z axis, pitch rate, quaternion 0, quaternion 1, quaternion 2, quaternion 3, x position (global frame), z position (global frame)
    # X0 gives the initial state of ou vehicle.
    X0    = [unsw.M * a, 0, 0, quat[0], quat[1], quat[2], quat[3], start[0], start[1]]

    #Calculate number of steps in our integration based on the amount of time and the time steps
    #Create empty array for storing states
    #Set initial state X0 to the first value of empty array.
    steps = int(t/dt)
    X     = np.zeros((steps, 9))
    C     = np.zeros(steps)
    X[0]  = X0

    #for each timestep, integrate to find the state at the next timestep.
    for i in range(1, steps):
        [X[i], C[i]] = FlightDynamics.step3DOF(X[i - 1], C[i-1], unsw, flap_l, flap_r, dt, 9.81, tgt)

    #Split up the final filled array into the components for plotting graphs for visual assessment
    time_arr = np.arange(0, steps) * dt
    u = X[:, 0]
    w = X[:, 1]
    q = X[:, 2]
    e0 = X[:, 3]
    e1 = X[:, 4]
    e2 = X[:, 5]
    e3 = X[:, 6]
    x = X[:, 7]
    z = X[:, 8]
    V = np.sqrt(np.power(u, 2) + np.power(w, 2))

    phi = np.zeros(len(time_arr))
    theta = np.zeros(len(time_arr))
    psi = np.zeros(len(time_arr))

    for i in range(len(time_arr)):
        [phi[i], theta[i], psi[i]] = FlightDynamics.q2e([e0[i], e1[i], e2[i], e3[i]])

    #plotting various
    # pyplot.figure(1)
    # pyplot.plot(time_arr, V)
    # pyplot.plot(time_arr, u)
    # pyplot.plot(time_arr, w)
    # pyplot.legend(['V', 'u', 'w'])
    # pyplot.xlabel('Time (s)')
    # pyplot.ylabel('Speed (m/s)')
    #
    # pyplot.figure(2)
    # pyplot.plot(time_arr, theta)
    # pyplot.legend(['theta'])
    # pyplot.xlabel('Time (s)')
    # pyplot.ylabel('Orientation Angle (rad)')
    #
    # pyplot.figure(3)
    # pyplot.plot(x, z)
    # pyplot.plot(tgt[0], tgt[1], 'x')
    # pyplot.xlabel('X Position (m)')
    # pyplot.ylabel('Z Position (m)')
    # pyplot.axis('equal')
    # pyplot.grid('on')
    #
    # pyplot.figure(4)
    # pyplot.plot(time_arr, q)
    # pyplot.xlabel('time (s)')
    # pyplot.ylabel('pitch rate (rad/s)')
    #
    # pyplot.figure(5)
    # pyplot.plot(time_arr, C)
    # pyplot.xlabel('time (s)')
    # pyplot.ylabel('elevator deflection (rad)')
    #
    # pyplot.show()

if __name__ == '__main__':
    tgt     = [120000, 30000]
    start   = [0, 40000]
    dt      = 0.001
    t       = 5
    eul     = [0, 0, 0]
    LonSim(tgt, start, eul, dt, t)


