import Vehicle
import numpy
import pathlib
import os

def LoadUNSW5_Control():
    M = 7               #Mach number
    gamma = 1.4         #Ratio of Specific Heats (AIR)
    cbar = 0.6          #Mean chord (m)
    span = 0.4446       #Span of the vehicle (m)
    sref = 0.218        #Reference area of the vehicle (m^2)
    xref = 0.57685      #Center of mass x coordinate (m)
    yref = 0            #Center of mass y coordinate (m)
    zref = -0.047       #Center of mass z coordinate (m)
    m = 35              #mass value (kg)
    compression = 1     #panel method reference (compression)
    expansion = 1       #panel method reference (expansion)
    Ixx = 0.141         #Mass Moment Ixx (m^4)
    Iyy = 2.531         #Mass Moment Iyy (m^4)
    Izz = 2.596         #Mass Moment Izz (m^4)
    Ixz = 0.205         #Mass Momemt Ixz (m^4)
    actuation_rate = 200*numpy.pi/180    #maximum actuation rate of the control surfaces (rad/s)

    #File Path to the stl files.
    current     = str(pathlib.Path(__file__).parent.resolve())
    filepath    = os.path.join(current, "Vehicles", "final_body.stl")
    filepath_fr = os.path.join(current, "Vehicles", "final_fr.stl")
    filepath_fl = os.path.join(current, "Vehicles", "final_fl.stl")

    # M, gamma, cbar, span, sref, xref, yref, zref, m, stlfile, compression, expansion, Ixx, Iyy, Izz, Ixz
    unsw    = Vehicle.Vehicle(M, gamma, cbar, span, sref, xref, yref, zref, m, filepath, compression, expansion, Ixx, Iyy, Izz, Ixz, actuation_rate)
    flap_l  = Vehicle.Vehicle(M, gamma, cbar, span, sref, xref, yref, zref, m, filepath_fl, compression, expansion, Ixx, Iyy, Izz, Ixz, actuation_rate)
    flap_r  = Vehicle.Vehicle(M, gamma, cbar, span, sref, xref, yref, zref, m, filepath_fr, compression, expansion, Ixx, Iyy, Izz, Ixz, actuation_rate)

    return [unsw, flap_l, flap_r]