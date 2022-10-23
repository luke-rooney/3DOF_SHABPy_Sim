#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:51:02 2022

@author: lukerooney
"""
from stl import mesh
import MethodsModule
import FlightDynamics
import numpy as np

class Vehicle():
    def __init__(self, M, gamma, cbar, span, sref, xref, yref, zref, m, stlfile, compression, expansion, Ixx, Iyy, Izz, Ixz, actuation_rate):
        self.M     = M
        self.gamma = gamma
        self.cbar  = cbar
        self.span  = span
        self.sref  = sref
        self.yref  = yref
        self.xref  = xref
        self.zref  = zref
        self.m     = m
        self.mesh  = mesh.Mesh.from_file(stlfile)
        self.C     = FlightDynamics.getInertiaCoeffs(Ixx, Iyy, Izz, Ixz)
        self.actuation_rate = actuation_rate
        self.maxdef = 12*np.pi/180
        self.mindef = -12*np.pi/180

        if compression == 1:
            self.compression = MethodsModule.NewtonianMethod(M, gamma)
        elif compression == 2:
            self.compression = MethodsModule.NewtonianPrandtlMeyer(M, gamma)
        elif compression == 3:
            self.compression = MethodsModule.ModifiedNewtonian(M, gamma)
        elif compression == 4:
            self.compression = MethodsModule.HankeyFlatSurface(M, gamma)
        elif compression == 5:
            self.compression = MethodsModule.VanDykeUnified(M, gamma)
        elif compression == 6:
            self.compression = MethodsModule.BusemannSecondOrderTheory(M, gamma)
        else:
            self.compression = MethodsModule.NewtonianMethod(M, gamma)

        if expansion == 1:
            self.expansion = MethodsModule.NewtonianMethod(M, gamma)
        elif expansion == 2:
            self.expansion = MethodsModule.NewtonianPrandtlMeyer(M, gamma)
        elif expansion == 3:
            self.expansion = MethodsModule.ModifiedNewtonian(M, gamma)
        elif expansion == 4:
            self.expansion = MethodsModule.HankeyFlatSurface(M, gamma)
        elif expansion == 5:
            self.expansion = MethodsModule.VanDykeUnified(M, gamma)
        elif expansion == 6:
            self.expansion = MethodsModule.BusemannSecondOrderTheory(M, gamma)
        else:
            self.expansion = MethodsModule.NewtonianMethod(M, gamma)

    def UpdatePanelMethod(self, method):
        if method == 1:
            self.compression = MethodsModule.NewtonianMethod(self.M, self.gamma)
            self.expansion   = MethodsModule.NewtonianMethod(self.M, self.gamma)
        elif method == 2:
            self.compression = MethodsModule.NewtonianPrandtlMeyer(self.M, self.gamma)
            self.expansion   = MethodsModule.NewtonianPrandtlMeyer(self.M, self.gamma)
        elif method == 3:
            self.compression = MethodsModule.ModifiedNewtonian(self.M, self.gamma)
            self.expansion   = MethodsModule.ModifiedNewtonian(self.M, self.gamma)
        elif method == 4:
            self.compression = MethodsModule.HankeyFlatSurface(self.M, self.gamma)
            self.expansion   = MethodsModule.HankeyFlatSurface(self.M, self.gamma)
        elif method == 5:
            self.compression = MethodsModule.VanDykeUnified(self.M, self.gamma)
            self.expansion   = MethodsModule.VanDykeUnified(self.M, self.gamma)
        elif method == 6:
            self.compression = MethodsModule.BusemannSecondOrderTheory(self.M, self.gamma)
            self.expansion   = MethodsModule.BusemannSecondOrderTheory(self.M, self.gamma)
        else:
            self.compression = MethodsModule.NewtonianMethod(self.M, self.gamma)
            self.expansion   = MethodsModule.NewtonianMethod(self.M, self.gamma)
