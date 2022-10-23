#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 12:34:03 2022

@author: lukerooney
"""

import abc
import math
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

#Create Superclass PanelMethod which all panel methods will inherit.
class PanelMethod(metaclass=abc.ABCMeta):
    def __init__(self, M, gamma):
        self.M = M
        self.gamma = gamma
    
    def setM(self, M):
        self.M = M
        
    def setGamma(self, gamma):
        self.gamma = gamma
        
    def calculatecompression(self, delta):
        pass

    def calculateexpansion(self, delta):
        pass

#Newtonian Method -- simplest method. 
class NewtonianMethod(PanelMethod):
    def __init__(self, M, gamma):
        super().__init__(M, gamma)
    
    def calculatecompression(self, delta):

        return 2*np.power(np.sin(delta), 2)

    def calculateexpansion(self, delta):

        return np.zeros(len(delta))


#Newtonian Prandtl Meyer Method. - Improved Newtonian
def InversePrandtlMeyer(nu):
    ipm1       = 1.3604
    ipm2       = 0.0962
    ipm3       = -0.5127
    ipm4       = -0.6722
    ipm5       = -0.3278
    numax      = 0.5*math.pi*(math.sqrt(6)-1)
    y          = (nu/numax)**(2/3)
    return (1 + y*(1.0+y*(ipm1+y*(ipm2+y*ipm3)))/(1.0+y*(ipm4+y*ipm5)))


def newtonianprandtlmey(M, gamma, delta):
    cpstag = 2.0
    msq = M * M
    pcap = (2 / (gamma + 1)) / (msq ** (gamma / (gamma - 1))) * (
                (2 * gamma * msq - gamma + 1) / (gamma + 1)) ** (1 / (gamma - 1))
    p1 = 1
    m2 = 0
    p2 = 0
    emlow = 0.91 + 0.3125 * gamma
    emup = emlow + 0.4
    msubq = emlow
    count = 1

    while (abs(msubq - m2) > 10 ** (-4)) and (abs(p2 - p1) > 10 ** (-6)):
        q = (2 / (2 + (gamma - 1) * msubq ** 2)) ** (gamma / (gamma - 1))
        pc = q * (1 - (gamma ** 2 * msubq ** 4 * q) / (4 * (1 - q) * (msubq ** 2 - 1)))
        p1 = p2
        p2 = pc
        m1 = m2
        m2 = msubq

        msubq = m1 + (pcap - p1) * (m2 - m1) / (p2 - p1)
        msubq = min(msubq, emup)
        msubq = max(msubq, emlow)

        if (count == 1):
            msubq = emup
        elif (count == 20):
            break

        count += 1

    if (q > pcap):
        return cpstag * np.sin(delta) ** 2
    else:
        sdeltq = np.sqrt((q - pcap) / (1.0 - pcap))
        nu = np.asin(sdeltq) + delta
        mdelta = InversePrandtlMeyer(nu)
        ppo = (1 + 0.5 * (gamma - 1) * mdelta ** 2) ** (-(gamma / (gamma - 1)))
        ppfs = ppo / pcap

        return (2 / (gamma * M ** 2)) * (ppfs - 1)


class NewtonianPrandtlMeyer(PanelMethod):
    def __init__(self, M, gamma):
        super().__init__(M, gamma)

    def calculateexpansion(self, delta):
        return newtonianprandtlmey(self.M, self.gamma, delta)

    def calculatecompression(self, delta):
        return newtonianprandtlmey(self.M, self.gamma, delta)
            
            
#modified newtonian method
class ModifiedNewtonian(PanelMethod):
    def __init__(self, M, gamma):
        super().__init__(M, gamma)

    def calculateexpansion(self, delta):
        return np.zeros(len(delta))
            
    def calculatecompression(self, delta):
        gamma1   = self.gamma + 1
        gammam1  = self.gamma - 1
        expt     = self.gamma/(self.gamma - 1)

        msq = self.M**2
        q   = 2/(self.gamma*msq)
        rayleigh_pitot = (((gamma1**2*msq)/(4*self.gamma*msq - 2*gammam1))**expt)*((1 - self.gamma + 2*self.gamma*msq)/gamma1)
        cpmax = q * (rayleigh_pitot - 1)
        return cpmax*np.sin(delta)**2

def hankey(M, delta):
    cosdel = np.cos(delta)
    delta_deg = 180 / np.pi * delta

    if delta_deg < 10:
        hankey = (0.195 + 0.222594 / M ** 0.3 - 0.4) * delta_deg + 4.0
    else:
        hankey = 1.95 + 0.3925 / (M ** 0.3 * np.tan(delta))

    return hankey * cosdel * cosdel


#Hankey Flat surface method, simple but dated and with limitations
class HankeyFlatSurface(PanelMethod):
    def __init__(self, M, gamma):
        super().__init__(M, gamma)
            
    def calculatecompression(self, delta):
        cosdel      = np.cos(delta)
        delta_deg   = 180/np.pi*delta

        hankey      = np.zeros(len(delta))

        lowinds = np.where(delta <= 10)[0]  # ~index
        highinds = np.where(delta > 10)[0]

        hankey[lowinds]  = (0.195+0.222594/self.M**0.3-0.4)*delta_deg[lowinds] + 4.0
        hankey[highinds] = 1.95 + 0.3925/(self.M**0.3*np.tan(delta[highinds]))
            
        return hankey*cosdel*cosdel

    def calculateexpansion(self, delta):
        cosdel    = np.cos(delta)
        delta_deg =  180/np.pi*delta
        hankey    = (0.195+0.222594/self.M**0.3-0.4)*delta_deg + 4.0
        return hankey * cosdel * cosdel


#Busemann Second Order Theory - method with limited expansion capability
class BusemannSecondOrderTheory(PanelMethod):
    def __init__(self, M, gamma):
        super().__init__(M, gamma)
            
    def calculatecompression(self, delta):
        gamma1  = self.gamma + 1
        C1      = 2/(math.sqrt(self.M**2 - 1))
        C2      = (gamma1*self.M**4 - 4*self.M**2 + 4) / (2*(self.M**2 - 1)**2)
        return C1*delta + C2*delta**2

    def calculateexpansion(self, delta):
        return np.zeros(len(delta))

#van dyke method 
class VanDykeUnified(PanelMethod):
    def __init__(self, M, gamma):
        super().__init__(M, gamma)

    def calculatecompression(self, delta):
        gamma1 = self.gamma + 1
        gammam1 = self.gamma - 1
        expt = self.gamma / (gammam1)
        msq = self.M ** 2
        betasq = msq - 1
        hsq = betasq * np.power(delta, 2)
        h = np.power(hsq, 0.5)
        gammaTerm = 0.5 * gamma1

        cp = np.power(delta,2)*(gammaTerm + np.sqrt(gammaTerm**2 + 4.0/hsq))

        return cp


    def calculateexpansion(self, delta):
        zeroinds = np.where(delta == 0)[0]

        gamma1  = self.gamma + 1
        gammam1 = self.gamma - 1
        expt    = self.gamma/(gammam1)
        msq     = self.M**2
        betasq  = msq - 1
        hsq     = betasq*delta**2
        h       = np.sqrt(hsq)
        gammaTerm = 0.5*gamma1

        bracket  = np.zeros(len(delta), dtype=complex)
        cp       = np.zeros(len(delta), dtype=complex)

        bracket  = np.power((1.0 - 0.5*gammam1*h), (2.0*expt)) - 1.0
        cp       = 2.0*np.power(delta, 2)*bracket/(self.gamma*hsq)
        cpVac    = -2.0/(self.gamma*msq)

        cp[(np.isreal(cp)) & (cp <= cpVac)] = cpVac
        cp[(np.isnan(cp))]                  = cpVac
        cp[zeroinds]        = 0

        return cp


