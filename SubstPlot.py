#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:59:54 2019

This code takes a 2D array and converts it to a jpeg 

@author: yajun
"""

############## import packages
from skimage import color
from skimage import io

from numpy import ceil 
from numpy import matrix, linspace

import matplotlib.pyplot as plt
from PIL import Image

import os, os.path
import numpy as np

from substitution import SUBSTITUTION, SUBS

######## generate color spectrums for picking tiling colorings.

def ColorizeAlphabet(N=60): # gives a sequence of RGB colors (mirror with N total samples (it will roound up to a multiple of 7)
    # npc = number of samples per color, of which there are 7, doubled for conjugate symmetry
#    if reflect:
#        npc = ceil(N/12); 
#    else:
#        npc = ceil(N/6);
#    N = 6*npc;
    L = linspace(0,6,N);
    cA = [] # the rainbow of colors! it will be "reflected" over after.
    # RGB COLORS
    VIOLET = matrix([139,0,255]); INDIGO = matrix([39,0,51]); BLUE = matrix([0,0,255]); GREEN = matrix([0,255,0]); YELLOW = matrix([255,255,0]); ORANGE = matrix([255,127,0]); RED = matrix([255,0,0]);
    COLORS = [VIOLET, INDIGO, BLUE, GREEN, YELLOW, ORANGE, RED];
    
#    for i in range(6):
#        for l in L:
#            RB.append(tuple((COLORS[i] + l*(COLORS[i+1]-COLORS[i])).tolist()[0]))

    for l in L[0:-1]:
        cA.append(tuple((COLORS[int(l)]+(l-int(l))*(COLORS[int(l)+1]-COLORS[int(l)])).tolist()[0]))
    cA.append(tuple(COLORS[-1].tolist()[0])) 
    return cA
     
########


# A Nonbijective (2,2)-substitution in the plane with Lebesgue component in the spectrum
RS2 = SUBSTITUTION([2,2],[[0,0,0,4,4,4,4,0],[1,1,5,1,5,5,1,5],[2,6,2,2,6,2,6,6],[7,3,3,3,3,7,7,7]])


figsavepath = os.path.abspath('/Users/tehspoke/Google Drive/ALAN STUFF/Python/Substitutions')

# draw a patch of diameter D at the point k in a supertile of generation n for SUB iterated on gamma
def drawSub(SUB=RS2,n=4,k=[0,0],D=[8,12],gamma=0):
    s=SUB.s
    s=25
    Alpha = [i for i in range(s)];
    cAlpha = ColorizeAlphabet(s)
    
    A = np.random.randint(255,size=(k[0]+D[0],k[1]+D[1],3)) # 3D array for RGB

    img = Image.fromarray(A, 'RGB') # change to 'L' is only need grayscale
    img.save(figsavepath+'/'+'my.jpg') # specify format and maybe location for your save figure
    img.show()
