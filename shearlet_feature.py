import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as myimg
import matplotlib.colors as mycolor
from scipy import stats
from mahotas import features
from quantization import linearQuantization, nonLinearQuantization
import pyshearlab 
import matplotlib.cm as cm
import matplotlib.cbook as cbook
import matplotlib.colors as colors
import glob
import cv2 as cv
import math
import csv

#######################################################################################
##... THIS FUNCTION IN FOR CONSTRUCTING MOSAICS FROM INNER NUCLEUS TEXTURE ......######
#                                                                                 #####
# inputs: 1. patch. Is the complete image patch which contains one White Blood cell ###
#         2. mask.  Is a mask for the cell nucleus                                   ###
# Output: sh_descript. Which contains a vector which contains mean and stdeviation  ###
#         for each shearlet sub-bands                                               ###
#######################################################################################

def shearTexture(patch,mask):
    patch = np.double(patch)
    row,col = np.shape(patch)
    m = min(row,col)

    
    ###  This part is for finding the contour of the mucleus
    Area = 1
    contours,_= cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
                
        Area1 = cv.contourArea(contour)
        if Area1 > Area:
            Area = Area1
            con = contour
    print('Contour_area------------------>',Area)
    
    x,y,w,h = cv.boundingRect(con)
    #cv.rectangle(patch,(x,y),(x+w,y+h),(0,255,0),2)
    M = cv.moments(con)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    h,w = 1,1
    minRect = mask[cy-h: cy+h,cx-w:cx+w]
    print(minRect)
    while 0 not in mask[cy-h: cy+h,cx-w:cx+w]:
        print('Optimizing inner patch...',np.shape(mask[cy-h: cy+h,cx-w:cx+w]))
        h = h+1
        w = w+1
    #cv.rectangle(patch,(cx-w,cy-h),(cx+w,cy+h),(255),2)



    
    #########################################################################################
    ##       This code section is for extracting the inner texture of nucleus and        ####
    ##        and for constructing a Mosaic as a repetition fo such pattern               ###
    #########################################################################################
    
    ###  Extracting nucleus patch according to width and high of the previously found contour 
    ima = patch[cy-h: cy+h,cx-w:cx+w]
    dim = ima.shape
    m = np.min(dim)

    ## Checking for size requirement according to the number of shearlet scales 
    print('Original nucleus patch size:', dim)
    rep = np.uint8(np.floor(250/(m))/2)+1
    print(rep)
    if rep>10:
        rep=10
    patch_row = ima
    
    
    ##  constructing  the inner nucleus txture Mosaic by repeating the patter until acomplish the shearlet size requierement
    for u in range(0,rep):
        patch_row = np.append(patch_row,np.flip(patch_row,1), axis=1)
        patch_row = np.append(patch_row,np.flip(patch_row,0), axis=0)
    
    print('Final mosaicing patch size:',patch_row.shape)
    
    if np.shape(patch_row)[0]>1000:
        patch_row = patch_row[0:1000,0:1000]
    elif np.shape(patch_row)[0]!=np.shape(patch_row)[1]:
        print('|||Edgy patch|||')
        size = np.shape(patch_row)
        minimus = min(size)
        patch_row = patch_row[0:minimus,0:minimus]
        
    print('Final mosaicing patch size:',patch_row.shape)

    
    ### Verifying the number of non zero intensity bins which establish the maximum number of quantization levels
    #n = plt.hist(patch_row.ravel(),256,[np.min(patch_row),np.max(patch_row)])
    #n_zero = np.count_nonzero(n[0])
    
    ########################################################################
    ##         FOLLOWING CODE IS FOR PATCH MOSAIC QUANTIZATION            ##
    #    quanti_patch contains a copy of the patch mosaic (patch_row)      #
    #      quanto is a variable that stores the quantization factor        #
    #                                                                      #
    # q_levels: is the number of levels for quantization                   #
    # move    : is the displacement of the non linear quantization function#
    # ######################################################################
    q_levels = 32
    move = 5.5
    ### .... This part perform  a nonlinear quantization 
    quanti = nonLinearQuantization(patch_row,move)

    ### .....................
    #plt.subplot(2,3,1),plt.imshow(mask)
    #plt.subplot(2,3,2),plt.imshow(patch)
    #plt.subplot(2,3,3),plt.imshow(ima)
    #plt.subplot(2,3,4),plt.imshow(patch_row)
    #plt.subplot(2,3,5),plt.imshow(quanti)
    
    # ... This part is for linear quantization
    quanti_patch = linearQuantization(patch_row,q_levels)
    
    ## .......
    
    #plt.subplot(2,3,6),plt.imshow(quanti_patch)
    #plt.show()
    
    ###########################################################################
    ###               EXTRACTING shearlet coefficients                      ###
    #   where coeffs stores the shearlet coefficients and scale_number      ###
    ###########################################################################

    print('Extracting shearlet coefficients...')
    m = np.min(patch_row.shape)
    scale_number = 4
    #print('k shear parameter......', math.ceil((1:scale_number)/2))
    shearletSystem = pyshearlab.SLgetShearletSystem2D(0,m,m,scale_number)
    coeffs = pyshearlab.SLsheardec2D(np.double(quanti_patch),shearletSystem)

    
    f,c,s = coeffs.shape
    #### print(f,c,s)
    #Xrec = pyshearlab.SLshearrec2D(coeffs,shearletSystem)
    ###for i in range(45,49):
        ###plt.imshow(np.abs(coeffs[:,:,i]))
        ###plt.show()
    

    sh_descript = []
    plt.imshow(patch)
    plt.show()
    plt.imshow(ima)
    plt.show()
    plt.imshow(patch_row)
    plt.show()
    count =0
    for t in range(0,s):
        if (t ==1) or (t ==2) or (t ==4) or (t ==28) or (t ==29) or (t ==31) or (t ==32) or (t ==33) or (t ==30):
            fig, ax = plt.subplots()
            pos = ax.imshow(coeffs[10:-10,10:-10,t],norm=colors.CenteredNorm() ,cmap='coolwarm')#,norm=colors.LogNorm(vmin=np.min(np.min(coeffs[:,:,t])), vmax=np.max(np.max(coeffs[:,:,t]))))#,cmap='PuBu_r')
            ax.set_title('f sub-band: %i' %t)
            #pcm = ax.pcolor(coeffs[:,:,t], cmap='PuBu_r', shading='nearest', norm =colors.LogNorm(vmin=np.min(np.min(coeffs[:,:,t]))))
            fig.colorbar(pos, ax=ax, extend='max')
            plt.show()
        sh_descript = np.append(sh_descript,np.mean(coeffs[:,:,t].ravel()))
        sh_descript = np.append(sh_descript,np.std(coeffs[:,:,t].ravel()))
        #sh_descript = np.append(sh_descript,stats.kurtosis(coeffs[:,:,t].ravel()))
        #sh_descript = np.append(sh_descript,stats.skew(coeffs[:,:,t].ravel()))
        #plt.hist(coeffs[:,:,t].ravel(),256,[-m,m]); plt.show()

    return(sh_descript) 
