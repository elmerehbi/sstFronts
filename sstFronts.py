# Developed by: Ibrahim EL MEREHBI, 2017
# See Jupyter notebook: "00_Using-the%20library" to learn how to use the library.

import netCDF4 as nc
import numpy as np
import scipy
from scipy import interpolate
from mpl_toolkits import basemap as bm
import matplotlib
matplotlib.use('agg')
from matplotlib import  pyplot as plt
import os
import sys
import socket
from pathlib import Path
import datetime as dt
import warnings
warnings.filterwarnings("ignore")
# import cmocean # oceanography-specific colormaps
import seaborn as sns
# import pandas as pd

def dist_spheric(lat1, lon1, lat2, lon2):
    """
    function that takes angles in deg
    return mastrix dist of size NI*ND
    NI: size of lat2
    ND size of lat1
    arguments must be vectors (NOT 2D arrays)
    """
    R= 6371.008*10**3
    NI=np.size(lat2)
    ND=np.size(lat1)

    l=np.abs(np.dot(np.transpose(lon2),np.ones((1,ND)))- np.dot(np.ones((NI,1)),lon1))
    l[l>=180]=360-l[l>=180]

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    l=np.radians(l)

    dist=R*np.arctan2(np.sqrt((np.sin(l)*np.dot(np.transpose(np.cos(lat2)),np.ones((1,ND))))**2 \
                        +(np.dot(np.transpose(np.sin(lat2)),np.cos(lat1)) \
                         - np.dot(np.transpose(np.cos(lat2)),np.sin(lat1))*np.cos(l))**2 \
                        ),
                          np.dot(np.transpose(np.sin(lat2)),np.sin(lat1))+\
                          np.dot(np.transpose(np.cos(lat2)),np.cos(lat1))*np.cos(l) \
                          )
    return(dist)
#-----------------------------------------------------------------------------

def gnomonic(lon,lat,lon0,lat0,R):
    """
    Gnomonic projection is distance preserving for
    interpolation purposes.

    see reference: http://mathworld.wolfram.com/GnomonicProjection.html
    """
    lat = lat*np.pi/180
    lon = lon*np.pi/180
    lat0= lat0*np.pi/180
    lon0= lon0*np.pi/180
    cosc = np.sin(lat0)*np.sin(lat) + np.cos(lat0)*np.cos(lat)*np.cos(lon-lon0)
    xg = R*np.cos(lat)*np.sin(lon-lon0)/cosc
    yg = R*(np.cos(lat0)*np.sin(lat) - np.sin(lat0)*np.cos(lat)*np.cos(lon-lon0) )/cosc
    return(xg,yg)
#-----------------------------------------------------------------------------

def ignomonic(x,y,lon0,lat0,R):
    """
    Inverse Gnomonic projection.

    see reference: http://mathworld.wolfram.com/GnomonicProjection.html
    """
    lat0= lat0*np.pi/180
    lon0= lon0*np.pi/180
    rho=np.sqrt(x**2+y**2)
    c=np.arctan2(rho,R)

    if (rho!=0):
        lat=np.arcsin(np.cos(c)*np.sin(lat0) + (y*np.sin(c)*np.cos(lat0))/rho)
        if ((lat!=np.pi/2) & (lat!=-np.pi/2) ):
            lon=lon0+np.arctan2(x*np.sin(c),(rho*np.cos(lat0)*np.cos(c)-y*np.sin(lat0)*np.sin(c)))
        elif (lat==np.pi/2):
            lon=lon0+np.arctan2(x,-y)
        elif (lat==-np.pi/2):
            lon=lon0+np.arctan2(x,y)
    elif (rho==0):
        lat=lat0
        lon=lon0

    lat = lat*180/np.pi
    lon = lon*180/np.pi
    return(lon,lat)
#-----------------------------------------------------------------------------
def nextpow2(x):
    n=1
    NFFT=2**n
    while (NFFT<x):
        n=n+1
        NFFT=2**n
    return(n)
#-----------------------------------------------------------------------------

from scipy.spatial import Delaunay
import time as tm

def get_tri_coef(X, Y, newX, newY, verbose=0):

    """
    Inputs:
        origin lon and lat 2d arrays (X,Y)
        child lon and lat 2d arrays (newX,newY)

    Ouputs:
        elem - pointers to 2d gridded data (at lonp,latp locations) from
            which the interpolation is computed (3 for each child point)
        coef - linear interpolation coefficients
    Use:
        To subsequently interpolate data from Fp to Fc, the following
        will work:      Fc  = sum(coef.*Fp(elem),3);  This line  should come in place of all
        griddata calls. Since it avoids repeated triangulations and tsearches (that are done
        with every call to griddata) it should be much faster.
    """

    Xp = np.array([X.ravel(), Y.ravel()]).T
    Xc = np.array([newX.ravel(), newY.ravel()]).T


    #Compute Delaunay triangulation
    if verbose==1: tstart = tm.time()
    tri = Delaunay(Xp)
    if verbose==1: print('Delaunay Triangulation', tm.time()-tstart)

    #Compute enclosing simplex and barycentric coordinate (similar to tsearchn in MATLAB)
    npts = Xc.shape[0]
    p = np.zeros((npts,3))

    points = tri.points[tri.vertices[tri.find_simplex(Xc)]]

    if verbose==1: tstart = tm.time()
    for i in range(npts):

        if verbose==1: print(np.float(i)/npts)

        if tri.find_simplex(Xc[i])==-1:  #Point outside triangulation
             p[i,:] = p[i,:] * np.nan

        else:

            if verbose==1: tstart = tm.time()
            A = np.append(np.ones((3,1)),points[i] ,axis=1)
            if verbose==1: print('append A', tm.time()-tstart)

            if verbose==1: tstart = tm.time()
            B = np.append(1., Xc[i])
            if verbose==1: print('append B', tm.time()-tstart)

            if verbose==1: tstart = tm.time()
            p[i,:] = np.linalg.lstsq(A.T,B.T)[0]
            if verbose==1: print('solve', tm.time()-tstart)


    if verbose==1: print('Coef. computation 1', tm.time()-tstart)

    if verbose==1: tstart = tm.time()
    elem = np.reshape(tri.vertices[tri.find_simplex(Xc)],(newX.shape[0],newY.shape[1],3))
    coef = np.reshape(p,(newX.shape[0],newY.shape[1],3))
    if verbose==1: print('Coef. computation 2', tm.time()-tstart)

    return(elem,coef)


#---------------------------------------------------------------------------

def curlzroms(pm,pn,u,v):
    #varX: zonal u at u point
    #varY: meridional v at v point
    #pn-> 1/dy at rho rho point (dy:dist between v point)
    #pm -> 1/dx at rho point (dx: dist between u points)
    curlz=0.25*(pm[:-1,:-1]+pm[:-1,1:]+pm[1:,:-1]+pm[1:,1:])*np.diff(v,1,1)-\
    0.25*(pn[:-1,:-1]+pn[:-1,1:]+pn[1:,:-1]+pn[1:,1:])*np.diff(u,1,0)
    return(curlz)

#-----------------------------------------------------------

def project_section(SEC, dgc):
    """
    Input
        sec: a section (SEC[i])
        dgc: geospatial resolution

    Returns
        longs, latgs: Mercator coordinates corresponding to interpolated section
    """
    LONGS, LATGS = [], []

    for isec in np.arange(len(SEC)):
        sec = SEC[isec]

        lon0=sec[0]
        lat0=sec[1]
        lon1=sec[2]
        lat1=sec[3]

        R=6367442.76

        distsecx=dist_spheric(lat0,lon0,lat1,lon1)
        dthetax=dgc/R

        npsecx = np.int(distsecx/dgc)

        # gnomonic projection
        xg1,yg1 = gnomonic(lon1,lat1,lon0,lat0,R)

        # inverse gnomonic projection
        a=yg1/xg1
        xgs=np.zeros((1,npsecx))
        xgs[0,0]=0
        xgs[0,1:]=-R*np.tan(np.arange(1,npsecx)*dgc/R)/(1+np.abs(a)**2)**0.5
        ygs=a*xgs
        longs=np.zeros((1,npsecx))
        latgs=np.zeros((1,npsecx))
        for ig in range(0,np.size(xgs)):
            longs[0,ig],latgs[0,ig] = ignomonic(xgs[0,ig],ygs[0,ig],lon0,lat0,R)

        LONGS += [longs]
        LATGS += [latgs]

    return(LONGS, LATGS)

# def interpole_section(sec, dgc, sst, lon, lat, longs, latgs, elem, coef):
#     """
#     Interpolate the SST on the section with geospatial resolution dgc

#     Returns:
#          - along_section_sst: interpolated SSTs along section
#          - latgs/longs: gnomonic-projected latitude/longitude
#          - at_d: off-shore distance along section
#     """
#     longs, latgs = project_section(sec, dgc)

#     [elem,coef] = get_tri_coef(lon, lat, longs, latgs)
#     along_section_sst = np.zeros((longs.shape))
#     along_section_sst = np.sum(coef * sst.ravel()[elem],2)

#     # using 2D interpolation

#     # calculate off-shore distance along section
#     n = longs.shape[1]
#     at_d = np.zeros(n)
#     for ix in np.arange(n):
#         at_d[ix] = dist_spheric(latgs[0,ix], longs[0,ix], sec[1], sec[0])

# #     print(elem)
# #     print(coef)
#     return(along_section_sst, latgs, longs, at_d)

def interpole_section(sec, sst, lon, lat, longs, latgs, elem, coef):
    """
    Interpolate the SST on the section with geospatial resolution dgc

    Returns:
         - along_section_sst: interpolated SSTs along section
         - latgs/longs: gnomonic-projected latitude/longitude
         - at_d: off-shore distance along section
    """

    along_section_sst = np.zeros((longs.shape))
    along_section_sst = np.sum(coef * sst.ravel()[elem],2)[0]

    # using 2D interpolation

    # calculate off-shore distance along section
    n = longs.shape[1]
    at_d = np.zeros(n)
    for ix in np.arange(n):
        at_d[ix] = dist_spheric(latgs[0,ix], longs[0,ix], sec[1], sec[0])

    assert(along_section_sst.shape == at_d.shape), "along_section_sst.shape and at_d.shape should have same dimensions"
    return(along_section_sst, latgs, longs, at_d)


"""
definitions needed to find the points of intersection
"""
#
# line segment intersection using vectors
# see Computer Graphics by F.S. Hill
#
def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return
def seg_intersect(a1, a2, b1, b2):
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp )
    return (num / denom.astype(float))*db + b1

def find_intersects(at_d, sst_grad, indx):
    """
    Calculates and returns all points of intersection with the horizontal threhold line
    at_d: off-shore distances along the profile section
    sst_grad: SST gradient
    indx: indices of points with opposite signs (i.e. those that form a line that croses the x-axis)
    """

    # assert (len(indx) % 2 == 0), "indx is not even!"

    # points making up the thresold line
    # specific to a horizontal line!!!
    p3 = np.array([0, 0])
    p4 = np.array([at_d.max(), 0])

    n = int(len(indx)/2)
    intersects = np.zeros((2,n))

    for ix in np.arange(0, 2*n-1, 2):
#         print(n, ix, ix+1)
#     for ix in np.arange(0, n-1, 1):
#         print(n, ix, ix+1)
        p1 = np.array([at_d[indx[ix]], sst_grad[indx[ix]]])
        p2 = np.array([at_d[indx[ix+1]], sst_grad[indx[ix+1]]])
        intersects[:,int(np.floor(ix/2))] =  seg_intersect(p1, p2, p3, p4)
#     print(intersects)

    return(intersects)

def define_sections(sst_dir, sarpline, dgcxcross=5e3, dgcycross=50e3, figs_dir="../figs", verbose=0):
    """
    Defines sections perpendicular to the western coast of South African

    verbose:
        - 0 for no output
        - 1 for printed output
        -'show figures' to show plots of (1) comparison between Delaunay & Nearest neighbour interpolation, (2) same but with SST vs Longitude, and (3) plot of sections defined on the SST map
    """

    print('interpolating...')

    SEC=[]

    # Read sarp line
    lonsarp=[]
    latsarp=[]
#     # sarpline='/Users/herbette/Bob-Ibrahim/Data/Africana203_SARP.txt'
#     sarpline='../data/along-track_altimetry/Africana203_SARP.txt'
    with open(sarpline) as f:
        for line in f:
            latsarp.append(float(line.strip().split(';')[1]))
            lonsarp.append(float(line.strip().split(';')[2]))


    # Choose the Southern Cross-shelf section to be the SARP line
    lonsarpe=np.max(lonsarp)
    latsarpe=latsarp[np.argmax(lonsarp)]
    lonsarpw=np.min(lonsarp)
    latsarpw=latsarp[np.argmin(lonsarp)]
    SEC[0:0]=[np.array([lonsarpe,latsarpe,lonsarpw,latsarpw])]

#     # sst_dir =  "/net/leo/local/tmp/1/herbette/ODYSEA/odyssea_saf_extraction/2016/"
#     sst_dir =  "../data/ODYSEA/odyssea_saf_extraction/1601/"

    pathlist = Path(sst_dir).glob('**/**/*SAF*.nc')
    for path in pathlist:
            # because path is object not string
            nc_path = str(path)
            abs_path = os.path.abspath(nc_path)
    #         print(abs_path)

    lon, lat, sst, landmask = read_SST(abs_path)

    #SARP Line
    SEC[1:1]=[np.array([lonsarpe,latsarpe,16,-28])] # EAST SECTION

    #SEC[1:1]=[np.array([18.3,-32.5,15.5,-33.5])]

    #Define point of SARP line
    lon0=SEC[0][0]
    lat0=SEC[0][1]
    lon1=SEC[0][2]
    lat1=SEC[0][3]

    #USER
    R=6367442.76
#     #dgc=7e3 # 7 km (horizontal resolution of the model)
#     #dgc=3e3
#     dgcxcross=5e3
#     dgcycross=50e3

    dgc=dgcxcross
    distsecx=dist_spheric(lat0,lon0,lat1,lon1)
    dthetax=dgc/R
    npow=nextpow2(distsecx/dgc)
    if np.abs(distsecx-2**npow*dgc) > np.abs(distsecx-2**(npow-1)*dgc):
        npsecx=2**(npow-1)+1
    else:
        npsecx=2**npow+1

    #USER

    npsecx=128
    npsecxp=128-20+1 #I take some points to the West towards sea
    npsecxm=20 #I take some points towards inland

    xg1,yg1 = gnomonic(lon1,lat1,lon0,lat0,R)
    a=yg1/xg1
    xgs=np.zeros((1,npsecx))
    xgs[0,0]=0
    xgs[0,1:npsecxm]=R*np.tan(np.arange(1,npsecxm)*dgc/R)/(1+np.abs(a)**2)**0.5
    xgs[0,npsecxm:]=-R*np.tan(np.arange(1,npsecxp)*dgc/R)/(1+np.abs(a)**2)**0.5
    xgs[0,:]=np.sort(xgs,axis=1)[0,::-1]
    if verbose ==1: print(xgs)

    ygs=a*xgs
    longs=np.zeros((1,npsecx))
    latgs=np.zeros((1,npsecx))
    for ig in range(0,np.size(xgs)):
        longs[0,ig],latgs[0,ig] = ignomonic(xgs[0,ig],ygs[0,ig],lon0,lat0,R)

    # Define point of EAST section
    #start from most eastern point found on the southern section
    lon0=longs[0,0]
    lat0=latgs[0,0]
    lon1=lon0+(SEC[1][2]-lonsarpe)
    lat1=lat0+(SEC[1][3]-latsarpe)
    if verbose == 1: print('shape(lon1) =', np.shape(lon1))
    dgc=dgcycross
    distsecy=dist_spheric(lat0,lon0,lat1,lon1)
    dthetay=dgc/R
    npow=nextpow2(distsecy/dgc)
    if np.abs(distsecy-2**npow*dgc) > np.abs(distsecy-2**(npow-1)*dgc):
        npsecy=2**(npow-1)+1
    else:
        npsecy=2**npow+1

    xg1,yg1 = gnomonic(lon1,lat1,lon0,lat0,R)
    a=yg1/xg1
    xge=np.zeros((1,npsecy))
    xge[0,0]=0
    xge[0,1:]=-R*np.tan(np.arange(1,npsecy)*dgc/R)/(1+np.abs(a)**2)**0.5
    yge=a*xge

    longe=np.zeros((1,npsecy))
    latge=np.zeros((1,npsecy))
    for ig in range(0,np.size(xge)):
        longe[0,ig],latge[0,ig] = ignomonic(xge[0,ig],yge[0,ig],lon0,lat0,R)

    #define all arrays
    dlon=np.diff(longe,1,1)
    dlat=np.diff(latge,1,1)
    longw=np.zeros((1,npsecy))
    longw[0,1:]=longs[0,-1]+np.cumsum(dlon)
    latgw=np.zeros((1,npsecy))
    latgw[0,1:]=latgs[0,-1]+np.cumsum(dlat)

    londom=np.zeros((npsecy,npsecx))
    latdom=np.zeros((npsecy,npsecx))
    londom[0,:]=longs
    latdom[0,:]=latgs
    for i in range(0,npsecx):
        #print(ig)
        londom[1:,i]=longs[0,i]+np.cumsum(dlon)
        latdom[1:,i]=latgs[0,i]+np.cumsum(dlat)

    #Interplate on the new domain
    # Find coeff of interpolation
    [Lon,Lat]=np.meshgrid(lon,lat)
    [elem,coef] = get_tri_coef(Lon,Lat,londom,latdom)
    #Use the coeff to interpolate
    sstdom= np.sum(coef*sst.data[:,:].ravel()[elem],2)
    landmaskdom= np.sum(coef*landmask[:,:].ravel()[elem],2) #does not work well

    ma=np.sum(coef*sst.mask[:,:].ravel()[elem],2)

    #-----------------------------------------------------------------------------
    # nearest neighbour interpolation
    nearestindexinterp=[]
    lon2=np.array(Lon.ravel()).reshape((1,np.size(Lon)))
    lat2=np.array(Lat.ravel()).reshape((1,np.size(Lat)))
    for j in range(np.shape(londom)[0]):
        loni=np.reshape(londom[j,:],(1,np.size(londom[j,:])))
        lati=np.reshape(latdom[j,:],(1,np.size(latdom[j,:])))
        lon1=loni
        lat1=lati
        nearestindexinterp+=nearest_neighbour_interp(lon1,lat1,lon2,lat2, Lon)

    #-----------------------------------------------------------------------------

    sstdom[2,:]
    londom[2,:]
    sstdom2=np.zeros(np.shape(londom))
    dmindom2=np.zeros(np.shape(londom))
    landmaskdom2=np.zeros(np.shape(londom))
    for j in range(np.shape(londom)[0]):
        sstdom2[j,:]=sst[nearestindexinterp[2*j][0],nearestindexinterp[2*j][1]]
        dmindom2[j,:]=nearestindexinterp[2*j+1]
        landmaskdom2[j,:]=landmask[nearestindexinterp[2*j][0],nearestindexinterp[2*j][1]]

    #sstdom2ma=np.ma.masked_where((dmindom2>10000.) | (sstdom2<0) ,sstdom2)
    sstdom2ma=np.ma.masked_where(landmaskdom2==2 ,sstdom2)

    sstdomma=np.ma.masked_where((landmaskdom2==2)|(np.isnan(sstdom))|(sstdom<=0),sstdom) #does not work well

    #Find the index of the beginning of the coast line on every section
    ixbsec=[]
    for j in range(np.shape(londom)[0]):
        if verbose == 1: print('\tj =', j)
        ixbsec.append(np.max(np.where(sstdomma.mask[j,:-60])[0])+1)
    if verbose == 1: print('ixbsec =', ixbsec)

    #Find the index of the last available sst data on the section
    ixesec=[]
    for j in range(np.shape(londom)[0]):
        if verbose == 1: print('\tj =', j)
        ixesec.append(np.max(np.where(londom[j,:]>np.min(lon))))
    if verbose ==1: print('ixesec =', ixesec)

    #-----------------------------------------------------------------------------

    if verbose == 'show figures':
        #sstdomma=np.ma.masked_array(sstdom,np.isnan(sstdom))
        plt.figure(figsize=(20,15))
        plt.subplot(1,3,1)
        plt.title('Delaunay triangulation')
        cax=plt.pcolormesh(londom,latdom,sstdomma)
        plt.colorbar(cax)
        plt.subplot(1,3,2)
        plt.title('Nearest neighbour')
        cax=plt.pcolormesh(londom,latdom,sstdom2ma)#,vmin=np.min(sstdomma),vmax=np.max(np.max(sstdomma)))
        plt.colorbar(cax)

        plt.subplot(1,3,3)
        plt.title('Difference')
        sstdoma_difference = sstdomma - sstdom2ma
        plt.pcolormesh(londom, latdom, sstdoma_difference)
        plt.colorbar()

        print('plotting interpolation comparison')
        plt.savefig(os.path.join(os.path.abspath(figs_dir), "interpolation_comparison_2D.png"))

        plt.figure()
        plt.plot(londom[3,:],sstdom2ma[3,:], 'b-', label='nearest neighbour')
        plt.plot(londom[3,:],sstdomma[3,:],'r-', label='Delaunay')
        plt.title('Nearest neighbour vs Delaunay Triangulation')
        plt.xlabel('SST')
        plt.ylabel('Longitude')
        plt.legend()

        plt.savefig(os.path.join(os.path.abspath(figs_dir), "interpolation_comparison_1D.png"))

    #-----------------------------------------------------------------------------
    # a209=np.loadtxt('/Users/herbette/Python-Scripts/track209.asc')
    a209=np.loadtxt('../data/along-track_altimetry/track209.asc')
    #Get only values in range of interest
    i209=np.where( (a209[:,0]>np.min(lon)) & (a209[:,0]<np.max(lon)) & (a209[:,1]>np.min(lat)) & (a209[:,1]<np.max(lat)) )
    # a057=np.loadtxt('/Users/herbette/Python-Scripts/track057.asc')
    a057=np.loadtxt('../data/along-track_altimetry/track057.asc')
    #Get only values in range of interest
    i057=np.where( (a057[:,0]>np.min(lon)) & (a057[:,0]<np.max(lon)) & (a057[:,1]>np.min(lat)) & (a057[:,1]<np.max(lat)) )
    # a133=np.loadtxt('/Users/herbette/Python-Scripts/track133.asc')
    a133=np.loadtxt('../data/along-track_altimetry/track133.asc')
    #Get only values in range of interest
    i133=np.where( (a133[:,0]>np.min(lon)) & (a133[:,0]<np.max(lon)) & (a133[:,1]>np.min(lat)) & (a133[:,1]<np.max(lat)) )

    # plot figure to confirm the defined sections
    if verbose == 'show figures':
        plt.figure(figsize=(30,30))
        cax=plt.pcolormesh(Lon,Lat,sst,cmap=mycmap())
        #plt.plot(Lon,Lat,'r+')
        cbar = plt.colorbar(cax)

        plt.plot(londom,latdom,'k+')
        for j in range(np.shape(londom)[0]):
            plt.plot(londom[j,ixbsec[j]],latdom[j,ixbsec[j]],'ro')
            plt.plot(londom[j,ixesec[j]],latdom[j,ixesec[j]],'ro')

        #plt.plot(lonsarp,latsarp,'w+')
        plt.plot(lonsarpe,latsarpe,'bo')
        plt.plot(lonsarpw,latsarpw,'bo')

        plt.plot(a209[i209,0],a209[i209,1],'bo')
        plt.plot(a057[i057,0],a057[i057,1],'bo')
        plt.plot(a133[i133,0],a133[i133,1],'bo')

        plt.savefig('../figs/defined_sections.png', dpi=300)

    altimetry_tracks = np.array([np.array([a057[i057][:,0], a133[i133][:,0], a209[i209][:,0]]), np.array([a057[i057][:,1], a133[i133][:,1], a209[i209][:,1]])])

    #-----------------------------------------------------------------------------

    # get only end points of the sections to work with other functions
    SEC=[]
    longsbob=[]
    latgsbob=[]
    for j in range(np.shape(londom)[0]):
        SEC[j:j]=[np.array([londom[j,ixbsec[j]],latdom[j,ixbsec[j]],londom[j,ixesec[j]],latdom[j,ixesec[j]]])]
        longsbob[j:j]=[londom[j,ixbsec[j]:ixesec[j]]]
        latgsbob[j:j]=[latdom[j,ixbsec[j]:ixesec[j]]]

    return(SEC, longsbob, latgsbob, sstdomma, landmaskdom2, londom, ixbsec, ixesec, elem, coef, altimetry_tracks)

def read_SST(nc_path):
    """
    directory_in_str = "<path_to_data>/data/ODYSEA/odyssea_saf_extraction/2016/"
    nc_path: path to the netCDF file
    """

    import netCDF4 as nc

    df = nc.Dataset(nc_path, 'r')

    # read the needed data
    lon = df['lon'][:]
    lat = df['lat'][:]
    sst = df['analysed_sst'][0] - 273
    sst = np.ma.masked_array(sst)
#     analysis_error = df['analysis_error'][0]
    landmask = df['mask'][0]
#     # sif = df['sea_ice_fraction'][0]

    return(lon, lat, sst, landmask)

def assertCoordinates(lon, lat):
    """
    Input: 2 lists or arrays
        - lon
        - lat

    Output: assertion error if not in range.

    Example call:

        assertCoordinates(lon = [28, 30], lat = [-10, 200])
    """

    lon = np.array(lon); lat = np.array(lat)

    assert ( (lon >= 0).all() and (lon <= 360).all() ), "longitudes are corrpoted."
    assert ( (lat >= -180).all() and (lat <= 180).all() ), "latitudes are corrpoted."


# def map_borders(lon, lat):
#     """
#     Returns minimums and maximums of lat & lon
#     """

#     min_lon = np.min(lon)
#     max_lon = np.max(lon)
#     min_lat = np.min(lat)
#     max_lat = np.max(lat)

#     return(min_lon, max_lon, min_lat, max_lat)

def read_bathymetry(bathy_path, region_borders=None, downsampling=None):
    """
    Input:
        - bathy_path
        - region_borders
        - downsampling: number of points to skip between two consecutive points
    Output
        - bathymetry array containing a longitude, latitude, and eleveation arrays

    Example call:
        bathymetry = read_bathymetry(bathy_path, region_borders = [15, -36, 19, -32], downsampling=20)
        bathy_lon = bathymetry[0]
        bathy_lat = bathymetry[1]
        bathy_elev = bathymetry[2]

    Tests
        region_borders = [15, -36, 19, -32]
        bathymetry = read_bathymetry(bathy_path, region_borders, downsampling=100)
        bathymetry = read_bathymetry(bathy_path, downsampling=100)
        srtm_lon, srtm_lat, srtm_elev = bathymetry
    """

    srtm = nc.Dataset(bathy_path, 'r')

    if region_borders:
        assert( (region_borders[0] < region_borders[2]) & (region_borders[1] < region_borders[3]) ), 'Wrong format for region_border.'
        llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat = region_borders
        indx_lon = (srtm['longitude'][:] >= llcrnrlon) & (srtm['longitude'][:] <= urcrnrlon)
        indx_lat = (srtm['latitude'][:] >= llcrnrlat) & (srtm['latitude'][:] <= urcrnrlat)
        srtm_lon = srtm['longitude'][indx_lon]
        srtm_lat = srtm['latitude'][indx_lat]
        srtm_elev = srtm['elevation'][indx_lat, indx_lon]

    elif (region_borders==None) & (downsampling == None):
        downsampling = 100
        print('downsampling bathymetry by %g points' % downsampling)
        srtm_lon = srtm['longitude'][:][::downsampling]
        srtm_lat = srtm['latitude'][:][::downsampling]
        srtm_elev = srtm['elevation'][:][::downsampling,::downsampling]

    elif (region_borders==None):
        print('downsampling bathymetry by %g points' % downsampling)
        srtm_lon = srtm['longitude'][:][::downsampling]
        srtm_lat = srtm['latitude'][:][::downsampling]
        srtm_elev = srtm['elevation'][:][::downsampling,::downsampling]

    assertCoordinates(srtm_lon, srtm_lat), "Problem with srtm_lon or srtm_lat"

    return(srtm_lon, srtm_lat, srtm_elev)


def compose_date(year, month=None, day=None):
    """
    Checks the input and asserts they are in the appropriate length then returns the start and end dates accordingly.

    Unit tests:
        print(compose_date(2016) == (20160101, 20161231))
        print(compose_date(2016, 2) == (20160201, 20160231))
        print(compose_date(2016, 3, 8) == (20160308, 20160308))

    """
    # year: YYYY, month: MM, # day: DD

    if (year !=None):
        assert ((int(year) >= 2010) and (int(year) <= 2017)), 'Year is wrong!'
        year = str(year)
        assert(len(year) == 4), "Year should be quadruple (e.g. YYYY)"

    if (month != None):
        assert ((int(month) >= 1) and (int(month) <= 12)), 'Month is wrong!'
        month = str(month)
        if len(month) == 1:
            month = str(0) + month
        assert(len(month) == 2), "Month should be double (e.g. 02)"

    if (day != None):
        assert ((int(day) >= 1) and (int(day) <= 31)), 'Day is wrong!'
        day = str(day)
        if len(str(day)) == 1:
            day = str(0) + str(day)
        assert(len(day) == 2), "Day should be double (e.g. 02)"

    ######### compose the start & end dates #########
    if ((month != None) and (day != None)):
        year = str(year)
        month = str(month)
        day = str(day)
        start_date = np.int( year + month + day )
        end_date = start_date

    elif (month != None): # process Year-Month
        year = str(year)
        month = str(month)
        start_date = np.int( year + month + '01')
        end_date = np.int( year + month + '31')

    elif year: #process year
        year = str(year)
        start_date = np.int( year + '01' + '01')
        end_date = np.int( year + '12' + '31')


    assert(len(str(start_date)) == 8)
    assert(len(str(end_date)) == 8)

    return(start_date, end_date)

# # unit tests
# print(compose_date(2016) == (20160101, 20161231))
# print(compose_date(2016, 2) == (20160201, 20160231))
# print(compose_date(2016, 3, 8) == (20160308, 20160308))

def nearest_neighbour_interp(loni,lati,lon,lat, Lon):
    """
    Nearest Neighbour Interpolation
        Lon: 2D array (result of meshgrid!)
    """
    d=dist_spheric(loni,lati,lon,lat)
    return([np.unravel_index(np.argmin(d,axis=0), np.shape(Lon)),np.min(d,axis=0)])

def write_front_properties(front_properties, year, results_dir="../results"):

    # create necessary directories if they don't exist
    make_directories([results_dir], year)

    try:
        timestamp_lst, front_lon, front_lat, front_sst, front_d, front_lon_mean, front_lat_mean, \
    front_lon_std, front_lat_std, front_nb = front_properties

    except(ValueError or NameError): # if stats are not included
        timestamp_lst, front_lon, front_lat, front_sst, front_d, front_nb = front_properties
    else:
        print("I don't know what to do here. Fix me \\\///")
        raise()

    # write results to files
    # to read the results: front2_lat = np.loadtxt("../results/front2_lat.txt", delimiter=',')


    header = "columns correspond to sections; rows correspond to different days"

    map_filename = 'front%g_sst.dat' % front_nb
    map_filename = os.path.join(os.path.abspath(results_dir), str(year), map_filename)
    np.savetxt(map_filename, front_sst, delimiter='\t', header=header)

    map_filename = 'front%g_lon.dat' % front_nb
    map_filename = os.path.join(os.path.abspath(results_dir), str(year), map_filename)
    np.savetxt(map_filename, front_lon, delimiter='\t', header=header)

    map_filename = 'front%g_lat.dat' % front_nb
    map_filename = os.path.join(os.path.abspath(results_dir), str(year), map_filename)
    np.savetxt(map_filename, front_lat, delimiter='\t', header=header)

    map_filename = 'front%g_d.dat' % front_nb
    map_filename = os.path.join(os.path.abspath(results_dir), str(year), map_filename)
    np.savetxt(map_filename, front_d, delimiter='\t', header=header)

    map_filename = 'front%g_timestamp.dat' % front_nb
    map_filename = os.path.join(os.path.abspath(results_dir), str(year), map_filename)
    np.savetxt(map_filename, timestamp_lst, fmt='%.8g', delimiter='\t', header=header)


    # try:
    #     # write stats to file
    #     front_stats_arr = front_lon_mean, front_lat_mean, front_lon_std, front_lat_std
    #     header = "Basic statistics of the front:\n(columns correspond to each section)\n"
    #     header += "1. mean(lon)\n2. mean(lat)\n3. std(lon)\n4. std(lat)"
    #     header += "\nread with Numpy as: np.loadtxt('../results/front1_stats.txt', delimiter='\t')\n"
    #     np.savetxt('../results/front%g_stats.txt' %(front_nb), front_stats_arr, delimiter='\t', header=header)
    # except(NameError):
    #     pass


def mycmap(cmapfile='../data/NCV_rainbow2.rgb'):
    # cmapfile='/Users/herbette/Bob-Ibrahim/Python/NCV_rainbow2.rgb'

    import matplotlib.colors as mpcol
    ctmp1=[]
    ctmp2=[]
    ctmp3=[]
    with open(cmapfile) as f:
        next(f)
        next(f)
        for line in f:
            ctmp1.append(float(line.strip().split('  ')[0]))
            ctmp2.append(float(line.strip().split('  ')[1]))
            ctmp3.append(float(line.strip().split('  ')[2]))
    ctmp = []
    i = 0
    while i < (len(ctmp1)):
        ctmp.append(np.array([ctmp1[i]/256,ctmp2[i]/256,ctmp3[i]/256]))
        i += 1

    #ctmp=np.vstack([ctmp1,ctmp2,ctmp3])
    rainbowcm = mpcol.ListedColormap(ctmp, name='rainbowcm', N=None)
    return(rainbowcm)

def plot_SST_map(sst, lon, lat, LONGS, LATGS, altimetry_tracks, front_lon, front_lat, front_sst, front_nb, SEC, bathymetry, region_borders, timestamp, year=None, cmap=mycmap(), figs_dir="../figs/", i_ts = None, imageExt=None):
    """
    generates an SST map with the sections and main front locations overlayed.

    Input:
    ======
        - `sst`: SST of the section
        - `lon`: longitudes corresponding to the SSTs
        - `lat`: latitdes corresponding to the SSTs
        - `front_lon` and `front_lat` are outputs of front_stats()
        - `front_sst`: SSTs at front locations (used to plot corresponding isocontours)
        - `bathy_path`: path to the bathymetry file
        - `i_ts`: index to the time series of interest (i.e. row) (optional; by default uses last time series)
        - `timestamp`: timestamp extracted from the filename (Optional)


    Output:
    =======
        SST map with sections and main front position overlayed
    """
    assert np.array(front_lon).shape == np.array(front_lat).shape, "Input shapes are not consistent."

    # h,w = plt.figaspect(0.5)
    # plt.figure(figsize=((h,w)))
    plt.figure(figsize=((20,15)))

    nsec = len(SEC)

    if len(SEC) == 4:
        col = ['b', 'r', 'g', 'k']
    else:
        col = None

    srtm_lon, srtm_lat, srtm_elev = bathymetry

    try:
        llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat = region_borders

        # map center
        lon_center = np.mean((llcrnrlon, urcrnrlon))
        lat_center = np.mean((llcrnrlat, urcrnrlat))

        proj = 'tmerc' # map projection
        parallels = np.arange(llcrnrlat, urcrnrlat, 1).round()
        meridians = np.arange(llcrnrlon, urcrnrlon, 1).round()

    except:
        print("Cannot define region.")
        raise

    m = bm.Basemap(llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, projection=proj, lat_0=lat_center, lon_0=lon_center, resolution='h')
#     x_section, y_section = m(lon, lat)
#     m.pcolormesh(x_section, y_section, sst, cmap=plt.cm.rainbow)
    m.pcolormesh(lon, lat, sst, latlon=True, cmap=cmap)
    cb = m.colorbar(location='bottom', pad='5%')
    cb.set_label('SST [$^\circ$C]', fontsize=20)
    plt.clim(12.5, 24)

#     # get contour coordinates & plot
#     p = cs2.collections[0].get_paths()[0]
#     v = p.vertices
#     xc = v[:,0]
#     yc = v[:,1]
#     cs3= m(xc, yc, inverse=True)
#     m.plot(xc, yc, 'r-', markersize=5, zorder=100, label = 'contour');

    # add CP & CC to map
    CC = [17.9, -32.95] # Cape Columbine
    CP = [18.37, -34.1] # Cape Peninsula
    x,y = m(CP[0], CP[1])
    plt.text(x, y, 'CP')
    m.plot(x, y, 'ko')
    x,y = m(CC[0], CC[1])
    plt.text(x, y, 'CC')
    m.plot(x, y, 'ko')

#     ###### add isobaths to map ######
#     levels = - np.array([100, 500, 1000, 2000])[::-1] # flipped; must have them in increasing order
#     srtm_lon, srtm_lat = np.meshgrid(srtm_lon, srtm_lat)
#     srtm_lon, srtm_lat = m(srtm_lon, srtm_lat)
#     bc = plt.contour(srtm_lon, srtm_lat, srtm_elev, levels, colors='k', linewidths=0.7,latlon=False) #cmocean.cm.deep
#     plt.clabel(bc, fmt='%2.0f', fontsize=12) # labels for isobaths

    # isoSSTs
#     cs1 = m.contour(lon, lat, sst, 8, latlon=True, c='k', linestyles='dashed')
#     plt.clabel(cs1, fmt='%2.0f', fontsize=12) # labels for isoSSTs
#   ignore first SEC corresponding to along-track
    cs2 = m.contour(lon, lat, sst, levels = np.sort(front_sst[-1][0:]), latlon=True, colors='k', linestyles='dashed')
    plt.clabel(cs2, fmt='%2.0f', fontsize=12) # labels for isoSSTs
    cb.add_lines(cs2) # add isobaths to colorbar

#     if list(front_sst): # add isocontours for front for 3 main sections (blue, red, black)
#         levels = np.sort( np.array(front_sst)[-1] ) # make sure levels are ascending
#         fc = plt.contour(x_section, y_section, sst, levels, colors = 'k', latlon=False)
#         plt.clabel(fc, fmt='%2.0f', fontsize=12) # labels for isoSSTs

    try:
        m.drawcoastlines()
        m.drawcountries()
        m.drawparallels(parallels, labels = [1,0,0,0])
        m.drawmeridians(meridians, labels = [0,0,0,1])
        m.fillcontinents(color='0.8', lake_color='aqua')
    except:
        pass

    # if i_ts is not given use the last time series
    if i_ts is None:
        i_ts = -1

    # warning if sections extend beyond plotted region
    if ((np.min(front_lon[i_ts]) < lon.min()) or (np.max(front_lon[i_ts]) > lon.max()) \
    or (np.min(front_lat[i_ts]) < lat.min()) or (np.max(front_lat[i_ts]) > lat.max())):
#     if ((np.min(front_lon[i_ts,:]) < lon.min()) or (np.max(front_lon[i_ts,:]) > lon.max()) \
#     or (np.min(front_lat[i_ts,:]) < lat.min()) or (np.max(front_lat[i_ts,:]) > lat.max())):
        print('\n')
        print("\tWARNING: At least one section exceeds plotted region! Some peaks may be missing from the map!")

    # # plot sections
    # for isec in np.arange(nsec):
    #     #longs, latgs = project_section(SEC[isec], dgc)
    #     x, y = m(LONGS[isec], LATGS[isec])

    #     if col: # if color is defined (dependent on length of SEC)
    #         m.plot(x, y, linestyle='dashed', color = col[isec], alpha=0.8)
    #     else:
    #         m.plot(x, y, 'b--', alpha=0.8)

    # # plot altimetry tracks
    # for i in np.arange(len(altimetry_tracks[0])):
    #     x, y = m(altimetry_tracks[0][i], altimetry_tracks[1][i])
    #     m.plot(x, y, 'b--', markersize=1.5)

    # add time series fronts locations
    # print(front_sst[i_ts], front_lon[i_ts], front_lat[i_ts])
    x, y = m(front_lon[i_ts], front_lat[i_ts])
    m.plot(x, y, 'ro', markersize=5, zorder=100, label = 'front positions');

    # add timestamp
    # plt.text(llcrnrlon+0.10, llcrnrlat+0.10, timestamp)
    # plt.legend()
    plt.title('front %g - ' %(front_nb) + str(timestamp), fontsize=22)

    if imageExt is None:
        imageExt = 'png'
    map_filename = 'map_sst-front%g_'  % front_nb
    map_filename += str(timestamp) + '.' + imageExt

    if year:
        plt.savefig(os.path.join(os.path.abspath(figs_dir), str(year), map_filename), dpi=300, bbox_inches = 'tight')
    elif year == None:
        # climatology case
        plt.savefig(os.path.join(os.path.abspath(figs_dir), map_filename), dpi=300, bbox_inches = 'tight')


def plot_ts_map(lon_section, lat_section, sst_section, front_lon, front_lat, front_SST, bathymetry, region_borders, front_nb, front_lon_mean = None, front_lat_mean = None, front_lon_std = None, front_lat_std = None, imageExt=None):
    """
    Plots a map of the (time series) front locations
    `front_lon` and `front_lat` are outputs of front_stats()
    `bathy_path`: path to the bathymetry file

    Requires the following to be declared

    Optional:
        front_*_mean: means of front position
        front_*_std: stds of front positions
    """

    assert front_lon.shape == front_lat.shape, "front_lon and front_lat do not have the same shape."
    # assert bathy_path exists

    # create a gridSpec for map & profiles
    # plt.figure(figsize=(20,10))
    fig, axs = plt.subplots(1, 1, figsize=(10,12))

    # # min & max SST to scale y-axis for all profiles.
    # [Tmin, Tmax] = [np.floor(np.min(sst.data[0])), np.ceil(np.max(sst.data[0]))]

    # don't have access to lon_section here!!!! <<<<<<<<<<<<<<<<<<<<<<<
    try:
#         llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat = map_borders(lon_section, lat_section)
        llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat = region_borders
        assertCoordinates(lon = [llcrnrlon, urcrnrlon], lat = [llcrnrlat, urcrnrlat])

        # map center
        lon_center = np.mean((llcrnrlon, urcrnrlon))
        lat_center = np.mean((llcrnrlat, urcrnrlat))

        proj = 'tmerc' # map projection
        parallels = np.arange(llcrnrlat, urcrnrlat, 1).round()
        meridians = np.arange(llcrnrlon, urcrnrlon, 1).round()

    except:
        print("Cannot define region.")
        raise

    assert ( (llcrnrlon >= 0) & (llcrnrlon <= 360) ), "longitudes are corrupted."
    assert ( (urcrnrlon >= 0) & (urcrnrlon <= 360) ), "longitudes are corrupted."
    assert ( (llcrnrlat >= -180) & (llcrnrlat <= 180) ), "latitudes are corrupted."
    assert ( (urcrnrlat >= -180) & (urcrnrlat <= 180) ), "latitudes are corrupted."


    srtm_lon, srtm_lat, srtm_elev = bathymetry

    # map center
    lon_center = np.mean((llcrnrlon, urcrnrlon))
    lat_center = np.mean((llcrnrlat, urcrnrlat))

    m = bm.Basemap(llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, projection=proj, lat_0=lat_center, lon_0=lon_center, resolution='h')
    srtm_lon, srtm_lat = m(srtm_lon, srtm_lat)
    m.pcolormesh(srtm_lon, srtm_lat, srtm_elev, cmap=plt.cm.terrain)
    cb = m.colorbar(location='bottom', pad='5%')
    cb.set_label('Bathymetry [m]', fontsize=20)

    ###### add isobaths to map ######
    levels = - np.array([100, 200, 500, 1000, 2000, 3000, 3500, 4000])[::-1] # flipped; must have them in increasing order
    bc = plt.contour(srtm_lon, srtm_lat, srtm_elev, levels, colors='k', linewidths=0.7) #cmocean.cm.deep
    # cs = m.contour(lon_section, lat_section, sst_section, latlon=True, c='k', linestyles='dashed')
    plt.clabel(bc, fmt='%2.0f', fontsize=12) # labels for isobaths

    #   ignore first SEC corresponding to along-track
    cs2 = m.contour(lon_section, lat_section, sst_section, levels = np.sort(front_SST[-1][1:]), latlon=True, colors = 'k', linestyles='solid')
    plt.clabel(cs2, fmt='%2.0f', fontsize=12) # labels for isoSSTs

#     # create basemap to map the lon/lat so we can plot the isocontours corresponding to the front
#     m2 = bm.Basemap(llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, projection=proj, lat_0=lat_center, lon_0=lon_center, resolution='h')
#     x_section, y_section = m2(lon, lat)
#     m2.pcolormesh(x_section, y_section, sst, cmap=plt.cm.rainbow)

#     if front_SST: # add isocontours for front for 3 main sections (blue, red, black)
#         levels = np.sort( np.array(front_SST)[-1] ) # make sure levels are ascending
#         fc = plt.contour(x_section, y_section, sst, levels)

#     plt.clabel(fc, fmt='%2.0f', fontsize=12) # labels for isoSSTs

    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(parallels, labels = [1,0,0,0])
    m.drawmeridians(meridians, labels = [0,0,0,1])
    m.fillcontinents(color='0.8', lake_color='aqua')

    # warning if sections extend beyond plotted region
    if ((np.min(front_lon) < lon_section.min()) or (np.max(front_lon) > lon_section.max()) \
    or (np.min(front_lat) < lat_section.min()) or (np.max(front_lat) > lat_section.max())):
        print('\n')
        print("\tWARNING: At least one section exceeds plotted region! Some peaks may be missing from the map!")

    # add time series fronts locations
    x, y = m(front_lon, front_lat)
    plt.plot(x, y, 'ro', markersize=2.5, zorder=100, label = 'front positions');

    try: # plot front mean if means are provided
        [x, y] = m(front_lon_mean, front_lat_mean)
        m.plot(x, y, 'yo', zorder = 10000, label = 'mean positions');
    except:
        print('No mean values given.')
        pass

    # remove the multiple instances of the same label (i.e. front positions)
    handles,labels = axs.get_legend_handles_labels() #get existing legend item handles and labels
    j=np.arange(len(labels)) #make an index for later
    filter=np.array([]) #set up a filter (empty for now)
    # print([set(labels)])
    unique_labels=list(set(labels)) #find unique labels
    for ul in unique_labels: #loop through unique labels
        filter=np.append(filter,[j[np.array(labels)==ul][0]]) #find the first instance of this label and add its index to the filter
    handles=[handles[int(f)] for f in filter] #filter out legend items to keep only the first instance of each repeated label
    labels=[labels[int(f)] for f in filter]
    axs.legend(handles,labels, loc = 'lower left')
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='12')

    plt.title('Time series of the SST front %g' %(front_nb), fontsize=22)
    plt.tight_layout()

    if imageExt is None:
        imageExt = 'png'
    map_filename = '../figs/map_sst-front%g_ts' %(front_nb)
    map_filename += '.' + imageExt
    plt.savefig(map_filename, dpi=300)

def plot_bathymetry(bathymetry=None, bathy_path=None, region_borders=None, figsize=None):
    """
    Make a quick abthymetry plot
    Uses pyplot, not basemap
    """

    if bathymetry != None:

        if region_borders != None:
            srtm_lon, srtm_lat, srtm_elev = bathymetry
            assert( (region_borders[0] < region_borders[2]) & (region_borders[1] < region_borders[3]) ), 'Wrong format for region_border.'
            llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat = region_borders
            indx_lon = (srtm_lon[:] >= llcrnrlon) & (srtm_lon[:] <= urcrnrlon)
            indx_lat = (srtm_lat[:] >= llcrnrlat) & (srtm_lat[:] <= urcrnrlat)
            srtm_lon = srtm_lon[indx_lon]
            srtm_lat = srtm_lat[indx_lat]
            srtm_elev = srtm_elev[indx_lat, indx_lon]

        elif region_borders == None:
            srtm_lon, srtm_lat, srtm_elev = bathymetry


    elif (bathymetry==None) and bathy_path:
        from sstFronts import read_bathymetry
        print('Reading bathymetry...')

        if region_borders != None:
            bathymetry = read_bathymetry(bathy_path, region_borders=region_borders)
        elif region_borders == None:
            bathymetry = read_bathymetry(bathy_path)

        srtm_lon, srtm_lat, srtm_elev = bathymetry

    else:
        print('Either bathymetry or bathy_path must be used.')


    from matplotlib import pyplot as plt
    plt.figure(figsize=figsize)
    cs = plt.pcolormesh(srtm_lon, srtm_lat, srtm_elev, cmap = plt.cm.terrain, rasterized=True)
    cbar = plt.colorbar(cs, orientation='vertical', extend='both')
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_xlabel('meters',fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show()

    plt.savefig('../figs/bathymetry.png', dpi=300)
    return(bathymetry)

def plot_study_area(bathy_path, region_borders=None, study_area = None, figsize=None):
    """
    Make a quick bathymetry map of the study area
    """
    from sstFronts import read_bathymetry
    from matplotlib import pyplot as plt
    import numpy as np

    if region_borders == None and study_area != None:

        # assert( (study_area[0] < study_area[2]) & (study_area[1] < study_area[3]) ), 'Wrong format for region_border.'
        assert( study_area[:2] < study_area[2:] ), 'Wrong format for region_border.'

        bathymetry = read_bathymetry(bathy_path, region_borders=study_area)
        srtm_lon, srtm_lat, srtm_elev = bathymetry

        plt.figure(figsize=figsize)
        cs = plt.pcolormesh(srtm_lon, srtm_lat, srtm_elev, cmap = plt.cm.terrain, rasterized=True)
        cbar = plt.colorbar(cs, orientation='vertical', extend='both')
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.set_xlabel('meters',fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)

        ##### add isobaths to map ######
        levels = - np.array([100, 500, 1000, 2000])[::-1] # flipped; must have them in increasing order
        srtm_lon, srtm_lat = np.meshgrid(srtm_lon, srtm_lat)
    #         srtm_lon, srtm_lat = m(srtm_lon, srtm_lat)
        bc = plt.contour(srtm_lon, srtm_lat, srtm_elev, levels, colors='k', linewidths=0.7,latlon=False) #cmocean.cm.deep
        plt.clabel(bc, fmt='%2.0f', fontsize=12) # labels for isobaths
        map_filename = 'bathymetry_study-area'

    elif region_borders != None and study_area != None:

        # assert( (region_borders[0] < region_borders[2]) & (region_borders[1] < region_borders[3]) ), 'Wrong format for region_border.'
        # assert( (study_area[0] < study_area[2]) & (study_area[1] < study_area[3]) ), 'Wrong format for region_border.'
        # assert( (study_area < region_borders) ), 'Wrong format for region_border.'
        assert( region_borders[:2] < region_borders[2:] ), 'Wrong format for region_border.'
        assert( study_area[:2] < study_area[2:] ), 'Wrong format for region_border.'
        assert( (region_borders[:2] < study_area[:2]) and (region_borders[2:] > study_area[2:]) ), 'Wrong format for region_border.'

        bathymetry = read_bathymetry(bathy_path, region_borders=region_borders)
        srtm_lon, srtm_lat, srtm_elev = bathymetry

        bathymetry = read_bathymetry(bathy_path, region_borders=region_borders)
        srtm_lon, srtm_lat, srtm_elev = bathymetry

        plt.figure(figsize=figsize)
        cs = plt.pcolormesh(srtm_lon, srtm_lat, srtm_elev, cmap = plt.cm.terrain, rasterized=True)
        cbar = plt.colorbar(cs, orientation='vertical', extend='both')
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.set_xlabel('meters',fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)

        llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat = study_area
        plt.hlines(llcrnrlat, llcrnrlon, urcrnrlon, linestyles = "dashed")
        plt.hlines(urcrnrlat, llcrnrlon, urcrnrlon, linestyles = "dashed")
        plt.vlines(llcrnrlon, llcrnrlat, urcrnrlat, linestyles = "dashed")
        plt.vlines(urcrnrlon, llcrnrlat, urcrnrlat, linestyles = "dashed")
        map_filename = 'bathymetry_region'

    elif region_borders == None and study_area == None:
        # plot whole map

        bathymetry = read_bathymetry(bathy_path)
        srtm_lon, srtm_lat, srtm_elev = bathymetry

        plt.figure(figsize=figsize)
        cs = plt.pcolormesh(srtm_lon, srtm_lat, srtm_elev, cmap = plt.cm.terrain, rasterized=True)
        cbar = plt.colorbar(cs, orientation='vertical', extend='both')
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.set_xlabel('meters',fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        map_filename = 'bathymetry_global'


    else:
        print('Input error. Need to define either or both of region_borders and study_region')

    plt.savefig('../figs/' + map_filename + '.png', bbox_inches='tight', dpi=300)

    return(bathymetry)


def sst_fronts(sst, lon, lat, LONGS, LATGS, SSTGS, SEC, timestamp, front_nb = 1, verbose = 0):
    """
    finds the following about the main SST fronts:
        - location: d along section, lat & lon
        - SST, delta d, detla SST, & grad SST values
        - relative delta SST

    ToDo:
        - if threshold is low no intersections are found and indx is empty or of length 1. Need to take care of this. For example: file 20160707 with threshold = 0; no fronts detected for section 17.
        - more work is needed to differentiate the 1st & 2nd fronts. Sometimes the first front is further offshore in one SST image than the 2nd front in another SSt image.
    """
    assert verbose in [0, 1, 2, 3, 'fig'], "Wrong verbose choice."

    # lists to preserve indices, lon & lat of peaks
    lon_peaks = []
    lat_peaks = []
    idx_tot = [] # list for arrays of peak indices (in longs, latgs) for each section

    col = ['b', 'r', 'g', 'k']
    nsec=len(LONGS)
    if verbose == 'fig' or verbose == 3: fig = plt.figure(figsize=(12,6*nsec))

    # loop for each section
    delta_sst_tmp = []
    delta_d_tmp = []
    reference_delta_sst_tmp = []
    front_delta_sst_tmp = []
    front_d_tmp = [] # temporary list for (along section) distance of the front for current time series
    front_sst_grad_tmp = []
    front_sst_tmp = []
    front_lon_tmp = []
    front_lat_tmp = []

    #interpolation

    for isec in np.arange(nsec):
#     for isec in np.arange(1):

        if verbose == 1: print('section # ', isec+1)

        #along_section_sst, latgs, longs, at_d = interpole_section(SEC[isec], dgc, sst, lon, lat)
        along_section_sst=SSTGS[isec]
        latgs=LATGS[isec]
        longs=LONGS[isec]
        # print(np.shape(longs))
        n = longs.shape[0]
        at_d = np.zeros(n)
        for ix in np.arange(n):
            at_d[ix] = dist_spheric(latgs[ix], longs[ix],latgs[0],longs[0])

        # calculate SST gradient & subtract threshold
        sst_grad_threshold = 0.00002

        sst_grad = np.gradient(along_section_sst, at_d) - sst_grad_threshold
        # print('sst_grad =',sst_grad)
        # print('sst_sec =',along_section_sst)
        # find points forming a line that crosses the x-axis
        indx = []

        # take first point as begining of first interval if grad is increasing & positive
        if (sst_grad[0] * sst_grad[1] > 0) & (sst_grad[0] > 0) & (sst_grad[0] < sst_grad[1]): # if grad doesn't intersect threshold
            indx += [0] # add the index of the first point
            flag = 'odd' # flag to calculate the intersects (mainly to know how to subset)
            if verbose == 2: print('\tselecting first point!')
        else:
            flag = 'even'

        # get the remaining points defining the peaks' intervals
        for ix in np.arange(len(sst_grad)-1):

            if len(indx) == 1:
                if sst_grad[ix] * sst_grad[ix+1] < 0:
                    indx += [ix, ix+1]
                    if verbose == 2: print('\tselecting its pair')

            else:
                if sst_grad[ix] * sst_grad[ix+1] < 0:
                    indx += [ix, ix+1]
                    if verbose == 2: print('\tselecting paired points.')

        if verbose == 1 or verbose == 2: print('\tindx = ', indx)

        # calculate intersection points for pairs of points excluding first pair
        if flag == 'odd':
            intersects_rest = find_intersects(at_d, sst_grad, indx[1:])
            intersects = np.zeros((2, intersects_rest.shape[1]+1))
            intersects[:,0] = np.array([ at_d[ indx[0] ], sst_grad[ indx[0] ] ]) # add the first pair (not intersection points!)
            intersects[:,1:] = intersects_rest
        else:
            intersects = find_intersects(at_d, sst_grad, indx)

        if verbose == 2: print('\t',intersects)

        # interpolate SST at intersection points
        f = interpolate.interp1d(x = at_d, y = along_section_sst, kind = 'linear') # construct interpolation fn
        d_new = intersects[0,:] # along_section distance for interesects
        sst_new = f(d_new) # interpolate SST at intersection points (i.e. at threshold)
#         flat = interpolate.interp1d(x = at_d, y = latgs[0], kind = 'linear') # interpolation fun for latgs
#         flon = interpolate.interp1d(x = at_d, y = longs[0], kind = 'linear') # interpolation fun for longs
#         latgs_new = flat(d_new)
#         longs_new = flon(d_new)

        # calculate delta_sst
        delta_sst_tmp += [np.diff(sst_new)]
        delta_d_tmp += [np.diff(d_new)]
        if verbose == 2: print('\tdelta_sst_tmp:', delta_sst_tmp)
        if verbose == 2: print('\tdelta sst: ', delta_sst)

        # Reference delta SST <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#         sst_min  = along_section_sst.data[0].min() # min(interpolated SST along section)
#         sst_max  = along_section_sst.data[0].max() # max(interpolated SST along section)
        sst_min  = sst.min() # min(interpolated SST along section)
        sst_max  = sst.max() # max(interpolated SST along section)
        reference_delta_sst_tmp += [sst_max - sst_min]

#        try:
#            # Maximum normalized delta SST
#            front_delta_sst_tmp += [np.max(delta_sst_tmp[-1] / reference_delta_sst_tmp[-1])]
#        except (ValueError):
#            print('Warning: No fronts detected for transect # %g. Adding a NaN' %isec)
#            front_delta_sst_tmp += [np.nan]
#            pass

        # avoid haulting of the algorithm due to no fronts being detected
        if len(delta_sst_tmp[-1]) > 1:

            # Maximum normalized delta SST
            front_delta_sst_tmp += [np.max(delta_sst_tmp[-1] / reference_delta_sst_tmp[-1])]

            # find the distance (along section) of the maximum grad SST corresponding to the maximum delta SST
            normalized_delta_sst_tmp = delta_sst_tmp[-1] / reference_delta_sst_tmp[-1]

            if front_nb == 1:
                kmax = np.argmax( normalized_delta_sst_tmp ) # index of maximum delta SST

            elif front_nb == 2:
    #             if len(normalized_delta_sst_tmp) >= 2:
                max2 = np.sort( normalized_delta_sst_tmp )[-2] # 2nd maximum
                kmax = np.argwhere( normalized_delta_sst_tmp == max2 )
    #             else: #no 2nd front

                if len(kmax) == 1:
                    kmax = kmax[0][0]
                    if verbose == 3: print(np.array(sst_grad)[kmax])
                elif len(kmax) > 1:
                    print('Multiple 2nd maximums. What should I do?')
                elif len(kmax) < 1:
                    print('No maximum found. Empty list?')

    # #         elif front_nb = 'both':
    # #             # recursion
    # #             myfunc(, front_nb = 1)
    # #             myfunc(, front_nb = 2)

    #         else:
    #             print('Wrong front number!')

            if verbose == 3: print(kmax)


            # print('at_d',at_d)
            # print('d_new',d_new)
            # print('along_sec_sst',along_section_sst)
            # print('sstgrad',sst_grad)
            # find d, SST, lon, & lat of front
            cond = (at_d < d_new[kmax+1]) & (at_d > d_new[kmax]) # boolean coresponding to the maximum delta SST
            # print('cond',cond)
            igradmax = np.argmax( sst_grad[cond] ) # index of the maximum grad SST
            front_d_tmp += [at_d[cond][igradmax]] # distance (along section) of the maximum grad SST correstssponding to the maximum delta SST
            front_sst_grad_tmp += [sst_grad[cond][igradmax]] # distance (along section) of the maximum grad SST corresponding to the maximum delta SST
            front_sst_tmp += [along_section_sst[cond][igradmax]] # find SSTs at front locations <<<<<<<<<<<<<<

            # find the lon/lat of the peak for the current section
            front_lon_tmp += [longs[cond][igradmax]]
            front_lat_tmp += [latgs[cond][igradmax]]

        if verbose == 3: print(front_lon_tmp[-1], front_lat_tmp[-1])

        if verbose == 3: print('d of front', front_d_tmp)

        if verbose == 2:
            print()
            print('\tMax. normalized delta SST = ', front_delta_sst)
            print()

        # plot SST, interpolated SSTs, & front location for validation
        if verbose == 'fig' or verbose == 3:
            print('\tplotting SST & Gradient profiles...')
            ax1 = plt.subplot(nsec, 1, isec+1)
            ax1.plot(longs, along_section_sst, 'k.-')
            ax1.plot(front_lon_tmp[-1], front_sst_tmp[-1], 'ro')
            plt.gca().invert_xaxis()
            ax2 = plt.twinx()
            ax2.plot(longs, sst_grad, marker='.', markerfacecolor='darkblue')
            ax2.plot(front_lon_tmp[-1], front_sst_grad_tmp[-1], 'ro')
            ax3 = plt.twiny()
            ax3.set_xlim(latgs[0], latgs[-1]);

            ax1.set_xlabel('Lon', fontsize=14)
            ax2.set_xlabel('Lat', fontsize=14)
            ax1.set_ylabel('SST', fontsize=14)
            ax2.set_ylabel('$\\vec{\\nabla}$ SST', fontsize=14)
            ax1.yaxis.label.set_color('black')
            ax2.yaxis.label.set_color('darkblue')
            ax1.tick_params(axis='y', colors = 'black')
            ax2.tick_params(axis='y', colors = 'darkblue')
            plt.title(isec+1, fontsize=22)


#             # add labels for pl2 & pl4 (only those have labels)
#             lns = pl2+pl5
#             labs = [l.get_label() for l in lns]
#             ax1.legend(lns, labs, loc=0)
            profiles_dir = os.path.join("../figs", 'profiles')
            if not os.path.exists(profiles_dir):
                os.makedirs(profiles_dir)
            profile_filename = os.path.join(profiles_dir, 'SST-grad_sections_front%g_' %(front_nb) + str(timestamp) + '.png') # overplots (if several time series!)
            print(profile_filename)
            plt.savefig(profile_filename)

        if verbose == 3: print(delta_sst_tmp)

    return(delta_d_tmp, delta_sst_tmp, reference_delta_sst_tmp, front_delta_sst_tmp, front_d_tmp, front_sst_grad_tmp, front_lon_tmp, front_lat_tmp, front_sst_tmp)

def sst_fronts_old(sst, lon, lat, SEC, dgc, LONGS, LATGS, elem, coef, front_nb = 1, verbose = 0):
    """
    finds the following about the main SST fronts:
        - location: d along section, lat & lon
        - SST, delta d, detla SST, & grad SST values
        - relative delta SST

    ToDo:
        - more work is needed to differentiate the 1st & 2nd fronts. Sometimes the first front is further offshore in one SST image than the 2nd front in another SSt image.
    """
    assert verbose in [0, 1, 2, 3, 'fig'], "Wrong verbose choice."

    # lists to preserve indices, lon & lat of peaks
    lon_peaks = []
    lat_peaks = []
    idx_tot = [] # list for arrays of peak indices (in longs, latgs) for each section

    nsec = len(SEC)
    col = ['b', 'r', 'g', 'k']

    if verbose == 'fig' or verbose == 3: fig = plt.figure(figsize=(12,6*nsec))

    # loop for each section
    delta_sst_tmp = []
    delta_d_tmp = []
    reference_delta_sst_tmp = []
    front_delta_sst_tmp = []
    front_d_tmp = [] # temporary list for (along section) distance of the front for current time series
    front_sst_grad_tmp = []
    front_sst_tmp = []
    front_lon_tmp = []
    front_lat_tmp = []

    #interpolation


    for isec in np.arange(nsec):
#     for isec in np.arange(1):

        print('section # ', isec+1)

        #along_section_sst, latgs, longs, at_d = interpole_section(SEC[isec], dgc, sst, lon, lat)
        along_section_sst=sstdom[isec,:]
        latgs=londom, longs, at_d

        # calculate SST gradient & subtract threshold
        sst_grad_threshold = 0.

        sst_grad = np.gradient(along_section_sst.data[0], at_d) - sst_grad_threshold

        # find points forming a line that crosses the x-axis
        indx = []

        # take first point as begining of first interval if grad is increasing & positive
        if (sst_grad[0] * sst_grad[1] > 0) & (sst_grad[0] > 0) & (sst_grad[0] < sst_grad[1]): # if grad doesn't intersect threshold
            indx += [0] # add the index of the first point
            flag = 'odd' # flag to calculate the intersects (mainly to know how to subset)
            if verbose == 2: print('\tselecting first point!')
        else:
            flag = 'even'

        # get the remaining points defining the peaks' intervals
        for ix in np.arange(len(sst_grad)-1):

            if len(indx) == 1:
                if sst_grad[ix] * sst_grad[ix+1] < 0:
                    indx += [ix, ix+1]
                    if verbose == 2: print('\tselecting its pair')

            else:
                if sst_grad[ix] * sst_grad[ix+1] < 0:
                    indx += [ix, ix+1]
                    if verbose == 2: print('\tselecting paired points.')

        if verbose == 1 or verbose == 2: print('\tindx = ', indx)

        # calculate intersection points for pairs of points excluding first pair
        if flag == 'odd':
            intersects_rest = find_intersects(at_d, sst_grad, indx[1:])
            intersects = np.zeros((2, intersects_rest.shape[1]+1))
            intersects[:,0] = np.array([ at_d[ indx[0] ], sst_grad[ indx[0] ] ]) # add the first pair (not intersection points!)
            intersects[:,1:] = intersects_rest
        else:
            intersects = find_intersects(at_d, sst_grad, indx)

        if verbose == 2: print('\t',intersects)

        # interpolate SST at intersection points
        f = interpolate.interp1d(x = at_d, y = along_section_sst.data[0], kind = 'linear') # construct interpolation fn
        d_new = intersects[0,:] # along_section distance for interesects
        sst_new = f(d_new) # interpolate SST at intersection points (i.e. at threshold)
#         flat = interpolate.interp1d(x = at_d, y = latgs[0], kind = 'linear') # interpolation fun for latgs
#         flon = interpolate.interp1d(x = at_d, y = longs[0], kind = 'linear') # interpolation fun for longs
#         latgs_new = flat(d_new)
#         longs_new = flon(d_new)

        # calculate delta_sst
        delta_sst_tmp += [np.diff(sst_new)]
        delta_d_tmp += [np.diff(d_new)]
        if verbose == 2: print('\tdelta_sst_tmp:', delta_sst_tmp)
        if verbose == 2: print('\tdelta sst: ', delta_sst)

        # Reference delta SST <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#         sst_min  = along_section_sst.data[0].min() # min(interpolated SST along section)
#         sst_max  = along_section_sst.data[0].max() # max(interpolated SST along section)
        sst_min  = sst.data[0].min() # min(interpolated SST along section)
        sst_max  = sst.data[0].max() # max(interpolated SST along section)
        reference_delta_sst_tmp += [sst_max - sst_min]

        # Maximum normalized delta SST
        front_delta_sst_tmp += [np.max(delta_sst_tmp[-1] / reference_delta_sst_tmp[-1])]

        # find the distance (along section) of the maximum grad SST corresponding to the maximum delta SST
        normalized_delta_sst_tmp = delta_sst_tmp[-1] / reference_delta_sst_tmp[-1]

        if front_nb == 1:
            kmax = np.argmax( normalized_delta_sst_tmp ) # index of maximum delta SST

        elif front_nb == 2:
#             if len(normalized_delta_sst_tmp) >= 2:
            max2 = np.sort( normalized_delta_sst_tmp )[-2] # 2nd maximum
            kmax = np.argwhere( normalized_delta_sst_tmp == max2 )
#             else: #no 2nd front

            if len(kmax) == 1:
                kmax = kmax[0][0]
                if verbose == 3: print(np.array(sst_grad)[kmax])
            elif len(kmax) > 1:
                print('Multiple 2nd maximums. What should I do?')
            elif len(kmax) < 1:
                print('No maximum found. Empty list?')

# #         elif front_nb = 'both':
# #             # recursion
# #             myfunc(, front_nb = 1)
# #             myfunc(, front_nb = 2)

#         else:
#             print('Wrong front number!')

        if verbose == 3: print(kmax)

        # find d, SST, lon, & lat of front
        cond = (at_d < d_new[kmax+1]) & (at_d > d_new[kmax]) # boolean coresponding to the maximum delta SST
        igradmax = np.argmax( sst_grad[cond] ) # index of the maximum grad SST
        front_d_tmp += [at_d[cond][igradmax]] # distance (along section) of the maximum grad SST correstssponding to the maximum delta SST
        front_sst_grad_tmp += [sst_grad[cond][igradmax]] # distance (along section) of the maximum grad SST corresponding to the maximum delta SST
        front_sst_tmp += [along_section_sst.data[0][cond][igradmax]] # find SSTs at front locations <<<<<<<<<<<<<<

        # find the lon/lat of the peak for the current section
        front_lon_tmp += [longs[0][cond][igradmax]]
        front_lat_tmp += [latgs[0][cond][igradmax]]
        if verbose == 3: print(front_lon_tmp[-1], front_lat_tmp[-1])

        if verbose == 3: print('d of front', front_d_tmp)

        if verbose == 2:
            print()
            print('\tMax. normalized delta SST = ', front_delta_sst)
            print()

        # plot SST, interpolated SSTs, & front location for validation
        if verbose == 'fig' or verbose == 3:
            print('\tplotting...')
            ax1 = plt.subplot(nsec, 1, isec+1)
            ax1.plot(longs[0], along_section_sst.data[0], 'k.-')
            ax1.plot(front_lon_tmp[-1], front_sst_tmp[-1], 'ro')
            plt.gca().invert_xaxis()
            ax2 = plt.twinx()
            ax2.plot(longs[0], sst_grad, marker='.', markerfacecolor='darkblue')
            ax2.plot(front_lon_tmp[-1], front_sst_grad_tmp[-1], 'ro')
            ax3 = plt.twiny()
            ax3.set_xlim(latgs[0][0], latgs[0][-1]);

            ax1.set_xlabel('Lon', fontsize=14)
            ax2.set_xlabel('Lat', fontsize=14)
            ax1.set_ylabel('SST', fontsize=14)
            ax2.set_ylabel('$\\vec{\\nabla}$ SST', fontsize=14)
            ax1.yaxis.label.set_color('black')
            ax2.yaxis.label.set_color('darkblue')
            ax1.tick_params(axis='y', colors = 'black')
            ax2.tick_params(axis='y', colors = 'darkblue')


#             # add labels for pl2 & pl4 (only those have labels)
#             lns = pl2+pl5
#             labs = [l.get_label() for l in lns]
#             ax1.legend(lns, labs, loc=0)
            profile_filename = '../figs/SST-grad_sections_front%g_' %(front_nb) + str(timestamp) + '.png'# overplots (if several time series!)
            plt.savefig(profile_filename)

        if verbose == 3: print(delta_sst_tmp)

    return(delta_d_tmp, delta_sst_tmp, reference_delta_sst_tmp, front_delta_sst_tmp, front_d_tmp, front_sst_grad_tmp, front_lon_tmp, front_lat_tmp, front_sst_tmp)

def make_directories(directories_list=None, year=None):
    """
    Creates necessary directories for each year
    """

    if not directories_list:
        directories_list = ["../figs", "../results"]
        # print(directories_list)

    if directories_list:
        # directories: results & figs
        for directory in directories_list:
            if not os.path.exists(directory):
                print('creating %s' %directory)
                os.makedirs(directory)


    if year: # directories for year
        for directory in directories_list:
            directory = os.path.join(directory, str(year))
            if not os.path.exists(directory):
                print('creating %s' %directory)
                os.makedirs(directory)

def main(sst_dir, bathymetry, sarpline, year, month=None, day=None, region_borders=None, front_nb = 1, figs_dir="../figs", verbose=None):
    """
    Input:
        - sst_dir: directory containing all (years of) data
        - bathy_path: path to the bathymetry netCDF file
        - year: numeric
        - month, day: numeric (optional; if not given whole year is processed)
        - section: list of length 4 for section min & max lon & lat in the form: [lon1, lon2, lat1, lat2]
        - front_nb: front number (1 or 2)

    Calls several necessary functions to:
        - loop over each file and read the data
        - finds the requested (1st or 2nd) front
        - plots an SST map and bathymetry map with the time series & mean of the front under ../figs
        - calculates statistics
        - writes data to files under ../results


    run only once:
    - read bathymetry & define region borders
    - your sections [londom, latdom]
    - the coordinates on the first and last sea point of your sections
    - the interpolation coefficients that you will need to interpolate every sst image on these sections

    loop on all files of sst.
    - read the sst data
    - interpolate the sst data on the sections using the interpolation coefficients calculated in the preliminary
    step (outside the loop)
    - applies your metres to detect the front position
    """

    ####### assert year, month, & day are in range and have appropraite length
    SEC, longsbob, latgsbob, sstdomma, landmaskdom2, londom, ixbsec, ixesec, elem, coef, altimetry_tracks = define_sections(sst_dir, sarpline)

#     year = 2016
#     month = 1
#     day = 1
#     front_nb=1
#     # region_borders = None
#     region_borders = [Lon.min(), Lat.min(), Lon.max(), Lat.max()]
#     verbose = 0
#     dgc = 5e3

    start_date, end_date = compose_date(year, month, day)

    delta_d = []
    delta_sst = []
    reference_delta_sst = []
    front_delta_sst = []
    front_d = [] # for (along section) distance of the main front
    front_sst_grad = []
    front_sst = []
    front_lon = []
    front_lat = []
    timestamp_lst = []

    import sys
    import socket
    host = socket.gethostname();
    from pathlib import Path

    nfile = 0 # keeps track of files processed


    pathlist = sorted(Path(sst_dir).glob('**/**/*SAF*.nc'))

    #     # number of files to process
    #     nfiles = 0
    #     for path in pathlist:
    #         nfiles+=1
    #     print('Number of files to process: %g' %(nfiles)); print()

    global timestamp

    for path in pathlist:
        # because path is object not string
        nc_path = str(path)
        absolute_nc_path = os.path.abspath(nc_path)

        # subset date from filename according to the sst_dir

        timestamp = np.int(absolute_nc_path[-59:-51])

        if (timestamp >= start_date) & (timestamp <= end_date):
    #         print(timestamp, '\t files remaining: ', nfiles - nfile)
    #        print()
            print(timestamp)

        # read data
        #         print('Day %g/%g\n=======' %(i,nfiles))
            lon, lat, sst, landmask = read_SST(nc_path)
#             print(nc_path)
            # subset a section to zoom in on area of interest
            lon, lat = np.meshgrid(lon, lat) # create meshgrid

            # subset a region to zoom in on area of interest
            if region_borders:
                min_region_lon, max_region_lon = region_borders[::2]
                min_region_lat, max_region_lat = region_borders[1::2]
            else:
                min_region_lon = np.min(lon)
                max_region_lon = np.max(lon)
                min_region_lat = np.min(lat)
                max_region_lat = np.max(lat)
                region_borders = [min_region_lon, min_region_lat, max_region_lon, max_region_lat]

#             print(region_borders)

            # # assert the min & max of the SSTs region is the same or smaller than the bahymetry data given
            # assert(min_region_lon >= bathymetry.)

            lon_ind = np.where((lon[0,:]>= min_region_lon) & (lon[0,:] <= max_region_lon))
            lat_ind = np.where((lat[:,0] >= min_region_lat) & (lat[:,0] <= max_region_lat))
            lon_region = lon[np.min(lat_ind):np.max(lat_ind), np.min(lon_ind):np.max(lon_ind)]
            lat_region = lat[np.min(lat_ind):np.max(lat_ind), np.min(lon_ind):np.max(lon_ind)]
            sst_region = sst[np.min(lat_ind):np.max(lat_ind), np.min(lon_ind):np.max(lon_ind)]

            # might be worth:
            # requires modifying the two functions
            # # do array operation in main()
            # longs, latgs = project_section(sec, dgc)
            # # do loop over sections in sst_fronts()
            # along_section_sst, latgs, longs, at_d = interpole_section(sec, dgc, sst, lon, lat):

            sstdom = np.sum(coef*sst.data[:,:].ravel()[elem],2)
            sstdomma = np.ma.masked_where((landmaskdom2==2)|(np.isnan(sstdom))|(sstdom<=0),sstdom) #does not work well
            SSTGS=[]
            LONGS=longsbob
            LATGS=latgsbob
            for j in range(np.shape(londom)[0]):
                SSTGS[j:j]=[sstdomma[j,ixbsec[j]:ixesec[j]]]

             # call sst_fronts() to find main front
            delta_d_tmp, delta_sst_tmp, reference_delta_sst_tmp, front_delta_sst_tmp, front_d_tmp, \
            front_sst_grad_tmp, front_lon_tmp, front_lat_tmp, front_sst_tmp = \
            sst_fronts(sst_region, lon_region, lat_region, LONGS, LATGS, SSTGS, SEC, timestamp, front_nb=1)

            # stack arrays of front properties (each row corresponds to a time series)
            delta_d += [delta_d_tmp]
            delta_sst += [delta_sst_tmp]
            reference_delta_sst += [reference_delta_sst_tmp]
            front_delta_sst += [front_delta_sst_tmp]
            front_d += [front_d_tmp]
            front_sst_grad += [front_sst_grad_tmp]
            front_sst += [front_sst_tmp]
            front_lon += [front_lon_tmp]
            front_lat += [front_lat_tmp]
            timestamp_lst += [timestamp]

            # plot & save SST map #<<<<<<<< recheck which front to take
            plot_SST_map(sst_region, lon_region, lat_region, LONGS, LATGS, altimetry_tracks, front_lon, front_lat, front_sst, front_nb, SEC, bathymetry, region_borders, timestamp, year, cmap = mycmap(), figs_dir=figs_dir)

        else:
    #             print('timestamp problem. Timestamp:', timestamp)
            pass

    # convert list of arrays to an array of arrays
    delta_d = np.array(delta_d)
    delta_sst = np.array(delta_sst)
    reference_delta_sst = np.array(reference_delta_sst)
    front_delta_sst = np.array(front_delta_sst)
    front_d = np.array(front_d)
    front_sst_grad = np.array(front_sst_grad)
    front_sst = np.array(front_sst)
    front_lon = np.array(front_lon)
    front_lat = np.array(front_lat)
    timestamp = np.array(timestamp_lst)
    assert((front_lon.shape != (0,)) and front_lat.shape != (0,)), "Empty front lon & lat arrays."

    # pad the arrays to be able to use np.savetxt to write them (need arrays)
    front_lon = pad_an_array(front_lon)
    front_lat = pad_an_array(front_lat)
    front_sst = pad_an_array(front_sst)
    front_d = pad_an_array(front_d)

    #     print("delta_d: ", type(delta_d), np.shape(delta_d))
    #     print("delta_d[-1]: ", type(delta_d[-1]), np.shape(delta_d[-1]))

    if verbose == 'final results': print(front_lon.shape, front_lat.shape)
    #     ########### Stats ###########
    #     # calculate front statistics
    #     front_lon, front_lat, front_lon_mean, front_lat_mean, front_lon_std, front_lat_std = front_stats(front_lon, front_lat)

    # write properties to files
    try:
        # if stats have been computed
        front_properties = timestamp_lst, front_lon, front_lat, front_sst, front_d, front_lon_mean, front_lat_mean, front_lon_std, front_lat_std, front_nb
    except NameError:
        # else if stats have not been computed
        front_properties = timestamp_lst, front_lon, front_lat, front_sst, front_d, front_nb
    finally:
        # write the properties
        write_front_properties(front_properties, year)
        print('Writing front properties to files...')

    # #     ########### plot & save bathymetry map with  ###########
    # #     plot_ts_map(lon_region, lat_region, sst_region, front_lon, front_lat, front_sst, bathy_path, front_nb, front_lon_mean, front_lat_mean)
    # #     plot_ts_map(lon_region, lat_region, sst_region, front_lon, front_lat, front_sst, bathymetry, region_borders, front_nb)

#     # temporary return to re-develop the panel plot function. remove afterwards. No need for it.
#     return(lon_region, lat_region, sst_region, front_lon, front_lat, front_d, front_sst, timestamp)

def run_all_years(sst_dir, figs_dir = "../figs/", results_dir="../results"):
    """
    Create a folder for each year and runs the main() function

    ToDo: add results directory (like figs!)
    """


    # bathy_path =  "/net/leo/local/tmp/1/herbette/SRTM/SRTM30_0_360_new.nc"
    bathy_path =  "../data/SRTM/SRTM30_0_360_new.nc"
    sarpline= "../data/along-track_altimetry/Africana203_SARP.txt"
    bathymetry = read_bathymetry(bathy_path)

    directories_list = [figs_dir, results_dir]
    make_directories(directories_list)

    global year

    for year in os.listdir(sst_dir):

        make_directories(year=year)

        main(sst_dir, bathymetry, sarpline, year=year)

def plot_climatology_sst_map(sst,
                             lon,
                             lat,
                             LONGS,
                             LATGS,
                             altimetry_tracks,
                             front_lon,
                             front_lat,
                             front_sst,
                             front_nb,
                             SEC,
                             bathymetry,
                             region_borders,
                             dates,
                             isobaths=True,
                             cmap=None,
                             colorbar_range=None,
                             figs_dir="../figs/"):
    """
    generates an SST map with the sections and main front locations overlayed.

    Input:
    ======
        - `sst`: SST of the section
        - `lon`: longitudes corresponding to the SSTs
        - `lat`: latitdes corresponding to the SSTs
        - `LONGS` & `LATGS`: Longitudes & Latitudes of the corresponding sections (start & end points) given by define_sections()
        - `altimetry_tracks`: array of 2D arrays. Each 2D array corresponds to the lon/lat of one altimetry track given by define_sections().
        - `front_lon` and `front_lat`: longitudes & latitudes of the fronts
        - `front_sst`: SSTs at front locations (used to plot corresponding isocontours)
        - `dates`: [start_date, end_date, years] passed by the climatology() function

    Output:
    =======
        SST map with sections and main front position overlayed
    """

    assert np.array(front_lon).shape == np.array(front_lat).shape, "Input shapes are not consistent."

    start_date, end_date, years = dates
    start_year, end_year = years[0], years[-1]

    plt.figure(figsize=((20, 15)))

    nsec = len(SEC)

    if len(SEC) == 4:
        col = ['b', 'r', 'g', 'k']
    else:
        col = None

    srtm_lon, srtm_lat, srtm_elev = bathymetry

    try:
        llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat = region_borders

        # map center
        lon_center = np.mean((llcrnrlon, urcrnrlon))
        lat_center = np.mean((llcrnrlat, urcrnrlat))

        proj = 'tmerc'  # map projection
        parallels = np.arange(llcrnrlat, urcrnrlat, 1).round()
        meridians = np.arange(llcrnrlon, urcrnrlon, 1).round()

    except:
        print("Cannot define region.")
        raise

    m = bm.Basemap(
        llcrnrlon,
        llcrnrlat,
        urcrnrlon,
        urcrnrlat,
        projection=proj,
        lat_0=lat_center,
        lon_0=lon_center,
        resolution='h')
    x_section, y_section = m(lon, lat)
    m.pcolormesh(lon, lat, sst, latlon=True, cmap=cmap)
    cb = m.colorbar(location='bottom', pad='5%')
    min_sst, max_sst = colorbar_range
    plt.clim(min_sst, max_sst)
    cb.set_label('SST [$^\circ$C]', fontsize=20)

    #     # get contour coordinates & plot
    #     p = cs2.collections[0].get_paths()[0]
    #     v = p.vertices
    #     xc = v[:,0]
    #     yc = v[:,1]
    #     cs3= m(xc, yc, inverse=True)
    #     m.plot(xc, yc, 'r-', markersize=5, zorder=100, label = 'contour');

    # add CP & CC to map
    CC = [17.9, -32.95]  # Cape Columbine
    CC = [18.00, -32.85]  # Cape Columbine
    CP = [18.5, -34]  # Cape Peninsula
    x, y = m(CC[0], CC[1])
    m.plot(x, y, 'ko')
    x, y = m(CC[0] + 0.07, CC[1] - 0.1)
    plt.text(x, y, 'CC', fontsize=14)
    x, y = m(CP[0], CP[1])
    m.plot(x, y, 'ko')
    x, y = m(CP[0] + 0.05, CP[1] - 0.)
    plt.text(x, y, 'CP', fontsize=14)

    ###### add isobaths to map ######
    if isobaths == True:
        levels = -np.array([
            100, 200, 500, 1000, 2000
        ])[::-1]  # flipped; must have them in increasing order
        srtm_lon, srtm_lat = np.meshgrid(srtm_lon, srtm_lat)
        bc = m.contour(
            srtm_lon,
            srtm_lat,
            srtm_elev,
            levels,
            latlon=True,
            colors='k',
            linestyles='dotted',
            linesize=1.0)  #cmocean.cm.deep
        plt.clabel(bc, fmt='%2.0f', fontsize=12)  # labels for isobaths

    # isoSSTs
    #     cs1 = m.contour(lon, lat, sst, 8, latlon=True, c='k', linestyles='dashed')
    #     plt.clabel(cs1, fmt='%2.0f', fontsize=12) # labels for isoSSTs
    #   ignore first SEC corresponding to along-track
    cs2 = m.contour(
        lon,
        lat,
        sst,
        levels=np.sort(front_sst[-1][0:]),
        latlon=True,
        colors='k',
        linestyles='dashed')
    plt.clabel(cs2, fmt='%2.0f', fontsize=12)  # labels for isoSSTs
    cb.add_lines(cs2)  # add isoSSTs to colorbar

    if list(
            front_sst
    ):  # add isocontours for front for 3 main sections (blue, red, black)
        levels = np.sort(
            np.array(front_sst)[-1])  # make sure levels are ascending
        fc = plt.contour(
            x_section, y_section, sst, levels, colors='k', latlon=False)
        plt.clabel(fc, fmt='%2.0f', fontsize=12)  # labels for isoSSTs

    try:
        m.drawcoastlines()
        m.drawcountries()
        m.drawparallels(parallels, labels=[1, 0, 0, 0])
        m.drawmeridians(meridians, labels=[0, 0, 0, 1])
        m.fillcontinents(color='0.8', lake_color='aqua')
    except:
        pass

    # warning if sections extend beyond plotted region
    if ((np.min(front_lon) < lon.min()) or (np.max(front_lon) > lon.max()) \
    or (np.min(front_lat) < lat.min()) or (np.max(front_lat) > lat.max())):
        print('\n')
        print(
            "\tWARNING: At least one section exceeds plotted region! Some peaks may be missing from the map!"
        )

    # # plot sections
    # for isec in np.arange(nsec):
    #     #longs, latgs = project_section(SEC[isec], dgc)
    #     x, y = m(LONGS[isec], LATGS[isec])

    #     if col: # if color is defined (dependent on length of SEC)
    #         m.plot(x, y, linestyle='dashed', color = col[isec], alpha=0.8)
    #     else:
    #         m.plot(x, y, 'm--', alpha=0.8)

    #     # plot altimetry tracks
    #     for i in np.arange(len(altimetry_tracks[0])):
    #         x, y = m(altimetry_tracks[0][i], altimetry_tracks[1][i])
    #         m.plot(x, y, 'b--', markersize=1.5)

    # add time series fronts locations
    # print(front_sst, front_lon, front_lat)
    #     x, y = m(front_lon, front_lat)
    x, y = m(front_lon[0], front_lat[0])
    m.plot(x, y, 'ro', markersize=5, zorder=100, label='front positions')

    #     # fix the colorbar range
    #     if colorbar_range != None:
    #         min_sst, max_sst = colorbar_range
    #         print(min_sst, max_sst)
    #         plt.clim(min_sst, max_sst)

    #############################################
    months_names = {
        '01': 'Jan',
        '02': 'Feb',
        '03': 'March',
        '04': 'April',
        '05': 'May',
        '06': 'Jun',
        '07': 'Jul',
        '08': 'Aug',
        '09': 'Sept',
        '10': 'Oct',
        '11': 'Nov',
        '12': 'Dec'
    }

    if start_date and end_date:
        #         start_date = str(start_date)
        #         end_date = str(end_date)
        #         assert(len(str(start_date))==8 and len(str(end_date)) == 8)
        #         # convert dates from int/str to datetime
        #         plot_start_date = dt.datetime.strftime(dt.datetime(int(start_date[:4]), int(start_date[4:6]), int(start_date[6:])), "%d-%b-%Y")
        #         plot_end_date = dt.datetime.strftime(dt.datetime(int(end_date[:4]), int(end_date[4:6]), int(end_date[6:])), "%d-%b-%Y")
        #         start_date = dt.datetime.strftime(dt.datetime(int(start_date[:4]), int(start_date[4:6]), int(start_date[6:])), "%Y%m%d")
        #         end_date = dt.datetime.strftime(dt.datetime(int(end_date[:4]), int(end_date[4:6]), int(end_date[6:])), "%Y%m%d")

        #         print(str(start_date)[4:6], str(end_date)[4:6])
        plot_start_month = months_names[str(start_date)[4:6]]
        plot_end_month = months_names[str(end_date)[4:6]]
        plt.title(
            'Climatology SST map \n %s-%s (%s-%s)' %
            (plot_start_month, plot_end_month, start_year, end_year),
            fontsize=22)
        map_filename = "climatology_" + str(start_date)[4:] + "-" + str(
            end_date)[4:] + '.png'
    else:
        plt.title('Climatology SST map', fontsize=22)
        map_filename = "climatology_" + '.png'

    climatology_dir = "climatology"
    climatology_dir = os.path.join(figs_dir, climatology_dir)
    if not os.path.exists(climatology_dir):
        print('creating %s' % climatology_dir)
        os.makedirs(climatology_dir)

#     plt.tight_layout(pad = 1.2)
    plt.savefig(
        os.path.join(os.path.abspath(climatology_dir), map_filename),
        bbox_inches='tight',
        dpi=300)

def climatology(sst_dir, bathy_path, sarpline, region_borders=None, start_month=None, end_month=None, period=None, front_nb=1, results_dir="../results", verbose=0):
    """
    Subsets data within date range & takes its mean

    - region_borders: region to zoom on. List of the corners as [lon_min, lat_min, lon_max, lat_max] (e.g. [13., -38., 21., -25])
    - start_month, end_month: start & end months of the year as numbers (e.g. "01", "02"). If specified 'period' shouldn't be specified.
    - period = [1, 2, 6] for climatology for 1 month, 2 months, or 6 months. Note: results are over written. If specified, start_month & end_month should not be specified.
    """

    assert ((start_month != None) and (end_month != None) or (period != None)), "Either 'period' or 'start_month' and 'end_month' should be specified"

    SEC, longsbob, latgsbob, sstdomma, landmaskdom2, londom, ixbsec, ixesec, elem, coef, altimetry_tracks = define_sections(sst_dir, sarpline)

    if region_borders != None:
        bathymetry = read_bathymetry(bathy_path, region_borders)
    else:
        bathymetry = read_bathymetry(bathy_path)

    # ------ find pathlist for given date range --------
    # months for start and end of seasons

    if (start_month != None) and (end_month != None):
        assert (len(str(start_month)) in [1, 2]
                and len(str(end_month)) in [1, 2])
        seasons = [[start_month, end_month]]

    elif period == 1:  # monthly
        # create list of start & end months
        a = np.arange(1, 13)
        a = list(map(str, a))
        for i, aa in enumerate(a):
            if len(aa) == 1:
                a[i] = str(0) + aa
        seasons = np.array(list(zip(a, a))).tolist()

    elif period == 2:  # bi-monthly
        seasons = [['01', '02'], ['03', '04'], ['05', '06'], ['07', '08'],
                   ['09', '10'], ['11', '12']]

    elif period == 6:  # six months
        seasons = [['01', '06'], ['07', '12']]

    # start_day = end_day =  '15'
    start_day = '01'
    end_day = '31'
    years = sorted(os.listdir(sst_dir), reverse=False)
    start_year, end_year = years[0], years[-1]

    delta_d = []
    delta_sst = []
    reference_delta_sst = []
    front_delta_sst = []
    front_d = []
    front_sst_grad = []
    front_sst = []
    front_lon = []
    front_lat = []
    timestamp_lst = []

    if verbose == 1: print('start_date | end_date')
    for season in seasons:
        print('calculating for season = ', season)
        start_month, end_month = season

        sst_pathlist = []
        for year in years:

            year = int(year)
            start_date = np.int(str(year) + start_month + start_day)
            end_date = np.int(str(year) + end_month + end_day)
            if verbose == 1: print(start_date, end_date)
            assert (
                len(str(start_date)) == 8 and len(str(end_date)) == 8
            ), 'Start & End dates have wrong lengths. Should be of form yyyymmdd'

            pathlist = sorted(Path(sst_dir).glob("**/*SAF*.nc"))

            for sst_path in pathlist:
                absolute_sst_path = os.path.abspath(str(sst_path))
                timestamp = np.int(absolute_sst_path[-59:-51])
                if (timestamp >= int(start_date)) & (timestamp <=
                                                     int(end_date)):
                    sst_pathlist += [absolute_sst_path]
                    if verbose == 1: print(start_date, timestamp, end_date)
    #                 print(sst_path)

    # read one (the last here) file to check dimensions of arrays
        lon, lat, sst, landmask = read_SST(
            absolute_sst_path
        )  # <<<<<< no need to loop over this; once is enough; get out of loops

        nf = len(sst_pathlist)  # number of files

        # create one array for each variable
        nlon = lon.shape[0]
        nlat = lat.shape[0]
        n, m = sst.shape
        lon = np.zeros((nf, nlon))
        lat = np.zeros((nf, nlat))
        sst = np.zeros((nf, n, m))
        landmask = np.zeros((nf, n, m))

        if nf > 0:  # don't continue if no files exist

            for i, sst_path in enumerate(sst_pathlist):
                timestamp = sst_pathlist[i][-59:-51]

                # try/except statement to deal with unreadable files
                try:
                    # read & store data in the arrays
                    lon_tmp, lat_tmp, sst_tmp, landmask_tmp = read_SST(sst_path)
                    lon[i, :] = lon_tmp
                    lat[i, :] = lat_tmp
                    #                 sst_tmp.data[sst_tmp.data == -32768] = np.nan
                    #                 sst[i,:,:] = np.ma.masked_values(sst_tmp.data, -32768.0)
                    sst[i, :, :] = sst_tmp
                    landmask[i, :, :] = landmask_tmp
                    assert (sst_tmp.fill_value == -32768.0), print(
                        'fill value for sst_tmp[%g] = %g' %
                        (i, sst_tmp.fill_value))

                    if verbose == 1:
                        print('sst min = ',
                              np.min(sst), 'sst max = ', np.max(sst))

                    if verbose == 1: print()
                    if verbose == 1: print('sst.shape = ', sst.shape)
                    if verbose == 1: print('lon.shape = ', lon.shape)
                    if verbose == 1: print('lat.shape = ', lat.shape)
                    if verbose == 1: print('landmask.shape = ', landmask.shape)

                except OSError:
                    print("\tFile with timestamp %s couldn't be read." %
                          timestamp)
                    #             sst[i] = np.nan # set array to nan to avoid biasing the mean
                    #             sst[i] = np.np.ma.masked_values(sst[i], 0) # mask the array to avoid biasing the mean
                    sst[i] += sst_tmp.fill_value  # add the fill value to the zeros array to mask them & avoid bias

            assert (lon.shape == (nf, nlon, )), print('shape(lon) = ', lon.shape)
            assert (lat.shape == (nf, nlat, ))
            assert (sst.shape == (nf, n, m))
            assert (landmask.shape == (nf, n, m))

        #mask the array before proceeding, to plot well
        sst = np.ma.masked_values(sst, -32768.0)

        mean_sst = np.ma.mean(sst, axis=0)
        mean_lat = np.ma.mean(lat, axis=0)
        mean_lon = np.ma.mean(lon, axis=0)
        mean_landmask = np.ma.mean(landmask, axis=0)

        if verbose == 1:
            print('min sst = ', min_sst, 'max sst = ', max_sst, 'mean sst = ', mean_sst)

        #         if nf > 1: #if more than one file, take the mean excluding nan's
        #             mean_lon = np.nanmean(lon, axis=0)
        #             mean_lat = np.nanmean(lat, axis=0)
        #             mean_sst = np.nanmean(sst, axis=0)
        #             mean_landmask = np.nanmean(landmask, axis=0)
        #         elif nf == 1: # if one file, don't take the mean!
        #             mean_lon = lon
        #             mean_lat = lat
        #             mean_sst = sst
        #             mean_landmask = landmask

        #         ToDO: replace the initialization of the arrays above by lists. That's better because th number of fronts != number of sections due to lack of detection on some transects which we avoid in the preceeding if statement.
        assert (mean_lon.shape == (nlon, )), print('shape(mean_lon) = ',
                                                   mean_lon.shape)
        assert (mean_lat.shape == (nlat, ))
        assert (mean_sst.shape == (n, m))
        assert (mean_landmask.shape == (n, m))

        if verbose == 1: print('sst shape =', sst.shape)

        ############################ fronts ######################################
        lon, lat, sst, landmask = mean_lon, mean_lat, mean_sst, mean_landmask
        if verbose == 1: print('shape(lon) = ', lon.shape)
        if verbose == 1: print('shape(lat) = ', lat.shape)
        if verbose == 1: print('shape(sst) = ', sst.shape)
        if verbose == 1: print('shape(landmask) = ', landmask.shape)

        delta_d_season = []
        delta_sst_season = []
        reference_delta_sst_season = []
        front_delta_sst_season = []
        front_d_season = []  # for (along section) distance of the main front
        front_sst_grad_season = []
        front_sst_season = []
        front_lon_season = []
        front_lat_season = []
        timestamp_lst_season = []

        # ------------------------------------------------------

        lon, lat = np.meshgrid(lon, lat)  # create meshgrid

        ################## The subsetting should be done at the beginning and front detection done on the subset.
        ################## This needs to be re-arrganged & re-written
        # subset a region to zoom in on area of interest
        if region_borders != None:
            min_region_lon, max_region_lon = region_borders[::2]
            min_region_lat, max_region_lat = region_borders[1::2]
        else:
            min_region_lon = np.min(lon)
            max_region_lon = np.max(lon)
            min_region_lat = np.min(lat)
            max_region_lat = np.max(lat)
            region_borders = [min_region_lon, min_region_lat, max_region_lon, max_region_lat]

        # # assert the min & max of the SSTs region is the same or smaller than the bahymetry data given
        # assert(min_region_lon >= bathymetry.)

        lon_ind = np.where((lon[0, :] >= min_region_lon) & (lon[0, :] <= max_region_lon))
        lat_ind = np.where((lat[:, 0] >= min_region_lat) & (lat[:, 0] <= max_region_lat))
        lon_region = lon[np.min(lat_ind):np.max(lat_ind), np.min(lon_ind):np.max(lon_ind)]
        lat_region = lat[np.min(lat_ind):np.max(lat_ind), np.min(lon_ind):np.max(lon_ind)]
        sst_region = sst[np.min(lat_ind):np.max(lat_ind), np.min(lon_ind):np.max(lon_ind)]

        sstdom = np.sum(coef * sst.ravel()[elem], 2)
        sstdomma = np.ma.masked_where(
            (landmaskdom2 == 2) | (np.isnan(sstdom)) | (sstdom <= 0),
            sstdom)  #does not work well
        SSTGS = []
        LONGS = longsbob
        LATGS = latgsbob
        for j in range(np.shape(londom)[0]):
            SSTGS[j:j] = [sstdomma[j, ixbsec[j]:ixesec[j]]]

        # call sst_fronts() to find main front
        delta_d_tmp, delta_sst_tmp, reference_delta_sst_tmp, front_delta_sst_tmp, front_d_tmp, \
        front_sst_grad_tmp, front_lon_tmp, front_lat_tmp, front_sst_tmp = \
        sst_fronts(sst_region, lon_region, lat_region, LONGS, LATGS, SSTGS, SEC, timestamp, front_nb=1, verbose = 0)

        # stack arrays of front properties (each row corresponds to a time series)
        delta_d_season += [delta_d_tmp]
        delta_sst_season += [delta_sst_tmp]
        reference_delta_sst_season += [reference_delta_sst_tmp]
        front_delta_sst_season += [front_delta_sst_tmp]
        front_d_season += [front_d_tmp]
        front_sst_grad_season += [front_sst_grad_tmp]
        front_sst_season += [front_sst_tmp]
        front_lon_season += [front_lon_tmp]
        front_lat_season += [front_lat_tmp]

        # plot & save SST map #<<<<<<<< recheck which front to take
        dates = [start_date, end_date, years]
        plot_climatology_sst_map(
            sst_region,
            lon_region,
            lat_region,
            LONGS,
            LATGS,
            altimetry_tracks,
            front_lon_season,
            front_lat_season,
            front_sst_season,
            front_nb,
            SEC,
            bathymetry,
            region_borders,
            dates=dates,
            cmap=mycmap(),
            colorbar_range=[12.5, 24])

        # convert list of arrays to an array of arrays
        delta_d += [np.array(delta_d_season[0])]
        delta_sst += [np.array(delta_sst_season[0])]
        reference_delta_sst += [np.array(reference_delta_sst_season[0])]
        front_delta_sst += [np.array(front_delta_sst_season[0])]
        front_d += [np.array(front_d_season[0])]
        front_sst_grad += [np.array(front_sst_grad_season[0])]
        front_sst += [np.array(front_sst_season[0])]
        front_lon += [np.array(front_lon_season[0])]
        front_lat += [np.array(front_lat_season[0])]

        #     assert((front_lon.shape != (0,)) and front_lat.shape != (0,)), "Empty front lon & lat arrays."

        if verbose == 'final results': print(front_lon.shape, front_lat.shape)

    delta_d = np.array(delta_d)
    delta_sst = np.array(delta_sst)
    reference_delta_sst = np.array(reference_delta_sst)
    front_delta_sst = np.array(front_delta_sst)
    front_d = np.array(front_d)
    front_sst_grad = np.array(front_sst_grad)
    front_sst = np.array(front_sst)
    front_lon = np.array(front_lon)
    front_lat = np.array(front_lat)
    timestamp_lst = np.array(timestamp_lst)

    front_properties = {
        'front_lon': front_lon,
        'front_lat': front_lat,
        'front_sst': front_sst,
        'front_d': front_d,
        'front_nb': front_nb
    }

    write_climatology_front_properties(front_properties, seasons, dates, results_dir)

    print('Done.')

    return (mean_lon, mean_lat, mean_sst, mean_landmask)

def pad_an_array(arr):
    """
    Takes an array created created by converted appended lists into an array.
    Returns an array padded with NaNs where data is missing making resulting in an even array.
    This was necessary to be able to write the results to ascii files.
    """

    A = np.full((len(arr), max(map(len, arr))), np.nan)
    for i, aa in enumerate(arr):
        A[i, :len(aa)] = aa

#     ## less efficient
#     # find length of longest array
#     max_len_of_array = 0
#     for aa in arr:
#         len_of_array = aa.shape[0]
#         if len_of_array > max_len_of_array:
#             max_len_of_array = len_of_array

#     n = arr.shape[0]

#     A = np.zeros((n, max_len_of_array)) * np.nan
#     for i, aa in enumerate(zip(arr)):
#         A[i][:aa[0].shape[0]] = aa[0]

    return (A)

def write_climatology_front_properties(front_properties,
                                       seasons,
                                       dates,
                                       results_dir="../results"):
    """
    Takes one argument, front_properties, as a dictionary
    e.g.:

    front_properties = {'front_lon':front_lon, 'front_lat':front_lat, 'front_sst':front_sst,
                        'front_d':front_d, 'front_nb':front_nb}

    write_climatology_front_properties(front_properties)
    """

    climatology_results_dir = os.path.join(results_dir, 'climatology')
    make_directories([climatology_results_dir])

    np.savetxt(
        os.path.join(climatology_results_dir, "seasons"),
        np.array(seasons),
        fmt='%s',
        header=
        "Seasons corresponding to each line in the other climatology results files"
    )

    header = "columns correspond to detected fronts; rows correspond to different days\nnb: some sections may not have detected front"
    start_date, end_date, start_year, end_year = str(dates[0])[4:], str(
        dates[1])[4:], dates[2][0], dates[2][-1]
    years_period = start_year + '-' + end_year
    season_period = start_date + '-' + end_date

    for front_property in front_properties.keys():
        try:
            # pad arrays with NaNs
            front_properties[front_property] = pad_an_array(
                front_properties[front_property])
            filename = 'climatology_%s_%s_%s.%s' % (
                'front' + str(front_properties['front_nb']), front_property,
                years_period, 'txt')
            filename = os.path.join(climatology_results_dir, filename)
            #         print("Writing '%s' to '%s'" % (front_property, filename))
            print("Writing '%s'" % (filename))
            np.savetxt(
                filename,
                front_properties[front_property],
                delimiter='\t',
                header=header)

        except TypeError:
            pass
        except:
            raise
