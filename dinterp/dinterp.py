r"""
Displacement interpolation (dinterp)

Donsub Rim (Columbia U.) July 2018
see LICENSE
"""

import numpy as np
import matplotlib.pyplot as plt
import _dinterp
#from scipy.interpolate import interp1d, splev, splrep
#from scipy.integrate import quad
 
 
 
def computeCDF(x,f,g,tol=0.):
    
    N = f.shape[0]

    F,s0 = cumsum1(f)
    G,s1 = cumsum1(g)

    xF,xG,Fv = merge_monotone(x,F,G,tol=tol)

    return xF,xG,Fv


def cumsum1(f):

    F = np.cumsum(f)
    s0 = F[-1]
    F = F/F[-1]
    F = np.concatenate(([0.],F),axis=0)

    return F,s0

def merge_monotone(x,F,G,tol=0.):
    
    xF,xG,Fv,mt = _dinterp.dinterp.merge_monotone(x,x,F,G,tol)

    xF = xF[:mt]
    xG = xG[:mt]
    Fv = Fv[:mt]
    
    return xF,xG,Fv

def merge_monotone2(xf,xg,F,G,tol=0.):
    
    xF,xG,Fv,mt = _dinterp.dinterp.merge_monotone(xf,xg,F,G,tol)

    xF = xF[:mt]
    xG = xG[:mt]
    Fv = Fv[:mt]
    
    return xF,xG,Fv


def merge_monotone_list(x_list,F_list,tol=0.):
    
    xv0,xv1,Fv = merge_monotone2(x_list[0],x_list[1],F_list[0],F_list[1],\
                                 tol=tol)

    l_old = len(Fv)
    l = l_old + 1
    J = len(x_list)
    print('loop = ' + str(J))
    for j in range(1,J):
        for k in range(2):
            xv01, xv1, Fv1 = merge_monotone2(xv0,x_list[j],Fv,F_list[j],tol=tol)
            xv0, xv1, Fv = merge_monotone2(xv01,x_list[j],Fv1,F_list[j],tol=tol)
    
    xv_list = [xv0]
    for j in range(1,J):
       xv0, xv1, Fv = merge_monotone2(xv0,x_list[j],Fv,F_list[j],tol=tol)
       xv_list.append(xv1)

    return xv_list,Fv


def dinterp(x,xF,xG,Fv,alpha):
    

    #if xF.shape[0] != xG.shape[0]:
    #    n0 = np.abs(xF.shape[0] - xG.shape[0])
    #    ones = np.ones(n0)
    #    if xF.shape[0] > xG.shape[0]:
    #        xG = np.concatenate((xG,ones*xG[-1]),axis=0)   
    #        Gv = np.concatenate((Gv,ones),axis=0)   

    #    elif xG.shape[0] > xF.shape[0]:
    #        xF = np.concatenate((xF,ones*xF[-1]),axis=0)   
    #        Fv = np.concatenate((Fv,ones),axis=0)   

    xH = (1.-alpha)*xF + alpha*xG
    xp = np.concatenate((x[:-1],[x[-1]-1e-4,x[-1]]),axis=0)
    #xp = np.concatenate((x,[x[-1] + (x[-1] - x[-2])]),axis=0)
    #xp = np.linspace(x[0],x[-1],x.shape[0]+1)
    
    #Hv1 = np.interp(xp,x3,Fv1,right=Fv1[-1] + (Fv1[-1] - Fv1[-2]))
    Hv = np.interp(xp,xH,Fv)
    h = np.diff(Hv)/np.diff(xp)
    #h = np.diff(Hv)

    return x,h


def get_intercept(y1,y0,x1,x0):
    return (y1 - y0)/(x1 - x0)*(-x0) + y0




def drt_computeFz(da0,da1):

    N = da0[0].shape[0]
    M = da0[0].shape[1]
    xp1 = np.linspace(0.,N,N+1);
    x = xp1[:-1]

    s0 = []
    s1 = []

    x0 = []
    x1 = []

    F0 = []
    F1 = []
    
    for q in range(4):
        
        da0q = da0[q]
        da1q = da1[q]
        
        s0q = np.zeros(M)
        s1q = np.zeros(M)

        x0q_list = []
        x1q_list = []

        F0q_list = []
        F1q_list = []

        for k in range(M):
        
            v0 = da0q[:,k].copy()
            v1 = da1q[:,k].copy()

            sv0 = np.sum(v0)
            sv1 = np.sum(v1)

            s0q[k] = sv0
            s1q[k] = sv1
    
            xF,xG,Fv,Gv = computeFz(xp1,v0,v1)
            
            x0q_list.append(xF.copy())
            x1q_list.append(xG.copy())
            
            F0q_list.append(Fv.copy())
            F1q_list.append(Gv.copy())

            if 0:
                # for debugging
                f = plt.figure()
                plt.plot(xF,Fv)
                plt.plot(xG,Fv)
                f.savefig('poke_' + str(q) + '_' + str(k) + '.png')
                plt.close(f)

        s0.append(s0q.copy())
        s1.append(s1q.copy())

        x0.append(x0q_list)
        x1.append(x1q_list)

        F0.append(F0q_list)
        F1.append(F1q_list)


    return x,s0,s1,x0,x1,F0,F1



def drt_dinterp(cdfs,a):

    x = cdfs[0]
    s0 = cdfs[1]
    s1 = cdfs[2]
    
    x0 = cdfs[3]
    x1 = cdfs[4]
    
    F0 = cdfs[5]
    F1 = cdfs[6]


    # scaling
    sa0_list, sb0_list, sc0_list, sd0_list = s0
    sa1_list, sb1_list, sc1_list, sd1_list = s1  

    # x-coordinates (CDFs)
    xa0_list, xb0_list, xc0_list, xd0_list = x0 
    xa1_list, xb1_list, xc1_list, xd1_list = x1

    # F-coordinates (CDFs)
    Fa0_list, Fb0_list, Fc0_list, Fd0_list = F0
    Fa1_list, Fb1_list, Fc1_list, Fd1_list = F1
    
    N = x.shape[0] 
    M = len(sa0_list)

    daad = np.zeros((N,M))
    dabd = np.zeros((N,M))
    dacd = np.zeros((N,M))
    dadd = np.zeros((N,M))

    for k in range(M):
    
        xxa,ga = dinterp(x,xa0_list[k],xa1_list[k],Fa0_list[k],Fa1_list[k],a);
        xxb,gb = dinterp(x,xb0_list[k],xb1_list[k],Fb0_list[k],Fb1_list[k],a);
        xxc,gc = dinterp(x,xc0_list[k],xc1_list[k],Fc0_list[k],Fc1_list[k],a);
        xxd,gd = dinterp(x,xd0_list[k],xd1_list[k],Fd0_list[k],Fd1_list[k],a);
     
        daad[:,k] = ga*((1.-a)*sa0_list[k] + a*sa1_list[k])
        dabd[:,k] = gb*((1.-a)*sb0_list[k] + a*sb1_list[k])
        dacd[:,k] = gc*((1.-a)*sc0_list[k] + a*sc1_list[k])
        dadd[:,k] = gd*((1.-a)*sd0_list[k] + a*sd1_list[k])
    
    return daad,dabd,dacd,dadd


def diffp(a,p=1):
    r"""
        differentiate with padding: (output vector length unchanged)
          
    """
    return np.diff(np.concatenate((a[:p],a)))


def supports(a):
    r"""
        return begin/ending indices of connected supports of a vector a

    """
    b = 1.*(a > 0.)
    dpb = np.diff(np.concatenate(([0.],b),axis=0))
    I = np.arange(len(a))
    ibegin = I[dpb > 0.]
    iend   = I[dpb < 0.]
    if len(ibegin) > len(iend):
        iend = np.concatenate((iend, [len(a)]))
    return (ibegin,iend)


def get_parts(v,fil=[.1,.2,.6,.2,.1]):
    r"""
        return positive / negative parts of a vector

    """
    #a = np.convolve(v,fil,'same')
    vp = v*(v >= 0.)
    vm = v*(v <  0.)
    return vp,vm


def get_pieces(v):
    r"""
        obtain pieces (supports) of a vector where the other pieces
        have been zero'ed out

    """
    ibegin,iend = supports(v)
    pieces_list = []
    w = None
    for k in range(len(ibegin)):
        ii = np.zeros(v.shape,dtype=bool)
        ii[ibegin[k]:iend[k]] = True
        w = v*ii
        pieces_list.append(w)
    return pieces_list


def add_gc(v,n=(2,2)):
    r"""
    pre- and post-pend ghost cells

    """
    return np.concatenate((np.array([0.]*n[0]),v,np.array([0.]*n[1])),axis=0)


def del_gc(v,n=(2,2)):
    r"""
        remove ghost cells

    """
    if (n[0] > 0) and (n[1] > 0):
        return v[n[0]:-n[1]]
    elif (n[0] == 0) and (n[1] > 0):
        return v[:-n[1]]
    elif (n[0] > 0) and (n[1] == 0):
        return v[n[0]:]






#
#def drt_diff(da):
#
#    daa = da[0]
#    dab = da[1]
#    dac = da[2]
#    dad = da[3]
#
#    
#    ddaa = np.diff(\
#           np.concatenate((np.zeros((1,daa.shape[1])),daa),axis=0),\
#                   axis=0)
#
#    ddab = np.diff(\
#           np.concatenate((np.zeros((1,dab.shape[1])),dab),axis=0),\
#                   axis=0)
#
#    ddac = np.diff(\
#           np.concatenate((np.zeros((1,dac.shape[1])),dac),axis=0),\
#                   axis=0)
#
#    ddad = np.diff(\
#           np.concatenate((np.zeros((1,dad.shape[1])),dad),axis=0),\
#                   axis=0)
#
#    return (ddaa,ddab,ddac,ddad)
#
#
#def drt_parts(da):
#
#    daa = da[0]
#    dab = da[1]
#    dac = da[2]
#    dad = da[3]
#
#    dims = daa.shape
#
#    pdaa = np.zeros(dims)
#    pdab = np.zeros(dims)
#    pdac = np.zeros(dims)
#    pdad = np.zeros(dims)
#    
#    mdaa = np.zeros(dims)
#    mdab = np.zeros(dims)
#    mdac = np.zeros(dims)
#    mdad = np.zeros(dims)
#
#    pdaa = daa*(daa >= 0.)
#    pdab = dab*(dab >= 0.)
#    pdac = dac*(dac >= 0.)
#    pdad = dad*(dad >= 0.)
#
#    mdaa = -daa*(daa < 0.)
#    mdab = -dab*(dab < 0.)
#    mdac = -dac*(dac < 0.)
#    mdad = -dad*(dad < 0.)
#    
#    return (pdaa,pdab,pdac,pdad), (mdaa,mdab,mdac,mdad)
#
#def drt_integrate(da):
#
#    daa = da[0]
#    dab = da[1]
#    dac = da[2]
#    dad = da[3]
#
#    dims = daa.shape
#
#    sdaa = np.cumsum(daa,axis=0)
#    sdab = np.cumsum(dab,axis=0)
#    sdac = np.cumsum(dac,axis=0)
#    sdad = np.cumsum(dad,axis=0)
#    
#    return (sdaa,sdab,sdac,sdad)
#
#def drt_add(da0,da1):
#
#    return (da0[0]+da1[0], da0[1]+da1[1], da0[2]+da1[2], da0[3]+da1[3])
#
#def drt_minus(da0,da1):
#
#    return (da0[0]-da1[0], da0[1]-da1[1], da0[2]-da1[2], da0[3]-da1[3])
