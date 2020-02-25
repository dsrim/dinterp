r"""
Displacement interpolation (dinterp)

Donsub Rim (Columbia U.) July 2018
see LICENSE
"""

import numpy as np
import matplotlib.pyplot as plt
 
 
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
    
    xF,xG,Fv,mt = _dinterpc.merge_monotone(x,x,F,G,tol)

    xF = xF[:mt]
    xG = xG[:mt]
    Fv = Fv[:mt]
    
    return xF,xG,Fv

def merge_monotone2(xf,xg,F,G,tol=0.):
    
    xF,xG,Fv,mt = _dinterpc.merge_monotone(xf,xg,F,G,tol)

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
    

    xH = (1.-alpha)*xF + alpha*xG
    xp = np.concatenate((x[:-1],[x[-1]-1e-4,x[-1]]),axis=0)
    Hv = np.interp(xp,xH,Fv)
    h = np.diff(Hv)/np.diff(xp)

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


