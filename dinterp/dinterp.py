r"""
Displacement interpolation (dinterp)

Donsub Rim (Courant Institute, NYU) July 2019
see LICENSE
"""

import numpy as np
import matplotlib.pyplot as plt
import os,sys
from . import _dinterpc
 
 
def computeFz(x,f,g,tol=0.):
    
    N = f.shape[0]

    F,s0 = cumsum1(f)
    G,s1 = cumsum1(g)

    xF,xG,Fv = merge_monotone(x,F,G,tol=tol)

    # TODO output is reundant..
    return xF,xG,Fv,Fv

def deim(U):
    r"""
    discrete empirical interpolation method
    """
    
    U_new = np.zeros(U.shape)
    P = np.zeros(U.shape,dtype=bool)
    
    U_new[:,0] = U[:,0]
    pi = np.argmax(np.abs(U[:,0]))
    P[pi,0] = 1
    
    m = U.shape[1]

    for i in range(1,m):
        c = np.linalg.solve(np.dot(P[:,:i].T,U_new[:,:i]),\
                            np.dot(P[:,:i].T,U[:,i]))
        r = U[:,i] - np.dot(U_new[:,:i],c)
        pi = np.argmax(np.abs(r))
        P[pi,i] = True
        U_new[:,i] = U[:,i]
    
    return U,P

def deim_coeff(U,P,b):
    r"""
    compute DEIM coefficients
        c = (P^T U)^{-1} b
    """

    A = U.T[P.T]    # access index sing boolean array
    c = np.linalg.solve(A,b)
    return c

def print_prog(text):

    n = len(text)
    sys.stdout.write('\r| ' + text + ' '*(80-n-1))
    sys.stdout.flush()


def computeCDFs(x_list,f_list,tol=0.):
    
    F_list = []
    for k,f in enumerate(f_list):
        F,_ = cumsum1(f)
        F_list.append(F)
    xv_list,Fv = merge_monotone_list(x_list,F_list,tol=tol)

    return xv_list,Fv


def computeCDFsp(x_list,df_list):

    n_p = len(df_list)
    m_p = len(df_list[0])
    
    xv_list = []
    Fv_list = []

    if m_p > 0:
        for k in range(m_p):
            # k: piece no.
            x1_list = []
            F_list = []
            for j in range(n_p):
                # j: no of fctns

                df = df_list[j][k]
                x1_list.append(x_list[j].copy())
                F_list.append(df)
            
            xv,Fv = computeCDFs(x1_list,F_list)

            xv_list.append(xv)
            Fv_list.append(Fv)

    return xv_list,Fv_list


def computeCDFsbp(x_list,dfp_list,dfm_list):

    n_p = len(dfp_list)
    m_p = len(dfp_list[0])
    
    n_m = len(dfm_list)
    m_m = len(dfm_list[0])

    xvp_list = []
    Fvp_list = []

    if m_p > 0:
        for k in range(m_p):
            # k: pieces
            x1_list = []
            F_list = []
            for j in range(n_p):
                # j: no of fctns

                dfp = dfp_list[j][k]
                x1_list.append(x_list[j].copy())
                F_list.append(dfp)
            
            xvp,Fvp = computeCDFs(x1_list,F_list)

            xvp_list.append(xvp)
            Fvp_list.append(Fvp)

    xvm_list = []
    Fvm_list = []

    if m_m > 0:
        for k in range(m_m):
            # k: pieces

            x1_list = []
            F_list = []
            for j in range(n_m):
                # j: no of fctns

                dfm = dfm_list[j][k]
                x1_list.append(x_list[j].copy())
                F_list.append(dfm)
            
            xvm,Fvm = computeCDFs(x1_list,F_list)

            xvm_list.append(xvm)
            Fvm_list.append(Fvm)

    return xvp_list,Fvp_list,xvm_list,Fvm_list


def cumsum1(f):
    """
    Compute cumulative sum then normalize.

    """

    # TODO raise exception if f has neg. entry

    F0 = np.cumsum(f)
    s0 = F0[-1]
    F0 /= s0
    F = np.zeros(len(F0)+1)
    F[1:] = F0

    return F,s0

def merge_monotone(x,F,G,tol=0.):
    xF,xG,Fv = _dinterpc.merge_monotone(x,x,F,G,tol)
    return xF,xG,Fv

def merge_monotone(xF,xG,F,G,tol=0.):
    xF,xG,Fv = _dinterpc.merge_monotone(xF,xG,F,G,tol)
    return xF,xG,Fv

def merge_monotone_list(x_list,F_list,tol=0.):
    
    xv0,xv1,Fv = merge_monotone(x_list[0],x_list[1],F_list[0],F_list[1],tol=tol)
    Fv = Fv.copy()

    l_old = len(Fv)
    l = l_old + 1
    J = len(x_list)
    for j in range(1,J):
        for k in range(2):
            xv0,xv1,Fv = merge_monotone(xv0,x_list[j],Fv,F_list[j],tol=tol)
    
    xv_list = [xv0]
    for j in range(1,J):
        xv0,xv1,Fv = merge_monotone(xv0,x_list[j],Fv,F_list[j],tol=tol)
        xv_list.append(xv1)
    return xv_list,Fv

def dinterp(x,xF,xG,Fv,Gv,alpha):
    
    xH = (1.-alpha)*xF + alpha*xG
    xp = np.concatenate((x[:-1],[x[-1]-1e-4,x[-1]]),axis=0)
    Hv = np.interp(xp,xH,Fv)
    h = np.diff(Hv)/np.diff(xp)
    return x,h


# displacement interpolation by pieces (DIP)

class DIP(object):
    r"""

    Displacement Interpolation by Pieces (DIP)

    """

    def __init__(self):
        
        self._pivot = 0
        self._signature = None
        self._signature_nz = None
        self._pieces = None
        self._pieces_nz = None
        self._data = None
        self._ddata = None
        self._dims = None
        self._x = None
        self._x1 = None
        self._xv_pieces_list = None
        self._xv0 = None
        self._Fv_pieces_list = None
        self._xvN_pieces_list = None      # length 
        self._scaling_list = None
        self._lr_index_pieces = None
        self._lr_coords = None
        self._lr_modes = None
        self._lr_singvals = None
        self._lr_v = None
        self._lr_w = None
        self._lr_method = None
        self._mu = None                  # param. values for each snapshot
        self._deim_modes = None
        self._deim_mu = None             # local mu-coordinates
        self._deim_coords = None
        self._deim_index = None          # snapshot nos.
        self._deim_ipts = None     # DEIM interp pts
        self._eps = np.finfo(float).eps
        self._save_vars = \
            ['pivot',
             'signature'       ,
             'signature_nz'    ,
             'pieces'          ,
             'pieces_nz'       ,
             'dims'            ,
             'data'            ,
             'ddata'           ,
             'x'               ,
             'x1'              ,
             #'xv_pieces_list'  ,
             'xv0'  ,
             'Fv_pieces_list'  ,
             'xvN_pieces_list' ,
             'scaling_list'    ,
             'deim_modes'      ,
             'deim_ipts'       ,
             'lr_coords'       ,
             'lr_modes'        ,
             'lr_singvals'     ,
             "lr_index_pieces"  ,
             'lr_v'            ,
             'lr_w'            ,
             'lr_method']


    def save(self,path='_output',prefix='dip0_'):
        
        save_attributes_dct = self._save_vars

        for name in save_attributes_dct:
            
            fname = os.path.join(path,prefix+name+'.npy')
            a = getattr(self,"_" + name)
            np.save(fname,a)
            print_prog('saving: ' + name + ' -> ' + fname)
    
    def load(self,path='_output',prefix='dip0_'):

        load_attributes_dct = self._save_vars

        for name in load_attributes_dct:
            
            fname = os.path.join(path,prefix+name+'.npy')
            print_prog('loading: self._' + name + ' <- ' + fname)
            setattr(self,"_" + name,np.load(fname))


    def set_data(self,A):
        r"""
        A is the input snapshot

        """

        self._dims = A.shape
        self._data = A
        

    def compute_lr_model(self,method='svd',tol=1e-3,rrank=4,crank=4,basis=None):
        r"""

        compute low-rank model either using SVD or CUR
            * if method='svd', tol sets the singval cut-off tolerance
        
        """

        # TODO check if we already did compute_transport()
        self._lr_method = method

        # form snapshot of the transport
        i0 = 0
        L = len(self._xv_pieces_list)
        fXV = []
        lip_list = []
        la = 0
        lb = 0
        for i in range(L):
            fXV_diff = np.array(self._xv_pieces_list[i])\
                     - np.array(self._xv_pieces_list[i][i0])
            fXV.append(fXV_diff)
            lb += fXV_diff.shape[1]
            lip_list.append((la,lb))
            la = lb
        fXV = np.hstack(fXV).T      #TODO: avoid transpose, name confusing?

        self._lr_index_pieces = lip_list
        # compute SVD of the snapshot
        if method == 'svd':

            U,s,V = np.linalg.svd(fXV,full_matrices=0)
            self._lr_singvals = s
            S = np.diag(s) 
            if basis == None:
                m = np.sum(s/s[0] > tol)
                self._lr_modes = U[:,:m]
                self._lr_coords = np.dot(S[:m,:m],V[:m,:])   
            else:
                self._lr_modes = U[:,:basis]
                self._lr_coords = np.dot(S[:basis,:basis],V[:basis,:])   

        # compute CUR of the snapshot
        elif method == 'cur':
            from pymf.cur import CUR

            cur_model = CUR(fXV,rrank=rrank,crank=crank)
            cur_model.factorize()

            self._cur_model = cur_model

            U = self._cur_model.U
            S = self._cur_model.S
            V = self._cur_model.V
            
            self._lr_modes = U
            self._lr_coords = np.dot(S,V)   # 1st column is null

    def _print(self,msg):
        r"""

        Flush line then print some given msg 

        """

        hfill = " "*(80 - len(msg)-1)
        sys.stdout.write('\r' + msg + hfill)
        sys.stdout.flush()

    def compute_transport(self,tol=1e-10):
        r"""

        Compute transport maps between the snapshots: collect corresponding
        pieces and their CDFs then merging them to calculate monotone 
        rearrangement

        """


        N,M = self._dims
        n_p = len(self._signature_nz[0])    # total # of (nonzero) pieces

        x  = np.linspace( 0.0,N    ,N+1)    # center pts
        x1 = np.linspace(-0.5,N+0.5,N+2)    # endpts

        df_list = []
        scaling_list = []

        for k in range(n_p):
            dvpiece_list = []        
            scaling_piece_list = []
            for j in range(M):
                piecej = self._pieces_nz[j][k]
                sigj = self._signature_nz[j][k]
                v = np.zeros(N+1)
                v[piecej[0]:piecej[1]] = \
                        float(sigj)*self._ddata[piecej[0]:piecej[1],j]
                v *= (np.abs(v) >= tol)
                sc0 = np.sum(v)
                scaling_piece_list.append(sc0)
                v /= sc0
                dvpiece_list.append(v)
                if not (v >= 0).all():
                    print('warning: non-zero piece: ',j,k,piecej)
                if not ((np.sum(v) - 1.0) < 1e-10):
                    print('warning: not normalized, ',j,k,piecej)
        
            df_list.append(dvpiece_list)
            scaling_list.append(scaling_piece_list)

        # merge CDF for each piece 
        xv_pieces_list = []
        Fv_pieces_list = []
        for k in range(n_p):
            x1_list = [x1]*M
            xv_list,Fv = computeCDFs(x1_list,df_list[k])
            xv_pieces_list.append(xv_list)
            Fv_pieces_list.append(Fv)
        
        self._xv_pieces_list = xv_pieces_list
        self._xv0 = [xv_pieces_list[j][0] for j in range(len(xv_pieces_list))]
        self._Fv_pieces_list = Fv_pieces_list
        self._scaling_list = scaling_list
        self._xvN_pieces_list = \
          [len(self._xv_pieces_list[j][0]) for j in range(len(self._xv_pieces_list))]
    
    def compute_dinterp_vectors(self):
        """

        compute linear basis of unit displacements at the interp pts

        """

        P0 = self._deim_ipts

        N = self._data.shape[0]
        Nt = self._lr_modes.shape[1]

        xh = np.linspace(0.5,N-0.5,N)

        ipt0 = np.array([xh[P0[:,j]] for j in range(P0.shape[1])]).flatten()

        eij = np.eye(Nt)
        vi_list = []
        wgt_list = []
        
        for j in range(Nt):
            vi,wgti =  self.dinterp_lr_ipts(eij[:,j],P0=P0)
            vi_list.append(vi - ipt0.flatten())
            wgt_list.append(wgti - 1.0)
        
        vij = np.vstack(vi_list).T
        wgtij = np.vstack(wgt_list).T  

        self._lr_v = vij
        self._lr_w = wgtij

    def compute_pieces(self):
        """
        Compute monotone pieces, support intervals and signature, sorted
        according to location (center of intervals)

        *test: 08

        """
        
        (N,M) = self._dims      # snapshot dimensions
        
        df_list  = []           # list of fctn pieces
        ind_list = []           # list of indices of supp
        sgn_list = []           # list of signs (pos v neg)
    
        self._ddata = np.zeros((N+1,M))
        
        for j in range(M):
            f = self._data[:,j]
            df = np.zeros(N+1)
            df[1:-1] = np.diff(f)
            self._ddata[:,j] = df
            
            dfp,dfm,dfz = self.get_parts_pm(df)

            df0_pieces_all = []
            df0_indices_all = []
            df0_sgn_all = []
            for df0,sgn in [(dfp,1),(dfz,0),(dfm,-1)]:
                df0_pieces, df0_indices = self.get_fpieces(np.abs(df0))
                df0_pieces_all += df0_pieces
                df0_indices_all += df0_indices
                df0_sgn_all += [sgn]*len(df0_pieces)
            
            # sort according to location
            supp_locs = np.zeros(len(df0_indices_all))
            for k,intvl in enumerate(df0_indices_all):
                supp_locs[k] = 0.5*sum(intvl)
            args = np.argsort(supp_locs)

            df_list.append([df0_pieces_all[i] for i in args])
            ind_list.append(np.array([df0_indices_all[i] for i in args]))
            sgn_list.append(np.array([df0_sgn_all[i] for i in args]))

        # store results
        self._signature = sgn_list
        self._pieces = ind_list         # pieces xh
        self._signature_nz = [self._signature[k]\
                             [self._signature[k] != 0] for k in range(M)]
        self._pieces_nz = [self._pieces[k]\
                          [self._signature[k] != 0] for k in range(M)]
    
    
    def _get_piece_pt(self,pt,j0=0):

        piece_array = np.array(self._pieces[j0])
        npiece_array = np.arange(len(self._signature[j0]))
        npiece_array_nz = np.cumsum(np.abs(self._signature[j0]))

        loc = np.any(piece_array <= pt, axis=1) \
            * np.any(piece_array > pt, axis=1)

        if not loc.any():
                loc = np.any(piece_array == pt, axis=1)

        return npiece_array[loc]

    def _get_lengths_list(self):
        
        lengths = self._xvN_pieces_list
        return np.hstack([0, np.cumsum(lengths)])

    def _get_lr_mode_pieces(self):

        lengths_list = self._get_lengths_list()

        modes_list = []
        for n in range(len(lengths_list)-1):
            n0 = lengths_list[n]
            n1 = lengths_list[n+1]
            modes_list.append(self._lr_modes[n0:n1,:])
            
        return modes_list


    def dinterp_lr(self,coords,pivot=None,return_cdf=False):
        r"""

        Compute DIP using low-rank approximation obtained through
        compute_lr_model(), return individual interpolated pieces 
        Default returns the DIP space.

        Parameters
        ----------
        coords: array_like
            Transport coordinates for applying transport modes

        Returns
        -------
        list
            List of interpolants or single array if (sum_pieces==True)

        Other parameters
        ----------------
        pivot : int
            Manually pick pivot snapshot (default is set in self._pivot)
        
        return_cdf: [ False | 'grid' | 'full' ]
            'grid' integrates then interpolates to the uniform grid
            'full' returns individual CDFs of interpolated pieces in accurate
            form, without returning to the uniform grid

        """

        if pivot == None:
            j0 = self._pivot
        else:
            j0 = pivot
        
        Nz = len(self._pieces_nz[0])            # no of non-zero pieces
        (N,M) = self._dims

        C = self._lr_modes
        X = np.dot(C,coords)                    # transport coordinates

        lengths = self._xvN_pieces_list
        lengths_list = np.hstack([0, np.cumsum(lengths)])

        xv_pivot_pieces = [self._xv_pieces_list[k][j0] for k in range(Nz)]
        xv0 = np.hstack(xv_pivot_pieces)
        xv1 = xv0 + X

        xhe = np.linspace(-0.5,N+0.5,N+2)       # extended half-grid
        
        h_list = []
        for l in range(len(lengths_list)-1):

            ind0 = lengths_list[l]
            ind1 = lengths_list[l+1]

            xv1p = xv1[ind0:ind1]
            Fvp = self._Fv_pieces_list[l]

            if (return_cdf == "full"):
                h_list.append([xv1p,Fvp])
            else:
                H = np.interp(xhe,xv1p,Fvp)
                h = np.diff(H)
                sgn = self._signature_nz[j0][l]
                h *= sgn
                if (return_cdf == "grid"):
                    h = np.cumsum(h)[:-1]
                h_list.append(h)
    
        return h_list

    def dinterp_lr_pts1(self,pts,coords):

        N = self._dims[0]
        pieces0 = self._pieces[0]
        pn = _dinterpc.compute_pc4pt(pts.astype(np.float),
                                             pieces0.astype(np.float))

        dX = np.dot(self._lr_modes,coords)

        Tpts = np.zeros(pts.shape)
        K = self._pieces[0].shape[0]
        for k in range(K):
            ppts0 = pts[pn == k]
            k0 = self._nzpiece(k)
            if k0 > -1:
                ia,ib = self._lr_index_pieces[k0]
                xv0 = self._xv0[k0]
                dX0 = dX[ia:ib]
                Tppts0 = _dinterpc.compute_transport_pt(ppts0,xv0,dX0)
                Tpts[pn == k] = Tppts0
            elif ((k > 0) and (k < K-1)):
                xa = self._pieces[0][k0][0]
                xa = np.array([xa - 0.5])
                ka = self._nzpiece(k-1)
                ia,ib = self._lr_index_pieces[ka]
                xva = self._xv0[ka]
                dX0 = dX[ia:ib]
                Ta = _dinterpc.compute_transport_pt(xa,xva,dX0)
                
                xb = self._pieces[0][k0][1]
                xb = np.array([xb - 0.5])
                kb = self._nzpiece(k+1)
                ia,ib = self._lr_index_pieces[kb]
                xvb = self._xv0[kb]
                dX0 = dX[ia:ib]
                Tb = _dinterpc.compute_transport_pt(xb,xvb,dX0)

                alph = (ppts0 - xa)/(xb - xa)
                Tppts0 = (1.0 - alph)*Ta + Tb
                Tpts[pn == k] = Tppts0
            elif (k == 0):
                xa = -0.5
                Ta = -0.5
                
                xb = self._pieces[0][k][1]
                xb = np.array([xb - 0.5])
                kb = self._nzpiece(k+1)
                ia,ib = self._lr_index_pieces[kb]
                xvb = self._xv0[kb]
                dX0 = dX[ia:ib]
                Tb = _dinterpc.compute_transport_pt(xb,xvb,dX0)

                alph = (ppts0 - xa)/(xb - xa)
                Tppts0 = (1.0 - alph)*Ta + alph*Tb
                Tpts[pn == k] = Tppts0
            elif (k == K-1):
                xa = self._pieces[0][k][0]
                xa = np.array([xa - 0.5])
                ka = self._nzpiece(k-1)
                ia,ib = self._lr_index_pieces[ka]
                xva = self._xv0[ka]
                dX0 = dX[ia:ib]
                Ta = _dinterpc.compute_transport_pt(xa,xva,dX0)
                
                xb = N + 0.5
                Tb = N + 0.5

                alph = (ppts0 - xa)/(xb - xa)
                Tppts0 = (1.0 - alph)*Ta + alph*Tb
                Tpts[pn == k] = Tppts0

        return Tpts

    def dinterp_lr_pts(self,pts,coords,return_dX=False):
        """
        Return mapped points lying inside the domain [0,N] under the low-rank
        transport map

        Parameters
        ----------
        pts : array_like
            Array of points to compute the mapped points for
        
        coords: array_like
            Coordinates (transport modes) for computing the transport map

        Other Parameters
        ----------------
        trj : list
            Return trajectory of the transport map 
            (x0 -> CDF -> CDF -> T(x0))
            
            trj[j] = np.array([xt,s0,s1,y,k]))

            where 
                xt : the mapped point of the input value `pts[j]`
                s0 : slope of the pivot CDF at x0
                s1 : slope of the interp CDF at xy
                y  : value of pivot CDF at x0 and interp CDF at xt
                k  : piece number 
        
        """

        (N,M) = self._dims
        
        xhe = np.linspace(-0.5,N+0.5,N+2)       # half-grid extended
        
        pieces0 = self._pieces[0]

        C = self._lr_modes
        dX = np.dot(C,coords)                   # transport map

        trj = []
        for j in range(len(pts)):
            x0 = pts[j]
            
            # find the interval for pt 
            loc = (x0 >= xhe[pieces0[:,0]])*(x0 < xhe[pieces0[:,1]])
            k = np.argmax(loc)
            sgn = self._signature[0][loc]

            if sgn == 0:
                # zero signature case
                xl0 = xhe[pieces0[loc,0]]
                xr0 = xhe[pieces0[loc,1]]

                if xl0 <= -0.5:
                    # left endpt is on bdry
                    xl1 = xl0
                else:
                    xl1,yl,(sl0,sl1) = self.compute_transport_pt(xl0,dX,k-1)
                if xr0 >= N+0.5:
                    # right endpt is on bdry
                    xr1 = xr0
                else:
                    xr1,yr,(s0,s1) = self.compute_transport_pt(xr0,dX,k+1)

                y0 = (x0 - xl0)/(xr0 - xl0)
                xt = (1-y0)*xl1 + y0*xr1
                s0 = 1.0/(xr0 - xl0)
                s1 = 1.0/(xr1 - xl1)
                trj.append(np.array([xt,s0,s1,y0,k]))
            else:
                # nonzero signature case
                xt,y,(s0,s1) = self.compute_transport_pt(x0,dX,k)
                trj.append(np.array([xt,s0,s1,y,k]))

        if return_dX:
            # return dX for plotting
            return trj, dX
        else:
            return trj


    def dinterp_lr_pt(self,pts,coords,return_trajectory=False):
        """
        Return mapped points lying inside the domain [0,N] under the low-rank
        transport map

        Parameters
        ----------
        pts : array_like
            Array of points to compute the mapped points for
        
        coords: array_like
            Coordinates (transport modes) for computing the transport map

        Other Parameters
        ----------------
        return_trajectory : bool
            Return trajectory of the transport map 
            (x0 -> CDF -> CDF -> T(x0))

        DEPRECATED: use dinterp_lr_pts()
        
        """

        j0 = 0
        piece_array = self._pieces[j0]
        npiece_array = np.arange(len(self._signature[j0]))
        npiece_array_nz = np.cumsum(np.abs(self._signature[j0])) - 1
        
        if return_trajectory:
            new_pts_list = []
        else:
            new_pts_array = np.zeros(len(pts))
            wgt_list = []

        N = self._dims[0]
        C = self._lr_modes
        X = np.dot(C,coords)  # transport modes

        lengths = self._xvN_pieces_list
        lengths_list = np.hstack([0, np.cumsum(lengths)])

        L = len(self._xv_pieces_list)
        xv_pivot_pieces = [self._xv_pieces_list[k][j0] for k in range(L)]
        xv0 = np.hstack(xv_pivot_pieces)
        
        for j in range(len(pts)):

            pt = pts[j]
            
            # find the interval for pt 
            loc = (piece_array[:,0] <= pt)*(piece_array[:,1] > pt)
            if not loc.any():
                loc = (piece_array[:,1] == pt)
            
            left_pt  = piece_array[loc,0]
            right_pt = piece_array[loc,1]
            
            if self._signature[j0][loc] == 0:
                # if the point is in the null part
                l1 = int(npiece_array[loc])
                xl = self._pieces[j0][l1][0] 
                xr = self._pieces[j0][l1][1]

                # get new left location
                locl = (piece_array[:,0] <= xl)*(piece_array[:,1] > xl)
                if locl[0]:
                    locl = 0
                    xlt = 0.
    
                else:
                    kl = int(npiece_array_nz[locl])  # nz index of the piece
                    kl0 = int(lengths_list[kl])
                    kl1 = int(lengths_list[kl+1])
                    Xl1 = X[kl0:kl1] + xl
                    Xl1[0] = 0
                    Xl1[-1] = N
                    
                    Fv = self._Fv_pieces_list[kl]
                    dfp = self._get_snapshot_dpiece(j0,kl)
                    dfp1,_ = cumsum1(dfp)
                    dfp1 /= dfp1[-1]

                    xlt = Xl1[\
                       np.argmax(np.isclose(Fv,np.min([1.,Fv.max()]),rtol=1e-10))]
                
                # get new right location
                locr = (piece_array[:,0] <= xr)*(piece_array[:,1] > xr)
                if locr[-1]:
                    locr = N
                    xrt = float(N)
                else:
                    kr = int(npiece_array_nz[locr])  # nz index of the piece
                    kr0 = int(lengths_list[kr])
                    kr1 = int(lengths_list[kr+1])
                    Xr1 = X[kr0:kr1] + xr
                    Xr1[0] = 0
                    Xr1[-1] = N
                    
                    Fv = self._Fv_pieces_list[kr]
                    dfp = self._get_snapshot_dpiece(j0,kr)
                    dfp1,_ = cumsum1(dfp)
                    dfp1 /= dfp1[-1]

                    xrt = Xr1[np.argmax(Fv > 0.)-1]

                alph = (pt - xl)/(xr - xl)
                xt = (1-alph)*xlt + alph*xrt
                if return_trajectory:
                    w = float(xr-xl)/float(xrt-xlt)
                    new_pts_list.append([xt,xl,xr,0,0,0,[0.,2.,xt-w,w],0])
                else:
                    new_pts_array[j] = xt
                    wgt_list.append((1./(xrt-xlt),1./(xr-xl)))
                
            
            else:
                # if the point is in the non-null part
                k = int(npiece_array_nz[loc])  # nz index of the piece
                k0 = int(lengths_list[k])
                k1 = int(lengths_list[k+1])
                X1 = X[k0:k1] + pts[j]
                X1[0] = 0
                X1[-1] = N
                
                Fv = self._Fv_pieces_list[k]
                dfp = self._get_snapshot_dpiece(j0,k)
                dfp1,_ = cumsum1(dfp)
                dfp1 /= dfp1[-1]
                xt,scFx,wgt = self._compute_xt(pt,X1,Fv,dfp1)
                if return_trajectory:
                    #TODO: change to consistent scaling
                    xx = self._xv_pieces_list[k][j0]
                    X0 = (xx - xx.min())/(xx.max() - xx.min()) * N
                    X1 = X[k0:k1] + X0
                    new_pts_list.append([pt,xt,scFx,X1,Fv,dfp1,wgt,k])
                else:
                    new_pts_array[j] = xt
                    wgt_list.append((wgt[4],wgt[1]))   # bad: (old,new)-slope
                
        if return_trajectory:
            return new_pts_list
        else:
            return new_pts_array,wgt_list

    def compute_dipts(self,P0):
        r"""

        Compute displacement of interpolation points under unit transpot mode

        """
        Nt = self._lr_modes.shape[1]
        N = self._dims[0]
        xh = np.linspace(0.5,N-0.5,N)
        ipt0 = np.array([xh[P0[:,j]] for j in range(P0.shape[1])]).flatten()
        
        vi = np.zeros((Nt,P0.shape[1]))
        for j in range(Nt):
            ej = np.zeros(Nt)
            ej[j] = 1.0             # canonical basis
            iptj, _ = self.dinterp_lr_ipts(ej,P0=P0)
            vi[j,:] = (iptj - ipt0)

        return vi

    def dinterp_lr_ipts(self,coord,P0=None):
        r"""

        map interpolation points

        Parameters
        ----------
        coord : array_like
            transport coordinate

        Returns
        -------
        new_ipts : array_like
            mapped points
        wgts : array_like
            change of volume near the interpolation point

        """

        N = self._data.shape[0]
        xh = np.linspace(0.5,N-0.5,N)
        
        P = P0.shape[1]
        ipts = np.zeros(P)
        ipts_intvl = np.zeros((2,P))
        for p in range(P):
            ipts[p] = xh[P0[:,p]]

        for p in range(P):
            ipts_intvl[0,p] = ipts[p] - 0.5      # left-pt of unit interval
            ipts_intvl[1,p] = ipts[p] + 0.5      # right-pt of unit interval

        trj = self.dinterp_lr_pts(ipts,coord)
        new_ipts = np.array([float(trj[j][0]) for j in range(len(trj))])
        
        trj_intvl = self.dinterp_lr_pts(ipts_intvl.flatten(),coord)
        wgts = np.array([float(trj_intvl[j][0]) for j in range(len(trj_intvl))])
        wgts = wgts.reshape((2,P))
        wgts = np.diff(wgts,axis=0).flatten()
        
        return new_ipts,wgts
    

    def dinterp_lr_grid(self,coord=None,refine=1):
        r"""

        transport a uniform grid (w given refinement ratio),
        calls dinterp_lr_pt for each point.

        - output original grid pts and new grid pts.
        - returns: the tuple (new_grid_pts, old_grid_pts)

        """

        if (type(coord) == type(None)):
            coord = np.zeros(self._lr_coords.shape[1])

        N = self._data.shape[0]
        N1 = int(np.round(refine*N))
        xh = np.linspace(0.5,N-0.5,N1)      # endpts of cells
        xi = np.linspace(0.0,N    ,N1+1)      # endpts of cells
        trj = self.dinterp_lr_pts(xh,coord)
        xte = np.array([float(trj[j][0]) for j in range(len(trj))])
        trj_intvl = self.dinterp_lr_pts(xi,coord)
        wgts = np.array([float(trj_intvl[j][0]) for j in range(len(trj_intvl))])
        wgts = np.diff(wgts)
        
        return xte,wgts

    def _nzpiece(self,k):
        """
        Return non-zero piece index 
        
        """
        
        if (self._signature[0][k] != 0):
            return k - np.sum(self._signature[0][:k] == 0)
        else:
            return -1

    def _zpiece(self,k1):
        """
        Return piece index incl. zeros
        
        """
        
        ii = np.cumsum(self._signature[0] != 0) - 1
        ind = ii == k1
        if ind.any():
            return np.argmax(ind)
        else:
            return -1

    def compute_transport_pt(self,x0,dX,k):
        """
        Compute the transported map for one point

        Parameters
        ----------
        xcoord : float
            location in the domain to apply the map to
        C : array_like
            transport map (fctn of F)
        k : int
            piece number

        Returns
        -------
        xt : transported point
        y0 : quantile
        (s0,st) : slope of CDF at x and xt
        
        """

        N = self._dims[0]
        ia,ib = self._pieces[0][k]

        xp  = np.linspace( 0.0,N    ,N+1)
        xhe = np.linspace(-0.5,N+0.5,N+2)

        # break ties at edge points
        eps = (N+2)*self._eps
        if (x0 == xhe[ia]):
            x0 += eps
        elif (x0 == xhe[ib]):
            x0 -= eps
        
        df = self._get_snapshot_fpiece(0,k)     # k-th piece of 0-th snapshot
        df = np.abs(df)
        df1,sc = cumsum1(df)

        y0 = np.interp(x0,xhe,df1)              # value on pivot CDF
        i0 = np.argmin(np.abs(x0 - xp))
        s0 = df[i0]/sc                          # guranteed to be nonzero

        knz = self._nzpiece(k)
        if knz < 0:
            return -1
        ia,ib = self._lr_index_pieces[knz]

        dX0 = dX[ia:ib]
        Fv = self._Fv_pieces_list[knz]
        dXi = dX0[1:-1]
        Fvi = Fv[1:-1]
        ii = np.arange(len(Fvi))

        dx0 = np.interp(y0,Fvi,dXi)

        xt = x0 + dx0

        # compute slope
        if (y0 < Fvi).any():
            ia = np.max(ii[y0 >= Fvi]) 
            ib = np.min(ii[y0 <  Fvi])
        else:
            # last interval
            ia = np.max(ii[y0 >  Fvi]) 
            ib = np.min(ii[y0 <= Fvi])
        st = 1.0/s0 + (dXi[ib] - dXi[ia])/(Fvi[ib] - Fvi[ia])
        st = 1.0/st 

        return xt,y0,(s0,st)
    
    def _compute_xt(self,xcoord,C1,Fv,dfp1):
        r"""
    
        Compute the transported location of xcoord

        DEPRECATED: use compute_transport_pt
    
        """
        
        N = self._dims[0]
        x = np.linspace(0,N,N+1) 
        scFx = np.interp(xcoord,x,dfp1)
        ii = np.sum(Fv <= scFx) 
        scFx1 = np.interp(xcoord-0.25,x,dfp1)
        
        if (ii < len(Fv)) and (Fv[ii] > Fv[ii-1]):
            xt = np.interp(scFx, [Fv[ii-1],Fv[ii]], [C1[ii-1],C1[ii]])
            tol = 1e-16
            if np.abs(C1[ii] - C1[ii-1]) > tol:
                # new slope, as seen from the y-axis
                w = (C1[ii] - C1[ii-1])/(Fv[ii] - Fv[ii-1]) + 0.25/(scFx - scFx1)
                scFx2 = scFx - 1.0/w
                if scFx2 < 0.0:
                    x0 = np.min(x[dfp1 > 0]-1.0)
                    x1 = np.min(C1[Fv > 0]-1.0)
                    xt2 = x0*(xt - 1.0)/x1
                else:
                    ii2 = np.sum(dfp1 <= scFx2)
                    xt2 = np.interp(scFx2,[dfp1[ii2-1],dfp1[ii2]],\
                                          [   x[ii2-1],   x[ii2]])
                w0 = (scFx - scFx1)/0.25
                wgt = [w,1.0/w,xt2,scFx2,w0]
            else:
                wgt = [0,np.inf,0,0,np.inf]
        else:
            wgt = [0,np.inf,0,0,np.inf]
            if (ii < len(Fv)):
                print('case2')
                xt = C1[ii]
            else:
                print('case3')
                xt = C1[-1]
    
        return xt,scFx,wgt

    def _get_snapshot_fpiece(self,j,k,nz_index=False):
        r"""

        return cut-off k-th piece of the j-th snapshot

        """

        N = self._dims[0] 
        v = np.zeros(N+1)

        if nz_index:
            ind = self._pieces_nz[j][k]
            v[ind[0]:ind[1]] = self._ddata[ind[0]:ind[1],j]
        else:
            sgn = self._signature[j][k]
            ind = self._pieces[j][k]

            if sgn == 0:
                v[ind[0]:ind[1]] = 1.0
            else:
                v[ind[0]:ind[1]] = self._ddata[ind[0]:ind[1],j]

        return v

    def _get_snapshot_dpiece(self,j,k,nz_index=True):
        r"""

        return cut-off k-th piece of the j-th snapshot

        DEPRECATED: use _get_snapshot_fpiece()

        """

        # TODO: distinguish zero / nonzero indexing

        N = self._dims[0] 
        v = np.zeros(N+1)

        if nz_index:
            ind = self._pieces_nz[j][k]
            v[ind[0]:ind[1]] = self._ddata[ind[0]:ind[1],j]
        else:
            sgn = self._signature[j][k]
            ind = self._pieces[j][k]

            if sgn == 0:
                v[ind[0]:ind[1]] = 1.0
            else:
                v[ind[0]:ind[1]] = self._ddata[ind[0]:ind[1],j]

        return v
        
    def check_signature(self):
        r"""

        return true if the given snapshot satisfies the signature condition

        """

        flag = True

        if type(self._pieces) == type(None):
            self.compute_pieces()
        
        M = self._dims[1]
        sigj_old = self._signature[0]
        
        for j in range(1,M):
            sigj = self._signature[j]

            if ((len(sigj_old) != len(sigj)) or (sigj_old != sigj).all()):
                flag = False
                break

            sigj_old = sigj

        return flag
        
    def get_fpieces(self,v):
        r"""
        Compute pieces (supports) of an array where the other pieces have been 
        zero'ed out
    
        """
        
        ibegin,iend = self.supp_intvl(v)
        pieces_list = []
        apieces_list = []
        
        w = None
        for k in range(len(ibegin)):
            ii = np.zeros(v.shape,dtype=bool)
            ii[ibegin[k]:iend[k]] = True
            w = v*ii
            pieces_list.append(w)
            apieces_list.append([ibegin[k],iend[k]])
        
        return pieces_list,apieces_list

    def get_parts_pm(self,v,tol=1e-10):
        r"""
        Return positive & negative parts of an array
    
        """
        vp = v*(v >  tol)
        vm = v*(v < -tol)
        vz = (np.abs(v) <= tol)
        return vp,vm,vz

    def supp_intvl(self,v,tol=1e-10):
        """
        Return beginning- and ending- indices of connected supports 
        * assumes uniform grid
        * returned indices are on the end-pts off by half-grid 
        * test no: 07
    
        """
        N = len(v)
        b = 1*(v > tol)
        be = np.zeros(N+2,dtype=np.int)
        be[1:-1] = b
        dpb = np.diff(be)      # diff w/ padding
        ii = np.arange(N+1)
        ibegin = ii[dpb > 0.0]
        iend   = ii[dpb < 0.0]
        return (ibegin,iend)

    ### plotting tools
    
    def plot_transport_modes(self,name_postfix=''):
        r"""

        generate (two) plots on useful info. about the low-rank map:
        then save plots to files. 
        run after: compute_pieces(),
                   compute_transport(),
                   compute_lr_mode().
        - plot 1
          * singular vectors for each piece
          * singular values
        - plot 2
          * the evolution of the pivot under each of the modes

        """

        if self._lr_method == 'svd':
            # plot filenames
            plot1_fname = 'transport_svd' + name_postfix + '.png'
            plot2_fname = 'transport_dinterp_std' + name_postfix + '.png' 
            
            # plot1: singular values / vectors
            # recover the svd
            fUX = self._lr_modes
            s0 = np.linalg.norm(self._lr_coords,axis=1)
            s0 = s0 / s0[0]
            Ni_list = self._get_lengths_list()
            fFv_list = self._Fv_pieces_list

            # no. of singular vectors to plot
            M = self._lr_coords.shape[0]
            Mfig = min([3,M])
            
            # plot
            f1,ax1 = self._newfig(dims=(1,Mfig+1))
            ax1[0].semilogy(s0,'-rs')
            ax1[0].set_title('transport SVD');
            for m in range(Mfig):
                for k in range(len(Ni_list)-1):
                    N0 = Ni_list[k]
                    N1 = Ni_list[k+1]
                    ax1[m+1].plot(fUX[N0:N1,m],fFv_list[k]);
                    ax1[m+1].set_title('singular vector {:d}'.format(m));
            f1.tight_layout()
            f1.savefig(plot1_fname,dpi=300)
        
            # plot2: disp interp under each singular mode
            f2,ax2 = self._newfig(dims=(1,Mfig))
            coords = np.eye(M)          # standard basis for coords
            x = self._x                 # x - coordinate
            xl = x.min()          # left endpt
            xr = x.max()          # right endpt
            for m in range(Mfig):
                vmin = 2.
                vmax = np.abs(self._lr_coords[m,:]).max() / 5
                alph_list = np.linspace(vmin,vmax,5)
                for alph in alph_list:
                    c0 = (alph-vmin)/(vmax-vmin)    # set color
                    coord0 = alph*coords[:,m]
                    h2 = self.dinterp_lr(coord0,sum_pieces=True)
                    ax2[m].plot(x,h2,color=(0.1,.5*c0,1.-.5*c0))
                    ax2[m].set_title('all pieces under mode # {:1d}'.format(m+1))
            f2.tight_layout()
            f2.savefig(plot2_fname,dpi=300)

        return f1,f2

    def plot_snapshot_fpiece(self,j,k,step=True):
        r"""
        plot cut-off snapshot (j-th snapshot, k-th piece)

        Parameters
        ----------
        j : int
            snapshot number
        k : int
            piece number

        Returns
        -------
        fig : :class:`matplotlib.figure.Figure` object

        """
        
        N = self._dims[0]

        # plot snapshot restricted to a piece
        fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(10,3))
        n_pieces_nz = len(self._pieces_nz[0])
        sgn = self._signature[j][k]
        leg_list = []
        if step:
            xi = np.linspace(0.0,N,N+1)
            ax.step(xi,self._get_snapshot_fpiece(j,k),".",where="mid")
        else:
            xi = np.linspace(0.0,N,N+1)
            ax.plot(xi,self._get_snapshot_fpiece(j,k),".",linestyle="-")
        ax.set_title('snapshot:{:d} | piece:{:d} | sgn:{:d}'.format(j,k,sgn))

        return fig

    def plot_transport_grid(self, coord=None,refine=1,ax=None,\
                                  cmap='YlGnBu',dom=(0.0,1.0),xlim=(0.0,1.0)):
        r"""

        plot trajectory of the uniform grid under transport

        """

        from matplotlib import cm

        if (type(coord) == type(None)):
            coord = np.ones(self._lr_coords.shape[1])

        N = self._dims[0]
        cm0 = cm.get_cmap(cmap)

        xe = np.linspace(0.5,N-0.5,int(np.round(N*refine)))
        xte,_ = self.dinterp_lr_grid(coord,refine=refine)

        if type(ax) == type(None):
            fig,ax = self._newfig(size=(13,1.5))
        diffmax = np.abs(xte - xe).max()
        for j in range(len(xte)):
            diff0 = np.abs(xte[j] - xe[j])
            beta = np.min([(diff0/diffmax + 0.2),1.0])
            color0 = cm0(beta)
            x0 = xe[j]/float(N)*dom[1] + dom[0]
            xt = xte[j]/float(N)*dom[1] + dom[0]
            ax.plot([x0,xt],[0.0,1.0],marker='.',\
                    color=color0,linewidth=1.0,markersize=0.5)
        ax.set_yticks([0.0,1.0],minor=False)
        ax.set_yticklabels([0.0,1.0])
        ax.set_xlim(xlim)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\alpha$")

        return ax

    def plot_transport_trajectory(self, pt=np.array([0.2]),\
                                  coord=None):
        r"""
        Plot the trajectory of a point in the domain x0 to another pt T(x0)
        input pt should lie within the domain

        saves the plot, returns figure

        TODO: check, make example

        """

        N = self._dims[0]
        xi  = np.linspace( 0.0,N    ,N+1)
        xie = np.linspace(-1.0,N+1  ,N+2)
        xh  = np.linspace( 0.5,N-0.5,  N)
        xhe = np.linspace(-0.5,N+0.5,N+2)

        if len(pt) > 1:
            return -1
        if type(coord) == type(None):
            coord = -80.0*np.ones(self._lr_modes.shape[1])

        trj_list,dX = self.dinterp_lr_pts(pt,coord,return_dX=True)

        x0 = float(pt)
        xt,s0,s1,y0,k = trj_list[0]
        k  = int(k)

        sgn = self._signature[0][k]
        dfp = np.abs(self._get_snapshot_fpiece(0,k))
        dfp1,_ = cumsum1(dfp)

        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,3))
        ax.step(xi,dfp/dfp.max(),where="mid",color=(0.,0.,0.,.25))  # plot dpiece 
        ax.plot(xhe,dfp1)                            # plot CDF of the dpiece
        
        if sgn == 0:
            xx1 = np.array([-0.5,xt - y0/s1,xt + (1.0 - y0)/s1,N+0.5])
            dgp1 = np.array([0.0,0.0,1.0,1.0])
            ax.plot(xx1,dgp1,'--')
        else:
            k1 = self._nzpiece(k)
            Fv = self._Fv_pieces_list[k1]
            X0 = self._xv_pieces_list[k1][0]     # pivot CDF, piece k
            a,b = self._lr_index_pieces[k1]
            X1 = X0 + dX[a:b]
    
            ax.plot(X1,Fv,'--')                         # plot transported CDF
        
        ax.legend(['c-o piece (normalized)'.format(k),\
                   'CDF c-o piece '.format(k),\
                   'CDF d-i piece'])
    
        # plot trajectory
        ax.plot(x0,0.0,'rx')
        ax.plot(xt,0.0,'rx')
        ax.plot([x0,x0],[0.0,y0],'r')
        ax.plot([xt,xt],[0.0,y0],'r')
        ax.plot([x0,xt],[ y0,y0],'r')
    
        # plot slopes of CDFs
        dx = 0.25
        ax.plot([x0-dx,x0+dx],[y0-dx*s0,y0+dx*s0],'b')
        ax.plot([xt-dx,xt+dx],[y0-dx*s1,y0+dx*s1],'g')

        if x0 > xt:
            halign =['left', 'right']
        else:
            halign =['right', 'left']
        ax.annotate(r'$x_0$', xy=(x0,0.0),xytext=(x0,-0.1),\
                    horizontalalignment=halign[0])
        ax.annotate(r'$T(x_0)$', xy=(xt,0.),xytext=(xt,-0.1),\
                    horizontalalignment=halign[1])
        ax.set_ylim([-0.2,1.1])

        return fig

    def plot_transported_ipts(self,coord0,P0=None):
        r"""

        plot two sets of DEIM interpolation points (before / after transport)
        P0: 2D boolean array, if None use original DEIM interp pts

        TODO: write doc
        """

        plot_fname = 'transport_ipts.png'
    
        N = self._data.shape[0]
        if (self._x != None).all():
            x = self._x
            x1 = self._x1
        else:
            x = np.linspace(0.,N-1,N) + .5
            x1 = np.linspace(0.,N,N+1)

        if P0 == None:
            P0 = self._deim_ipts

        xr = x[-1]
        xl = x[0]

        ii = np.linspace(0,N-1,N)
        
        P = P0.shape[1]
        ipts = np.zeros(P)
        for p in range(P):
            ipts[p] = ii[P0[:,p]]

        
        new_ipts,_ = self.dinterp_lr_pt(ipts,coord0)
        new_ipts = np.array(new_ipts)
        xipts = ipts/(N-1)*(xr-xl) + xl
        xnew_ipts = new_ipts/(N-1)*(xr-xl) + xl
        
        f,ax = self._newfig()
        ax.plot(xipts, .1+ 0.*ipts,'b.')
        ax.plot(xnew_ipts, 0.*new_ipts,'r.')
        ax.set_xlim([x[0],x[-1]])
        ax.set_yticks([0.,0.1])
        ax.set_yticklabels(['transported','original'])
        ax.set_title('interpolation pts, ' + \
                     'coord = ({:.2f},{:.2f})'.format(coord0[0],coord0[1]))

        f.savefig(plot_fname,dpi=300)

        return f

    def _newfig(self,dims=(1,1),size=(9,3)):
        r"""

        plotting helper

        """

        f,ax = plt.subplots(ncols=dims[0],\
                            nrows=dims[1],\
                            figsize=(size[0]*dims[0]+1,\
                                     size[1]*dims[1]))

        return f,ax


