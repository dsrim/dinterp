/*  C extensions for dinterp, tools for displacement interpolation
    Donsub Rim, Courant Institute, <sul.rim@gmail.com>
    See LICENSE  */

#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "arrayobject.h"
#include <stdio.h>
#include <math.h>

/* Compute 1D barcyentric coordinate */
double coord(double a, double b, double c){
    
    double lam;
    
    if (b > a) lam = (c - a)/(b - a);
    else lam = 0.0;

    return lam;
    }

/* Find convex combiation between a and b using coordinate c */
double interp(double a, double b, double c){
    return (1.0 - c)*a + c*b;
    }

/* swap two pointers to ints */
void swap_iptrs(int** ipa, int** jpa){

    int* tmp = *ipa;
    *ipa = *jpa;
    *jpa = tmp;
    }

/* swap two pointers to npy_intp */
void swap_npy_intp(npy_intp** ipa, npy_intp** jpa){

    npy_intp *tmp = *ipa;
    *ipa = *jpa;
    *jpa = tmp;
    }

void swap_PyArray_ptrs(PyArrayObject** apa, PyArrayObject** bpa){
    /* swap two pointers to doubles (arrays) */

    PyArrayObject* tmp = *apa;
    *apa = *bpa;
    *bpa = tmp;
    }

/**
 * Find where to insert entries 
 * @param xp  numpy array with x coordinates for first array
 * @param xq  numpy array with x coordinates for second array
 * @param p   numpy array with y coordinates for first array
 * @param q   numpy array with y coordinates for second array
 * @param ins_p2q  array of ints indicated where to insert entries of 
 *                 p into q
 * @param ip  numpy array with y coordinates for second array
 * @param size_p  size of the numpy array xp,p 
 * @param size_q  size of the numpy array xq,q
 */
void insert_entries(PyArrayObject* xp, PyArrayObject* xq, 
                    PyArrayObject* p, PyArrayObject* q, 
                    int* ins_p2q, double* ip, int size_p, int size_q){

    Py_INCREF(xp);
    Py_INCREF(xq);
    Py_INCREF(p);
    Py_INCREF(q);

    int k,l;
    double qkm1,qk,pl,xqkm1,xqk;
    double z;
    npy_intp ikm1[1]={0};
    npy_intp ik[1]={0};
    npy_intp il[1]={0};

    l=0;
    for(k=0; k<size_q; k++){
        while((ins_p2q[l] == k) && (l < size_p)){
            ikm1[0] = (npy_intp) k - (k > 0);
            ik[0] = (npy_intp) k;
            il[0] = (npy_intp) l;
        
            qkm1 = *(double*) PyArray_GetPtr(q,&ikm1[0]);
            qk   = *(double*) PyArray_GetPtr(q,&ik[0]);
            pl   = *(double*) PyArray_GetPtr(p,&il[0]);
            z = coord(qkm1,qk,pl);
            
            xqkm1 = *(double*) PyArray_GetPtr(xq,&ikm1[0]);
            xqk   = *(double*) PyArray_GetPtr(xq,&ik[0]);
            ip[k+l] = interp(xqkm1,xqk,z);
            l++;
            }

        ik[0] = (npy_intp) k;
        xqk = *(double*) PyArray_GetPtr(xq,&ik[0]);
        ip[k+l] = xqk;
        }

    while(k+l < size_p+size_q){
        ikm1[0] = (npy_intp) k-1;
        ip[k+l] = *(double*) PyArray_GetPtr(xq,&ikm1[0]);
        l++;
        }

    Py_DECREF(xp);
    Py_DECREF(xq);
    Py_DECREF(p);
    Py_DECREF(q);
    
    }

/** Merge two monotone non-increasing arrays, so that the share the points 
    on the y-axis
    @param self a python input
    @param args a python input arguments

 */
static PyObject *
    merge_monotone(PyObject *self, PyObject *args){

    PyArrayObject *xp, *xq, *p, *q;
    int i,j,k,l,n;
    const int ndim = 1;
    double tol = 0.0;

    if (!PyArg_ParseTuple(args,"O!O!O!O!d",
                          &PyArray_Type,&xp,&PyArray_Type,&xq,
                          &PyArray_Type,&p, &PyArray_Type,&q,&tol))
        return NULL;

    if (PyArray_NDIM(p) > 1  || PyArray_NDIM(q) > 1 || 
        PyArray_NDIM(xq) > 1 || PyArray_NDIM(xq) > 1 ){
    
        return NULL;
        }

    Py_INCREF(xp);
    Py_INCREF(xq);
    Py_INCREF(p);
    Py_INCREF(q);

    npy_intp* dimsp;
    npy_intp* dimsq;

    dimsp = PyArray_DIMS(p);
    dimsq = PyArray_DIMS(q);
    
    int size_p = (int) *dimsp;
    int size_q = (int) *dimsq;
    int size_o;
    
    double * ip = (double *) malloc((size_p + size_q)*sizeof(double));
    double * iq = (double *) malloc((size_p + size_q)*sizeof(double));
    double * vals = (double*) malloc((size_p + size_q)*sizeof(double));
    
    /* stores indices to insert to */
    int * ins_p2q = (int *) malloc(size_p*sizeof(int)); 
    int * ins_q2p = (int *) malloc(size_q*sizeof(int));

    int *iptr, *jptr; 
    npy_intp *irptr, *jrptr; 
    int *isizeptr, *jsizeptr;
    PyArrayObject *aptr, *bptr; 
    int *a_ins_ptr, *b_ins_ptr; 

    a_ins_ptr = ins_p2q;
    b_ins_ptr = ins_q2p;

    aptr = p;
    bptr = q;

    i = 0;
    j = 0;
    npy_intp ir[1] = {0};
    npy_intp jr[1] = {0};

    k = 0;
    l = 0;

    iptr = &i;
    jptr = &j;
    irptr = &ir[0];
    jrptr = &jr[0];

    isizeptr = &size_p;
    jsizeptr = &size_q;
    
    /* find insertion indices */
    while(*jptr < *jsizeptr){
        while(*jptr < *jsizeptr && 
              *((double*) PyArray_GetPtr(aptr,irptr)) 
              >= *((double*) PyArray_GetPtr(bptr,jrptr))){
            vals[k++] = *(double*) PyArray_GetPtr(bptr,jrptr);
            b_ins_ptr[*jptr] = *iptr;
            *jptr += 1;
            *jrptr += (*jptr < *jsizeptr);
        }
        
        /* swap pointers */
        swap_iptrs(&iptr,&jptr);
        swap_iptrs(&isizeptr,&jsizeptr);
        swap_iptrs(&a_ins_ptr,&b_ins_ptr);
        swap_npy_intp(&irptr,&jrptr);
        swap_PyArray_ptrs(&aptr,&bptr);
    }

    insert_entries(xp,xq,p,q,ins_p2q,iq,size_p,size_q);
    insert_entries(xq,xp,q,p,ins_q2p,ip,size_q,size_p);

    free(ins_p2q);
    free(ins_q2p);
    
    /* count w/o duplicate entries */
    i=0;
    for(l=0; l<size_p + size_q; l++){
        i++;
        k=1;
        while(l+k < size_p + size_q && vals[l+k] - vals[l] <= tol)  k++;
        if (k > 1){
            n=l+k-1;
            if(ip[n] > ip[l] + tol || iq[n] > iq[l] + tol) i++;
            }
        l+=k-1;
        }

    size_o = i;
    npy_intp dimso[1]={size_o};
    PyArrayObject *ip_arr, *iq_arr, *vl_arr;

    ip_arr =  (PyArrayObject*) PyArray_NewFromDescr(&PyArray_Type, 
                               PyArray_DescrFromType(NPY_DOUBLE),
                               ndim, dimso, NULL,NULL,0,NULL);
    
    iq_arr =  (PyArrayObject*) PyArray_NewFromDescr(&PyArray_Type,
                               PyArray_DescrFromType(NPY_DOUBLE),
                               ndim, dimso, NULL,NULL,0,NULL);

    vl_arr =  (PyArrayObject*) PyArray_NewFromDescr(&PyArray_Type,
                               PyArray_DescrFromType(NPY_DOUBLE),
                               ndim,dimso, NULL,NULL,0,NULL);

    npy_intp ii[1];
    i=0;
    for(l=0; l<size_p + size_q; l++){
        ii[0] = (npy_intp) i;
        *(double*) PyArray_GetPtr(ip_arr, ii) = ip[l];  
        *(double*) PyArray_GetPtr(iq_arr, ii) = iq[l];  
        *(double*) PyArray_GetPtr(vl_arr, ii) = vals[l];  
        i++;
        k=1;
        while(l+k < size_p + size_q && vals[l+k] - vals[l] <= tol) k++;
        if (k > 1){
            n=l+k-1;
            if(ip[n] > ip[l] + tol || iq[n] > iq[l] + tol){
                ii[0] = (npy_intp) i;
                *(double*) PyArray_GetPtr(ip_arr, ii) = ip[n];
                *(double*) PyArray_GetPtr(iq_arr, ii) = iq[n];
                *(double*) PyArray_GetPtr(vl_arr, ii) = vals[n];
                i++;
                }
            }
        l+=k-1;
        }
    
    Py_DECREF(xp);
    Py_DECREF(xq);
    Py_DECREF(p);
    Py_DECREF(q);

    free(ip);
    free(iq);
    free(vals);

    return Py_BuildValue("OOO",ip_arr,iq_arr,vl_arr); 
    }


/** 
    
 */
static PyObject *
    compute_pc4pt(PyObject *self, PyObject *args){

    PyArrayObject *pts, *pieces0;
    int i, pn, npts, npieces0;
    double ptsi, ai, bi;
    const int ndim = 1;

    if (!PyArg_ParseTuple(args,"O!O!",
                          &PyArray_Type,&pts,&PyArray_Type,&pieces0))
        return NULL;

    npy_intp* dims;
    npy_intp i0[1] = {0};
    npy_intp i1[2] = {0,0};
    
    dims = PyArray_DIMS(pts);
    npts = (int) dims[0];

    PyArrayObject *pt2pieces;
    pt2pieces =  (PyArrayObject*) PyArray_NewFromDescr(&PyArray_Type, 
                               PyArray_DescrFromType(NPY_INT),
                               ndim, dims, NULL,NULL,0,NULL);
    
    dims = PyArray_DIMS(pieces0);
    npieces0 = (int) dims[0];
    
    i = 0;
    pn = 0;
    while ((i < npts) && (pn < npieces0)){
        i0[0] = i;
        ptsi = *(double*) PyArray_GetPtr(pts,&i0[0]);

        i1[0] = pn; i1[1] = 0;
        ai = *(double*) PyArray_GetPtr(pieces0,&i1[0]) - 0.5;
        i1[0] = pn; i1[1] = 1;
        bi = *(double*) PyArray_GetPtr(pieces0,&i1[0]) - 0.5;
        
        if (ptsi >= ai && ptsi < bi){
            *(int*) PyArray_GetPtr(pt2pieces, &i0[0]) = pn;
            // printf("%6.2f <= %6.4f < %6.2f \n",ai,ptsi,bi);
            i++;
            }
        else {
            pn++;
            if (pn == npieces0){
                if (fabs(ptsi - ai) < 1e-10){
                    *(int*) PyArray_GetPtr(pt2pieces, &i0[0]) = pn;
                    }
                }
            }
        }

    return Py_BuildValue("O",pt2pieces); 
    }

/** 
    
 */
static PyObject *
    compute_transport_pt(PyObject *self, PyObject *args){

    PyArrayObject *ppts, *xv0, *X;
    int i, j, nppts, nxv0;
    double alph, a, b, Ta, Tb;
    double Xjm1, Xj, pptsi;
    const int ndim = 1;

    if (!PyArg_ParseTuple(args,"O!O!O!",
              &PyArray_Type,&ppts,&PyArray_Type,&xv0,&PyArray_Type,&X))
        return NULL;

    Py_INCREF(ppts);
    Py_INCREF(xv0);
    Py_INCREF(X);
    
    npy_intp j0[1] = {0};
    npy_intp* dims; 

    dims = PyArray_DIMS(ppts);
    nppts = dims[0];
    
    npy_intp dimso[1] = {nppts}; 

    PyArrayObject *Tppts;
    Tppts =  (PyArrayObject*) PyArray_NewFromDescr(&PyArray_Type, 
                              PyArray_DescrFromType(NPY_DOUBLE),
                              ndim, dimso, NULL,NULL,0,NULL);

    dims = PyArray_DIMS(xv0);
    nxv0 = (int) dims[0];

    j = 0;
    for (i=0; i < nppts; i++){
        j0[0] = (npy_intp) i;
        pptsi =  *(double*) PyArray_GetPtr(ppts,&j0[0]);
        j0[0] = (npy_intp) j;
        while ((pptsi >= *(double*) PyArray_GetPtr(xv0,&j0[0]))
                && (j < nxv0-1)){ 
            j++;
            j0[0] = (npy_intp) j;
            }

        if (j < nxv0-1){
            j0[0] = (npy_intp) j-1;
            a = *(double*) PyArray_GetPtr(xv0,&j0[0]);
            j0[0] = (npy_intp) j;
            b = *(double*) PyArray_GetPtr(xv0,&j0[0]);
            alph = (pptsi - a)/(b - a);
            // printf("i = %d/%d, %1.4f <= %1.4f < %1.4f \n", i, nppts, a, pptsi, b);
            }
        else {
            /* endpt */
            j0[0] = (npy_intp) j;
            a = *(double*) PyArray_GetPtr(xv0,&j0[0]);
            b = 1.0*a;
            alph = 1.0;
            }

        j0[0] = (npy_intp) j-1;
        Xjm1 = *(double*) PyArray_GetPtr(X,&j0[0]);
        j0[0] = (npy_intp) j;
        Xj =  *(double*) PyArray_GetPtr(X,&j0[0]);

        Ta = a + Xjm1;
        Tb = b + Xj;
        j0[0] = (npy_intp) i;
        *(double*) PyArray_GetPtr(Tppts,&j0[0]) = (1-alph)*Ta + alph*Tb;
    }
    
    Py_DECREF(ppts);
    Py_DECREF(xv0);
    Py_DECREF(X);

    return Py_BuildValue("O", Tppts); 
    // return Py_BuildValue("i", nxv0); 
    }

static PyMethodDef DinterpModuleMethods[] = {
   {"merge_monotone", merge_monotone, METH_VARARGS, 
                                "merge two numpy arrays which are CDFs"},
   {"compute_pc4pt", compute_pc4pt, METH_VARARGS,"find piece for given pt"},
   {"compute_transport_pt", compute_transport_pt, METH_VARARGS,
    "compute transported pt under low-rank transport map"},
   {NULL, NULL, 0, NULL}   /* Sentinel */
   };

static struct PyModuleDef DinterpModule = {
        PyModuleDef_HEAD_INIT,
        "dinterpc",
        "Dinterp C extension",
        -1,
        DinterpModuleMethods
    };

PyMODINIT_FUNC PyInit__dinterpc(void) {
    Py_Initialize();
    import_array();
    return PyModule_Create(&DinterpModule);
    }

