#define NPY_NO_DEPRECATED_API 7
#define SQR(a)  ( (a) * (a) )
#define LOG2(a)  ( (log(a)) * 1.44269504088896340736 )

#include "Python.h"
#include "numpy/arrayobject.h"
#include <numpy/npy_math.h>

#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

typedef npy_float32 float32_t;
typedef npy_float64 float64_t;
typedef npy_uint32 uint32_t;

static PyObject *CovarianceExtError;

int good_array(PyObject* o, int typenum, npy_intp size_want, int ndim_want, npy_intp* shape_want) {
    int i;

    if (!PyArray_Check(o)) {
        PyErr_SetString(CovarianceExtError, "not a NumPy array" );
        return 0;
    }

    if (PyArray_TYPE((PyArrayObject*)o) != typenum) {
        PyErr_SetString(CovarianceExtError, "array of unexpected type");
        return 0;
    }

    if (!PyArray_ISCARRAY((PyArrayObject*)o)) {
        PyErr_SetString(CovarianceExtError, "array is not contiguous or not well behaved");
        return 0;
    }

    if (size_want != -1 && size_want != PyArray_SIZE((PyArrayObject*)o)) {
        PyErr_SetString(CovarianceExtError, "array is of unexpected size");
        return 0;
    }


    if (ndim_want != -1 && ndim_want != PyArray_NDIM((PyArrayObject*)o)) {
        PyErr_SetString(CovarianceExtError, "array is of unexpected ndim");
        return 0;
    }

    if (ndim_want != -1 && shape_want != NULL) {
        for (i=0; i<ndim_want; i++) {
            if (shape_want[i] != -1 && shape_want[i] != PyArray_DIMS((PyArrayObject*)o)[i]) {
                PyErr_SetString(CovarianceExtError, "array is of unexpected shape");
                return 0;
            }
        }
    }
    return 1;
}

static void calc_covariance_matrix(
                float64_t *E,
                float64_t *N,
                npy_intp *shape_coord,
                uint32_t *map,
                uint32_t nleafs,
                float64_t ma,
                float64_t mb,
                uint32_t nthreads,
                uint32_t adaptive_subsampling,
                float64_t *cov_arr) {
    npy_intp l1row_beg, l1row_end, l1col_beg, l1col_end, il1row, il1col;
    npy_intp l2row_beg, l2row_end, l2col_beg, l2col_end, il2row, il2col;
    npy_intp icl1, icl2, nrows, ncols;
    npy_intp il1, il2, npx, l_length;
    uint32_t leaf_subsampling[nleafs], l1hit, l2hit, tid;
    float64_t cov;

    nrows = (npy_intp) shape_coord[0];
    ncols = (npy_intp) shape_coord[1];

    // Defining adaptive subsampling
    for (il1=0; il1<nleafs; il1++) {
        if (adaptive_subsampling) {
            l_length = map[il1*4+1] - map[il1*4+0];
            leaf_subsampling[il1] = ceil(LOG2(l_length));
        } else {
           leaf_subsampling[il1] = 1;
        }
    }
    if (nthreads == 0)
        nthreads = omp_get_num_procs();

    // printf("coord_matrix: %ldx%ld\n", nrows, ncols);
    // printf("subsampling: %d\n", subsampling);
    // printf("nthreads: %d\n", nthreads);
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel \
        shared (E, N, map, cov_arr, nrows, ncols, nleafs, leaf_subsampling) \
        private (l1row_beg, l1row_end, l1col_beg, l1col_end, il1row, il1col, icl1, \
                 l2row_beg, l2row_end, l2col_beg, l2col_end, il2row, il2col, icl2, il2, \
                 npx, cov, l1hit, l2hit, tid) \
        num_threads (nthreads)
    {
        tid = omp_get_thread_num();
        #pragma omp for schedule (dynamic)
        for (il1=0; il1<nleafs; il1++) {
            l1row_beg = map[il1*4+0];
            l1row_end = map[il1*4+1];
            l1col_beg = map[il1*4+2];
            l1col_end = map[il1*4+3];
            // printf("l(%lu): %lu-%lu:%lu-%lu (ss %d)\n", il1, l1row_beg, l1row_end, l1col_beg, l1col_end, leaf_subsampling[il1]);
            for (il2=il1; il2<nleafs; il2++) {
                l2row_beg = map[il2*4+0];
                l2row_end = map[il2*4+1];
                l2col_beg = map[il2*4+2];
                l2col_end = map[il2*4+3];

                l1hit = 0;
                l2hit = 0;

                cov = 0.;
                npx = 0;
                while(! (l1hit && l2hit)) {
                    // printf("tid %d :: l(%lu-%lu) :: %lu:%lu (ss %d) %lu:%lu (ss %d)\n", tid, il1, il2, (l1row_end-l1row_beg), (l1col_end-l1col_beg), leaf_subsampling[il1], (l2row_end-l2row_beg), (l2col_end-l2col_beg), leaf_subsampling[il2]);
                    for (il1row=l1row_beg; il1row<l1row_end; il1row++) {
                        if (il1row > nrows) continue;
                        for (il1col=l1col_beg; il1col<l1col_end; il1col+=leaf_subsampling[il1]) {
                            if (il1col > ncols) continue;
                            icl1 = il1row*ncols + il1col;
                            if (npy_isnan(E[icl1]) || npy_isnan(N[icl1])) continue;
                            l1hit = 1;

                            for (il2row=l2row_beg; il2row<l2row_end; il2row++) {
                                if (il2row > nrows) continue;
                                for (il2col=l2col_beg; il2col<l2col_end; il2col+=leaf_subsampling[il2]) {
                                    if (il2col > ncols) continue;
                                    icl2 = il2row*ncols + il2col;
                                    if (npy_isnan(E[icl2]) || npy_isnan(N[icl2])) continue;
                                    l2hit = 1;

                                    cov += exp(-sqrt(SQR(E[icl1]-E[icl2]) + SQR(N[icl1]-N[icl2])) / mb);
                                    npx++;
                                }
                            }
                        }
                    }
                    #pragma omp critical
                    {
                        if (! l1hit) {
                            leaf_subsampling[il1] = floor(leaf_subsampling[il1]/2);
                            printf("no hit!\n");
                        }
                        if (! l2hit) {
                            leaf_subsampling[il2] = floor(leaf_subsampling[il2]/2);
                            printf("no hit!\n");
                        }
                    }
                }
                cov_arr[il1*(nleafs)+il2] = ma * (cov/npx);
                cov_arr[il2*(nleafs)+il1] = cov_arr[il1*(nleafs)+il2];
            }
        }
    }
    Py_END_ALLOW_THREADS
}

static PyObject* w_calc_covariance_matrix(PyObject *dummy, PyObject *args) {
    PyObject *E_arr, *N_arr, *map_arr;
    PyArrayObject *c_E_arr, *c_N_arr, *c_map_arr, *cov_arr;

    float64_t *x, *y, *covs, ma, mb;
    uint32_t *map, nthreads, adaptive_subsampling;
    npy_intp shape_coord[2], shape_dist[2], nleafs;
    npy_intp shape_want_map[2] = {-1, 4};

    if (! PyArg_ParseTuple(args, "OOOddII", &E_arr, &N_arr, &map_arr, &ma, &mb, &nthreads, &adaptive_subsampling)) {
        PyErr_SetString(CovarianceExtError, "usage: distances(X, Y, map, covmodel_a, covmodel_b, nthreads, adaptive_subsampling)");
        return NULL;
    }

    if (! good_array(E_arr, NPY_FLOAT64, -1, 2, NULL))
        return NULL;
    if (! good_array(N_arr, NPY_FLOAT64, -1, 2, NULL))
        return NULL;
    if (! good_array(map_arr, NPY_UINT32, -1, 2, shape_want_map))
        return NULL;

    c_E_arr = PyArray_GETCONTIGUOUS((PyArrayObject*) E_arr);
    c_N_arr = PyArray_GETCONTIGUOUS((PyArrayObject*) N_arr);
    c_map_arr = PyArray_GETCONTIGUOUS((PyArrayObject*) map_arr);


    if (PyArray_SIZE(c_E_arr) != PyArray_SIZE(c_N_arr)) {
        PyErr_SetString(CovarianceExtError, "X and Y must have the same size!");
        return NULL;
    }

    x = PyArray_DATA(c_E_arr);
    y = PyArray_DATA(c_N_arr);
    map = PyArray_DATA(c_map_arr);
    nleafs = PyArray_SIZE(c_map_arr)/4;

    shape_coord[0] = (npy_intp) PyArray_DIMS(c_E_arr)[0];
    shape_coord[1] = (npy_intp) PyArray_DIMS(c_E_arr)[1];
    shape_dist[0] = nleafs;
    shape_dist[1] = nleafs;

    cov_arr = (PyArrayObject*) PyArray_EMPTY(2, shape_dist, NPY_FLOAT64, 0);
    // printf("size distance matrix: %lu\n", PyArray_SIZE(cov_arr));
    // printf("size coord matrix: %lu\n", PyArray_SIZE(E_arr));
    covs = PyArray_DATA(cov_arr);

    calc_covariance_matrix(x, y, shape_coord, map, nleafs, ma, mb, nthreads, adaptive_subsampling, covs);
    return (PyObject*) cov_arr;
}

static PyMethodDef CovarianceExtMethods[] = {
    {"covariance_matrix", w_calc_covariance_matrix, METH_VARARGS,
     "Calculates the covariance matrix for full resolution." },

    {NULL, NULL, 0, NULL}         /* Sentinel */
};

PyMODINIT_FUNC
initcovariance_ext(void)
{
    PyObject *m;

    m = Py_InitModule("covariance_ext", CovarianceExtMethods);
    if (m == NULL) return;
    import_array();

    CovarianceExtError = PyErr_NewException("covariance_ext.error", NULL, NULL);
    Py_INCREF(CovarianceExtError);  /* required, because other code could remove `error`
                               from the module, what would create a dangling
                               pointer. */
    PyModule_AddObject(m, "CovarianceExtError", CovarianceExtError);
}
