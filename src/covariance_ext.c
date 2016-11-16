#define NPY_NO_DEPRECATED_API 7

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

static float64_t sqr(float64_t x) {
    return x*x;
}

static void calc_distances(float64_t *X, float64_t *Y, npy_intp *shape_coord, uint32_t *map, uint32_t nleafs, uint32_t subsampling, float64_t *dists) {
    uint32_t il1, il2, ndist;
    uint32_t l1x_beg, l1x_end, l1y_beg, l1y_end, il1x, il1y;
    uint32_t l2x_beg, l2x_end, l2y_beg, l2y_end, il2x, il2y;
    float64_t dist;
    npy_intp icl1, icl2, idist, coord_rows, coord_cols;


    coord_rows = shape_coord[0];
    coord_cols = shape_coord[1];
    printf("coord_matrix: %dx%d\n", coord_rows, coord_cols);
    printf("subsampling: %d\n", subsampling);

    for (il1=0; il1<nleafs; il1++) {
        l1y_beg = map[il1*4+0];
        l1y_end = map[il1*4+1];
        l1x_beg = map[il1*4+2];
        l1x_end = map[il1*4+3];
        printf("l(%d): %d-%d:%d-%d\n", il1, l1x_beg, l1x_end, l1y_beg, l1y_end);

        for (il2=il1; il2<nleafs; il2++) {
            l2y_beg = map[il2*4+0];
            l2y_end = map[il2*4+1];
            l2x_beg = map[il2*4+2];
            l2x_end = map[il2*4+3];

            dist = 0.;
            ndist = 0;
            // printf("Calculating for %dx%d (%d)\n", il1, il2, idist);
            for (il1x=l1x_beg; il1x<l1x_end; il1x++) {
                if (il1x > coord_cols) continue;
                for (il1y=l1y_beg; il1y<l1y_end; il1y+=subsampling) {
                    if (il1y > coord_rows) continue;
                    icl1 = il1x + coord_cols*il1y;
                    if (npy_isnan(X[icl1])) continue;

                    for (il2x=l2x_beg; il2x<l2x_end; il2x++) {
                        if (il2x > coord_cols) continue;
                        for (il2y=l2y_beg; il2y<l2y_end; il2y+=subsampling) {
                            if (il2y > coord_rows) continue;
                            icl2 = il2x + coord_cols*il2y;
                            if (npy_isnan(X[icl2])) continue;

                            dist += sqrt(sqr(X[icl1] - X[icl2]) + sqr(Y[icl1] - Y[icl2]));
                            ndist++;
                        }
                    }
                }
            }
            dists[il1*(nleafs)+il2] = dist/ndist;
            dists[il2*(nleafs)+il1] = dist/ndist;
            // printf("l%d-l%d: %f\n", il1, il2, dist);
        }
    }
}

static PyObject* w_distances(PyObject *dummy, PyObject *args) {
    PyObject *x_arr, *y_arr, *map_arr;
    PyArrayObject *c_x_arr, *c_y_arr, *c_map_arr, *dists_arr;

    float64_t *x, *y, *dists;
    uint32_t *map, subsampling;
    npy_intp *shape_coord[2], shape_dist[2], nleafs, ncomb;
    npy_intp shape_want_map[2] = {-1, 4};

    if (! PyArg_ParseTuple(args, "OOOI", &x_arr, &y_arr, &map_arr, &subsampling)) {
        PyErr_SetString(CovarianceExtError, "usage: distances(X, Y, map, subsampling)");
        return NULL;
    }

    if (! good_array(x_arr, NPY_FLOAT64, -1, 2, NULL))
        return NULL;
    if (! good_array(y_arr, NPY_FLOAT64, -1, 2, NULL))
        return NULL;
    if (! good_array(map_arr, NPY_UINT32, -1, 2, shape_want_map))
        return NULL;

    c_x_arr = PyArray_GETCONTIGUOUS((PyArrayObject*) x_arr);
    c_y_arr = PyArray_GETCONTIGUOUS((PyArrayObject*) y_arr);
    c_map_arr = PyArray_GETCONTIGUOUS((PyArrayObject*) map_arr);


    if (PyArray_SIZE(c_x_arr) != PyArray_SIZE(c_y_arr)) {
        PyErr_SetString(CovarianceExtError, "X and Y must have the same size!");
        return NULL;
    }

    x = PyArray_DATA(c_x_arr);
    y = PyArray_DATA(c_y_arr);
    map = PyArray_DATA(c_map_arr);
    nleafs = PyArray_SIZE(c_map_arr)/4;

    shape_coord[0] = PyArray_DIMS(c_x_arr)[0];
    shape_coord[1] = PyArray_DIMS(c_x_arr)[1];
    shape_dist[0] = nleafs;
    shape_dist[1] = nleafs;
    printf("nleafs: %d\n", nleafs);

    dists_arr = (PyArrayObject*) PyArray_EMPTY(2, shape_dist, NPY_FLOAT64, 0);
    printf("size distance matrix: %d\n", PyArray_SIZE(dists_arr));
    printf("size coord matrix: %d\n", PyArray_SIZE(x_arr));
    dists = PyArray_DATA(dists_arr);

    calc_distances(x, y, shape_coord, map, nleafs, subsampling, dists);
    return (PyObject*) dists_arr;
}

static PyMethodDef CovarianceExtMethods[] = {
    {"leaf_distances", w_distances, METH_VARARGS,
     "Calculate mean distances between leafs!"},

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
