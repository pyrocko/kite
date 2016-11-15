#define NPY_NO_DEPRECATED_API 7

#include "Python.h"
#include "numpy/arrayobject.h"

#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

typedef npy_float32 float32_t;
typedef npy_float64 float64_t;
typedef npy_uint32 uint32_t;

static PyObject *CovarianceExtError;

int good_array(PyObject* o, int typenum, ssize_t size_want, int ndim_want, npy_intp* shape_want) {
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

    if (ndim_want != -1) {
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


npy_intp get_nleafs(uint32_t *map, npy_intp ncombinations) {
    npy_intp i;
    uint32_t max = 0;
    for (i=0; i < ncombinations; i++) {
        if (map[i+6] < max) max = map[i+6];
    }
    return max;
}

static void calc_distances(float64_t *X, float64_t *Y, npy_intp *shape, uint32_t *map, float64_t *distances) {
    (void) sqr;
    printf("%lu\n", shape[0]);
}

static PyObject* w_distances(PyObject *dummy, PyObject *args) {
    PyObject *x_arr, *y_arr, *map_arr;
    PyArrayObject *c_x_arr, *c_y_arr, *c_map_arr, *dists_arr;

    float64_t *x, *y, *dists;
    uint32_t *map;
    npy_intp shape_coord[2], shape_dist[2], nleafs;

    npy_intp shape_want_map[2] = {-1, 6};

    if (! PyArg_ParseTuple(args, "OOO", &x_arr, &y_arr, &map_arr)) {
        PyErr_SetString(CovarianceExtError, "usage: distances(X, Y, map");
        return NULL;
    }

    if (! good_array(x_arr, NPY_FLOAT64, -1, 2, NULL)) return NULL;
    if (! good_array(y_arr, NPY_FLOAT64, -1, 2, NULL)) return NULL;
    if (! good_array(map_arr, NPY_UINT32, -1, 2, shape_want_map)) return NULL;

    c_x_arr = PyArray_GETCONTIGUOUS((PyArrayObject*) x_arr);
    c_y_arr = PyArray_GETCONTIGUOUS((PyArrayObject*) y_arr);
    c_map_arr = PyArray_GETCONTIGUOUS((PyArrayObject*) map_arr);

    if (PyArray_SHAPE(c_x_arr) != PyArray_SHAPE(c_y_arr)) {
        PyErr_SetString(CovarianceExtError, "X and Y must have the same shape!");
        return NULL;
    }

    x = PyArray_DATA(c_x_arr);
    y = PyArray_DATA(c_y_arr);
    map = PyArray_DATA(c_map_arr);

    printf("shape: %lu\n", PyArray_SHAPE(c_map_arr)[0]);

    nleafs = get_nleafs(map, PyArray_SHAPE(c_map_arr)[0]);
    shape_dist[0] = nleafs;
    shape_dist[1] = nleafs;
    //shape_coord = PyArray_SHAPE(c_x_arr);

    dists_arr = (PyArrayObject*) PyArray_EMPTY(2, shape_dist, NPY_FLOAT64, 0);
    dists = PyArray_DATA(dists_arr);

    calc_distances(x, y, shape_coord, map, dists);
    return args;
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
