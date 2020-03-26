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
#if defined(_OPENMP)
    #include <omp.h>
#endif

struct module_state {
    PyObject *error;
};

uint32_t finished_combinations = 0;

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#define EXPONENTIAL 2
#define EXPONENTIAL_COSINE 4

typedef npy_float32 float32_t;
typedef npy_float64 float64_t;

typedef enum {
    SUCCESS = 0,
    SAUBSAMPLING_SPARSE_ERROR
} state_covariance;

int good_array(PyObject* o, int typenum, npy_intp size_want, int ndim_want, npy_intp* shape_want) {
    int i;

    if (!PyArray_Check(o)) {
        PyErr_SetString(PyExc_AttributeError, "not a NumPy array" );
        return 0;
    }

    if (PyArray_TYPE((PyArrayObject*)o) != typenum) {
        PyErr_SetString(PyExc_AttributeError, "array aof unexpected type");
        return 0;
    }

    if (!PyArray_ISCARRAY((PyArrayObject*)o)) {
        PyErr_SetString(PyExc_AttributeError, "array is not contiguous or not well behaved");
        return 0;
    }

    if (size_want != -1 && size_want != PyArray_SIZE((PyArrayObject*)o)) {
        PyErr_SetString(PyExc_AttributeError, "array is of unexpected size");
        return 0;
    }


    if (ndim_want != -1 && ndim_want != PyArray_NDIM((PyArrayObject*)o)) {
        PyErr_SetString(PyExc_AttributeError, "array is of unexpected ndim");
        return 0;
    }

    if (ndim_want != -1 && shape_want != NULL) {
        for (i=0; i<ndim_want; i++) {
            if (shape_want[i] != -1 && shape_want[i] != PyArray_DIMS((PyArrayObject*)o)[i]) {
                PyErr_SetString(PyExc_AttributeError, "array is of unexpected shape");
                return 0;
            }
        }
    }
    return 1;
}


// saves 50% computation
inline float64_t exp2(float64_t x) {
  x = 1.0 + x / 1024;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x;
  return x;
}

static float64_t covariance_model_exp(float64_t dist, float64_t *model_coeff) {
    return model_coeff[0] * exp(-dist / model_coeff[1]);
}

static float64_t covariance_model_exp_cosine(float64_t dist, float64_t *model_coeff) {
    return model_coeff[0] * exp(-dist / model_coeff[1]) * cos((dist-model_coeff[2])/model_coeff[3]);
}

static state_covariance calc_covariance_matrix(
                float64_t *E,
                float64_t *N,
                npy_intp *shape_coord,
                uint32_t *map,
                uint32_t nleaves,
                float64_t model_coeff[],
                uint32_t model_ncoeff,
                float64_t variance,
                uint32_t nthreads,
                uint32_t adaptive_subsampling,
                float64_t *cov_arr) {
    npy_intp l1row_beg, l1row_end, l1col_beg, l1col_end, il1row, il1col;
    npy_intp l2row_beg, l2row_end, l2col_beg, l2col_end, il2row, il2col;
    npy_intp icl1, icl2, nrows, ncols;
    npy_intp il1, il2, npx, l_length;
    uint32_t subsampling[nleaves], l1hit, l2hit, tid;
    float64_t cov, dist;
    float64_t (*cov_model) (float64_t, float64_t *);

    // Silence warnings
    (void) tid;
    (void) nthreads;
    finished_combinations = 0;
    cov_model = &covariance_model_exp;

    nrows = (npy_intp) shape_coord[0];
    ncols = (npy_intp) shape_coord[1];
    if (model_ncoeff == EXPONENTIAL)
        cov_model = &covariance_model_exp;
    else if (model_ncoeff == EXPONENTIAL_COSINE)
        cov_model = &covariance_model_exp_cosine;

    // Defining adaptive subsampling strategy
    for (il1=0; il1<nleaves; il1++) {
        if (adaptive_subsampling) {
            l_length = map[il1*4+1] - map[il1*4+0];
            subsampling[il1] = ceil(LOG2(l_length));
        } else {
            subsampling[il1] = 1;
        }
    }

    // printf("coord_matrix: %ldx%ld\n", nrows, ncols);
    // printf("subsampling: %d\n", subsampling);
    // printf("nthreads: %d\n", nthreads);
    Py_BEGIN_ALLOW_THREADS
    #if defined(_OPENMP)
        if (nthreads == 0)
            nthreads = omp_get_num_procs();
    #endif
    #if defined(_OPENMP)
        #pragma omp parallel \
            shared (E, N, map, model_coeff, cov_model, cov_arr, nrows, ncols, nleaves, subsampling, finished_combinations) \
            private (il1row, il1col, icl1, l1row_beg, l1row_end, l1col_beg, l1col_end, \
                     l2row_beg, l2row_end, l2col_beg, l2col_end, il2row, il2col, icl2, il2, \
                     npx, cov, l1hit, l2hit, tid, dist) \
            num_threads (nthreads)
        {
        tid = omp_get_thread_num();
        #pragma omp for schedule (dynamic)
    #endif
    for (il1=0; il1<nleaves; il1++) {
        l1row_beg = map[il1*4+0];
        l1row_end = map[il1*4+1];
        l1col_beg = map[il1*4+2];
        l1col_end = map[il1*4+3];
        // printf("l(%lu): %lu-%lu:%lu-%lu (ss %d)\n", il1, l1row_beg, l1row_end, l1col_beg, l1col_end, subsampling[il1]);
        for (il2=il1; il2<nleaves; il2++) {
            l2row_beg = map[il2*4+0];
            l2row_end = map[il2*4+1];
            l2col_beg = map[il2*4+2];
            l2col_end = map[il2*4+3];

            l1hit = 0;
            l2hit = 0;

            cov = 0.;
            npx = 0;
            while(! (l1hit && l2hit)) {
                // printf("thread %d :: l(%lu-%lu) :: %lu:%lu (ss %d) %lu:%lu (ss %d)\n", tid, il1, il2, (l1row_end-l1row_beg), (l1col_end-l1col_beg), subsampling[il1], (l2row_end-l2row_beg), (l2col_end-l2col_beg), subsampling[il2]);
                for (il1row=l1row_beg; il1row<l1row_end; il1row++) {
                    if (il1row > nrows)
                        continue;
                    for (il1col=l1col_beg; il1col<l1col_end; il1col+=subsampling[il1]) {
                        if (il1col > ncols)
                            continue;
                        icl1 = il1row*ncols + il1col;
                        if (npy_isnan(E[icl1]) || npy_isnan(N[icl1]))
                            continue;
                        l1hit = 1;

                        for (il2row=l2row_beg; il2row<l2row_end; il2row++) {
                            if (il2row > nrows)
                                continue;
                            for (il2col=l2col_beg; il2col<l2col_end; il2col+=subsampling[il2]) {
                                if (il2col > ncols)
                                    continue;
                                icl2 = il2row*ncols + il2col;
                                if (npy_isnan(E[icl2]) || npy_isnan(N[icl2]))
                                    continue;
                                l2hit = 1;
                                dist = sqrt(SQR(E[icl1]-E[icl2]) + SQR(N[icl1]-N[icl2]));
                                if (dist == 0.)
                                    cov += variance;
                                else
                                    cov += cov_model(dist, model_coeff);
                                npx++;
                            }
                        }
                    }
                }
                #if defined(_OPENMP)
                    #pragma omp critical
                    {
                #endif
                    if (! l1hit) {
                        subsampling[il1] = floor(subsampling[il1]/2);
                    }
                    if (! l2hit) {
                        subsampling[il2] = floor(subsampling[il2]/2);
                    }
                #if defined(_OPENMP)
                    }
                #endif
            }
            #if defined(_OPENMP)
                #pragma omp atomic
            #endif
            finished_combinations++;
            cov_arr[il1*(nleaves)+il2] = cov/npx;
            cov_arr[il2*(nleaves)+il1] = cov_arr[il1*(nleaves)+il2];
        }
    }
    #if defined(_OPENMP)
        }
    #endif
    Py_END_ALLOW_THREADS
    finished_combinations = 0;
    return SUCCESS;
}

static PyObject* w_calc_covariance_matrix(PyObject *m, PyObject *args, PyObject *kwds) {
    PyObject *E_arr, *N_arr, *map_arr, *coeff;
    PyArrayObject *c_E_arr, *c_N_arr, *c_map_arr, *cov_arr;

    float64_t *x, *y, *covs, variance, model_coeff[4];
    uint32_t *map, nthreads, adaptive_subsampling, i, model_ncoeff;
    npy_intp shape_coord[2], shape_dist[2], nleaves;
    npy_intp shape_want_map[2] = {-1, 4};

    nthreads = 0;
    adaptive_subsampling = 1;

    struct module_state *st = GETSTATE(m);
    state_covariance err;

    static char *kwlist[] = {
        "E",
        "N",
        "map",
        "coefficients",
        "variance",
        "nthreads",
        "adaptive_subsampling",
        NULL
    };

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "OOOOd|Ip", kwlist, &E_arr, &N_arr, &map_arr, &coeff, &variance, &nthreads, &adaptive_subsampling)) {
        PyErr_SetString(st->error, "usage: distances(East, North, map, model_coefficients, nthreads=0, adaptive_subsampling=True)");
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
        PyErr_SetString(st->error, "EastCoords and NorthCoords must have the same size!");
        return NULL;
    }

    x = PyArray_DATA(c_E_arr);
    y = PyArray_DATA(c_N_arr);
    map = PyArray_DATA(c_map_arr);
    nleaves = PyArray_SIZE(c_map_arr)/4;

    model_ncoeff = (uint32_t) PyTuple_Size(coeff);
    for (i=0; i < model_ncoeff; i++)
        model_coeff[i] = PyFloat_AsDouble(PyTuple_GetItem(coeff, i));

    shape_coord[0] = (npy_intp) PyArray_DIMS(c_E_arr)[0];
    shape_coord[1] = (npy_intp) PyArray_DIMS(c_E_arr)[1];
    shape_dist[0] = nleaves;
    shape_dist[1] = nleaves;

    cov_arr = (PyArrayObject*) PyArray_EMPTY(2, shape_dist, NPY_FLOAT64, 0);
    // printf("size distance matrix: %lu\n", PyArray_SIZE(cov_arr));
    // printf("size coord matrix: %lu\n", PyArray_SIZE(E_arr));
    covs = PyArray_DATA(cov_arr);

    err = calc_covariance_matrix(x, y, shape_coord, map, nleaves, model_coeff, (int) model_ncoeff, variance, nthreads, adaptive_subsampling, covs);
    if (err != SUCCESS) {
        PyErr_SetString(st->error, "Calculating covariance failed!");
        return NULL;
    }
    return (PyObject*) cov_arr;
}


static PyObject* w_get_finished_combinations(PyObject *m) {
    (void) m;
    return PyLong_FromSize_t(finished_combinations);
}

static PyMethodDef CovarianceExtMethods[] = {
    {
     "covariance_matrix",
     (PyCFunctionWithKeywords) w_calc_covariance_matrix,
     METH_VARARGS | METH_KEYWORDS,
     "Calculates the covariance matrix for full resolution."
    }, {
     "get_finished_combinations",
     (PyCFunction) w_get_finished_combinations,
     METH_NOARGS,
     "Get the number of calculated combinations. Total combinations are nleaves*(nleaves+1)/2."
    }, {NULL, NULL}         /* Sentinel */
};

static int calc_covariance_matrix_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int calc_covariance_matrix_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "covariance_ext",
        NULL,
        sizeof(struct module_state),
        CovarianceExtMethods,
        NULL,
        calc_covariance_matrix_traverse,
        calc_covariance_matrix_clear,
        NULL
};


#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_covariance_ext(void)
{
    PyObject *module = PyModule_Create(&moduledef);
    if (module == NULL)
        INITERROR;
    import_array();

    struct module_state *st = GETSTATE(module);
    st->error = PyErr_NewException("kite.covariance_ext.error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

    Py_INCREF(st->error);
    PyModule_AddObject(module, "error", st->error);

    return module;
}
