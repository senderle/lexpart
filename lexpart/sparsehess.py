import timeit
from math import exp, log, isnan, isinf

import numpy
import numba

_return_type = numba.types.Tuple((numba.float64,
                                  numba.float64[:],
                                  numba.float64[:, :]))


@numba.jit(_return_type(numba.float64[:], numba.float64[:]),
           nopython=True,
           nogil=True)
def hessian(ps, xs):
    """
    Calculate the hessian of a polynomial term (i.e. a monomial) at a
    given point, represented as two aligned sequences, the first,
    `ps`, containing the powers of the term, and the second, `xs`,
    containing the point at which to evaluate the monomial.

    Since differentiation is linear, this can be repeated for multiple
    terms, and the result can be summed to calculate the hessian of
    their sum (i.e. a polynomial).
    """

    f_out = numpy.prod(numpy.power(xs, ps))
    jac = numpy.empty((len(ps),))
    jac[:] = f_out
    hess = numpy.empty((len(ps), len(ps)))

    for i in range(len(ps)):
        jac[i] *= ps[i] / xs[i]
        for j in range(len(ps)):
            ps_j = ps[j] if i != j else ps[j] - 1
            hess[i, j] = jac[i] * ps_j / xs[j]

    return f_out, jac, hess


_return_type = numba.types.Tuple((numba.float64,
                                  numba.float64[:],
                                  numba.float64[:, :]))


@numba.jit(_return_type(numba.float64[:], numba.float64[:]),
           nopython=True,
           nogil=True)
def log_hessian(ps, xs):
    """
    Perform the same operation as `hessian` above, but do so in log
    space. This is slower but might (?) be more numerically stable
    in some cases.
    """
    log_ps = numpy.log(ps)
    log_xs = numpy.log(xs)
    f_out = (log_xs * ps).sum()
    jac = numpy.empty((len(ps),))
    jac[:] = f_out
    hess = numpy.empty((len(ps), len(ps)))

    for i in range(len(ps)):
        jac[i] += log_ps[i] - log_xs[i]
        for j in range(len(ps)):
            ps_j = ps[j] if i != j else ps[j] - 1
            hess[i, j] = exp(jac[i] - log_xs[j]) * ps_j

    return exp(f_out), numpy.exp(jac), hess


@numba.jit(numba.void(numba.float64[:],
                      numba.float64[:, :],
                      numba.float64[:, :, :],
                      numba.int64[:],
                      numba.float64[:],
                      numba.float64[:]),
           nopython=True,
           nogil=True,
           parallel=True)
def hessian_multi(f_out,
                  jacobian_out,
                  hessian_out,
                  ends,
                  counts,
                  potentials):
    """
    Calculate the hessian of multiple terms in parallel and store
    the results in preallocated buffers.
    """
    n_terms = ends.shape[0]
    starts = numpy.empty(ends.shape, dtype=numpy.int64)
    starts[0] = 0
    starts[1:] = ends[:-1]
    for term_ix in numba.prange(n_terms):
        start = starts[term_ix]
        end = ends[term_ix]
        term_counts = counts[start: end]
        term_potentials = potentials[start: end]
        f, jac, hess = hessian(term_counts, term_potentials)
        f_out[term_ix] = f
        jacobian_out[term_ix] = jac
        hessian_out[term_ix] = hess


# This describes the datatype of `hessian_projection` for ahead-of-time
# compilation. But just-in-time is more flexible and accommodates numpy
# zero-dim arrays more easily. This is included for the sake of
# documentation, but it is not used.
_hessian_projection_dtype = numba.int64(
        # Zero-dim arrays were faster than ordinary python scalars in tests.
        numba.types.Array(numba.float64, 0, 'A'),
        numba.float64[:],
        numba.float64[:, :],
        numba.int64[:],
        numba.int64[:],
        numba.int64[:],
        numba.float64[:],
        numba.float64[:, :],
        numba.float64)


@numba.jit(nopython=True,
           nogil=True)
def hessian_projection(
        fout,
        jacobian,
        rand_hess,
        ends,
        indices,
        counts,
        potentials,
        proj_vecs,
        scale_factor):
    """
    Calculate a random projection of the hessian for multiple polynomial
    terms. This is a simple and fairly effective form of dimension
    reduction. It's also, from the point of view of calculus, a matrix
    of random directional derivatives of the polynomial.
    """
    embed_size = proj_vecs.shape[1]
    err_count = 0
    start = 0
    n_docs = ends.shape[0]
    for doc_ix in range(n_docs):
        end = ends[doc_ix]
        hess_size = end - start

        # Calclulate the value of the polynomial term in log space.
        poly_pow = 0
        for i in range(hess_size):
            poly_pow += log(potentials[start + i]) * counts[start + i]

        # Briefly drop out of log space to save the scalar output
        # of the polynomial term.
        fout += exp(poly_pow) * scale_factor

        for i in range(hess_size):
            w_i = indices[start + i]
            counts_i = counts[start + i]
            potentials_i = potentials[start + i]

            # Calculate the value of the jacobian by dividing out the
            # value of the given variable (named `potentials_i` here) and
            # multiplying the term by the exponent (named `counts_i` here).
            jac_i = (poly_pow +
                     log(counts_i) -
                     log(potentials_i))

            # Briefly drop out of log space to update jacobian
            jacobian[w_i] += exp(jac_i) * scale_factor

            for j in range(hess_size):
                # Caclulate a hessian term by repeating the process used to
                # calculate the jacobian.
                hess_i_j = exp(jac_i +
                               log(counts[start + j]) -
                               log(potentials[start + j]))

                # If we are on a diagonal, we need a nonlinear correction
                # to get the second partial derivative instead of the mixed
                # partial derivative. (Second derivative of x^3 is 6x, but
                # without this correction we would get 9x.)
                if i == j:
                    hess_i_j *= (counts_i - 1) / counts_i

                # Drop unexpected NANs.
                if isnan(hess_i_j) or isinf(hess_i_j):
                    err_count += 1
                    continue

                # Perform the random projection.
                w_j = indices[start + j]
                for k in range(embed_size):
                    rand_hess[w_i, k] += (hess_i_j *
                                          proj_vecs[w_j, k] *
                                          scale_factor)

        start = end

    return err_count


# An experimental feature that corrects for the fact that the hessian of
# the partition function with respect to potential (mu) is not quite the
# same as the hessian of the partition function with respect to the
# simplified potential (u, where u = e^(beta * mu)). This gets the former
# from the latter. But empirically, there seems to be no difference in
# performance. Probably the following is true: this amounts to a difference
# in the degree of the terms of the derivative, which doesn't have much
# effect in the immediate neighborhood of the point of evaluation.
@numba.jit(nopython=True,
           nogil=True)
def elastic_hessian_projection(
        fout,
        jacobian,
        rand_hess,
        ends,
        indices,
        counts,
        potentials,
        proj_vecs,
        scale_factor):
    """
    Calculate a random projection of the hessian for multiple polynomial
    terms. This is a simple and fairly effective form of dimension
    reduction. It's also, from the point of view of calculus, a matrix
    of random directional derivatives of the polynomial.
    """
    embed_size = proj_vecs.shape[1]
    err_count = 0
    start = 0
    n_docs = ends.shape[0]
    for doc_ix in range(n_docs):
        end = ends[doc_ix]
        hess_size = end - start

        # Calclulate the value of the polynomial term in log space.
        poly_pow = 0
        for i in range(hess_size):
            poly_pow += log(potentials[start + i]) * counts[start + i]

        # Briefly drop out of log space to save the scalar output
        # of the polynomial term.
        fout += exp(poly_pow) * scale_factor

        for i in range(hess_size):
            w_i = indices[start + i]
            counts_i = counts[start + i]

            # Calculate the value of the jacobian by dividing out the
            # value of the given variable (named `potentials_i` here) and
            # multiplying the term by the exponent (named `counts_i` here).
            jac_i = (poly_pow +
                     log(counts_i))

            # Briefly drop out of log space to update jacobian
            jacobian[w_i] += exp(jac_i) * scale_factor

            for j in range(hess_size):
                # Caclulate a hessian term by repeating the process used to
                # calculate the jacobian.
                hess_i_j = exp(jac_i +
                               log(counts[start + j]))

                # If we are on a diagonal, we need a nonlinear correction
                # to get the second partial derivative instead of the mixed
                # partial derivative. (Second derivative of x^3 is 6x, but
                # without this correction we would get 9x.)
                if i == j:
                    hess_i_j *= (counts_i - 1) / counts_i

                # Drop unexpected NANs.
                if isnan(hess_i_j) or isinf(hess_i_j):
                    err_count += 1
                    continue

                # Perform the random projection.
                w_j = indices[start + j]
                for k in range(embed_size):
                    rand_hess[w_i, k] += (hess_i_j *
                                          proj_vecs[w_j, k] *
                                          scale_factor)

        start = end

    return err_count





def _hessian_test_wrapper_multi(PS, XS):
    n_terms, n_dims = PS.shape
    f_out = numpy.zeros((n_terms,), dtype=numpy.float64)
    jacobian_out = numpy.zeros((n_terms, n_dims), dtype=numpy.float64)
    hessian_out = numpy.zeros((n_terms, n_dims, n_dims), dtype=numpy.float64)
    ends = numpy.arange(0, n_dims * n_terms, n_dims) + n_dims
    counts = numpy.array(PS.ravel(), dtype=numpy.float64)
    potentials = numpy.array(XS.ravel(), dtype=numpy.float64)
    hessian_multi(f_out, jacobian_out, hessian_out,
                  ends, counts, potentials)
    return hessian_out[0]


def _estimated_hessian(ps, xs, delta=1e-4):
    def poly_eval(ps, xs):
        return numpy.prod(xs ** ps)

    def est_at(dim, dim2=None):
        if dim2 is None:
            p_diff = xs.copy()
            n_diff = xs.copy()
            p_diff[dim] += delta
            n_diff[dim] -= delta

            out = poly_eval(ps, p_diff)
            out -= 2 * poly_eval(ps, xs)
            out += poly_eval(ps, n_diff)
            out /= delta * delta
            return out
        else:
            dim1 = dim
            p_p_diff = xs.copy()
            p_n_diff = xs.copy()
            n_p_diff = xs.copy()
            n_n_diff = xs.copy()

            p_p_diff[dim1] += delta
            p_p_diff[dim2] += delta
            p_n_diff[dim1] += delta
            p_n_diff[dim2] -= delta
            n_p_diff[dim1] -= delta
            n_p_diff[dim2] += delta
            n_n_diff[dim1] -= delta
            n_n_diff[dim2] -= delta

            out = poly_eval(ps, p_p_diff)
            out += poly_eval(ps, n_n_diff)
            out -= poly_eval(ps, p_n_diff)
            out -= poly_eval(ps, n_p_diff)
            out /= delta * delta * 4
            return out

    result = numpy.zeros((len(xs), len(xs)), dtype=xs.dtype)
    for i in range(len(xs)):
        for j in range(len(xs)):
            if i == j:
                result[i, i] = est_at(i)
            else:
                result[i, j] = est_at(i, j)
    return result


def _test_hessian_speed():
    setup = """
from __main__ import (
    hessian,
    log_hessian,
    _hessian_test_wrapper_multi
)
import numpy

n_vars = 50
n_tests = 100000
PS = numpy.random.randint(1, 8, size=(n_tests, n_vars))
PS = numpy.asarray(PS, dtype=numpy.float64)
XS = numpy.random.random((n_tests, n_vars)) * 3
    """

    stmt_template = """
for ps, xs in zip(PS, XS):
    {}(ps, xs)
    """

    funcs = ['hessian', 'log_hessian']
    print()
    print('Hessian function performance tests:')
    for f in funcs:
        print()
        print(f'Timing {f}')
        stmt = stmt_template.format(f)
        timeit.main(['-s', setup, stmt])

    print()
    print('Timing _hessian_test_wrapper_multi')
    stmt = """
_hessian_test_wrapper_multi(PS, XS)
"""
    timeit.main(['-s', setup, stmt])


def _test_hessian_correctness():
    n_vars = 50
    n_tests = 100
    PS = numpy.random.randint(1, 8, size=(n_tests, n_vars))
    PS = numpy.asarray(PS, dtype=numpy.float64)
    XS = numpy.random.random((n_tests, n_vars)) * 2 + 1
    delta = 1e-6

    failed = 0
    for ps, xs in zip(PS, XS):
        _f, _j, hessian_result = hessian(ps, xs)
        _f, _j, log_hessian_result = log_hessian(ps, xs)
        est_result = _estimated_hessian(ps, xs)
        test_versions = ((hessian_result, "hessian"),
                         (log_hessian_result, "log_hessian"))
        for res, res_name in test_versions:
            try:
                assert numpy.allclose(est_result, res)
            except AssertionError:
                err = est_result - res
                err_abs = numpy.abs(err)
                abs_err_prop = err_abs.sum().sum() / est_result.sum().sum()

                est_result_nz = est_result.copy()
                est_result_nz[est_result == 0] = 1
                err_prop_mat = err_abs / est_result_nz
                err_max = err_prop_mat.max()
                err_median = numpy.median(err_prop_mat)
                worst_errors = (err_abs / est_result_nz) > (abs_err_prop * 5)
                worst_errors[est_result == 0] = False
                if abs_err_prop > (delta * 100):
                    print()
                    print("Hessian test failed for {}...".format(res_name))
                    print("Absolute proportional error: ", abs_err_prop)
                    print("Max error: ", err_max)
                    print("Median error: ", err_median)
                    if worst_errors.any() and res_name != "estimated_hessian":
                        loc = worst_errors.nonzero()
                        table = zip(est_result[worst_errors],
                                    res[worst_errors],
                                    loc[0],
                                    loc[1])
                        print("Worst error values -- ")
                        print("      "
                              "  Estimated value   "
                              "  Calculated value  "
                              "  Difference        "
                              "  at Row, Col       ")
                        table_fmt = (
                            "  {: 18.10f}"
                            "  {: 18.10f}"
                            "  {: 18.10f}"
                            "        {}")
                        for est, cal, r, c in table:
                            print(table_fmt.format(
                                est, cal, abs(est - cal), str((r, c))
                            ))

                    failed += 1

    if failed:
        print()
        print("{}/{} tests failed.".format(failed,
                                           n_tests * len(test_versions)))
        print()
        print("This probably doesn't indicate a major problem unless the")
        print("absolute proportional error is higher than 0.05, or is higher")
        print("than 0.001 for more than twenty or thirty tests.")
    else:
        print("All hessian tests passed.")


if __name__ == '__main__':
    _test_hessian_correctness()
    _test_hessian_speed()
