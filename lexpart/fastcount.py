import timeit
from collections import Counter

import numba
import numpy

from .util import random_window_gen


@numba.jit(nopython=True, nogil=True)
def fast_count(indices):
    """
    Take a sequence of integers that are the indices of a
    lookup table, and return two aligned arrays, one containing
    just one copy of each index, and the other containing the
    number of times that index appears in the input sequence.
    This could be done with a Counter, but it is much faster
    done this way when the input is larger than a few hundred
    values, and the indices aren't too sparsely distributed
    between 0 and the maximum.

    When the indices are too sparsely distributed, but you
    know the maximum possible index, and you are processing
    many sequences, `fast_count_prealloc` provides a useful
    alternative.
    """

    # Create an array to store counts. The value at index
    # i will be the number of times i occurs in `indices`.
    # Potentially very sparse.
    up = numpy.zeros(indices.max() + 1, dtype=numpy.int64)

    for i in indices:
        up[i] += 1

    # Count the number of distinct values greater than zero.
    # This is equal to the size of a python set containing
    # all distinct values in `indices`.
    set_size = (up > 0).sum()

    # Together, `dense_indices` and `counts` will be a dense
    # representation of the sparse array `up`. `dense_indices`
    # contains each index, just once. `counts` is aligned with
    # `dense_indices` such that each value is the count for the
    # corresponding value in `dense_indices`.
    dense_indices = numpy.zeros(set_size, dtype=numpy.int64)
    counts = numpy.zeros_like(dense_indices)

    # Counts should contain each value only once. We can
    # guarantee that by running through `indices` again.
    # We create a `down` array that is a copy of `up`.
    # Each time we encounter a value i in `indices`,
    # we decrease the corresponding value in `down` by
    # one. If the value is zero afterwards, we know we
    # have encountered the last instance of i in `indices`,
    # and therefore we can safely insert an (i, c) pair into
    # `counts`.
    down = up.copy()  # separate copy
    ins_pos = 0
    for i in indices:
        down[i] -= 1
        if down[i] == 0:
            counts[ins_pos] = up[i]
            dense_indices[ins_pos] = i
            ins_pos += 1
    return dense_indices, counts


@numba.jit(nopython=True, nogil=True)
def fast_count_prealloc(indices, up, down):
    """
    Do exactly the same thing as `fast_count`, but use
    preallocated buffers for the `up` and `down` arrays. When
    we are processing many small index sets drawn from the same
    very large collection of indices, the up and down arrays can
    get very large and sparse, and the cost of allocating them
    becomes prohibitive.
    """

    # Most documentation is omitted; see `fast_count` for details.
    # Apart from the preallocation of up and down, this function
    # is nearly identical.
    set_size = 0
    for i in indices:
        if up[i] == 0:     # Only count the first occurrence of an index
            set_size += 1  # to get the size of the set of distnict indices.
        up[i] += 1
        down[i] += 1       # Both go up at first.

    dense_indices = numpy.zeros(set_size, dtype=numpy.int64)
    counts = numpy.zeros_like(dense_indices)

    ins_pos = 0
    for i in indices:
        down[i] -= 1      # Down counts down while up stays up.
        if down[i] == 0:  # Now we know we've seen every `i` in the set.
            counts[ins_pos] = up[i]
            dense_indices[ins_pos] = i
            up[i] = 0     # Clean up the up array so it can be reused.
            ins_pos += 1
    return dense_indices, counts


@numba.jit(nopython=True, nogil=True)
def fast_count_batch(indices, ends):
    """
    Calculate counts for many sets of indices in single batch. This
    is efficient when the indices across the entire batch are densely
    distributed between 0 and the maximum value. It examines the whole
    batch and preallocates two arrays for storing counts. For any single
    set of indices, those arrays will only be sparsely populated, and
    reallocating them over and over would be very inefficient; the
    allocation overhead leads to an effective O(n) storage and lookup
    cost. By preallocating them, we avoid that problem, and thus get
    a convenient O(1) data structure for storing counts of indices.

    Each set of indices is identified by the `ends` array, which contains
    the end index of each set in the batch. This is essentially a
    "ragged array," where the rows have a variable number of columns.
    The first set is assumed to start at 0, and to iterate over the
    sets, we slice out `indices[0, end_1]`, then `indices[end_1, end_2]`,
    and so on.
    """

    # Counts will be stored in a new ragged array
    dense_ends = numpy.zeros_like(ends)
    dense_indices = numpy.zeros_like(indices)
    dense_counts = numpy.zeros_like(indices)

    # Preallocate the up and down arrays to be used by the counting
    # routine. This saves a ton of wastful memory allocation.
    pre_up = numpy.zeros(indices.max() + 1, dtype=numpy.int64)
    pre_down = pre_up.copy()

    # To iterate over a ragged array with endpoints indicated by
    # `ends`, we keep track of the start and loop over the ends,
    # updating `start` with the value of `end` at the end of
    # the loop. We are also filling a denser ragged array of
    # indices and counts, where in any single set, each index
    # only appears once. We must therefore maintain `dense_start`
    # and `dense_end` values. Their values are assigned based on
    # the output from the counting routine.
    start = 0
    dense_start = 0
    for i in range(len(ends)):
        end = ends[i]
        iset = indices[start: end]
        (iset_indices,
         iset_counts) = fast_count_prealloc(iset, pre_up, pre_down)

        # Now we know now long the row result is and can set
        # the value for dense_end.
        dense_end = dense_start + len(iset_indices)

        # Fill the dense arrays for this set.
        dense_ends[i] = dense_end
        dense_indices[dense_start: dense_end] = iset_indices
        dense_counts[dense_start: dense_end] = iset_counts

        # Finally we update both start values.
        start = end
        dense_start = dense_end

    return (dense_indices[:dense_end],
            dense_ends,
            dense_counts[:dense_end])


def _slow_count_batch(indices, ends):
    """A slow batch counter for testing."""
    start = 0
    dense_start = 0
    dense_ends = numpy.zeros_like(ends)
    dense_indices = numpy.zeros_like(indices)
    dense_counts = numpy.zeros_like(indices)

    for i in range(len(ends)):
        end = ends[i]
        batch = indices[start: end]
        ct = Counter(batch)
        batch_indices = numpy.array(list(ct.keys()))
        batch_counts = numpy.array(list(ct.values()))

        # Now we know now long the batch result is and can set
        # the value for dense_end.
        dense_end = dense_start + len(batch_indices)

        # Fill the dense_indices and dense_counts arrays.
        dense_indices[dense_start: dense_end] = batch_indices
        dense_counts[dense_start: dense_end] = batch_counts

        # Finally we update both start values and the dense_ends array.
        start = end
        dense_start = dense_end
        dense_ends[i] = dense_end
    return (dense_ends,
            dense_indices[:dense_end],
            dense_counts[:dense_end])


def fast_counter(indices):
    dense_indices, counts = fast_count(indices)
    return Counter(dict(zip(dense_indices, counts)))


def _slow_counter(objects):
    """A slow counter for testing."""
    obj_index, obj_counts = numpy.unique(objects, return_counts=True)
    return Counter(dict(zip(obj_index, obj_counts)))


def _test_batch_count_correctness():
    size = 5000
    indices = numpy.random.choice(range(0, 100), size=size)
    ends = numpy.fromiter(random_window_gen(50, 10, size), dtype=int)
    (dense_indices,
     dense_ends,
     dense_counts) = fast_count_batch(indices, ends)
    numba_counters = []
    s = 0
    for e in dense_ends:
        numba_counters.append(
            Counter(dict(zip(dense_indices[s:e], dense_counts[s:e])))
        )
        s = e

    regular_counters = []
    s = 0
    for e in ends:
        regular_counters.append(
            Counter(indices[s:e])
        )
        s = e

    for n, r in zip(numba_counters, regular_counters):
        try:
            assert n == r
        except AssertionError:
            print(sorted(n.items()))
            print(sorted(r.items()))
            raise


def _test_count_correctness():
    objects = numpy.random.choice(list('abcdefghijklmnopqrstuvwxyz'),
                                  size=3000)
    for i in range(100):
        sub_objects = objects[i * 30: (i + 1) * 30]
        try:
            assert _slow_counter(sub_objects) == Counter(sub_objects)
        except AssertionError:
            print(sorted(_slow_counter(sub_objects).items()))
            print(sorted(Counter(sub_objects).items()))
            raise


def _test_count_speed():
    setup = """
from collections import Counter

import numpy

from __main__ import fast_counter, _slow_counter

objects = numpy.random.choice(range(0, 100), size=50000)
"""
    stmt_fast = "x = fast_counter(objects)"
    stmt_slow = "x = _slow_counter(objects)"
    stmt_counter = "x = Counter(objects)"

    print("Timing fast_counter")
    timeit.main(['-s', setup, stmt_fast])
    print("Timing _slow_counter")
    timeit.main(['-s', setup, stmt_slow])
    print("Timing built-in Counter")
    timeit.main(['-s', setup, stmt_counter])


def _test_batch_count_speed():
    setup = """
from collections import Counter

import numpy

from __main__ import (
    fast_count_batch,
    _slow_count_batch,
    random_window_gen
)

size = 5000000
indices = numpy.random.choice(range(0, 300000), size=size)
ends = numpy.fromiter(random_window_gen(50, 10, size), dtype=int)
"""
    stmt_fast = "x = fast_count_batch(indices, ends)"
    stmt_slow = "x = _slow_count_batch(indices, ends)"

    print("Timing fast_count_batch")
    timeit.main(['-s', setup, stmt_fast])
    print("Timing _slow_count_batch")
    timeit.main(['-s', setup, stmt_slow])


if __name__ == "__main__":
    try:
        print("Testing fast_count correctness vs. Counter: ", end='')
        _test_count_correctness()
        print("test passed")
        print("Testing fast_count_batch_numba correctness vs. Counter: ",
              end='')
        _test_batch_count_correctness()
        print("test passed")
    except AssertionError:
        print("test failed")
        raise
    else:
        print()

    print("Testing fast_count speed vs. Counter")
    _test_count_speed()
    print()
    print("Testing fast_count_batch speed vs. Counter")
    _test_batch_count_speed()
