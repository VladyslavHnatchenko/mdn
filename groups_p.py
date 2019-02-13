# Group example with Parzen-window estimation
import numpy
from django_q.tasks import async_task, result_group, delete_group


# the estimation function
def parzen_estimation(x_samples, point_x, h):
    k_n = 0
    for row in x_samples:
        x_i = (point_x - row[:, numpy.newaxis]) / h
        for row in x_i:
            if numpy.abs(row) > (1 / 2):
                break
        else:
            k_n += 1
    return h, (k_n / len(x_samples)) / (h ** point_x.shape[1])


# create 100 calculations and return the collated result
def parzen_async():
    # clear the previous results
    delete_group('parzen', cached=True)
    mu_vec = numpy.array([0, 0])
    cov_mat = numpy.array([[1, 0], [0, 1]])
    sample = numpy.random. \
        multivariate_normal(mu_vec, cov_mat, 10000)
    widths = numpy.linspace(1.0, 1.2, 100)
    x = numpy.array([[0], [0]])
    # async_task them with a group label to the cache backend
    for w in widths:
        async_task(parzen_estimation, sample, x, w,
                group='parzen', cached=True)
    # return after 100 results
    return result_group('parzen', count=100, cached=True)
