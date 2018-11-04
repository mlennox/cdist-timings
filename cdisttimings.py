from scipy import spatial
from numpy import reshape, array, argmin, random
import timeit


def setup(dimension):
    vector_length = 100
    one = random.rand(dimension, vector_length)
    two = random.rand(dimension, vector_length)
    one_array = [reshape(row, (1, -1)) for row in one]
    return (one, one_array, two)


def loop_based(one, two):
    def closure_loop_based():
        most_similar = (-1, -1)
        smallest_cosine = -1
        for row_index in range(1, len(one)):
            row = one[row_index]
            distance_matrix = spatial.distance.cdist(row, two, "cosine").reshape(-1)
            min_index = argmin(distance_matrix)
            if smallest_cosine == -1 or smallest_cosine > distance_matrix[min_index]:
                smallest_cosine = distance_matrix[min_index]
                most_similar = (row_index, min_index)
        # print("Smallest at {0}".format(most_similar))

    return closure_loop_based


def vectorised(one, two, dimension):
    def closure_vectorised():
        most_similar = (-1, -1)
        smallest_cosine = -1
        distance_matrix = spatial.distance.cdist(one, two, "cosine").reshape(-1)
        # print(
        #     "vector distance shape (should be {0} x {0} = {1}) : {2}".format(
        #         dimension, dimension * dimension, distance_matrix.shape
        #     )
        # )
        min_index = argmin(distance_matrix)
        # print("min index", min_index)
        # print("at ({0}, {1})".format(min_index // dimension, min_index % dimension))

    return closure_vectorised


def run():
    loops = 10
    repeats = 100
    dimension = 1000
    loops_avgs = []
    vectorised_avgs = []

    print("Starting benchmarks, running {0} loops".format(loops))

    for loops_index in range(1, loops):
        print("Timing run {0}".format(loops_index))

        # we create a new set of test data each loop and pass same to both implementations
        one, one_array, two = setup(dimension)

        print("running loop implementation")
        timer = timeit.Timer(loop(one_array, two))
        duration_loop = timer.repeat(repeat=loops, number=1)
        # print("durations", duration_loop)
        loop_avg = sum(duration_loop) / len(duration_loop)
        print("average loop duration = {0}".format(loop_avg))

        print("running vectorised - - - - - - -")
        timer = timeit.Timer(vectorised(one, two, dimension))
        duration_vector = timer.repeat(repeat=loops, number=1)
        # print("durations", duration_vector)
        vectorised_avg = sum(duration_vector) / len(duration_vector)
        print("average vectorised duration = {0}".format(vectorised_avg))

        print("vectorised is {0} times faster".format(loop_avg / vectorised_avg))


run()
