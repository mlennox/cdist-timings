from scipy import spatial
from numpy import reshape, array, argmin, random
import timeit


def setup(dimension, vector_length):
    """
    Generates two arrays filled with random values of shape (dimension, vector_length)
    
    Arguments:
      dimension {int} -- the number of rows - this is meant to represent the number of word embeddings
      vector_length {int} -- the length of the word embedding vector
    
    Returns:
      tuple of the two arrays and the listified array for the loop implementation
    """

    one = random.rand(dimension, vector_length)
    two = random.rand(dimension, vector_length)
    one_array = [reshape(row, (1, -1)) for row in one]
    return (one, one_array, two)


def loop_based(one, two):
    # timeit needs a reference to an executable method
    # passing params would replace an executable with the result of the executed method
    # so we return a function closure instead
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
    """
    We run our timings `loops` times generating a new set of matrices each timing loop

    The timings are repeated with the same data `repeats` times

    We generate matrices that are the same dimensions in each timing loop but in reality
    there would be far more rows in the word embedding model

    Each timing loop we save the list of durations generated for each `repeats` and at
    the end of the `loops` we find the average of each implementations duration and 
    calculate how much faster the vectorised implementation is
    """
    loops = 2
    repeats = 30
    dimension = 1000
    vector_length = 100
    loop_based_duration_averages = []
    vectorised_duration_averages = []

    print("Starting benchmarks, running {0} loops".format(loops))

    for loops_index in range(0, loops):
        print("Timing run {0}".format(loops_index + 1))
        print("-  = - = - = - = - = - = - = - = - = - = - = - = - = - = -")

        # we create a new set of test data each loop and pass same to both implementations
        one, one_array, two = setup(dimension, vector_length)

        print("running loop implementation")
        timer = timeit.Timer(loop_based(one_array, two))
        # repeat the timing `repeats` times but only execute function once per repeat
        duration_loop = timer.repeat(repeat=repeats, number=1)
        # duration_loop will be a list of timings of length `repeats`
        loop_based_duration_averages.extend(duration_loop)
        print(
            "average loop duration = {0}".format(
                sum(duration_loop) / len(duration_loop)
            )
        )

        print("running vectorised implementation")
        timer = timeit.Timer(vectorised(one, two, dimension))
        duration_vector = timer.repeat(repeat=repeats, number=1)
        vectorised_duration_averages.extend(duration_vector)
        print(
            "average vectorised duration = {0}".format(
                sum(duration_vector) / len(duration_vector)
            )
        )

    loop_avg = sum(loop_based_duration_averages) / len(loop_based_duration_averages)
    vectorised_avg = sum(vectorised_duration_averages) / len(
        vectorised_duration_averages
    )
    print("vectorised is {0} times faster".format(loop_avg / vectorised_avg))


run()
