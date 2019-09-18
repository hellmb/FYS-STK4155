import numpy as np

def BenchmarksToFile(beta1, beta2):
    """
    write benchmarks to file
    """

    object = 0
    with open('Benchmarks/benchmarks_a.txt', 'w') as f:
        for item in beta1:
            if object == 0:
                f.write('[  ')
            f.write('%s  ' % item)
            object += 1
            if object == len(beta1):
                f.write(']')

    object = 0
    with open('Benchmarks/benchmarks_a.txt', 'a') as f:
        for item in beta2:
            if object == 0:
                f.write('\n[  ')
            f.write('%s  ' % item)
            object += 1
            if object == len(beta2):
                f.write(']')

    f.close()
