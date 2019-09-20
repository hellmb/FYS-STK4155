import numpy as np

def BenchmarksToFile(beta1, beta2, mse1, mse2, r2s1, r2s2):
    """
    write benchmarks to file
    """

    object = 0
    with open('Benchmarks/benchmarks_a.txt', 'w') as f:
        f.write('Predicted beta values from code:\n')
        for item in beta1:
            if object == 0:
                f.write('[  ')
            f.write('%s  ' % item)
            object += 1
            if object == len(beta1):
                f.write(']\n')

    object = 0
    with open('Benchmarks/benchmarks_a.txt', 'a') as f:
        f.write('\nPredicted beta values from scikit learn:\n')
        for item in beta2:
            if object == 0:
                f.write('[  ')
            f.write('%s  ' % item)
            object += 1
            if object == len(beta2):
                f.write(']\n')

    with open('Benchmarks/benchmarks_a.txt', 'a') as f:
        f.write('\nMean squared error (MSE) from code and scikit learn:\n')
        f.write('%s  %s\n' % (mse1, mse2))

    with open('Benchmarks/benchmarks_a.txt', 'a') as f:
        f.write('\nR2 score from code and scikit learn:\n')
        f.write('%s  %s\n' % (r2s1, r2s2))
        f.close()
