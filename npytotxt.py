__author__ = 'vinod'
import struct
import numpy as np
import os

def parseNPY(path, fileJustName):
    # load from the file
    inputFile = os.path.join(path, fileJustName + ".npy")
    matrices = np.load(inputFile)

    outputfile = os.path.join(path, fileJustName)
    for m in range(matrices.shape[0]):
        # file name for this matrix
        outFileFull = outputfile + "-" + str(m) + ".txt"
        # output matrix to a numbered file
        a = matrices[m:m+1]
        with open('atomscount.txt', 'a') as f:
            for item in a[0]:
                line = "{} {}\n".format(item[0], ' '.join(([str(x) for x in item[1]])))
                f.write(line)
        #np.savetxt(outFileFull, matrices[m:m+1], fmt="%s", delimiter="\n")


mypath = "/home/vinod/Downloads/qm7b_bob/"

for path, paths, filenames in os.walk(mypath):
    # translate all filenames.
    for filename in filenames:
        fileJustName, fileExtension = os.path.splitext(filename)
        if fileExtension == ".npy":
            print(os.path.join(path, fileJustName))
            parseNPY(path, fileJustName)