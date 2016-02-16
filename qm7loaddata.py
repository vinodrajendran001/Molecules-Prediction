import os
import cPickle as cp
import pickle


import numpy
import random
import json
from scipy.spatial.distance import cdist
from itertools import product

DATA_BASE_DIR = "/home/vinod/Downloads/qm7b_bob"
BHOR_TO_ANGSTROM = 0.529177249
HARTREE_TO_KCAL = 627.509
HARTREE_TO_EV = 27.211396132


#  C, H, Cl, N, O
ELE_TO_NUM = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
    'Cl': 17,
    'S': 16
}


def get_coulomb_feature(names, paths):

    cache = {}
    test = []
    for path in paths:
        if path in cache:
            continue
        elements, numbers, coords = read_file_data(path)
        mat = get_coulomb_matrix(numbers, coords)
        mat = numpy.pad(mat, ((0,29-mat.shape[0]),(0,29-mat.shape[0])), mode='constant')
        #mat = numpy.rollaxis(W, 2)
        #cache[path] = mat[numpy.tril_indices(mat.shape[0],mat.shape[1])]
        cache[path] = mat

    vectors = numpy.dstack([cache[path] for path in paths])
    vectors = numpy.rollaxis(vectors, 2)
    return vectors
    #return homogenize_lengths(vectors)


def get_random_coulomb_feature(names, paths, size=1):

    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        mat = get_coulomb_matrix(numbers, coords)
        vectors.append(mat)


    for mat in vectors[:]:
        shape = mat.shape
        for x in xrange(size - 1):
            order = numpy.arange(shape[0])
            out = numpy.random.permutation(order)
            perm = numpy.zeros(shape)
            perm[order, out] = 1
            vectors.append(perm.T * mat * perm)
    vectors = [mat[numpy.tril_indices(mat.shape[0])] for mat in vectors]
    return homogenize_lengths(vectors)



def randomize_coulomb_matrix(m):
    """
    Randomize a Coulomb matrix as decribed in Montavon et al., _New Journal
    of Physics_ __15__ (2013) 095003:

        1. Compute row norms for M in a vector row_norms.
        2. Sample a zero-mean unit-variance noise vector e with dimension
           equal to row_norms.
        3. Permute the rows and columns of M with the permutation that
           sorts row_norms + e.

    Parameters
    ----------
    m : ndarray
        Coulomb matrix.
    n_samples : int, optional (default 1)
        Number of random matrices to generate.
    seed : int, optional
        Random seed.
    """
    seed = None
    n_samples = m.shape[0]
    rval = []
    row_norms = numpy.asarray([numpy.linalg.norm(row) for row in m], dtype=float)
    rng = numpy.random.RandomState(seed)
    for i in xrange(n_samples):
        e = rng.normal(size=row_norms.size)
        p = numpy.argsort(row_norms + e)
        new = m[p][:, p]  # permute rows first, then columns
        rval.append(new)
    return rval


def homogenize_lengths(vectors):

    n = max(len(x) for x in vectors)
    feat = numpy.zeros((len(vectors), n))

    for i, x in enumerate(vectors):
        feat[i, 0:len(x)] = x

    return numpy.matrix(feat)



def get_coulomb_matrix(numbers, coords):

    ANGSTROM_TO_BHOR = 1. / BHOR_TO_ANGSTROM
    top = numpy.outer(numbers, numbers).astype(numpy.float64)
    r = get_distance_matrix(ANGSTROM_TO_BHOR * coords, power=1)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        numpy.divide(top, r, top)
    numpy.fill_diagonal(top, 0.5 * numpy.array(numbers) ** 2.4)
    top[top == numpy.Infinity] = 0
    top[numpy.isnan(top)] = 0

    return top


def get_bag_of_bonds_feature(names, paths):

    # Add all possible bond pairs (C, C), (C, O)...
    keys = set(tuple(sorted(x)) for x in product(ELE_TO_NUM, ELE_TO_NUM))
    # Add single element types (for the diag)
    keys |= set(ELE_TO_NUM)

    # Sort the keys to remove duplicates later ((C, H) instead of (H, C))
    sorted_keys = sorted(ELE_TO_NUM.keys())

    # Initialize the bags for all the molecules at the same time
    # This is to make it easier to make each of the bags of the same type the
    # same length at the end
    bags = {key: [] for key in keys}
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        # Sort the elements, numbers, and coords based on the element
        bla = sorted(zip(elements, numbers, coords.tolist()), key=lambda x: x[0])
        elements, numbers, coords = zip(*bla)
        coords = numpy.matrix(coords)

        ele_array = numpy.array(elements)
        ele_set = set(elements)
        mat = get_coulomb_matrix(numbers, coords)
        mat = numpy.array(mat)
        diag = numpy.diagonal(mat)

        for key in keys:
            bags[key].append([])

        for i, ele1 in enumerate(sorted_keys):
            if ele1 not in ele_set:
                continue
            # Select only the rows that are of type ele1
            first = ele_array == ele1
            # Select the diag elements if they match ele1 and store them,
            # highest to lowest
            bags[ele1][-1] = sorted(diag[first].tolist(), reverse=True)
            for j, ele2 in enumerate(sorted_keys):
                if i > j or ele2 not in ele_set:
                    continue
                # Select only the cols that are of type ele2
                second = ele_array == ele2
                # Select only the rows/cols that are in the upper triangle
                # (This could also be the lower), and are in a row, col with
                # ele1 and ele2 respectively
                mask = numpy.triu(numpy.logical_and.outer(first, second), k=1)
                # Add to correct double element bag
                # highest to lowest
                bags[ele1, ele2][-1] = sorted(mat[mask].tolist(), reverse=True)

    # Make all the bags of the same type the same length, and form matrix
    new = [homogenize_lengths(x) for x in bags.values()]
    new1 = [x for x in bags.values() if x != []]
    new1 = zip(*new1)
    new2 = [tuple(filter(None, tp)) for tp in new1]
    new3 = []
    for i, x in enumerate(new2):
        new3.append(tuple(x for seq in new2[i] for x in seq))
    return numpy.hstack(new)





def calculate_atomization_energies(atom_counts, energies):
    # H C N O F
    atom_energies = numpy.matrix([
        [-0.497912],
        [-37.844411],
        [-54.581501],
        [-75.062219],
        [-99.716370]
    ])
    return energies - atom_counts * atom_energies


def build_gdb13_data():
    atom_idxs = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    base_path = DATA_BASE_DIR
    #mkdir_p(base_path)

    energies = []
    atom_counts = []
    noofatoms = []

    for name in sorted(os.listdir(os.path.join(base_path, "dsgdb9nsd.xyz"))):
        xyz_path = os.path.join(base_path, "dsgdb9nsd.xyz", name)
        out_path = xyz_path.replace("dsgdb9nsd.xyz", "dsgdb9nsd.out")

        natoms = 0
        energy = None
        counts = [0 for _ in atom_idxs]
        with open(xyz_path, 'r') as xyz_f, open(out_path, 'w') as out_f:
            for i, line in enumerate(xyz_f):
                line = line.strip()
                if not i:
                    natoms = int(line)
                elif i == 1:
                    energy = float(line.split()[-3])
                    s = line.split()[2:]
                    for n, value in enumerate(s, start=1):
                        globals()["p%d"%n] = float(value)
                    results = open('/home/vinod/Downloads/Dataset/gdb13/properties.txt', 'a')
                    results.write("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n"
                                  % (p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15))
                    results.close()
                elif i - 2 < natoms:
                    line = line.replace("*^", "e")
                    ele, x, y, z, c = line.split()
                    counts[atom_idxs[ele]] += 1
                    out_f.write("%s %.8f %.8f %.8f %.6f\n" % (ele, float(x), float(y), float(z), float(c)))
        #energies.append(energy)
        atom_counts.append(counts)
        noofatoms.append(natoms)
    atom_counts = numpy.matrix(atom_counts)
    #atomization = calculate_atomization_energies(atom_counts, numpy.matrix(energies).T)
    #atomization *= HARTREE_TO_KCAL
    #numpy.savetxt(os.path.join(base_path, "energies.txt"), atomization, fmt='%0.8f')
    numpy.savetxt(os.path.join(base_path, "heavy_counts.txt"), atom_counts.sum(1), fmt='%d')



def load_gdb13_data():
    base_path = DATA_BASE_DIR
    base_path1 = os.path.join(DATA_BASE_DIR, "dsgdb9nsd.txt")
    if not os.path.isdir(base_path1) or not os.listdir(base_path1):
        build_gdb13_data()

    names = []
    datasets = []
    geom_paths = []
    meta = []
    lengths = []


    out_path = os.path.join(base_path, "dsgdb9nsd.txt")
    for name in sorted(os.listdir(out_path)):
        path = os.path.join(out_path, name)
        geom_paths.append(path)
        meta.append([1])
        datasets.append((1, ))

    names = ["test" if x > 7 else "train" for x in lengths]
    return names, datasets, geom_paths, meta


def get_distance_matrix(coords, power=-1, inf_val=1):

    dist = cdist(coords, coords)
    with numpy.errstate(divide='ignore'):
        numpy.power(dist, power, dist)
    dist[dist == numpy.Infinity] = inf_val
    return dist




def read_file_data(path):

    elements = []
    numbers = []
    coords = []

    with open(path, 'r') as f:
        for line in f:
            ele, x, y, z = line.strip().split()
            point = (float(x), float(y), float(z))
            elements.append(ele)
            numbers.append(ELE_TO_NUM[ele])
            #numbers.append(float(c))
            coords.append(point)
    #numbers = (numbers + [0] * 29)[:29]
    #coords = (coords + [(0,0,0)] * 29)[:29]
    return elements, numbers, numpy.matrix(coords)



if __name__ == '__main__':
    base_path = DATA_BASE_DIR
    names, datasets, paths, meta = load_gdb13_data()
    bagofbonds = get_bag_of_bonds_feature(names, paths)
    #random1 = randomize_coulomb_matrix(bagofbonds)
    numpy.savetxt(os.path.join(base_path, "bob.txt"), bagofbonds, fmt='%0.8f')

    #numpy.savetxt(os.path.join(base_path, "split.txt"), indices)
    #coulomb = numpy.loadtxt(os.path.join(base_path, "coulomb.txt")).astype(float).tolist()
    bob = numpy.loadtxt(os.path.join(base_path, "bob.txt")).astype(float).tolist()
    #split = numpy.loadtxt(os.path.join(base_path, "split.txt")).astype(int).tolist()
    #mydict = {'X': numpy.array(feat), 'T': numpy.array(prop_out), 'Z': numpy.array(properties), 'P': numpy.array(split), 'B':numpy.array(bob)}
    mydict = {'B':numpy.asarray(bob)}
    output = open('/home/vinod/Downloads/qm7b_bob/bob.pkl', 'wb')
    cp.dump(mydict, output)
    output.close()











