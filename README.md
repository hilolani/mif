# mif
A repository for calculating MiF (Markov inverse F-measure) using Python. MiF is a sophisticated measure of similarity (distance) between nodes in complex networks , and its algorithm was proposed in Akama et al., 2015.

cf. Hiroyuki Akama, Maki Miyake, Jaeyoung Jung, Brian Murphy, 2015. Using Graph Components Derived from an Associative Concept Dictionary to Predict fMRI Neural Activation Patterns that Represent the Meaning of Nouns, PLoS ONE, doi: https://doi.org/10.1371/journal.pone.0125725

MiF calculates distances (similarity) between vertices (nodes) within complex networks, simultaneously incorporating and unifying two perspectives: co-occurrence-based (local co-occurrence) and geodesic-based (global-geodesic). The former quantifies local similarity between nodes, specifically the overlap between edges. Similar to the Jaccard coefficient and Simpson's coefficient, it calculates the overlap rate of paths connecting adjacent or proximate points. The latter considers the shortest path length, coding the distance between points such that the shorter the shortest path and the greater the number of paths taking the shortest path, the smaller the distance. The MiF value falls within the interval [0,1].

This package proposes a new metric called the MiF Degradation Index, abbreviated as MiFDI. It involves selecting a specific vertex (node), such as the one with the smallest degree, and initiating a random walk from it. The random walk continues until all vertices (nodes) are reached, calculating and listing the MiF value between the specific starting vertex (node) and each reached vertex (node). In MiFDI, MiF values are recorded as logarithms, so they can be negative. At each step of the random walk, the average of log(MiF value) is output. You can choose whether to include or exclude self-loops. If excluded, the random walk stops at a node once it is reached and does not proceed further from that node.

# usage

Several adjacent matrices for demonstration purposes are stored in this repository as Matrix Market mtx files and can be used for calculations as follows.

The MiF() function specification is as follows. The first argument takes the adjacency matrix converted to a CSR sparse matrix by the adjacencyinfocheck() function. Not only Matrix Market format sparse matrices (.mtx files), but also sparse matrices in other formats, and even dense matrices (though not recommended), can be converted by the adjacencyinfocheck() function, so its output is fed into the MiF function. This function calculates the MiF value between vertices specified by the numbers placed in the second and third arguments. The fourth argument (corresponding to the beta value) and the fifth argument (corresponding to the gamma value) do not have default values set, so explicit input is required. However, the gamma_threshold, which terminates the random walk, is set to 10 by default in the seventh argument. The sixth argument defaults to 0, representing a 0-based index, and requires no special attention. If using a 1-based index, explicitly set this to 1. This also works if the first argument is a 1-based adjacency matrix. However, Scipy automatically converts mtx files (which are typically 1-based) to 0-based, so the need for explicit conversion is rare.

The MiF_broadcast() function calculates the MiF value between the starting vertex (specified by the first argument) and each reached vertex, within a default range of 10 steps. It uses the adjacency matrix converted to a CSR sparse matrix (first argument) and uses the vertex specified by the second argument as the starting point. The beta value is set to 0.5 by default in the third argument. The fourth argument, loop, is set to 0 by default. This means the random walk does not continue beyond the reached vertex, so no self-loop occurs there. Setting the fourth argument to 1 allows the random walk to continue from a vertex once it has been reached.

The MiFDI() function calculates the MiFDI values between the starting vertex (specified by the first argument) and each vertex reached, using the adjacent matrix converted to a csr sparse matrix (first argument). By default, it operates within a range of 10 steps. By default, the starting point is set to the vertex with the lowest degree, but the vertex with the highest degree can also be selected by specifying “max”. Note that the beta value for this function is set to 0.2 by default. The handling of the fourth argument, loop, is the same as for the MiF_broadcast() function.

    from mif import *

    mif = load_mif()

    mtxlist = [mif.gadget,mif.karateclub,mif.erdosReny,mif.scalefree,mif.homophilly,mif.heterophilly]
    
    adjacencyinfocheckedlist = [adjacencyinfocheck(i) for i in mtxlist]
    
    adjacencylist = ['gadget', 'karateclub', 'erdosReny', 'scalefree', 'homophilly', 'heterophilly']

    #Example of commands:

    MiF(adjacencyinfocheckedlist[1], 4, 32, 0.5, 3)

    MiF_broadcast(adjacencyinfocheckedlist[1], 3)

    MiF_broadcast(adjacencyinfocheckedlist[1], 3, loop = 1)

    MiFDI(adjacencyinfocheckedlist[1], loop = 1)

    MiFDI(adjacencyinfocheckedlist[1], startingvertices="max")
    
