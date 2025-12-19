# mif
A repository for calculating MiF (Markov inverse F-measure) using Python. MiF is a sophisticated measure of similarity (distance) between nodes in complex networks , and its algorithm was proposed in Akama et al., 2015.

cf. Hiroyuki Akama, Maki Miyake, Jaeyoung Jung, Brian Murphy, 2015. Using Graph Components Derived from an Associative Concept Dictionary to Predict fMRI Neural Activation Patterns that Represent the Meaning of Nouns, PLoS ONE, doi: https://doi.org/10.1371/journal.pone.0125725

MiF (Markov inverse F-measure)

While Re_MCL focuses on restructuring graph topology through Markov dynamics, MiF addresses a complementary problem: measuring similarity and distance within complex networks. MiF (Markov inverse F-measure) is a similarity (distance) measure between vertices in a graph, originally proposed by Akama et al. (2015).

MiF evaluates how closely two vertices are related by modeling how information flows between them through a Markov random walk. Unlike conventional graph similarity measures, MiF integrates both local and global structural information into a single framework.

From a local perspective, MiF considers co-occurrence-based similarity, which reflects how strongly two vertices overlap in their immediate neighborhoods. This idea is conceptually related to well-known measures such as the Jaccard and Simpson coefficients, which quantify similarity by the overlap of adjacent connections.

From a global perspective, MiF also incorporates geodesic-based similarity, taking into account the shortest path length between vertices. In general, vertices connected by shorter paths and by a larger number of such paths are regarded as more similar. MiF naturally balances these two perspectives, enabling robust similarity estimation even in complex network structures.

The MiF value is normalized to lie within the interval [0, 1], where larger values indicate stronger similarity.

Parameterization and network characteristics

MiF includes several free parameters that allow the metric to adapt to different network characteristics. In classical set-based similarity measures, normalization often relies on the size of the union of two sets. However, in graph-based settings, such normalization can become problematic due to degree imbalance, degree correlation, or scale-free structures.

To address this issue, MiF introduces a parameter β (0 < β < 1), which controls how vertex degrees contribute to normalization. By using a harmonic-mean–based formulation, MiF can emphasize or suppress degree effects. In practice, choosing β values close to zero allows heterophilic or homophilic properties of the network to be highlighted more clearly.

In addition, MiF considers the influence of longer paths in random walks. While shorter paths usually dominate similarity, longer paths and detours may still carry meaningful structural information. This effect is controlled by another parameter α, which gradually decreases the contribution of paths as their length increases.

MiF Degradation Index (MiFDI)

This package also introduces a derived metric called the MiF Degradation Index (MiFDI). MiFDI analyzes how similarity degrades as a random walk expands from a selected starting vertex.

In MiFDI, a random walk is initiated from a specific vertex (for example, a vertex with minimal degree). As the walk progresses and reaches other vertices, the MiF values between the starting vertex and each visited vertex are computed. These values are recorded on a logarithmic scale and averaged at each step of the walk.

Depending on the configuration, self-loops can be included or excluded. When self-loops are excluded, the random walk terminates propagation from a vertex once it has been reached. MiFDI provides a compact representation of how rapidly relational similarity decays across the network.

# usage
Run

    pip install git+https://github.com/hilolani/mif.git
    
to use this program. (If you are using Google Colab, put "!" before "pip")    

Several adjacent matrices for demonstration purposes are stored in this repository as Matrix Market mtx files and can be used for MiF calculations.

## Data Explanation:

This repository allows you to load several examples of typical complex networks in the form of sparse adjacency matrices (mtx files).

ErdosReny.mtx:
This is a random network with 100 vertices and 7866 edges.

gadget.mtx:
This is a random network with 106 vertices and 493 edges, closely following an exponential distribution.

karate.mtx:
This is the adjacency matrix for Zachary's karate club, a well-known graph representing a social friendship network. It has 34 nodes and exhibits strong small-world properties.

scalefree.mtx:
Created using the BA model, starting with a complete graph of 5 nodes. Nodes with degree 5 repeatedly undergo preferential attachment until the total number of vertices reaches 100.

homophilly.mtx, heterophilly.mtx:
The homophily and heterophily networks were created as follows.
First, for each node in a random graph with nearly uniform degree, we randomly assigned the degree from a scale-free graph (generated by the BA model) with the same number of nodes as its intrinsic weight. We then pruned edges based on the difference in these weights. This scale-free graph is identical to the one set in the same data directory. For the generation functions of homophily and heterophily, instead of setting a threshold on the weight difference between nodes for edge pruning as in standard threshold graph calculations, we set two separate symmetric sigmoid functions specifically for homophily and heterophily. The independent variable for these functions is the absolute value of the difference in the logarithm of the intrinsic weights. Pruning is then determined probabilistically.
Specifically, we computed the absolute difference of the log intrinsic weights (abslogdiff) for all pairs, then used the median of abslogdiff as the threshold. Median[abslogdiff], to calculate the pruning probability for heterophily using the sigmoid function P_hetero(X), which takes the absolute value of the difference in the logarithm of intrinsic weights as its independent variable, and the pruning probability for homophily using P_homo(X). By taking the median as the intermediate concavity bifurcation point, we balanced the number of pruned edges between homo- and hetero-connections to control the comparison conditions.

eat.mtx:
This is an undirected graph representing the associative relationships between words in the Edinburgh Associative Thesaurus (EAT), an associative concept dictionary.

## MiF functions:

### MiF:

    MiF(adjacencymatrixchecked, x, y, beta, gamma,index_base = 0, gamma_threshold = 10, logger=None):

The MiF() function specification is as follows. The first argument takes the adjacency matrix converted to a CSR sparse matrix by the adjacencyinfocheck() function. Not only Matrix Market format sparse matrices (.mtx files), but also sparse matrices in other formats, and even dense matrices (though not recommended), can be converted by the adjacencyinfocheck() function, so its output is fed into the MiF function. This function calculates the MiF value between vertices specified by the numbers placed in the second and third arguments. The fourth argument (corresponding to the beta value) and the fifth argument (corresponding to the gamma value) do not have default values set, so explicit input is required. However, the gamma_threshold, which terminates the random walk, is set to 10 by default in the seventh argument. The sixth argument defaults to 0, representing a 0-based index, and requires no special attention. If using a 1-based index, explicitly set this to 1. This also works if the first argument is a 1-based adjacency matrix. However, Scipy automatically converts mtx files (which are typically 1-based) to 0-based, so the need for explicit conversion is rare. For example, to calculate the MiF distance between vertex 3 and vertex 32 in the Karate Club network using a β value of 0.5 (default) and a γ value of 3 representing the number of steps: Use adjacencyinfocheckedlist[1], which is the adjacency matrix file karate.mtx stored in this repository converted to a CSR sparse matrix using the adjacencyinfocheck() function (see the example below).

    log_mif1 = MiF(adjacencylist[1], 4, 32, 0.5, 3)

    print(log_mif1)

This will do the job.

### MiF_broadcast:

    MiF_broadcast(adjacencymatrixchecked, startingvertex, beta = 0.5, gamma_threshold = 10, loop = 0,logger=None):

The MiF_broadcast() function calculates the MiF value between the starting vertex (specified by the first argument) and each reached vertex, within a default range of 10 steps. It uses the adjacency matrix converted to a CSR sparse matrix (first argument) and uses the vertex specified by the second argument as the starting point. The beta value is set to 0.5 by default in the third argument. The fourth argument, loop, is set to 0 by default. This means the random walk does not continue beyond the reached vertex, so no self-loop occurs there. Setting the fourth argument to 1 allows the random walk to continue from a vertex once it has been reached. For example, in the Karate Club network, to broadcast MiF values from vertex 3 to all points while allowing self-loops, use:

    log_without1 = MiF_broadcast(adjacencylist[1], 3)

    log_with1 = MiF_broadcast(adjacencylist[1], 3, loop = 1)

    print(f“MiF broadcast without loop: {log_without1}”)

    print(f“MiF broadcast with loop: {log_with1}”)

Omitting `loop = 1` disallows self-loops. Note that the MiF value for the starting vertex 3 is set to 0.

### MiFDI

    MiFDI(adjacencymatrixchecked, startingvertices="min", dangn = 0, beta = 0.2, gamma_threshold = 10, allstartinginfo = 0, loop = 0, logger=None):

The MiFDI() function calculates the MiFDI values between the starting vertices (specified by the first argument) and each vertex reached, using the adjacent matrix converted to a csr sparse matrix (first argument). MiFDI stands for MiF Deterioration Index, indicating that the longer a random walk persists, the greater the distance between the starting point and each reached node. By default, it operates within a range of 10 steps. The starting point is set to the vertex with the lowest degree (as the default setting), but the vertex with the highest degree can also be selected by specifying “max”. Note that the beta value for this function is set to 0.2 by default. The handling of the fourth argument, loop, is the same as for the MiF_broadcast() function. The `MiF_broadcast()` function and the `MiFDI()` function are similar in that they both calculate the MiF value between the starting point and other vertices. However, the decisive differences are:

1) MiFDI() determines the starting point by degree, so multiple starting points may exist and must be calculated separately, and
   
3) MiFDI() logarithmizes the MiF value
   
. In this function's specification, the argument dangn = 0 is the default setting. In this case, even if multiple starting points exist under conditions of maximum degree (startingvertices=“max”) or minimum degree (default setting startingvertices="min"), and even if the calculation process covers all starting points, the return value consists of only two items: the MiFDI information from the point with the youngest number (dangn = 0) and its logarithmized MiF value. Therefore, if a second starting point exists and you want its information, specify dangn = 1 as an argument. If you wish to return MiFDI information from all starting points collectively, please set the value of the sixth argument "allstartinginfo" of the MiFDI() to 1 by adding "allstartinginfo = 1" when executing this function. In the MiFDI() function, loop = 0 (default setting) disallows self-loops, while loop = 1 allows self-loops. This specification is identical to that of the MiF_broadcast() function. For example, to compute MiFDI from the vertex with the minimum degree for scalefree.mtx stored in this repository: 

    logdiwithout3, logdiwithoutmifdival3 = MiFDI(adjacencylist[3], dangn = 0)

    logdiwith3, logdiwithmifdival3 = MiFDI(adjacencylist[3], dangn = 0, loop = 1)

    print(f“MiFDI result without loop: {logdiwithout3}”)

    print(f“MiF values without loop: {logdiwithoutmifdival3}”)

    print(f“MiFDI result with loop: {logdiwith3}”)

    print(f“MiF values with loop: {logdiwithmifdival3}”)

Additionally, functions are provided to compare results with and without self-loop for each edge:

MiF_broadcast_diff_on_loop()

MiFDI_diff_on_loop()

    broadcast_diff = MiF_broadcast_diff_on_loop(log_with1, log_without1)

    di_diff = MiFDI_diff_on_loop(logdiwith3, logdiwithout3)

For details, please refer to the ipynb files in this repository.
    

# How to load the data

This repository utilizes Bunch objects from the scikit-learn library to enable direct use of toy datasets and other small datasets. Bunch objects extend dictionaries by allowing access to values via keys. A Bunch is a dictionary-like object that enables accessing elements using dot notation (e.g., bunch.key). Usage is as follows:

    from mif import *

    mif = load_mif()

    mtxlist = [mif.gadget,mif.karateclub,mif.erdosReny,mif.scalefree,mif.homophilly,mif.heterophilly]
    
    adjacencyinfocheckedlist = [adjacencyinfocheck(i) for i in mtxlist]
    
    adjacencylist = ['gadget', 'karateclub', 'erdosReny', 'scalefree', 'homophilly', 'heterophilly']

    #Example of commands:

    mif_val = MiF(adjacencyinfocheckedlist[1], 4, 32, 0.5, 3)

    all_result = MiF_broadcast(adjacencyinfocheckedlist[1], 3)

    all_result = MiF_broadcast(adjacencyinfocheckedlist[1], 3, loop = 1)

    all_result, mifdi_list = MiFDI(adjacencyinfocheckedlist[1], loop = 1)

    all_result, mifdi_list = MiFDI(adjacencyinfocheckedlist[1], startingvertices="max")
    
