import numpy as np
import os
import networkx as nx
import math
import itertools
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix,coo_matrix,csc_matrix,issparse,isspmatrix_csr,isspmatrix_csc,isspmatrix_coo,csgraph,lil_matrix,dok_matrix,dia_matrix,load_npz
from collections import defaultdict
from sklearn.utils import Bunch
try:
    # Python 3.9+
    from importlib.resources import files
except ImportError:
    # Python 3.7–3.8
    from importlib_resources import files
from . import data  # mif/data/
import shutil
from math import isclose
import logging
import json
import pickle
import sys
from typing import Optional, Dict

#If you use a Google Colab user, run the following.
#from google.colab import drive
#drive.mount('/content/drive')
#from google.colab import files

def fileOnColab(filename, basepath = "/content/drive/My Drive/Colab Notebooks"):
    filepath = os.path.join(basepath, filename)
    print(filepath)
    return filepath

formatter = logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S")
logger_a = logging.getLogger("MiF")
logger_a.setLevel(logging.INFO)

fh_a = logging.FileHandler("MiF.log", mode="w", encoding="utf-8")
fh_a.setFormatter(formatter)
ch_a = logging.StreamHandler(sys.stdout)
ch_a.setFormatter(formatter)

logger_a.addHandler(fh_a)
logger_a.addHandler(ch_a)
logger_a.propagate = False

logger_b = logging.getLogger("MatrixLoader")
logger_b.setLevel(logging.INFO)

fh_b = logging.FileHandler("matrix.log", mode="w", encoding="utf-8")
fh_b.setFormatter(formatter)
ch_b = logging.StreamHandler(sys.stdout)
ch_b.setFormatter(formatter)

logger_b.addHandler(fh_b)
logger_b.addHandler(ch_b)
logger_b.propagate = False

logger_c = logging.getLogger("re_mcl")
logger_c.setLevel(logging.INFO)

fh_c = logging.FileHandler("re_mcl.log", mode="w", encoding="utf-8")
fh_c.setFormatter(formatter)
ch_c = logging.StreamHandler(sys.stdout)
ch_c.setFormatter(formatter)

logger_c.addHandler(fh_c)
logger_c.addHandler(ch_c)
logger_c.propagate = False

def resolve_logger(logger: Optional[logging.Logger], context: str) -> logging.Logger:
    return logger if logger is not None else get_logger(context)

def get_logger(context: str) -> logging.Logger:
    context = context.lower()
    if context in ("mif", "mifdi", "distance", "similarity"):
        return logger_a
    elif context in ("matrix", "loader", "io"):
        return logger_b
    elif context in ("mcl", "re_mcl", "rmcl"):
        return logger_c    
    else:
        raise ValueError(f"Unknown logging context: {context}")

CONST_COEFFICIENT_ARRAY = (1, 1.618033988749895, 1.8392867552141607, 1.9275619754829254, 1.9659482366454855, 1.9835828434243263, 1.9919641966050354, 1.9960311797354144, 1.9980294702622872, 1.9990186327101014)

def load_mif(return_X_y=False, as_frame=False, scaled=False):
    base_path = os.path.join(os.path.dirname(__file__), "data")    
    return Bunch(
        erdosReny = os.path.join(base_path, "ErdosReny.mtx"),
        gadget = os.path.join(base_path, "gadget.mtx"),
        heterophilly = os.path.join(base_path, "heterophilly.mtx"),
        homophilly = os.path.join(base_path, "homophilly.mtx"),
        karateclub = os.path.join(base_path, "karateclub.mtx"),
        scalefree = os.path.join(base_path, "scalefree.mtx"),
        eat = os.path.join(base_path, "eat.mtx"),
        DESCR="This is a toy dataset consisting of six sparse matrices in Matrix Market format."
    )

class SafeCSR(csr_matrix):
    def __repr__(self):
        return f"<SafeCSR shape={self.shape}, nnz={self.nnz}, dtype={self.dtype}>"

    __str__ = __repr__ 

def adjacencyinfocheck(adjacencymatrix, logger = None):
    log = resolve_logger(logger, "matrix")
    print(f"log name: {log.name}")
    path_or_matrix = adjacencymatrix
    src = path_or_matrix if isinstance(path_or_matrix, str) else "<in-memory>"
    
    if isinstance(path_or_matrix, str) and os.path.exists(path_or_matrix):
        path = path_or_matrix
        ext = os.path.splitext(path)[1].lower()

        if ext == ".mtx":
            matrix = mmread(path).tocsr()
            log.info("Loaded .mtx file → CSR.")

        elif ext == ".npz":
            loaded = np.load(path, allow_pickle=True)
            if 'data' in loaded and 'indices' in loaded and 'indptr' in loaded:
                matrix = load_npz(path).tocsr()
                log.info("Loaded sparse .npz file → CSR.") 
            else:
                matrix = csr_matrix(loaded['arr_0'])
                log.info("Loaded dense .npz file → CSR.")
                
        elif ext == ".pkl":
            with open(path, "rb") as f:
                obj = pickle.load(f)
                matrix =adjacencyinfocheck(obj)
                
        elif ext == ".csv":
            arr = np.loadtxt(path, delimiter=",")
            matrix = csr_matrix(arr)
            log.info("Loaded .csv file → CSR.")
            
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if all(k in data for k in ("row", "col", "data", "shape")):
                row = np.array(data["row"], dtype=np.int64)
                col = np.array(data["col"], dtype=np.int64)
                vals = np.array(data["data"], dtype=np.float64)
                shape = tuple(data["shape"])
                matrix = coo_matrix((vals, (row, col)), shape=shape).tocsr()
                log.info("Loaded sparse JSON (COO format) → CSR.")
            else:
                arr = np.array(data, dtype=np.float64)
                matrix = csr_matrix(arr)
                log.info("Loaded dense JSON → CSR.")
                
        else:
            msg = f"Unsupported file extension: {ext}"
            log.error(msg)
            raise ValueError(msg)

    else:
        matrix = path_or_matrix

        if isinstance(matrix, csr_matrix):
          log.info(f"Matrix is already CSR format (shape={matrix.shape}, nnz={matrix.nnz})")
          
        elif issparse(matrix):
          matrix = csr_matrix(matrix)  
          log.info(f"Converting {type(matrix).__name__} to CSR format (shape={matrix.shape}, nnz={matrix.nnz})")
         
        elif isinstance(matrix, np.ndarray):
          matrix = csr_matrix(matrix)    
          log.info(f"Converting dense ndarray to CSR format (shape={matrix.shape})")
          
        else:
          msg = f"Unsupported input type: {type(matrix)}"
          log.error(msg)
          raise TypeError(msg)

    log.info(f"Matrix loaded successfully (type={type(matrix).__name__}, shape={matrix.shape}, nnz={matrix.nnz})") 
    return SafeCSR(matrix)


def MiF_ZeroBasedIndex(adjacencymatrixchecked, x, y, beta, gamma, logger=None):
    log = resolve_logger(logger, "MiF")
    print(f"log name: {log.name}")
    val = 0
    coefficientlist = list(CONST_COEFFICIENT_ARRAY)
    log.info(f"Here, all integer values are assumed to be 0-based indexes, i.e. data and parameters --including node numbers and gamma-- that are counted starting from 0.In other words, it is assumed that a sparse matrix with a 0-based index created in C, C++, Python,etc. was input here.")
    adj_matrix = adjacencymatrixchecked
    alphalist = [(1 / coefficientlist[gamma]) ** (i + 1) for i in range(0, gamma + 1)]
    i = None
    tmat = {0: adj_matrix}
    for k in range(0, gamma):
        tmat[k + 1] = tmat[k] @ tmat[0]
    for i in range(0, gamma + 1):
        matpower = tmat[i]
        matpowerxandy = matpower[x, y]
        sumupx = np.sum(matpower[x, :])
        sumupy = np.sum(matpower[y, :])
        numerator = alphalist[i] * matpowerxandy * (beta * sumupx + (1 - beta) * sumupy)
        denominator = sumupx * sumupy
        if denominator != 0:
            val += numerator / denominator
    return val

def MiF_OneBasedIndex(adjacencymatrixchecked, x, y, beta, gamma, logger=None):
    log = resolve_logger(logger, "MiF")
    print(f"log name: {log.name}")
    val = 0
    coefficientlist = list(CONST_COEFFICIENT_ARRAY)
    log.info(f"Here, all integer values are assumed to be 1-based indexes, i.e. data and parameters --including node numbers and gamma-- that are counted starting from 1.In other words, it is assumed that a sparse matrix with a 1-based index created in MATLAB, Mathematica, Julia, Fortran, R, etc. was input here.")
    adj_matrix = adjacencymatrixchecked
    alphalist = [(1 / coefficientlist[gamma - 1]) ** (i + 1) for i in range(0, gamma)]
    i = None
    tmat = {0: adj_matrix}
    for k in range(0, gamma - 1):
        tmat[k + 1] = tmat[k] @ tmat[0]
    for i in range(0, gamma):
        matpower = tmat[i]
        matpowerxandy = matpower[x - 1, y - 1]
        sumupx = np.sum(matpower[x - 1, :])
        sumupy = np.sum(matpower[y - 1 , :])
        numerator = alphalist[i] * matpowerxandy * (beta * sumupx + (1 - beta) * sumupy)
        denominator = sumupx * sumupy
        if denominator != 0:
            val += numerator / denominator
    return val

def MiF(adjacencymatrixchecked, x, y, beta, gamma,index_base = 0, gamma_threshold = 10, logger=None):
    log = resolve_logger(logger, "MiF")
    print(f"log name: {log.name}")
    if index_base == 0:
        return MiF_ZeroBasedIndex(adjacencymatrixchecked, x, y, beta, gamma,logger)
    elif index_base == 1:
        return MiF_OneBasedIndex(adjacencymatrixchecked, x, y, beta, gamma,logger)

def MiF_broadcast_withloop(adjacencymatrixchecked, startingvertex, beta = 0.5, gamma_threshold = 10, logger=None):
    log = resolve_logger(logger, "MiF")
    print(f"log name: {log.name}")
    adj_matrix = adjacencymatrixchecked
    Gobj = nx.from_scipy_sparse_array(adj_matrix)
    degdicformat = nx.degree(Gobj)
    deglst =list([degdicformat[i] for i in range(0,len(degdicformat))])
    log.info(f"the number od nodes: {len(deglst)}")
    alllistednodes = range(0,len(degdicformat))
    if int(startingvertex) > len(deglst):
       log.info(f"The starting node does not exist.")
    elif deglst[startingvertex] == 0:
       log.info(f"The starting node is isolated.")
    else:
       targettednodes = [i for i in alllistednodes if i not in [startingvertex]]
       gammaval = 0
       while gammaval < gamma_threshold:
          mifsteps = [[startingvertex, MiF(adj_matrix, startingvertex, j, beta, gammaval)] for j in targettednodes]
          reached = [[x, i] for i, x in enumerate(mifsteps) if x [1]!= np.float64(0.0)]
          resultinfo_tmp =  [[i[0][0], i[1], i[0][1]] for i in reached]
          gammaval = gammaval + 1
          if len(mifsteps) == len(reached):
             log.info(f"Gamma reached the maximum values, since all the nodes have been reached from the starting nodes.")
             return resultinfo_tmp
             break

def MiF_broadcast_withoutloop(adjacencymatrixchecked, startingvertex, beta = 0.5, gamma_threshold = 10, logger=None):
    log = resolve_logger(logger, "MiF")
    print(f"log name: {log.name}")
    adj_matrix = adjacencymatrixchecked
    Gobj = nx.from_scipy_sparse_array(adj_matrix)
    degdicformat = nx.degree(Gobj)
    deglst =list([degdicformat[i] for i in range(0,len(degdicformat))])
    log.info(f"the number od nodes: {len(deglst)}")
    alllistednodes = range(0,len(degdicformat))
    if int(startingvertex) > len(deglst):
       log.info(f"The starting node does not exist.")
    elif deglst[startingvertex] == 0:
       log.info(f"The starting node is isolated.")
    else:
       targettednodes = [i for i in alllistednodes if i not in [startingvertex]]
       gammaval = 0
       reachedlist =[]
       reachedlist =[]
       reachednodevalslist = []
       remainingnodes = targettednodes
       while gammaval < gamma_threshold:
           reachednodesfromstartingnodes = []
           reachednodesfromeachstartingnode = []
           mifsteps = [[startingvertex, j, MiF(adj_matrix, startingvertex, j, beta, gammaval)] for j in targettednodes]
           reached = [x for i, x in enumerate(mifsteps) if x[2]!= np.float64(0.0)]
           log.info(f"reached: {reached}")
           reachednodes = [x[1] for i, x in enumerate(reached)]
           log.info(f"list of nodes that, once reached, should be skipped without going any further, i.e. reachednodes: {reachednodes}")
           reachedlist.append(reachednodes)
           log.info(f"reachedlist: {reachedlist}")
           remainingnodes = [x for i, x in enumerate(remainingnodes) if x not in list(itertools.chain.from_iterable(reachedlist))]
           log.info(f"list of nodes that remain to be reached, i.e. remainingnodes: {remainingnodes}")
           gammaval += 1
           reachedlist = []
           reachednodevalslist = reachednodevalslist + reached
           if len(remainingnodes) == 0:
               log.info(f"Gamma reached the maximum values, since all the nodes have been reached from the starting nodes.")
               return reached
               break

def MiF_broadcast(adjacencymatrixchecked, startingvertex, beta = 0.5, gamma_threshold = 10, loop = 0,logger=None):
    log = resolve_logger(logger, "MiF")
    print(f"log name: {log.name}")
    if loop == 0:
        return MiF_broadcast_withoutloop(adjacencymatrixchecked, startingvertex, beta, gamma_threshold,logger)
    elif loop == 1:
        return MiF_broadcast_withloop(adjacencymatrixchecked, startingvertex, beta, gamma_threshold,logger)

def MiFDI_withloop(adjacencymatrixchecked, startingvertices = "min", beta = 0.2, gamma_threshold = 10, logger=None):
  log = resolve_logger(logger, "MiF")
  print(f"log name: {log.name}")
  adj_matrix = adjacencymatrixchecked
  Gobj = nx.from_scipy_sparse_array(adj_matrix)
  degdicformat = nx.degree(Gobj)
  deglst =list([degdicformat[i] for i in range(0,len(degdicformat))])
  alllistednodes = range(0,len(degdicformat))
  if startingvertices == "min":
      smallestdegval = min(deglst)
      mindegnodes = [i for i, x in enumerate(deglst) if x == min(deglst)]
      log.info(f"the smallest degree: {smallestdegval}")
      log.info(f"the node numbers with the smallest degree : {mindegnodes}")
      targettednodes = [i for i in alllistednodes if i not in mindegnodes]
      startingnodes = mindegnodes
  elif startingvertices == "max":
      largestdegval = max(deglst)
      maxdegnodes = [i for i, x in enumerate(deglst) if x == max(deglst)]
      log.info(f"the largest degree: {largestdegval}")
      log.info(f"the node numbers with the largest degree : {maxdegnodes}")
      targettednodes = [i for i in alllistednodes if i not in maxdegnodes]
      startingnodes = maxdegnodes
  gammaval = 0
  logmifmeanlist =[]
  while gammaval < gamma_threshold:
      mifsteps = [[startingnodes[i], MiF(adj_matrix,startingnodes[i], j, beta, gammaval)] for i in range(0, len(startingnodes)) for j in targettednodes]
      reached = [[x, i] for i, x in enumerate(mifsteps) if x [1]!= np.float64(0.0)]
      logresultinfo_tmp =  [[i[0][0], i[1], math.log(i[0][1])] for i in reached]
      logresultinfo = [[x[0], x[1] - startingnodes.index(x[0]) *  len(targettednodes), x[2]] for i, x in enumerate(logresultinfo_tmp)]
      log.info(f"Current gamma: {gammaval}, [Starting node, Reached node, Log(MiF)]: {logresultinfo}")
      meanlog =  np.mean([logresultinfo[l][2] for l in range(len(logresultinfo))])
      log.info(f"Current gamma: {gammaval}, Mean of the Log(MiF): {meanlog}")
      logmifmeanlist.append(meanlog)
      gammaval = gammaval + 1
      if len(mifsteps) == len(reached):
         log.info(f"Gamma reached the maximum values, since all the nodes have been reached from the starting nodes.")
         break
  mifdi = sum(logmifmeanlist)
  log.info(f"MiFDI value: {mifdi}")


def MiFDI_withoutloop(adjacencymatrixchecked, startingvertices = "min", beta = 0.2, gamma_threshold = 10, logger=None):
   log = resolve_logger(logger, "MiF")
   print(f"log name: {log.name}")
   adj_matrix = adjacencymatrixchecked
   Gobj = nx.from_scipy_sparse_array(adj_matrix)
   degdicformat = nx.degree(Gobj)
   deglst =list([degdicformat[i] for i in range(0,len(degdicformat))])
   smallestdegval = min(deglst)
   mindegnodes = [i for i, x in enumerate(deglst) if x == min(deglst)]
   log.info(f"the smallest degree: {smallestdegval}")
   log.info(f"the node numbers with the smallest degree : {mindegnodes}")
   alllistednodes = range(0,len(degdicformat))
   targettednodes = [i for i in alllistednodes if i not in mindegnodes]
   gammaval = 0
   logmiflist =[]
   reachedlist =[]
   reachednodevalslist = []
   remainingnodes = targettednodes
   while gammaval <  gamma_threshold:
       reachednodesfromminnodes = []
       reachednodesfromeachminnode = []
       for i in range(0, len(mindegnodes)):
            reachednodesfromeachminnode = []
            mifsteps = [[mindegnodes[i], j, MiF(adj_matrix, mindegnodes[i], j, beta, gammaval)] for j in remainingnodes]
            reached = [x for i, x in enumerate(mifsteps) if x[2]!= np.float64(0.0)]
            logresultinfo =  [[k[0], k[1], math.log(k[2])] for k in reached]
            log.info(f"Current gamma: {gammaval}, [Starting node, Reached node, Log(MiF)]: {logresultinfo}")
            logvals = [x[2] for i, x in enumerate(logresultinfo)]
            logmiflist.append(logvals)
            reachednodes = [x[1] for i, x in enumerate(logresultinfo)]
            reachedlist.append(reachednodes)
            reachednodesbyeachpath = [x[1] for i, x in enumerate(logresultinfo)]
            reachednodesfromeachminnode.append(reachednodesbyeachpath)
            reachedinfo = reachednodesfromeachminnode[0]
            reachednodesfromminnodes.append(reachedinfo)
       reachednodesskip = list(set(list(itertools.chain.from_iterable(reachednodesfromminnodes))))
       log.info(f"list of nodes that, once reached, should be skipped without going any further, i.e. reachednodesskip: {reachednodesskip}")
       remainingnodes = [x for i, x in enumerate(remainingnodes) if x not in reachednodesskip]
       log.info(f"list of nodes that remain to be reached, i.e. remainingnodes: {remainingnodes}")
       logmiflisteachgamma =list(itertools.chain.from_iterable(logmiflist))
       reachedlisteachgamma =list(itertools.chain.from_iterable(reachedlist))
       reachednodevals = [[reachedlisteachgamma[i],logmiflisteachgamma[i]] for i in range(0, len(logmiflisteachgamma))]
       gammaval += 1
       logmiflist = []
       reachedlist = []
       reachednodevalslist = reachednodevalslist + reachednodevals
       if len(remainingnodes) == 0:
          log.info(f"Gamma reached the maximum values, since all the nodes have been reached from the starting nodes.")
          break
   result_tmp = sorted(reachednodevalslist, key=lambda x: x[0])
   grouped = defaultdict(list)
   for key, value in result_tmp:
      grouped[key].append(value)
   result = [[k, np.mean(v)] for k, v in sorted(grouped.items())]
   allresult = result + [[mindegnodes[i], 0] for i in range(0, len(mindegnodes))]
   allresult = sorted(allresult, key=lambda x: x[0])
   log.info(f"allresult: {allresult}")
   mifval = [x[1] for i, x in enumerate(allresult) if x[1] != 0]
   mifdi = np.mean(mifval)
   log.info(f"MiFDI value: {mifdi}")
   return allresult, mifdi


def MiFDI(adjacencymatrixchecked, startingvertices="min", beta = 0.2, gamma_threshold = 10, loop = 0, logger=None):
    log = resolve_logger(logger, "MiF")
    print(f"log name: {log.name}")
    if loop == 0:
        return MiFDI_withoutloop(adjacencymatrixchecked, startingvertices, beta, gamma_threshold,logger)
    elif loop == 1:
        return MiFDI_withloop(adjacencymatrixchecked, startingvertices, beta, gamma_threshold,logger)


