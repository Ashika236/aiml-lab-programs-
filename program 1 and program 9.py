#!/usr/bin/env python
# coding: utf-8

# In[2]:



from math import ceil
import numpy as np
from scipy import linalg  


def lowess(x, y, f= 2. / 3., iter=3):
   
   n = len(x) # Number of x  points 
   r = int(ceil(f * n))  # Computing the residual of smoothing functions 
   h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)] # 
   w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)  # Weight Function 
   w = (1 - w ** 3) ** 3  # Tricube Weight Function
   ypred = np.zeros(n) # Initialisation of predictor 
   delta = np.ones(n)  # Initialisation of delta
  
   for iteration in range(iter):
       for i in range(n):
           weights = delta * w[:, i] # Cumulative Weights 
           b = np.array([np.sum(weights * y), np.sum(weights * y * x)]) # Matrix B
           A = np.array([[np.sum(weights), np.sum(weights * x)],
                         [np.sum(weights * x), np.sum(weights * x * x)]]) # Matrix A
                     
           beta = linalg.solve(A, b) # Beta,Solution of AX= B equation 
           ypred[i] = beta[0] + beta[1] * x[i]
            
       residuals = y - ypred   # Finding Residuals
       s = np.median(np.abs(residuals))  # Median of Residuals
       delta = np.clip(residuals / (6.0 * s), -1, 1)  # Delta
       delta = (1 - delta ** 2) ** 2   # Delta 

   return ypred

if __name__ == '__main__':  # Main Function
   
   import math
   
   n = 100  # Number of data points
  
   #Case1: Sinusoidal Fitting 
   x = np.linspace(0, 2 * math.pi, n)
   print(x)
   y = np.sin(x) + 0.3 * np.random.randn(n) 
   
   #Case2 : Straight Line Fitting
   #x=np.linspace(0,2.5,n) # For Linear
   #y= 1 + 0.25*np.random.randn(n) # For Linear
   
   f = 0.25
   ypred = lowess(x, y, f=f, iter=3)
   
   import pylab as pl
   pl.clf()
   pl.plot(x, y, label='Y NOISY')
   pl.plot(x, ypred, label='Y PREDICTED')
   pl.legend()
   pl.show()


# In[4]:


def aStarAlgo(start_node, stop_node):
    open_set = set(start_node) 
    closed_set = set()
    g = {} 
    parents = {}
    g[start_node] = 0
    parents[start_node] = start_node
    while len(open_set) > 0: 
        n = None
        for v in open_set:
            if n == None or g[v] + heuristic(v) < g[n] + heuristic(n): 
                 n = v
        if n == stop_node or Graph_nodes[n] == None: 
            pass
        else:
            for (m, weight) in get_neighbors(n):
                if m not in open_set and m not in closed_set: 
                    open_set.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                else:
                    if g[m] > g[n] + weight: 
                        g[m] = g[n] + weight 
                        parents[m] = n
                    if m in closed_set:
                        closed_set.remove(m) 
                        open_set.add(m)
        if n == None:
            print('Path does not exist!') 
            return None
        if n == stop_node:
            path = []
            while parents[n] != n: 
                path.append(n)
                n = parents[n] 
            path.append(start_node) 
            path.reverse()
            print('Path found: {}'.format(path)) 
            return path
        open_set.remove(n)
        closed_set.add(n)
    print('Path does not exist!') 
    return None

def get_neighbors(v): 
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return None
def heuristic(n):
    H_dist = { 
        'A': 11,
        'B': 6,
        'C': 99,
        'D': 1,
        'E': 7,
        'G': 0,
    }
    return H_dist[n]

Graph_nodes = {
'A': [('B', 2), ('E', 3)],
'B': [('C', 1),('G', 9)],
'C': None, 
'E': [('D', 6)],
'D': [('G', 1)],
}
aStarAlgo('A', 'G')


# In[ ]:




