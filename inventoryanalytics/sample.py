import scipy.stats as sp

if __name__ == '__main__':
    #q = 0.9
    #max_demand = lambda d: sp.poisson(d).ppf(q).astype(int)
    #pmf = lambda d, k : sp.poisson(d).pmf(k)/q 
    #D = [2,4,6,4]
    #print([[[(i,j), pmf(d, i)*pmf(d, j)] for i in range(0, max_demand(d)) for j in range(0, max_demand(d))] for d in D])
    a = list(k for k in range(1,2))
    print(a[0])