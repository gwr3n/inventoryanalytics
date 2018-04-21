import csv
import numpy as np
from typing import List
from scipy.optimize import linprog
import matplotlib.pyplot as plt

class abc:

    def __init__(self):
        self.data = abc.readData('./inventoryanalytics/abc/data/Flores1992.csv')

    @staticmethod
    def annotate_ABC(m):
        # annotate matrix with ABC
        for k in range(0,len(m)):
            if 1.0*k/len(m) < 0.2:
                m[k].append('A')
            elif 1.0*k/len(m) < 0.5:
                m[k].append('B')
            else:
                m[k].append('C')
        return m

    @staticmethod
    def annual_dollar_usage_instance():
        # prepare data
        d = abc()
        m = np.matrix(d.data)[1:,[1,2]].astype(np.float)

        # compute ADU
        adu = d.annual_dollar_usage(m)
        
        # append ADU classes
        for k in range(0,len(adu)):
            adu[k].append(d.data[k+1][8])

        adu = sorted(adu,key=lambda x: x[3], reverse=True)

        # annotate ABC
        abc.annotate_ABC(adu)

        print(np.matrix(adu))

    def annual_dollar_usage(self, m):
        # compute adu
        data = [[self.data[k+1][0],m[k,0],m[k,1],m[k,0]*m[k,1]] for k in range(0,len(m))]
        return data

    @staticmethod
    def ahp_weighted_instance():
        # prepare data
        d = abc()
        m = np.matrix(d.data)[1:,[2,3,6,7]].astype(np.float)
        w = [0.078,0.092,0.42,0.41] #average unit cost, annual dollar usage, criticality, lead time

        # compute AHP
        ahp = d.ahp_weighted(m, w)

        # append ADU classes
        for k in range(0,len(ahp)):
            ahp[k].append(d.data[k+1][8])

        ahp = sorted(ahp,key=lambda x: x[1], reverse=True)

        # annotate ABC
        abc.annotate_ABC(ahp)

        print(np.matrix(ahp))

    def ahp_weighted(self, m, w: List[float]):
        # compute ahp
        fmin, fmax = m.min(0), m.max(0)
        norm = lambda i, v : (v-fmin[0,i])/(fmax[0,i]-fmin[0,i])
        data = [[self.data[k+1][0],sum(w[i]*norm(i, m[k,i]) for i in range(0,4))] for k in range(1,len(m))]
        return data

    @staticmethod
    def dea_instance():
        # prepare data
        d = abc()
        m = np.matrix(d.data)[1:,[2,3,6,7]].astype(np.float)

        # compute DEA
        dea = d.dea(m)

        # append ADU classes
        for k in range(0,np.size(m,0)):
            dea[k].append(d.data[k+1][8])

        dea = sorted(dea,key=lambda x: x[1], reverse=True)

        # annotate ABC
        abc.annotate_ABC(dea)

        print(np.matrix(dea))

    def dea(self, m):
        # compute DEA
        dea = []
        fmin, fmax = m.min(0), m.max(0)
        norm = lambda j, v : (v-fmin[0,j])/(fmax[0,j]-fmin[0,j])
        lp_matrix = [[norm(j,m[k,j]) for j in range(0,np.size(m,1))] for k in range(0,np.size(m,0))]
        dea = [[self.data[i+1][0],abc.solveLP(i, lp_matrix)] for i in range(0,len(lp_matrix))]
        return dea

    @staticmethod
    def solveLP(i, dea):
        res = linprog([-k for k in dea[i]], dea, [1 for k in range(0,len(dea))])
        return -res.fun

    def k_nn(self):
        pass

    @staticmethod
    def abc_scatter():
        # prepare data
        d = abc()
        print(np.matrix(d.data))
        # plot points and annotations
        x, y = 2, 7 #1,2,6,7
        m = np.matrix(d.data)[1:,[x,y]].astype(np.float)
        my_colors = {'A':'red','B':'yellow','C':'green'}
        for k in range(0,len(d.data)-1):
            plt.scatter(m[:,0].flatten().tolist()[0][k],
                        m[:,1].flatten().tolist()[0][k],
                        color = my_colors.get(np.matrix(d.data)[1:,8].flatten().tolist()[0][k]))
            plt.annotate(np.matrix(d.data)[1:,8].flatten().tolist()[0][k], 
                         (m[:,0].flatten().tolist()[0][k],
                          m[:,1].flatten().tolist()[0][k]))
        # plot label and grid
        plt.xlabel(np.matrix(d.data)[0,x])
        plt.ylabel(np.matrix(d.data)[0,y])
        plt.grid(True)
        # show plot
        plt.show()

    @staticmethod
    def readData(file: str):
        d = []
        with open(file) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                d.append(row)
        return d

if __name__ == '__main__':
    d = abc()
    #d.annual_dollar_usage_instance()
    #d.ahp_weighted_instance()
    #d.dea_instance()
    d.abc_scatter()
    
    