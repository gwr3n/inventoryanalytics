import csv
import numpy as np
from typing import List

class abc:

    def __init__(self):
        self.data = abc.readData('./inventoryanalytics/abc/data/Flores1992.csv')

    @staticmethod
    def annual_dollar_usage_instance():
        # prepare data
        d = abc()
        m = np.matrix(d.data)[1:,[1,2]].astype(np.float)

        # compute ADU
        adu = d.annual_dollar_usage(m)
        
        # verify ABC
        for k in range(0,len(adu)):
            adu[k].append(d.data[k+1][8])

        adu = sorted(adu,key=lambda x: x[3], reverse=True)
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

        # verify ABC
        for k in range(0,len(ahp)):
            ahp[k].append(d.data[k+1][8])

        ahp = sorted(ahp,key=lambda x: x[1], reverse=True)
        print(np.matrix(ahp))

    def ahp_weighted(self, m, w: List[float]):
        # compute ahp
        fmin, fmax = m.min(0), m.max(0)
        norm = lambda i, v : w[i]*(v-fmin[0,i])/(fmax[0,i]-fmin[0,i])
        data = [[self.data[k+1][0],sum(norm(i, m[k,i]) for i in range(0,4))] for k in range(1,len(m))]
        return data

    def dea(self):
        pass
    
    @staticmethod
    def readData(file: str):
        d = []
        with open(file) as csvfile:
            #d = dict(filter(None, csv.reader(csvfile, delimiter=',')))
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                d.append(row)
        return d

if __name__ == '__main__':
    d = abc()
    #d.annual_dollar_usage_instance()
    d.ahp_weighted_instance()
    
    