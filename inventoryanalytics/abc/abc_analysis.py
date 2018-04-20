import csv
import numpy as np
from typing import List

class abc:

    def __init__(self):
        self.data = abc.readData('./inventoryanalytics/abc/data/Flores1992.csv')

    def annual_dollar_usage(self):
        data_adu = abc.compute_adu(self.data)
        print(np.matrix(data_adu))

    def ahp_weighted(self):
        data_ahp = abc.compute_ahp(self.data)
        print(np.matrix(data_ahp))

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

    @staticmethod
    def compute_adu(d):
        data = []
        for k in range(1,len(d)):
            try:
                adu = float(d[k][1])*float(d[k][2])
            except ValueError:
                pass
            data.append([d[k][0],d[k][1],d[k][2],adu])
        return sorted(data,key=lambda x: x[3], reverse=True)

    @staticmethod
    def compute_ahp(d):
        data = []
        weights = [0.078,0.092,0.42,0.41] #average unit cost, annual dollar usage, criticality, lead time
        matrix = np.matrix(d)
        data = matrix[1:,[2,3,6,7]].astype(np.float)
        fmax = data.max(0)
        print(fmax)
        fmin = data.min(0)
        print(fmin)
        out = []
        for k in range(1,len(d)-1):
            try:
                out.append([d[k+1][0],
                            weights[0]*(data[k,0]-fmin[0,0])/(fmax[0,0]-fmin[0,0])+
                            weights[1]*(data[k,1]-fmin[0,1])/(fmax[0,1]-fmin[0,1])+
                            weights[2]*(data[k,2]-fmin[0,2])/(fmax[0,2]-fmin[0,2])+
                            weights[3]*(data[k,3]-fmin[0,3])/(fmax[0,3]-fmin[0,3]),d[k+1][8]])
            except ValueError:
                pass
        return sorted(out,key=lambda x: x[1], reverse=True)

if __name__ == '__main__':
    d = abc()
    #d.annual_dollar_usage()
    d.ahp_weighted()
    
    