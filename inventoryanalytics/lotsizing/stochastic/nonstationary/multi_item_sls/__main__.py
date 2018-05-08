import matplotlib.pyplot as plt
import inventoryanalytics.lotsizing.stochastic.nonstationary.sdp_multi_item as sdp

'''
Optimal policy cost: 52.45080337182663
Optimal order quantity: (4, 4)

Optimal policy cost: 42.42570977263608
Optimal order quantity: (8, 8)
'''

instance = {"K": 10, "v": 0, "h": 1, "p": 5, "d": [3,6,9,6],
            "max_inv": 15, "q": 0.95, "initial_order": True}
lot_sizing = sdp.MultiItemStochasticLotSizing(**instance)
t = 0       # initial period
i = (0,0)   # initial inventory level
print("Optimal policy cost: "    + str(lot_sizing.f(i)))
print("Optimal order quantity: " + str(lot_sizing.q(t, i)))

#instance["initial_order"]=False
#lot_sizing = sdp.MultiItemStochasticLotSizing(**instance)
#points = [(i,j,lot_sizing.f((i,j))) for i in range(0,15) for j in range(0,15)]
#print(points)