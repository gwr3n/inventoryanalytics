'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import matplotlib.pyplot as plt
import inventoryanalytics.lotsizing.stochastic.nonstationary.sdp as sdp

instance = {"K": 100, "v": 0, "h": 1, "p": 10, "d": [20,40,60,40],
            "max_inv": 200, "q": 0.9999, "initial_order": True}
lot_sizing = sdp.StochasticLotSizing(**instance)
t = 0   # initial period
i = 0   # initial inventory level
#print("Optimal policy cost: "    + str(lot_sizing.f(i)))
#print("Optimal order quantity: " + str(lot_sizing.q(t, i)))
print(lot_sizing.extract_sS_policy())

instance["initial_order"]=False
lot_sizing = sdp.StochasticLotSizing(**instance)
plt.plot([lot_sizing.f(k) for k in range(-20,150)])
plt.ylabel('Optimal policy cost')
plt.show()