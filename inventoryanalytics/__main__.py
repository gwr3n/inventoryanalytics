import inventoryanalytics.lotsizing.stochastic.nonstationary.scarf1960 as scarf
import json

instance = {"K": 100, "v": 0, "h": 1, "p": 10, "C": None, "d": [20,40,60,40]}
lot_sizing = scarf.StochasticLotSizing(**instance)
initial_inventory_level = 0
print("Optimal policy cost: " + str(lot_sizing.f(initial_inventory_level)))
print("Optimal order quantity: " + str(lot_sizing.q(initial_inventory_level)))

with open('optimal_policy.txt', 'w') as f:
    json.dump(lot_sizing.cache_actions, f)
    f.close()