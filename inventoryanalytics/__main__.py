import inventoryanalytics.lotsizing.stochastic.nonstationary.scarf1960 as scarf

instance = {"K": 100, "v": 0, "h":1, "p":10, "C":None, "d":[20,40,60,40]}
lot_sizing = scarf.StochasticLotSizing(**instance)
initial_inventory_level = 0
print("Optimal policy cost: " + str(lot_sizing.f(initial_inventory_level)))
print("Optimal order quantity: " + str(lot_sizing.q(initial_inventory_level)))