import inventoryanalytics.lotsizing.stochastic.stationary.zf as ls

def zf_test_instances():
    data = {'mu': 10., 'K': 64, 'h': 1., 'b': 9.}
    zf = ls.ZhengFedergruen(**data)
    print(zf.findOptimalPolicy())
    print(zf.c(6,40))

    data['mu'] = 20
    zf = ls.ZhengFedergruen(**data)
    print(zf.findOptimalPolicy())
    print(zf.c(14,62))

    data['mu'] = 64
    zf = ls.ZhengFedergruen(**data)
    print(zf.findOptimalPolicy())
    print(zf.c(55,74))