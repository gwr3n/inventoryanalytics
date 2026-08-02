import unittest

import numpy as np
import pandas as pd

import inventoryanalytics.forecasting.AR as ar


class ARForecastingTest(unittest.TestCase):

    def test_sample_random_walk_is_deterministic(self):
        expected = [
            0.47143516373249306,
            -0.7195405309739714,
            0.7131664374521259,
            0.40051454136041303,
            -0.3200741920045986,
        ]
        self.assertEqual(list(ar.sample_random_walk(0, 5)), expected)


if __name__ == "__main__":
    unittest.main()
