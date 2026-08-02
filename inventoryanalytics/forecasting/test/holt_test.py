import unittest

import numpy as np
import pandas as pd

import inventoryanalytics.forecasting.holt as holt


class HoltForecastingTest(unittest.TestCase):

    def test_sample_random_walk_is_deterministic(self):
        expected = [
            0.571435163732493,
            -0.5195405309739715,
            1.0131664374521259,
            0.8005145413604131,
            0.17992580799540148,
        ]
        self.assertEqual(list(holt.sample_random_walk(0, 0.1, 5)), expected)

    def test_plot_data_shape_has_expected_columns(self):
        realisations = pd.Series([1, 2, 3, 4], index=[0, 1, 2, 3])
        forecasts = pd.Series([1.5, 2.5, 3.5], index=[0, 1, 2])
        self.assertEqual(len(realisations), 4)
        self.assertEqual(len(forecasts), 3)


if __name__ == "__main__":
    unittest.main()
