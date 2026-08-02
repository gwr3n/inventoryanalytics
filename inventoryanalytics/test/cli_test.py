import contextlib
import io
import json
import unittest

from inventoryanalytics.cli import ALGORITHMS, main


class CliTest(unittest.TestCase):
    def run_cli(self, *arguments):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(arguments)
        self.assertEqual(0, exit_code)
        return json.loads(output.getvalue())

    def test_catalog_contains_named_algorithms_only(self):
        names = {algorithm.name for algorithm in ALGORITHMS}

        self.assertIn("els", names)
        self.assertIn("naive", names)
        self.assertIn("holt-winters", names)
        self.assertIn("arima", names)
        self.assertIn("box-cox", names)
        self.assertIn("forecast-errors", names)
        self.assertIn("wagner-whitin", names)
        self.assertNotIn("State", names)
        self.assertNotIn("compute_els", names)

    def test_runs_els(self):
        result = self.run_cli(
            "-method", "els",
            "--n", "3",
            "--p", "[400,400,500]",
            "--d", "[50,50,60]",
            "--h", "[20,20,30]",
            "--s", "[0.1,0.1,0.1]",
            "--K", "[2000,2500,800]",
        )

        self.assertAlmostEqual(1.78310546875, result["cycle_length"])

    def test_runs_eoq(self):
        result = self.run_cli(
            "-method", "eoq", "--K", "100", "--h", "2", "--d", "100", "--v", "1"
        )

        self.assertAlmostEqual(100, result["order_quantity"])

    def test_runs_naive_forecast(self):
        result = self.run_cli("-method", "naive", "--series", "[10,12,14]", "--horizon", "2")

        self.assertEqual([14, 14], result["forecasts"])

    def test_runs_holt_winters_forecast(self):
        result = self.run_cli(
            "-method", "holt-winters",
            "--series", "[10,12,14,16,11,13,15,17,12,14,16,18]",
            "--horizon", "2",
            "--season-length", "4",
        )

        self.assertEqual(2, len(result["forecasts"]))


if __name__ == "__main__":
    unittest.main()