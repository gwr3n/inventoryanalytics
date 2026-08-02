import importlib
import sys
import types
import unittest
from unittest import mock


def _import_rq_module():
    module_name = "inventoryanalytics.lotsizing.stochastic.nonstationary.RQ"
    if module_name in sys.modules:
        return sys.modules[module_name]

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != "docplex":
            raise

        fake_docplex = types.ModuleType("docplex")
        fake_mp = types.ModuleType("docplex.mp")
        fake_model = types.ModuleType("docplex.mp.model")

        class DummyModel:
            def __init__(self, *args, **kwargs):
                pass

        fake_model.Model = DummyModel

        with mock.patch.dict(
            sys.modules,
            {
                "docplex": fake_docplex,
                "docplex.mp": fake_mp,
                "docplex.mp.model": fake_model,
            },
        ):
            return importlib.import_module(module_name)


rq = _import_rq_module()


class TestRQ(unittest.TestCase):

    def test_base_stochastic_lot_sizing_constructor(self):
        model = rq.StochasticLotSizing(K=10, h=1, p=5, d=[3, 4], I0=2)

        self.assertEqual(model.K, 10)
        self.assertEqual(model.h, 1)
        self.assertEqual(model.p, 5)
        self.assertEqual(model.d, [3, 4])
        self.assertEqual(model.I0, 2)

    def test_rq_cplex_initialization_sets_expected_parameters(self):
        with mock.patch.object(rq.RQ_CPLEX, "model", return_value=None) as mocked_model:
            model = rq.RQ_CPLEX(K=100, h=2, p=7, d=[20, 30], std_d=[3, 4], I0=5)

        mocked_model.assert_called_once_with()
        self.assertEqual(model.K, 100)
        self.assertEqual(model.h, 2)
        self.assertEqual(model.p, 7)
        self.assertEqual(model.d, [20, 30])
        self.assertEqual(model.I0, 5)
        self.assertEqual(model.std_demand, [3, 4])
        self.assertEqual(model.W, 5)
        self.assertEqual(len(model.prob), 5)
        self.assertEqual(len(model.E), 5)
        self.assertAlmostEqual(model.e, 0.022270929512393414)


if __name__ == "__main__":
    unittest.main()
