import unittest
import numpy as np

from sklearn.datasets import load_boston

from skcosmo.feature_selection import CUR, FPS
from skcosmo.feature_selection import BlockSelector, RollingSelector


class TestBase(unittest.TestCase):
    def setUp(self):
        self.X, _ = load_boston(return_X_y=True)

    def test_n_total(self):
        block_selector = BlockSelector(
            n_features_to_select=4,
            selector=CUR(n_features_to_select=4),
            max_block_size=2,
        )
        self.assertEqual(block_selector._n_total(self.X), self.X.shape[-1])


class TestBlockSelector(unittest.TestCase):
    def setUp(self):
        self.X, _ = load_boston(return_X_y=True)
        self.initial = 9

    def testFPS(self):
        original = FPS(n_features_to_select=4)
        original.fit(self.X)

        block_selector = BlockSelector(
            n_features_to_select=4,
            selector=FPS(n_features_to_select=4),
            max_block_size=2,
        )
        block_selector.fit(self.X)
        self.assertTrue(
            np.allclose(original.selected_idx_, block_selector.selected_idx_)
        )

    def test_CUR(self):
        original = CUR(n_features_to_select=4)
        original.fit(self.X)

        block_selector = BlockSelector(
            n_features_to_select=4,
            selector=CUR(n_features_to_select=4),
            max_block_size=2,
        )
        block_selector.fit(self.X)

        print(original.selected_idx_)
        self.assertTrue(
            np.allclose(original.transform(self.X), block_selector.transform(self.X))
        )


class TestRollingSelector(unittest.TestCase):
    def setUp(self):
        self.X, _ = load_boston(return_X_y=True)

    def test_CUR(self):
        original = CUR(n_features_to_select=12)
        original.fit(self.X)

        rolling_selector = RollingSelector(
            selector=CUR(),
        )

        for i in range(3):
            rolling_selector.fit(self.X, warm_start=(i > 0), n_features_to_select=4)

        Xr_rolling = rolling_selector.transform([self.X, self.X, self.X])
        Xr_original = self.X[:, original.selected_idx_]

        self.assertTrue(np.allclose(Xr_rolling, Xr_original))

    def test_bad_fit(self):
        rolling_selector = RollingSelector(
            selector=CUR(),
        )
        rolling_selector.fit(self.X, n_features_to_select=2)

        with self.assertRaises(ValueError) as cm:
            rolling_selector.fit(self.X[:4], n_features_to_select=2, warm_start=True)
            self.assertEqual(
                str(cm.message),
                "This feature set does not contain values for all previously fit samples. Expecting 506, got 4",
            )

    def test_bad_transform(self):
        rolling_selector = RollingSelector(
            selector=CUR(),
        )
        for i in range(3):
            rolling_selector.fit(self.X, warm_start=(i > 0), n_features_to_select=4)

        with self.assertRaises(ValueError) as cm:
            rolling_selector.transform(self.X)
            self.assertEqual(
                str(cm.message), "We are expecting 3 feature sets for X, got 506."
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
