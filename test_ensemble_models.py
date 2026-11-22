import unittest

import numpy as np

from ensemble_models import run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput


class RunEnsembleExperimentTests(unittest.TestCase):
    def test_array_inputs_and_test_target_isolation(self):
        rng = np.random.default_rng(42)
        features = rng.normal(size=(100, 4))
        target = np.tile([0, 1], 50)
        train_features, test_features = features[:80], features[80:]
        train_target, test_target = target[:80], target[80:]

        result = run_ensemble_experiment(
            EnsembleInput(train_features, train_target, test_features, test_target, 42)
        )
        altered_target_result = run_ensemble_experiment(
            EnsembleInput(
                train_features,
                train_target,
                test_features,
                1 - test_target,
                42,
            )
        )

        self.assertEqual(len(result.probabilities), len(test_target))
        self.assertEqual(len(result.predictions), len(test_target))
        self.assertTrue(set(np.unique(result.predictions)).issubset({0, 1}))
        self.assertTrue(np.all((result.probabilities >= 0) & (result.probabilities <= 1)))
        np.testing.assert_array_equal(result.predictions, altered_target_result.predictions)
        np.testing.assert_allclose(
            result.probabilities,
            altered_target_result.probabilities,
            rtol=0,
            atol=1e-14,
        )


if __name__ == "__main__":
    unittest.main()
