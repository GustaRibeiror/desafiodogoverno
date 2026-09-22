import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_oof import diagnose


class DiagnosticTests(unittest.TestCase):
    def test_calibration_uses_other_folds_and_pooled_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for i, scale in enumerate([1., 2., 3.]):
                p = Path(folder) / f'fold{i}.npz'
                np.savez(p, prediction=np.full(48, 12.), climatology=np.full(48, 10.),
                         truth=np.full(48, 10. + scale), lead=np.repeat(np.arange(1, 25), 2),
                         point_flat=np.array([0, 1]))
                paths.append(p)
            report = diagnose(paths)
            self.assertAlmostEqual(report['aggregate_pooled_rmse']['raw']['rmse_year2'], np.sqrt(2/3))
            self.assertEqual(report['folds'][0]['weights_from_other_folds'], [1.25] * 4)
            self.assertEqual(report['folds'][2]['weights_from_other_folds'], [.75] * 4)
            self.assertFalse(report['accept_calibration_vs_raw'])
            self.assertEqual(report['deployment_weights'], [1.] * 4)

    def test_rejects_duplicate_paths(self):
        with self.assertRaises(ValueError):
            diagnose(['same.npz'] * 3)


if __name__ == '__main__':
    unittest.main()
