"""Critical numerical/data invariants; run with python -m unittest discover -s tests."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from eleson.data import align_recording, recording_labels, sensor, trajectory_labels
from eleson.engine import make_splits
from eleson.metrics import binary_auc, classification_metrics
from eleson.model import ELESON, ReverseGradient, objective


class DataTests(unittest.TestCase):
    def test_sample_clock_and_epoch_bridge(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"acc.csv"
            p.write_text("currentTimeMillis,elapsedRealtime,imuTimestamp,accuracy,x,y,z\n"
                         "1000100,100,80,3,1,2,3\n1000100,100,90,3,2,3,4\n"
                         "1000110,110,100,3,3,4,5\n", encoding="utf-8")
            t, xyz, offset = sensor(p)
            np.testing.assert_array_equal(t, [80,90,100])
            self.assertEqual(offset, 1000000)
            self.assertEqual(xyz.shape, (3,3))

    def test_segment_interval_and_missing_outside(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root/"Segmented_Data/Mall1/Elevator/test/acc_a.csv"
            p.parent.mkdir(parents=True)
            (p.parent/"speak.csv").write_text("cross-layer label,1000100\ncross-layer label,1000300\n")
            labels = recording_labels(root,p,np.arange(0,500,100),0,1000000,{},"unknown")
            np.testing.assert_array_equal(labels, [-1,1,1,-1,-1])

    def test_trajectory_merge_conflict_and_terminal_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = root/"Trajectory_Data"
            folder.mkdir()
            (folder/"Label_file_1.csv").write_text("traj_id,id\nrelative_time,0,1280\nlabel,3,4\n")
            (folder/"Label_file_2.csv").write_text("traj_id,id\nrelative_time,0,1280\nlabel,3,1\n")
            labels, conflicts = trajectory_labels(root)
            self.assertEqual(conflicts, 1)
            p = folder/"id/acc_a.csv"
            result = recording_labels(root,p,np.array([0,1270,1280,2560]),0,None,labels,"unknown")
            np.testing.assert_array_equal(result, [1,1,-1,-1])

    def test_multiple_rides_still_labels_and_damaged_directory_encoding(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # Reproduce the UTF-8/CP850 corruption in the released archive.
            name = "01扶梯01".encode("utf-8").decode("cp850")
            p = root/"Segmented_Data/Mall2"/name/"acc_a.csv"
            p.parent.mkdir(parents=True)
            (p.parent/"speak.csv").write_text("still label,1000000\nstill label,1000100\n"
                "cross-layer label,1000100\ncross-layer label,1000300\n"
                "cross-layer label,1000400\ncross-layer label,1000600\n")
            y = recording_labels(root,p,np.arange(0,700,100),0,1000000,{},"unknown")
            np.testing.assert_array_equal(y,[0,2,2,-1,2,2,-1])

    def test_truncated_sensor_row_retains_good_samples(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"acc.csv"
            p.write_text("elapsedRealtime,x,y,z\n0,1,2,3\n10,1,2\n20,1,2,3\n")
            with self.assertWarns(UserWarning):
                t,_,_ = sensor(p)
            np.testing.assert_array_equal(t,[0,20])

    def test_resampling_does_not_interpolate_through_dropouts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = root/"Single_Label_Data/Elevator/id"
            folder.mkdir(parents=True)
            for name in ("acc", "gyro", "mag"):
                with (folder/f"{name}_a.csv").open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["elapsedRealtime","x","y","z"])
                    for t in [0,10,20,500,510,520]:
                        writer.writerow([t,0,0,9.8])
            x,y,_,_,_ = align_recording(root,folder/"acc_a.csv",{},max_gap_ms=100)
            self.assertTrue((y[3:50] == -1).all())
            self.assertTrue(np.isfinite(x).all())

    def test_first_magnetic_difference_is_observed_not_fabricated(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            folder=root/"Single_Label_Data/Elevator/id"
            folder.mkdir(parents=True)
            for name in ("acc", "gyro", "mag"):
                (folder/f"{name}_a.csv").write_text("elapsedRealtime,x,y,z\n0,1,0,0\n10,3,0,0\n20,6,0,0\n")
            x,_,_,t,_=align_recording(root,folder/"acc_a.csv",{})
            np.testing.assert_array_equal(t,[10,20])
            np.testing.assert_array_equal(x[:,6],[2,3])

    def test_group_split_has_no_capture_leakage(self):
        records = [dict(group=str(i//2), labels=[i%3]*5, source="Single_Label_Data",mall=None) for i in range(30)]
        folds = make_splits({"records":records})
        test_seen = []
        for split in folds:
            groups = {k:{records[i]["group"] for i in v} for k,v in split.items()}
            self.assertFalse(groups["train"] & groups["val"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["val"] & groups["test"])
            test_seen.extend(split["test"])
        self.assertEqual(sorted(test_seen), list(range(30)))


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gradient_reversal(self):
        x = torch.tensor([2.], requires_grad=True)
        w = torch.tensor([3.], requires_grad=True)
        (ReverseGradient.apply(x)*w).sum().backward()
        self.assertEqual(x.grad.item(), -3.)
        self.assertEqual(w.grad.item(), 2.)

    def test_forward_backward_and_evidence_simplex(self):
        torch.manual_seed(42)
        model = ELESON(hidden=4)
        x = torch.randn(6,200,7)
        y = torch.tensor([0,1,2,0,1,2])
        out = model(x, auxiliary=True)
        torch.testing.assert_close(out["confidence"].sum(-1)+out["uncertainty"], torch.ones(6))
        loss, terms = objective(out,x,y,torch.tensor([0,0,1,1,1,0]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for module in (model.motion,model.magnetic,model.generator,model.intervention,model.adversary,model.evidence):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertTrue(any(g.abs().sum() > 0 for g in grads))
        self.assertTrue(all(torch.isfinite(v) for v in terms.values()))

    def test_no_conveyors_and_single_class_batch(self):
        model = ELESON(hidden=4)
        x = torch.randn(2,200,7)
        y = torch.zeros(2,dtype=torch.long)
        out = model(x, auxiliary=True)
        loss,_ = objective(out,x,y,y)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_zero_evidence_has_zero_confidence(self):
        model = ELESON(hidden=4)
        with torch.no_grad():
            model.evidence[-2].weight.zero_()
            model.evidence[-2].bias.zero_()
        out = model(torch.randn(2,200,7))
        torch.testing.assert_close(out["uncertainty"], torch.ones(2))
        torch.testing.assert_close(out["confidence"], torch.zeros(2,3))
        self.assertTrue(torch.isfinite(out["probability"]).all())


class MetricTests(unittest.TestCase):
    def test_tie_correct_auc_and_missing_class(self):
        self.assertEqual(binary_auc([0,1],[.5,.5]), .5)
        self.assertEqual(binary_auc([0,1],[.1,.9]), 1.)
        self.assertIsNone(binary_auc([1,1],[.1,.9]))

    def test_ud_remains_false_negative_in_population_f1(self):
        m = classification_metrics([1,1,2],[1,1,2],[.9,.1,.9],.5)
        self.assertAlmostEqual(m["per_class"]["elevator"]["recall"], .5)
        self.assertEqual(m["selective_per_class"]["elevator"]["recall"], 1.)
        self.assertEqual(m["confusion_3x4"][1][3], 1)


if __name__ == "__main__":
    unittest.main()
