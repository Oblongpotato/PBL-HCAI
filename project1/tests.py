"""Regression tests for project 1.

Every case in DefectRegressionTests reproduces a defect found in review: each one used to
return HTTP 500 or silently corrupt the dataset. They are the tests that matter most.
"""

import io

from django.test import TestCase, override_settings
from django.urls import reverse

from utils import datasets

from .ml import SCORES
from .models import Dataset, TrainingRun

IRIS = "Resources/iris.csv"


def csv_upload(text, name="data.csv"):
    upload = io.BytesIO(text.encode())
    upload.name = name
    return upload


def iris_upload(name="iris.csv"):
    with open(IRIS, "rb") as handle:
        upload = io.BytesIO(handle.read())
    upload.name = name
    return upload


class UploadMixin:
    def upload(self, upload, task=""):
        return self.client.post(
            reverse("project1:index"), {"file": upload, "task": task}, follow=True
        )

    def train(self, **overrides):
        payload = {
            "model": "tree",
            "values": "",
            "test_size": "0.3",
            "random_state": "42",
            "scoring": "accuracy",
        }
        payload.update(overrides)
        return self.client.post(reverse("project1:train"), payload)


@override_settings(ALLOWED_HOSTS=["testserver"])
class UploadTests(UploadMixin, TestCase):
    def test_iris_is_summarised_and_detected_as_classification(self):
        response = self.upload(iris_upload())
        self.assertContains(response, "150 rows")
        self.assertContains(response, "4 features")
        dataset = Dataset.objects.get()
        self.assertEqual(dataset.task, "classification")
        self.assertEqual(dataset.target, "variety")

    def test_non_csv_extension_is_rejected(self):
        response = self.upload(csv_upload("a,b,label\n1,2,A\n", name="payload.exe"))
        self.assertContains(response, "not allowed")
        self.assertFalse(Dataset.objects.exists())

    def test_malformed_upload_reports_an_error_instead_of_crashing(self):
        upload = io.BytesIO(b"\x89PNG\x00\x01binary")
        upload.name = "image.csv"
        self.assertEqual(self.upload(upload).status_code, 200)
        self.assertFalse(Dataset.objects.exists())


@override_settings(ALLOWED_HOSTS=["testserver"])
class DefectRegressionTests(UploadMixin, TestCase):
    """One test per defect raised in the project 1 review."""

    def test_b1_unique_integer_features_are_not_mistaken_for_identifiers(self):
        # drop_id_columns used to remove every unique integer column, leaving no features.
        self.upload(csv_upload("a,b,label\n1,2,A\n2,3,B\n3,4,A\n4,5,B\n5,6,A\n6,7,B\n"))
        dataset = Dataset.objects.get()
        self.assertEqual(dataset.dropped_columns, [])
        self.assertEqual(dataset.n_features, 2)

        self.assertEqual(self.train().status_code, 200)
        self.assertTrue(TrainingRun.objects.exists())

    def test_b1_named_identifier_columns_are_still_dropped(self):
        rows = "\n".join("{0},{1},{2}".format(i, i % 5, "A" if i % 2 else "B") for i in range(1, 13))
        self.upload(csv_upload("Id,feature,label\n" + rows + "\n"))
        self.assertEqual(Dataset.objects.get().dropped_columns, ["Id"])

    def test_b1_running_index_is_still_dropped(self):
        rows = "\n".join("{0},{1},{2}".format(i, i % 5, "A" if i % 2 else "B") for i in range(1, 13))
        self.upload(csv_upload("serial,feature,label\n" + rows + "\n"))
        self.assertEqual(Dataset.objects.get().dropped_columns, ["serial"])

    def test_b2_categorical_target_cannot_be_forced_to_regression(self):
        response = self.upload(iris_upload(), task="regression")
        self.assertContains(response, "not numeric")
        self.assertFalse(Dataset.objects.exists())

    def test_b2_continuous_target_cannot_be_forced_to_classification(self):
        rows = "\n".join("{0},{1}".format(i, i * 1.37) for i in range(40))
        response = self.upload(csv_upload("a,y\n" + rows + "\n"), task="classification")
        self.assertContains(response, "looks continuous")
        self.assertFalse(Dataset.objects.exists())

    def test_b3_single_class_target_is_rejected(self):
        response = self.upload(csv_upload("a,b,label\n1,2,A\n2,3,A\n3,4,A\n4,5,A\n5,6,A\n"))
        self.assertContains(response, "single class")
        self.assertFalse(Dataset.objects.exists())

    def test_b4_one_row_dataset_is_rejected(self):
        response = self.upload(csv_upload("a,b,label\n1,2,A\n"))
        self.assertContains(response, "complete rows are needed")
        self.assertFalse(Dataset.objects.exists())

    def test_h1_uploads_are_not_reachable_through_the_media_url(self):
        self.upload(iris_upload("private.csv"))
        dataset = Dataset.objects.get()
        self.assertEqual(dataset.name, "private.csv")
        self.assertNotIn("private", dataset.file.name)
        self.assertIn("private_uploads", dataset.file.path)
        for path in ("/media/uploads/private.csv", "/media/" + dataset.file.name):
            self.assertEqual(self.client.get(path).status_code, 404)

    def test_h2_reported_score_is_not_the_selection_score(self):
        self.upload(iris_upload())
        self.train(values="1,2,3,4,5")
        run = TrainingRun.objects.get()
        self.assertNotEqual(run.selection_score, 0.0)
        self.assertNotIn(run.best_score, run.scores)

    def test_m1_root_url_reaches_the_project_hub(self):
        self.assertRedirects(self.client.get("/"), reverse("home:index"))

    def test_m4_overridden_task_drives_the_summary(self):
        rows = "\n".join("{0},{1},{2}".format(i % 7, i % 3, i * 0.5) for i in range(30))
        response = self.upload(csv_upload("a,b,y\n" + rows + "\n"), task="regression")
        self.assertNotContains(response, "<dt>Classes</dt>")

    def test_m6_expensive_forest_sweep_is_rejected(self):
        self.upload(iris_upload())
        response = self.train(model="forest", values="100,200,300,400")
        self.assertContains(response, "Keep the total under")
        self.assertFalse(TrainingRun.objects.exists())


@override_settings(ALLOWED_HOSTS=["testserver"])
class PipelineTests(UploadMixin, TestCase):
    def setUp(self):
        self.upload(iris_upload())

    def test_every_plot_kind_renders(self):
        for kind in ("pair", "target", "distribution", "correlation"):
            response = self.client.post(
                reverse("project1:visualize"),
                {"kind": kind, "x": "petal.length", "y": "petal.width"},
            )
            self.assertContains(response, "p1-" + kind)

    def test_pair_plot_requires_a_second_feature(self):
        response = self.client.post(
            reverse("project1:visualize"), {"kind": "pair", "x": "petal.length", "y": ""}
        )
        self.assertContains(response, "Pick a second feature")

    def test_classification_sweep_reaches_a_usable_score(self):
        self.train()
        self.assertGreaterEqual(TrainingRun.objects.get().best_score, 0.85)

    def test_automl_run_is_recorded_as_automated(self):
        self.client.post(reverse("project1:automl"))
        self.assertTrue(TrainingRun.objects.get().automated)

    def test_regression_dataset_trains_with_a_regression_model(self):
        frame, _ = datasets.drop_id_columns(datasets.read_csv(IRIS))
        reordered = frame[["sepal.length", "sepal.width", "variety", "petal.length", "petal.width"]]
        self.upload(csv_upload(reordered.to_csv(index=False), name="reg.csv"))
        self.assertEqual(Dataset.objects.latest("uploaded_at").task, "regression")
        self.train(model="ridge", scoring="r2")
        self.assertGreater(TrainingRun.objects.latest("created_at").best_score, 0.8)


class HelperTests(TestCase):
    def test_regression_target_is_detected(self):
        frame, _ = datasets.drop_id_columns(datasets.read_csv(IRIS))
        self.assertEqual(datasets.infer_task(frame["petal.width"]), "regression")
        self.assertEqual(datasets.infer_task(frame["variety"]), "classification")

    def test_minimising_scores_pick_the_smallest_value(self):
        self.assertFalse(SCORES["mse"]["maximise"])
        self.assertTrue(SCORES["r2"]["maximise"])
