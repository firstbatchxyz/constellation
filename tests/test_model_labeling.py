import unittest

from constellation.model_labeling import (
    label_sample_with_model,
    select_labels,
    taxonomy_candidates,
)
from constellation.schema import CanonicalSample, CanonicalTurn
from constellation.taxonomy import CapabilityTaxonomy, DomainTaxonomy


class FakeZeroShotClassifier:
    def __call__(
        self,
        text,
        candidate_labels,
        multi_label,
        hypothesis_template,
        batch_size,
    ):
        del text, multi_label, hypothesis_template, batch_size
        labels = list(candidate_labels)
        scores = []
        for label in labels:
            if label.startswith("STRUCTURED_REASONING") or label.startswith("SCIENCE"):
                scores.append(0.91)
            elif label.startswith("DEBUGGING"):
                scores.append(0.66)
            else:
                scores.append(0.05)
        return {"labels": labels, "scores": scores}


def sample():
    return CanonicalSample(
        id="science-1",
        source_dataset="fixture",
        sample_type="reasoning",
        messages=[
            CanonicalTurn(
                role="user",
                type="message",
                content="Explain the chemistry experiment and reason about the evidence.",
            )
        ],
        quality_score=1.0,
    )


class ModelLabelingTests(unittest.TestCase):
    def test_select_labels_applies_threshold_and_limit(self):
        labels = select_labels(
            {"A": 0.2, "B": 0.9, "C": 0.8},
            threshold=0.5,
            max_labels=1,
        )

        self.assertEqual(labels, ["B"])

    def test_taxonomy_candidates_include_descriptions(self):
        candidates = taxonomy_candidates(CapabilityTaxonomy.load())

        self.assertIn("STRUCTURED_REASONING", candidates)
        self.assertIn("Explicit stepwise reasoning", candidates["STRUCTURED_REASONING"])

    def test_label_sample_with_fake_zero_shot_classifier(self):
        labeled = label_sample_with_model(
            sample(),
            classifier=FakeZeroShotClassifier(),
            capability_taxonomy=CapabilityTaxonomy.load(),
            domain_taxonomy=DomainTaxonomy.load(),
            model_name="fake",
            capability_threshold=0.65,
            domain_threshold=0.65,
            max_capabilities=2,
            max_domains=1,
            max_chars=4000,
            batch_size=4,
        )

        self.assertEqual(labeled.capabilities, ["STRUCTURED_REASONING", "DEBUGGING"])
        self.assertEqual(labeled.domains, ["SCIENCE"])
        self.assertEqual(labeled.metadata["capability_labeling"]["method"], "zero_shot_nli_v1")
        self.assertEqual(labeled.metadata["domain_labeling"]["model"], "fake")


if __name__ == "__main__":
    unittest.main()
