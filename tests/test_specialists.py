import unittest

from constellation.specialists import load_specialist_targets


class SpecialistTargetTests(unittest.TestCase):
    def test_specialist_registry_has_at_most_twenty_targets(self):
        targets = load_specialist_targets()

        self.assertEqual(len(targets), 20)
        self.assertIn("science_reasoner", {target.id for target in targets})
        self.assertIn("writing_reviser", {target.id for target in targets})
        self.assertIn("humanities_analyst", {target.id for target in targets})
        self.assertLessEqual(len(targets), 20)


if __name__ == "__main__":
    unittest.main()
