import unittest

from readmission.data.preprocess import aggregate_patient_sequences, fit_vocab, strict_forward_labels, train_visit_corpus
from readmission.data.schema import PAD_INDEX, PAD_TOKEN, UNK_INDEX, UNK_TOKEN


class PreprocessTests(unittest.TestCase):
    def test_visits_are_sorted_and_aggregated_by_day(self):
        rows = [
            {"PID": "1", "DAY_ID": "5", "DX_GROUP_DESCRIPTION": "B", "SERVICE_LOCATION": "OFFICE"},
            {"PID": "1", "DAY_ID": "1", "DX_GROUP_DESCRIPTION": "A", "SERVICE_LOCATION": "OFFICE"},
            {"PID": "1", "DAY_ID": "1", "DX_GROUP_DESCRIPTION": "B", "SERVICE_LOCATION": "OFFICE"},
        ]
        vocab = fit_vocab(rows, {"1"})
        patient = aggregate_patient_sequences(rows, ["1"], vocab)[0]
        self.assertEqual(patient.days, (1, 5))
        self.assertEqual(patient.visits[0].codes, tuple(sorted([vocab["A"], vocab["B"]])))

    def test_pad_and_unk_are_reserved(self):
        rows = [{"PID": "1", "DAY_ID": "1", "DX_GROUP_DESCRIPTION": "A", "SERVICE_LOCATION": "OFFICE"}]
        vocab = fit_vocab(rows, {"1"})
        self.assertEqual(vocab[PAD_TOKEN], PAD_INDEX)
        self.assertEqual(vocab[UNK_TOKEN], UNK_INDEX)
        self.assertGreaterEqual(vocab["A"], 2)

    def test_strict_forward_label_excludes_same_day_inpatient(self):
        labels = strict_forward_labels([10, 20], [10, 25, 50], window=30)
        self.assertEqual(labels, [1, 1])
        self.assertEqual(strict_forward_labels([10], [10], window=30), [0])

    def test_unseen_codes_map_to_unk(self):
        train_rows = [{"PID": "1", "DAY_ID": "1", "DX_GROUP_DESCRIPTION": "TRAIN", "SERVICE_LOCATION": "OFFICE"}]
        valid_rows = [{"PID": "2", "DAY_ID": "1", "DX_GROUP_DESCRIPTION": "VALID_ONLY", "SERVICE_LOCATION": "OFFICE"}]
        vocab = fit_vocab(train_rows + valid_rows, {"1"})
        patient = aggregate_patient_sequences(train_rows + valid_rows, ["2"], vocab)[0]
        self.assertEqual(patient.visits[0].codes, (UNK_INDEX,))

    def test_train_corpus_uses_train_sequences_only(self):
        rows = [
            {"PID": "1", "DAY_ID": "1", "DX_GROUP_DESCRIPTION": "TRAIN", "SERVICE_LOCATION": "OFFICE"},
            {"PID": "2", "DAY_ID": "1", "DX_GROUP_DESCRIPTION": "VALID_ONLY", "SERVICE_LOCATION": "OFFICE"},
        ]
        vocab = fit_vocab(rows, {"1"})
        index_to_token = {idx: token for token, idx in vocab.items()}
        train_sequences = aggregate_patient_sequences(rows, ["1"], vocab)
        corpus = train_visit_corpus(train_sequences, index_to_token)
        self.assertEqual(corpus, [["TRAIN"]])


if __name__ == "__main__":
    unittest.main()
