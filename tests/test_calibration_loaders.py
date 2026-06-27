from calibration_loaders import CalibrationDataset, CalibrationExample, IGNORE_INDEX, collate_lm_batch


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(ch) for ch in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return type("Encoding", (), {"input_ids": ids})()


def test_response_only_masks_prompt_tokens():
    tokenizer = ToyTokenizer()
    dataset = CalibrationDataset([CalibrationExample("abc", "de", "abcde")], tokenizer, max_length=10, loss_on="response_only")
    item = dataset[0]
    assert item["input_ids"].tolist() == [97, 98, 99, 100, 101]
    assert item["labels"].tolist() == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 100, 101]


def test_response_only_truncation_preserves_response_labels():
    tokenizer = ToyTokenizer()
    dataset = CalibrationDataset([CalibrationExample("abcd", "ef", "abcdef")], tokenizer, max_length=3, loss_on="response_only")
    item = dataset[0]
    assert item["input_ids"].tolist() == [100, 101, 102]
    assert item["labels"].tolist() == [IGNORE_INDEX, 101, 102]


def test_collate_left_pads_labels_with_ignore_index():
    tokenizer = ToyTokenizer()
    dataset = CalibrationDataset([
        CalibrationExample("a", "b", "ab"),
        CalibrationExample("abc", "d", "abcd"),
    ], tokenizer, max_length=10, loss_on="response_only")
    batch = collate_lm_batch([dataset[0], dataset[1]], tokenizer)
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["labels"][0].tolist()[:2] == [IGNORE_INDEX, IGNORE_INDEX]
    assert batch["attention_mask"][0].tolist()[:2] == [0, 0]
