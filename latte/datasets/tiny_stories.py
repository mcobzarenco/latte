import os
from pathlib import Path
from typing import Literal

import numpy as np
import tiktoken
from datasets import DatasetDict, load_dataset
from tqdm import tqdm

DATASETS_ROOT = Path("datasets")
GPT2_TOKENIZER = tiktoken.get_encoding("gpt2")


def tokenize(example: dict[Literal["text"], str]) -> dict[str, list[int] | int]:
    ids = [GPT2_TOKENIZER.eot_token]
    ids.extend(GPT2_TOKENIZER.encode_ordinary(example["text"]))
    return {"ids": ids, "len": len(ids)}


def prepare() -> None:
    dataset = load_dataset("roneneldan/TinyStories")
    assert isinstance(dataset, DatasetDict)

    # rename "validation" -> "val"
    dataset["val"] = dataset["validation"]
    del dataset["validation"]
    assert dataset.keys() == {"train", "val"}

    tokenized = dataset.map(
        tokenize,
        remove_columns=["text"],
        desc="Tokenizing Tiny Stories dataset",
        num_proc=os.cpu_count(),
    )

    dataset_path = DATASETS_ROOT / "tiny-stories"
    dataset_path.mkdir(parents=True, exist_ok=True)

    for split, tokens in tokenized.items():
        filename = dataset_path / f"{split}.bin"

        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        array_len = np.sum(tokens["len"], dtype=np.uint64)
        array = np.memmap(filename, dtype=dtype, mode="w+", shape=(array_len,))

        total_batches, index = 1024, 0
        for batch_index in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = tokens.shard(
                num_shards=total_batches,
                index=batch_index,
                contiguous=True,
            ).with_format("numpy")
            array_batch = np.concatenate(batch["ids"])
            # Write into mmap
            array[index : index + len(array_batch)] = array_batch
            index += len(array_batch)

        array.flush()
