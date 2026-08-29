#!/usr/bin/env python3
"""Convert a trusted PKUSEG Python model into the PKUSEG-js binary format."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import struct
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


MAGIC = b"PKUSEGJS"
FORMAT_VERSION = 1
HEADER_BYTES = 64
UINT32_MASK = 0xFFFF_FFFF
MAXIMUM_LOAD_FACTOR = 0.62
HASH_VECTOR_TEXTS = ["$$", "c.我", "**Num", "自然语言处理", "😀"]


def rotate_left_32(value: int, bits: int) -> int:
    return ((value << bits) | (value >> (32 - bits))) & UINT32_MASK


def hash_pair(text: str) -> tuple[int, int]:
    """Hash Unicode code points exactly like src/hash.js."""

    first = 0x811C_9DC5
    second = 0x9747_B28C
    length = 0
    for character in text:
        code_point = ord(character)
        first = ((first ^ code_point) * 0x0100_0193) & UINT32_MASK

        mixed = (code_point * 0xCC9E_2D51) & UINT32_MASK
        mixed = rotate_left_32(mixed, 15)
        mixed = (mixed * 0x1B87_3593) & UINT32_MASK
        second ^= mixed
        second = rotate_left_32(second, 13)
        second = (second * 5 + 0xE654_6B64) & UINT32_MASK
        length += 1

    second ^= length
    second ^= second >> 16
    second = (second * 0x85EB_CA6B) & UINT32_MASK
    second ^= second >> 13
    second = (second * 0xC2B2_AE35) & UINT32_MASK
    second ^= second >> 16
    return first, second & UINT32_MASK


def aligned(value: int, alignment: int = 8) -> int:
    return math.ceil(value / alignment) * alignment


def hash_table_size(item_count: int) -> int:
    if item_count == 0:
        return 0
    minimum = max(2, math.ceil(item_count / MAXIMUM_LOAD_FACTOR))
    return 1 << (minimum - 1).bit_length()


def binary_layout(
    feature_table_size: int,
    unigram_table_size: int,
    dictionary_table_size: int,
    weight_count: int,
) -> tuple[dict[str, tuple[int, int, str]], int]:
    offset = HEADER_BYTES
    layout: dict[str, tuple[int, int, str]] = {}

    def add(name: str, length: int, dtype: str) -> None:
        nonlocal offset
        item_size = np.dtype(dtype).itemsize
        layout[name] = (offset, length, dtype)
        offset += length * item_size

    add("feature_first", feature_table_size, "<u4")
    add("feature_second", feature_table_size, "<u4")
    add("feature_identifiers", feature_table_size, "<u4")
    add("unigram_first", unigram_table_size, "<u4")
    add("unigram_second", unigram_table_size, "<u4")
    add("unigram_occupied", unigram_table_size, "u1")
    offset = aligned(offset)
    add("dictionary_first", dictionary_table_size, "<u4")
    add("dictionary_second", dictionary_table_size, "<u4")
    add("dictionary_occupied", dictionary_table_size, "u1")
    offset = aligned(offset)
    add("weights", weight_count, "<f8")
    return layout, offset


def mapped_array(
    mapped_file: np.memmap,
    layout: dict[str, tuple[int, int, str]],
    name: str,
) -> np.ndarray:
    offset, length, dtype = layout[name]
    return np.ndarray(
        shape=(length,),
        dtype=np.dtype(dtype),
        buffer=mapped_file,
        offset=offset,
    )


def insert_feature_mapping(
    mapping: dict[str, int],
    first_hashes: np.ndarray,
    second_hashes: np.ndarray,
    identifiers: np.ndarray,
) -> int:
    mask = len(identifiers) - 1
    maximum_probe = 0
    for position, (feature, identifier) in enumerate(mapping.items(), start=1):
        first, second = hash_pair(feature)
        slot = first & mask
        probes = 0
        while identifiers[slot] != 0:
            if first_hashes[slot] == first and second_hashes[slot] == second:
                raise RuntimeError(
                    "Dual-hash collision while exporting features: "
                    f"{feature!r} collides with feature id {identifiers[slot] - 1}"
                )
            slot = (slot + 1) & mask
            probes += 1
            if probes == len(identifiers):
                raise RuntimeError("Feature hash table is full")
        first_hashes[slot] = first
        second_hashes[slot] = second
        identifiers[slot] = identifier + 1
        maximum_probe = max(maximum_probe, probes)
        if position % 250_000 == 0:
            print(f"  indexed {position:,}/{len(mapping):,} features")
    return maximum_probe


def insert_membership(
    values: Iterable[str],
    value_count: int,
    first_hashes: np.ndarray,
    second_hashes: np.ndarray,
    occupied: np.ndarray,
    label: str,
) -> int:
    if value_count == 0:
        return 0
    mask = len(occupied) - 1
    maximum_probe = 0
    for position, value in enumerate(values, start=1):
        first, second = hash_pair(value)
        slot = first & mask
        probes = 0
        while occupied[slot] != 0:
            if first_hashes[slot] == first and second_hashes[slot] == second:
                raise RuntimeError(
                    f"Duplicate value or dual-hash collision in {label}: {value!r}"
                )
            slot = (slot + 1) & mask
            probes += 1
            if probes == len(occupied):
                raise RuntimeError(f"{label} hash table is full")
        first_hashes[slot] = first
        second_hashes[slot] = second
        occupied[slot] = 1
        maximum_probe = max(maximum_probe, probes)
        if position % 250_000 == 0:
            print(f"  indexed {position:,}/{value_count:,} {label} entries")
    return maximum_probe


def load_dictionary(path: Path) -> Iterator[str]:
    if path.suffix.lower() == ".pkl":
        with path.open("rb") as stream:
            value = pickle.load(stream)
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            yield from value.strip().split("\n")
            return
        if isinstance(value, (list, tuple, set)):
            yield from (str(item).strip() for item in value)
            return
        raise TypeError(f"Unsupported pickled dictionary type: {type(value)!r}")

    yield from path.read_text(encoding="utf-8").splitlines()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dictionary",
        type=Path,
        action="append",
        default=[],
        help="Optional pickle or UTF-8 dictionary; may be repeated",
    )
    parser.add_argument(
        "--task",
        choices=("auto", "segmentation", "postag"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    model_directory = arguments.model_dir.resolve()
    output_directory = arguments.output.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    print(f"Loading Python model from {model_directory}")
    with (model_directory / "features.pkl").open("rb") as stream:
        feature_data = pickle.load(stream)
    npz = np.load(model_directory / "weights.npz")
    sizes = np.asarray(npz["sizes"], dtype=np.int64)
    weights = np.asarray(npz["w"], dtype="<f8")
    n_tag = int(sizes[0])
    n_feature = int(sizes[1])

    feature_mapping = feature_data["feature_to_idx"]
    tag_mapping = feature_data["tag_to_idx"]
    inferred_task = "segmentation" if "unigram" in feature_data else "postag"
    task = inferred_task if arguments.task == "auto" else arguments.task
    if task != inferred_task:
        raise ValueError(
            f"Requested task {task!r} does not match model structure {inferred_task!r}"
        )
    if len(feature_mapping) != n_feature:
        raise ValueError(
            f"Feature count mismatch: {len(feature_mapping)} != {n_feature}"
        )
    if feature_mapping.get("$$") != 0:
        raise ValueError("PKUSEG-js requires the '$$' feature to have id 0")
    identifiers = sorted(feature_mapping.values())
    if identifiers != list(range(n_feature)):
        raise ValueError("Feature identifiers are not dense from zero")
    if weights.shape != (n_tag * (n_feature + n_tag),):
        raise ValueError("Unexpected PKUSEG weight shape")

    unigram_values = list(feature_data.get("unigram", []))
    dictionary_values: set[str] = set()
    for dictionary_path in arguments.dictionary:
        dictionary_values.update(
            value for value in load_dictionary(dictionary_path.resolve()) if value
        )

    feature_table_size = hash_table_size(len(feature_mapping))
    unigram_table_size = hash_table_size(len(unigram_values))
    dictionary_table_size = hash_table_size(len(dictionary_values))
    layout, total_bytes = binary_layout(
        feature_table_size,
        unigram_table_size,
        dictionary_table_size,
        len(weights),
    )

    binary_path = output_directory / "model.bin"
    print(
        "Writing compact model: "
        f"features={len(feature_mapping):,}, "
        f"unigrams={len(unigram_values):,}, "
        f"dictionary={len(dictionary_values):,}, "
        f"bytes={total_bytes:,}"
    )
    with binary_path.open("wb") as stream:
        stream.truncate(total_bytes)
    mapped_file = np.memmap(binary_path, dtype="u1", mode="r+", shape=(total_bytes,))

    header = bytearray(HEADER_BYTES)
    struct.pack_into(
        "<8s9I",
        header,
        0,
        MAGIC,
        FORMAT_VERSION,
        HEADER_BYTES,
        feature_table_size,
        unigram_table_size,
        dictionary_table_size,
        len(weights),
        n_feature,
        n_tag,
        1 if dictionary_values else 0,
    )
    mapped_file[:HEADER_BYTES] = np.frombuffer(header, dtype="u1")

    feature_probe = insert_feature_mapping(
        feature_mapping,
        mapped_array(mapped_file, layout, "feature_first"),
        mapped_array(mapped_file, layout, "feature_second"),
        mapped_array(mapped_file, layout, "feature_identifiers"),
    )
    unigram_probe = insert_membership(
        unigram_values,
        len(unigram_values),
        mapped_array(mapped_file, layout, "unigram_first"),
        mapped_array(mapped_file, layout, "unigram_second"),
        mapped_array(mapped_file, layout, "unigram_occupied"),
        "unigram",
    )
    dictionary_probe = insert_membership(
        sorted(dictionary_values),
        len(dictionary_values),
        mapped_array(mapped_file, layout, "dictionary_first"),
        mapped_array(mapped_file, layout, "dictionary_second"),
        mapped_array(mapped_file, layout, "dictionary_occupied"),
        "dictionary",
    )
    mapped_array(mapped_file, layout, "weights")[:] = weights
    mapped_file.flush()
    del mapped_file

    tags = [None] * n_tag
    for tag, identifier in tag_mapping.items():
        tags[identifier] = tag
    if any(tag is None for tag in tags):
        raise ValueError("Tag identifiers are not dense from zero")

    metadata = {
        "format": "pkuseg-js-model",
        "version": FORMAT_VERSION,
        "binary": binary_path.name,
        "sha256": sha256_file(binary_path),
        "task": task,
        "nFeature": n_feature,
        "nTag": n_tag,
        "tags": tags,
        "wordFeature": task == "segmentation",
        "wordMinimum": 2,
        "wordMaximum": 6,
        "featureCount": len(feature_mapping),
        "unigramCount": len(unigram_values),
        "dictionaryCount": len(dictionary_values),
        "hashAlgorithm": "pkuseg-codepoint-dual32-v1",
        "hashVectors": [
            {"text": text, "hash": list(hash_pair(text))}
            for text in HASH_VECTOR_TEXTS
        ],
        "maximumProbe": {
            "feature": feature_probe,
            "unigram": unigram_probe,
            "dictionary": dictionary_probe,
        },
        "sourceModel": model_directory.name,
    }
    (output_directory / "model.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Export complete: {binary_path}")
    print(f"SHA-256: {metadata['sha256']}")
    print(f"Maximum probes: {metadata['maximumProbe']}")


if __name__ == "__main__":
    main()
