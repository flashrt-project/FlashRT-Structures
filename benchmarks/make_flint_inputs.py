#!/usr/bin/env python3
"""Reconstruct the public llama.cpp code fixtures with the official Flint tokenizer."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-dir", required=True)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    recipe = json.loads(Path(__file__).with_name("flint_fixtures.json").read_text())
    from flash_rt.models.flint.gguf import GGUFFile
    from flash_rt.models.flint.tokenizer import FlintTokenizer

    output = []
    with GGUFFile(args.gguf) as gguf:
        tokenizer = FlintTokenizer(gguf)
        for record in recipe["records"]:
            messages = [dict(m) for m in record["messages"]]
            for source in record["sources"]:
                data = subprocess.check_output(
                    [
                        "git",
                        "-C",
                        args.llama_dir,
                        "show",
                        recipe["llama_revision"] + ":" + source["path"],
                    ]
                )
                if hashlib.sha256(data).hexdigest() != source["file_sha256"]:
                    raise ValueError("public corpus digest mismatch: " + source["path"])
                snippet = data.decode("utf-8")[: source["prefix_characters"]]
                for message in messages:
                    message["content"] = message["content"].replace(
                        source["marker"], snippet
                    )
            text = tokenizer.apply_chat_template(messages, tools=record["tools"])
            ids = tokenizer.encode(text)
            if len(ids) != record["expected_tokens"]:
                raise ValueError(
                    f"{record['id']}: tokenizer/checkpoint changed ({len(ids)} tokens)"
                )
            digest = hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest()
            if digest != record["expected_sha256"]:
                raise ValueError(
                    "tokenizer/checkpoint digest mismatch: " + record["id"]
                )
            output.append({"id": record["id"], "tokens": ids, "sha256": digest})
    Path(args.out).write_text(json.dumps(output))


if __name__ == "__main__":
    main()
