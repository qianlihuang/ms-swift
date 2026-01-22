#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import Iterable, List, Optional


def _load_keys_from_index(model_dir: str) -> Optional[List[str]]:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return None
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    weight_map = data.get("weight_map", {})
    return list(weight_map.keys()) if weight_map else None


def _load_keys_from_safetensors(model_dir: str) -> Optional[List[str]]:
    try:
        from safetensors import safe_open
    except Exception:
        return None

    keys: List[str] = []
    for name in os.listdir(model_dir):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(model_dir, name)
        try:
            with safe_open(path, framework="pt", device="cpu") as f:
                keys.extend(list(f.keys()))
        except Exception:
            continue
    return keys or None


def _load_keys_from_transformers(model_dir: str) -> Optional[List[str]]:
    try:
        from transformers import AutoModelForCausalLM
    except Exception:
        return None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype="auto",
            device_map="cpu",
        )
    except Exception:
        return None

    return [k for k, _ in model.named_parameters()]


def _pick_layer_prefix(keys: Iterable[str], layer_idx: int) -> Optional[str]:
    candidates = [
        f"model.layers.{layer_idx}.",
        f"transformer.layers.{layer_idx}.",
        f"gpt_neox.layers.{layer_idx}.",
        f"layers.{layer_idx}.",
    ]
    for c in candidates:
        if any(k.startswith(c) for k in keys):
            return c
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    args = parser.parse_args()

    model_dir = args.model_dir
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise SystemExit(f"config.json not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    num_layers = int(config.get("num_hidden_layers", 0))
    last_layer_idx = max(num_layers - 1, 0)

    print("== Config summary ==")
    print(f"model_type: {config.get('model_type')}")
    print(f"num_hidden_layers: {num_layers}")
    print(f"num_nextn_predict_layers: {config.get('num_nextn_predict_layers')}")
    print(f"last_layer_idx: {last_layer_idx}")

    keys = _load_keys_from_index(model_dir)
    if keys is None:
        keys = _load_keys_from_safetensors(model_dir)
    if keys is None:
        keys = _load_keys_from_transformers(model_dir)

    if not keys:
        print("\n[WARN] Could not read parameter keys. Provide safetensors or install transformers/safetensors.")
        return

    mtp_keys = [k for k in keys if re.search(r"mtp|nextn|predict", k, re.IGNORECASE)]
    layer_prefix = _pick_layer_prefix(keys, last_layer_idx)

    print("\n== Key scan ==")
    print(f"total_keys: {len(keys)}")
    print(f"mtp_like_keys: {len(mtp_keys)}")
    if mtp_keys:
        print("sample_mtp_keys:")
        for k in mtp_keys[:10]:
            print(f"  {k}")

    print("\n== Suggested trainable regex ==")
    if mtp_keys:
        print(r"trainable_parameters_regex=\"mtp|nextn|predict\"")
    else:
        if layer_prefix is None:
            print("[WARN] Could not detect last-layer prefix from keys.")
            print(r"trainable_parameters_regex=\"(layers|model\.layers)\.\d+\.|lm_head|model\.norm|model\.final_layernorm\"")
        else:
            safe_prefix = re.escape(layer_prefix)
            print(
                rf"trainable_parameters_regex=\"{safe_prefix}|lm_head|model\.norm|model\.final_layernorm\""
            )


if __name__ == "__main__":
    main()
