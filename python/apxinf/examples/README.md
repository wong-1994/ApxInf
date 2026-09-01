# apxinf examples

Minimal, dependency-light samples that **teach the interface**. Heavy evaluation
harnesses (LIBERO sim rollouts, resumable ledgers, benchmarks) stay in the
repo-level [`scripts/`](../../../scripts) — those are ops/eval tooling, not API
demos.

Each example is self-contained and runs straight from a source checkout: they
share a tiny `sys.path` shim + synthetic-observation builder in `_common.py`, so
no dataset or simulator is needed to see the API work (the actions are
meaningless — the point is the call shape).

| Example | Shows |
|---|---|
| [`pi05policy_infer.py`](pi05policy_infer.py) | Concrete `Pi05Policy.from_pretrained` — model-specific knobs (`action_dim`, `image_keys`). |
| [`autopolicy_infer.py`](autopolicy_infer.py) | Generic `AutoPolicy.from_pretrained` — dispatch by `config.json` model type. |
| [`walloss_policy_infer.py`](walloss_policy_infer.py) | Concrete in-process `WallossPolicy` — two-camera semantics, normalization key, deployable action width, and optional FP8 artifacts. |
| [`walloss_openpi_websocket_server.py`](walloss_openpi_websocket_server.py) | Serve WallOSS through the model-neutral OpenPI-compatible WebSocket transport. |
| [`openpi_server.py`](openpi_server.py) | Serve a policy in-process over OpenPI's websocket protocol (`apxinf.serving`). |
| [`openpi_client.py`](openpi_client.py) | Connect an OpenPI client, read metadata, send one observation, read actions. |

## Dependencies

- The `*_infer.py` examples and servers need the **`apxinf_py` CUDA
  binding** (`maturin develop --features cuda` in [`crates/apxinf-py`](../../../crates/apxinf-py))
  and a compatible checkpoint directory.
- WallOSS examples additionally need `pip install -e "python/apxinf[walloss]"`
  for its Qwen2.5-VL tokenizer/image processor and serialized normalizers.
- `openpi_server.py` also needs the transport deps
  ([`scripts/requirements-pi05-websocket.txt`](../../../scripts/requirements-pi05-websocket.txt)).
- `openpi_client.py` needs the upstream `openpi_client` package and a running
  server; it needs **no** CUDA or checkpoint of its own.

## Quick start

```sh
# Terminal 1 — serve a checkpoint
python examples/openpi_server.py --model-dir /path/to/checkpoint --precision bf16

# Terminal 2 — one round trip against it
python examples/openpi_client.py --host 127.0.0.1 --port 8000

# Or skip the network and call the policy in-process
python examples/pi05policy_infer.py --model-dir /path/to/checkpoint
python examples/autopolicy_infer.py --model-dir /path/to/checkpoint
python examples/walloss_policy_infer.py \
  --model-dir /path/to/wall-oss-0.5 --action-dim 7
```

`autopolicy_infer.py` keeps the checkpoint's full action width unless
`--action-dim` is explicit.  A width such as `7` is a robot/checkpoint convention,
not a generic model default.  WallOSS internally predicts 26 channels; trimming
is valid only when the chosen normalizer and the checkpoint's leading-channel
layout match the target robot.  Non-prefix layouts need a robot adapter rather
than a different `--action-dim`.
