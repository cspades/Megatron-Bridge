# Nemotron4 HF ↔ MegatronMIMO conversion

Nemotron4 reuses the generic MIMO converter and the standard Nemotron4 weight
mappings. It builds two components: `language` (the hybrid decoder, including
configured MTP layers) and `images` (Pixtral vision encoder plus MLP projector).
The encoder and projector have separate conversion routes sharing the image
component's process groups. Checkpoints use the Bridge MIMO provider and its
checkpoint save/load helpers.

Use the Nemotron4 Megatron-LM fork pinned by this branch. The preview's HF
implementation requires Transformers >= 5.17 and authenticated access to
`nvidia/nemotron4-nano-preview`. Run inside the development container with this
checkout and its MCore submodule on `PYTHONPATH`.

## Import and export

The following layout uses one GPU for language and one for images. The preview
contains roughly 100B total parameters, so the language GPU needs enough memory
for its BF16 weights plus conversion overhead.

```bash
uv run python -m torch.distributed.run --standalone --nproc_per_node=2 \
  examples/conversion/convert_megatron_mimo.py import \
  --hf-model nvidia/nemotron4-nano-preview \
  --megatron-path /tmp/nemotron4-mimo \
  --component language=tp=1 --component images=tp=1 \
  --trust-remote-code --torch-dtype bfloat16

uv run python -m torch.distributed.run --standalone --nproc_per_node=2 \
  examples/conversion/convert_megatron_mimo.py export \
  --hf-model nvidia/nemotron4-nano-preview \
  --megatron-path /tmp/nemotron4-mimo --hf-path /tmp/nemotron4-hf-export \
  --component language=tp=1 --component images=tp=1 \
  --trust-remote-code --torch-dtype bfloat16
```

## Constraints

- Language TP=PP=ETP=1; image TP=PP=CP=EP=ETP=1. Unsupported layouts fail before
  model construction. Language experts use the shared fused-expert conversion.
- MTP is preserved when present in the HF config, including repeated MTP layers.
- Residual controllers, GDP `A_log`, and router correction biases stay FP32
  while the decoder's ordinary weights use BF16.
- Quantile-balancing bin bounds have no HF counterpart and are not preserved
  through HF export/import. Router weights and expert correction biases are preserved.
- Conversion covers model weights, not optimizer/RNG state or training resume.
- These commands produce Bridge MIMO checkpoints. Loading them directly with the
  fork's separate `examples/mimo/pretrain_mimo.py` launcher is not established;
  its modality names and provider metadata differ.
- This example does not establish training, video/audio, or quantized conversion
  support.

For a four-GPU export with two language expert ranks and two image replicas,
use `--nproc_per_node=4 --component language=tp=1,ep=2 --component images=tp=1`.
The shared checkpoint loader reshares the saved model to the requested layout.

## Focused validation

The opt-in test creates a small HF model from the preview's actual remote code,
covering all four decoder block types, the vision tower, the projector, and MTP
both enabled and disabled. It checks every weight exactly before and after a
MIMO distributed-checkpoint save/reload and strict HF export.

```bash
NEMOTRON4_HF_MODEL=/path/to/local/preview-snapshot \
CUDA_VISIBLE_DEVICES=0,1 uv run python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m pytest \
  tests/functional_tests/test_groups/megatron_mimo/test_nemotron4_conversion.py -v
```
