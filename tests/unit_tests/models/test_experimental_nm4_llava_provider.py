# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for the experimental NM4 LLaVA provider."""

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from megatron.core.context_parallel import ContextParallelBatch
from megatron.core.models.multimodal import context_parallel as vision_cp
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.bridge.models import experimental_nm4_llava_provider as nm4
from megatron.bridge.models.experimental_nm4_llava_provider import ExperimentalNM4LlavaProvider, NM4LlavaModel


def _provider_with_recompute(**overrides):
    provider = object.__new__(ExperimentalNM4LlavaProvider)
    values = {
        "recompute_vision": False,
        "vision_recompute_granularity": None,
        "vision_recompute_method": None,
        "vision_recompute_num_layers": None,
        **overrides,
    }
    field_names = {field.name for field in fields(ExperimentalNM4LlavaProvider)}
    for key, value in values.items():
        if key not in field_names:
            raise ValueError(f"Unknown provider field: {key}")
        setattr(provider, key, value)
    return provider


def test_provider_exposes_vision_recompute_fields():
    field_names = {field.name for field in fields(ExperimentalNM4LlavaProvider)}

    assert {
        "recompute_vision",
        "vision_recompute_granularity",
        "vision_recompute_method",
        "vision_recompute_num_layers",
    } <= field_names


def test_provider_applies_requested_vision_recompute():
    provider = _provider_with_recompute(
        recompute_vision=True,
        vision_recompute_granularity="full",
        vision_recompute_method="block",
        vision_recompute_num_layers=30,
    )
    vision_config = SimpleNamespace()

    provider._apply_vision_recompute_settings(vision_config)

    assert vision_config.recompute_granularity == "full"
    assert vision_config.recompute_method == "block"
    assert vision_config.recompute_num_layers == 30


def test_provider_rejects_incomplete_vision_recompute():
    provider = _provider_with_recompute(recompute_vision=True)

    with pytest.raises(ValueError, match="Vision recompute requires"):
        provider._apply_vision_recompute_settings(SimpleNamespace())


@pytest.mark.unit
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("vision_dp_over_cp", [False, True])
@pytest.mark.parametrize("gtp_size", [1, 2])
def test_nm4_constructor_keeps_vision_cp_local(cp_size, vision_dp_over_cp, gtp_size):
    language_cp = MagicMock()
    language_cp.size.return_value = cp_size
    groups = SimpleNamespace(cp=language_cp, tp=MagicMock(), tp_cp=MagicMock(), hcp=[MagicMock()])
    groups.gtp_remat = MagicMock()
    groups.gtp_remat.size.return_value = gtp_size
    groups.expt_gtp_remat = MagicMock()
    original_gtp = groups.gtp_remat
    original_expt_gtp = groups.expt_gtp_remat
    original_tp_cp, original_hcp = groups.tp_cp, groups.hcp
    config = SimpleNamespace(sequence_parallel=False, context_parallel_size=cp_size, logit_dtype=torch.float32)
    with (
        patch.object(nm4, "HybridModel", return_value=torch.nn.Identity()) as language,
        patch.object(nm4, "ViTModel", return_value=torch.nn.Identity()) as vision,
        patch.object(nm4, "MultimodalProjector", return_value=torch.nn.Identity()) as projector,
        patch.object(nm4, "logit_dtype_kwarg", return_value={}),
        patch.object(torch.distributed, "get_rank", return_value=3),
        patch.object(torch.distributed, "new_group") as new_group,
    ):
        model = NM4LlavaModel(
            language_transformer_config=config,
            language_transformer_layer_spec=None,
            language_vocab_size=16,
            language_max_sequence_length=16,
            vision_transformer_config=SimpleNamespace(hidden_size=8),
            vision_transformer_layer_spec=None,
            vision_projection_config=None,
            vision_projection_layer_spec=None,
            image_token_index=1,
            img_h=8,
            img_w=8,
            patch_dim=2,
            spatial_merge_size=2,
            hybrid_layer_pattern=None,
            parallel_output=True,
            share_embeddings_and_output_weights=False,
            language_position_embedding_type="none",
            pre_process=True,
            post_process=True,
            add_encoder=True,
            add_decoder=True,
            pg_collection=groups,
            vision_dp_over_cp=vision_dp_over_cp,
        )
        vision_groups = vision.call_args.kwargs["pg_collection"]
        assert vision_groups is not groups
        assert vision_groups.tp is groups.tp
        assert language.call_args.kwargs["pg_collection"] is groups
        assert projector.call_args.kwargs["pg_collection"] is vision_groups
        assert vision_groups.gtp_remat is None
        assert vision_groups.expt_gtp_remat is None
        assert groups.gtp_remat is original_gtp
        assert groups.expt_gtp_remat is original_expt_gtp
        assert model.pg_collection is groups
        assert groups.cp is language_cp
        assert groups.tp_cp is original_tp_cp
        assert groups.hcp is original_hcp
        assert model.vision_dp_over_cp == (vision_dp_over_cp and cp_size > 1)
        if cp_size > 1:
            new_group.assert_called_once_with(ranks=[3], use_local_synchronization=True)
            assert vision_groups.cp is new_group.return_value
            assert vision_groups.tp_cp is groups.tp
            assert vision_groups.hcp is None
        else:
            new_group.assert_not_called()
            assert vision_groups.cp is language_cp


class _MergedVision(torch.nn.Module):
    """Small differentiable tower that enforces the NM4 merger's grid constraint."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.patch_counts = []

    def forward(self, images, *, imgs_sizes, packed_seq_params):
        self.patch_counts.append(images.shape[1])
        assert torch.all(imgs_sizes % 2 == 0)
        assert packed_seq_params.cu_seqlens_q[-1] == images.shape[1]
        return images.reshape(1, -1, 4, images.shape[-1]).mean(dim=2) * self.scale


def _vision_model(cp_size):
    model = object.__new__(nm4.NM4LlavaModel)
    torch.nn.Module.__init__(model)
    model.patch_dim = 1
    model.spatial_merge_size = 2
    model.context_parallel_lm = cp_size
    model.vision_dp_over_cp = cp_size > 1
    model.pg_collection = SimpleNamespace(cp=SimpleNamespace(size=lambda: cp_size))
    model.vision_model = _MergedVision()
    model.vision_projection = torch.nn.Linear(3, 2, bias=False)
    return model


@pytest.mark.unit
@pytest.mark.parametrize("cp_size,image_count", [(2, 1), (2, 2), (2, 3), (4, 1), (4, 7)])
def test_vision_cp_splits_before_encoding_and_projects_before_gather(monkeypatch, cp_size, image_count):
    model = _vision_model(cp_size)
    sizes = torch.tensor([[2, 2 + 2 * (index % 3)] for index in range(image_count)])
    total_patches = int(sizes.prod(dim=1).sum())
    images = torch.arange(total_patches * 3, dtype=torch.float32).reshape(1, -1, 3)
    monkeypatch.setattr(vision_cp, "get_context_parallel_world_size", lambda: cp_size)
    gathered = []

    def capture_gather(projected, padded):
        assert projected.shape[1] == 2  # Already projected, rather than encoder hidden width.
        assert padded == max(0, cp_size - image_count)
        gathered.append(projected)
        return projected

    monkeypatch.setattr(nm4, "gather_from_context_parallel_ranks_dynamic_res", capture_gather)
    for rank in range(cp_size):
        monkeypatch.setattr(vision_cp, "get_context_parallel_rank", lambda: rank)
        monkeypatch.setattr(nm4.parallel_state, "get_context_parallel_rank", lambda: rank)
        model._encode_images(images, sizes, None)

    padded = max(0, cp_size - image_count)
    real_shards = gathered[: cp_size - padded]
    sharded = torch.cat(real_shards)
    sharded.sum().backward()
    vision_grad = model.vision_model.scale.grad.clone()
    projection_grad = model.vision_projection.weight.grad.clone()
    model.zero_grad()
    model.vision_dp_over_cp = False
    unsharded = model._encode_images(images, sizes, None)
    unsharded.sum().backward()
    torch.testing.assert_close(sharded, unsharded)
    torch.testing.assert_close(vision_grad, model.vision_model.scale.grad)
    torch.testing.assert_close(projection_grad, model.vision_projection.weight.grad)
    assert sum(model.vision_model.patch_counts[: cp_size - padded]) == images.shape[1]
    assert model.vision_model.patch_counts[cp_size - padded : cp_size] == [4] * padded


@pytest.mark.unit
@pytest.mark.parametrize("cp_size,enabled", [(1, True), (2, False)])
def test_vision_cp_bypass_does_not_call_collectives(monkeypatch, cp_size, enabled):
    model = _vision_model(cp_size)
    model.vision_dp_over_cp = enabled and cp_size > 1

    def unexpected(*args, **kwargs):
        pytest.fail("Unsharded vision must not split or gather")

    monkeypatch.setattr(nm4, "split_to_context_parallel_ranks_dynamic_res", unexpected)
    monkeypatch.setattr(nm4, "gather_from_context_parallel_ranks_dynamic_res", unexpected)
    assert model._encode_images(torch.ones(1, 4, 3), torch.tensor([[2, 2]]), None).shape == (1, 2)


@pytest.mark.unit
def test_vision_cp_rejects_mismatched_process_group():
    model = _vision_model(2)
    model.pg_collection.cp = SimpleNamespace(size=lambda: 4)
    with pytest.raises(ValueError, match="does not match its process group"):
        model._encode_images(torch.ones(1, 4, 3), torch.tensor([[2, 2]]), None)


@pytest.mark.unit
def test_provider_enables_vision_dp_over_cp_by_default():
    defaults = {field.name: field.default for field in fields(ExperimentalNM4LlavaProvider)}
    assert defaults["vision_dp_over_cp"] is True


def _distributed_vision_cp_worker(rank, cp_size, init_method, backend="gloo"):
    from datetime import timedelta
    from unittest.mock import patch

    import torch.distributed as dist
    import torch.distributed.nn.functional  # Register the differentiable collective helpers.

    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        backend, rank=rank, world_size=cp_size, init_method=init_method, timeout=timedelta(seconds=60)
    )
    try:
        with (
            patch.object(vision_cp, "get_context_parallel_world_size", return_value=cp_size),
            patch.object(vision_cp, "get_context_parallel_rank", return_value=rank),
            patch.object(vision_cp, "get_context_parallel_group", return_value=dist.group.WORLD),
            patch.object(nm4.parallel_state, "get_context_parallel_rank", return_value=rank),
        ):
            # Gloo's all-to-all backward requires equal tensor sizes. Uneven
            # image counts and resolutions are covered by the split test above.
            counts = (1, cp_size, 2 * cp_size) if backend == "gloo" else (1, cp_size, cp_size + 1)
            for image_count in counts:
                torch.manual_seed(123)
                model = _vision_model(cp_size)
                if backend == "nccl":
                    model = model.cuda()
                    sizes = torch.tensor([[2, 2 + 2 * (index % 3)] for index in range(image_count)])
                else:
                    sizes = torch.full((image_count, 2), 2)
                total_patches = int(sizes.prod(dim=1).sum())
                images = torch.arange(
                    total_patches * 3, dtype=torch.float32, device=model.vision_model.scale.device
                ).reshape(1, -1, 3)
                sharded = model._encode_images(images, sizes, None)
                sharded.sum().backward()
                grads = []
                for parameter in model.parameters():
                    grad = parameter.grad.clone()
                    dist.all_reduce(grad)
                    grads.append(grad / cp_size)  # Every CP rank computed the full output loss.
                model.zero_grad()
                model.vision_dp_over_cp = False
                unsharded = model._encode_images(images, sizes, None)
                unsharded.sum().backward()
                torch.testing.assert_close(sharded, unsharded)
                for grad, parameter in zip(grads, model.parameters()):
                    torch.testing.assert_close(grad, parameter.grad)
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
@pytest.mark.parametrize("cp_size", [2, 4])
def test_vision_cp_real_gather_matches_features_and_gradients(tmp_path, cp_size):
    if not torch.distributed.is_gloo_available():
        pytest.skip("requires the Gloo backend")
    torch.multiprocessing.spawn(
        _distributed_vision_cp_worker,
        args=(cp_size, (tmp_path / "cp-init").as_uri()),
        nprocs=cp_size,
        join=True,
    )


@pytest.mark.gpu
def test_vision_cp_nccl_handles_uneven_features_and_gradients(tmp_path):
    if torch.cuda.device_count() < 2 or not torch.distributed.is_nccl_available():
        pytest.skip("requires two GPUs and NCCL")
    torch.multiprocessing.spawn(
        _distributed_vision_cp_worker,
        args=(2, (tmp_path / "cp-init").as_uri(), "nccl"),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize(
    ("linear_layout", "attention_layout"),
    [("zigzag", "zigzag"), ("contiguous", "zigzag")],
)
@pytest.mark.parametrize("post_process", [False, True])
def test_nm4_cp_padding_mask_uses_attention_layout(
    linear_layout: str, attention_layout: str, post_process: bool
) -> None:
    model = object.__new__(NM4LlavaModel)
    torch.nn.Module.__init__(model)
    model.pre_process = False
    model.context_parallel_lm = 2
    model.post_process = post_process
    model.sequence_parallel_lm = False
    model.config = SimpleNamespace(
        linear_cp_layout=linear_layout,
        attention_cp_layout=attention_layout,
        mtp_num_layers=0,
    )
    model.pg_collection = SimpleNamespace(
        cp=MagicMock(),
        tp=MagicMock(),
        tp_cp=MagicMock(),
    )
    model.pg_collection.cp.size.return_value = 2
    local_output = torch.arange(12).view(1, 4, 3).float()
    model.language_model = MagicMock(return_value=local_output)

    full_padding_mask = torch.tensor([[False, True, False, True, False, False, True, False]])
    full_valid_tokens = ~full_padding_mask
    boundary_valid_tokens = torch.tensor([[True, False, True, False]])
    attention_valid_tokens = torch.tensor([[True, False, False, True]])
    boundary_batch = {
        "tokens": torch.tensor([[0, 1, 2, 3]]),
        "decoder_input": None,
        "position_ids": None,
        "labels": None,
        "loss_mask": torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
        "mtp_input_mask": boundary_valid_tokens,
    }
    attention_batch = (
        boundary_batch
        if linear_layout == attention_layout
        else {
            **boundary_batch,
            "tokens": torch.tensor([[0, 1, 6, 7]]),
            "mtp_input_mask": attention_valid_tokens,
            "loss_mask": torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
        }
    )
    batches_by_layout = {linear_layout: boundary_batch, attention_layout: attention_batch}
    local_packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 4], dtype=torch.int32),
        max_seqlen_q=4,
        max_seqlen_kv=4,
    )
    cp_batch = ContextParallelBatch(
        boundary_layout=linear_layout,
        batches_by_layout=batches_by_layout,
        packed_seq_params_by_layout={layout: local_packed for layout in batches_by_layout},
    )
    full_packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 8], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 8], dtype=torch.int32),
        max_seqlen_q=8,
        max_seqlen_kv=8,
    )

    with (
        patch(
            "megatron.bridge.models.experimental_nm4_llava_provider.get_batches_on_this_cp_rank",
            return_value=cp_batch,
        ) as partition,
        patch(
            "megatron.bridge.models.experimental_nm4_llava_provider.convert_cp_layout",
            side_effect=lambda tensor, **kwargs: tensor.flip(0) if kwargs["source_layout"] == "contiguous" else tensor,
        ) as conversion,
    ):
        output, returned_mask = NM4LlavaModel.forward(
            model,
            input_ids=torch.arange(8).view(1, 8),
            padding_mask=full_padding_mask,
            loss_mask=full_valid_tokens.float(),
            packed_seq_params=full_packed,
        )

    expected_output = local_output.flip(1) if post_process and linear_layout == "contiguous" else local_output
    torch.testing.assert_close(output, expected_output)
    expected_mask = attention_batch["loss_mask"] if post_process else boundary_batch["loss_mask"]
    torch.testing.assert_close(returned_mask, expected_mask)
    assert conversion.call_count == int(post_process)
    if post_process:
        assert conversion.call_args.kwargs["source_layout"] == linear_layout
        assert conversion.call_args.kwargs["target_layout"] == "zigzag"
        assert conversion.call_args.kwargs["thd_plan"] is cp_batch.thd_plan
    assert partition.call_args.kwargs["boundary_layout"] == linear_layout
    assert model.language_model.call_args.kwargs["cp_batch"] is cp_batch
    assert model.language_model.call_args.kwargs["packed_seq_params"] is local_packed
    partition_batch = partition.call_args.args[0]
    torch.testing.assert_close(partition_batch["mtp_input_mask"], full_valid_tokens)
    expected_valid_tokens = boundary_valid_tokens if linear_layout == attention_layout else attention_valid_tokens
    torch.testing.assert_close(model.language_model.call_args.kwargs["padding_mask"], ~expected_valid_tokens)
    assert all("mtp_input_mask" not in batch for batch in cp_batch.batches_by_layout.values())
