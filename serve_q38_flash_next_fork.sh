#!/bin/bash
# Fork recipe: btbtyler09/Qwen3.8-Flash-Next-GPTQ-4bit (qwen4_exp) on 4x
# Radeon PRO V620 (gfx1030). Measured 2026-09-06 TP4 cards 0-3 (150 W caps):
#   MTP=3 PIECEWISE: decode @3.7k 24.7 t/s, c1 79.9, c8 152.5
#   (eager MTP=3: 10.5 t/s; eager no-MTP: 6.1)
#   EP tested and rejected: --enable-expert-parallel loses on this PCIe
#   host (19.4/-21% decode, 68.8/-14% c1, 129.2/-15% c8, greedy-clean) —
#   matches the BM35 finding; keep experts TP-sharded.
#   Parameter sweep 2026-09-06 — this recipe is the optimum of:
#   - MTP k=2/k=4 lose (k=3: c1 79.9 vs 70.1/69.1; c8 152.5 vs 144.6/138.0;
#     k=4 decode-only +3.7% not worth it)
#   - 0.95 util + max-num-seqs 16 collapses c8 to 63.2 (-59%)
#   - max-num-batched-tokens 8192 is a wash (-3.8% c1, -8% TTFT)
#   - --linear-backend exllama is REQUIRED: auto-select picks
#     RDNA2TritonW4A16 which wants a 95 GiB repack buffer (OOM on 32 GB)
#
# Required env:
# - VLLM_PLE_MMAP=1: the ~82GB generated n-gram hash table lives in pinned
#   host memory (UVA view); without it TP4 does not fit 32GB cards.
# - VLLM_USE_V2_MODEL_RUNNER=1: PLE forward kwargs (ngram_context /
#   query_start_loc) only exist on the V2 runner.
# - PIECEWISE cudagraphs: the host-table PLE gather runs between graph
#   pieces (FULL capture cannot replay host gathers).
MTP="${MTP:-3}"
export VLLM_PLE_MMAP=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENGINE_READY_TIMEOUT_S=1200
export GPU_MAX_HW_QUEUES=4
export HIP_FORCE_DEV_KERNARG=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_BLAS_PREFER_HIPBLASLT=0
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MOE=0
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"

SPEC_ARG=""
[ "${MTP}" != "0" ] && SPEC_ARG="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}}"

exec /mnt/Dev/vllm-rdna2/venv/bin/vllm serve btbtyler09/Qwen3.8-Flash-Next-GPTQ-4bit \
    --dtype float16 \
    --kv-cache-dtype auto \
    --linear-backend exllama \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 4096 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    ${SPEC_ARG} \
    --compilation-config '{"mode":3,"cudagraph_mode":"PIECEWISE"}' \
    --language-model-only \
    --skip-mm-profiling \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --gpu-memory-utilization 0.90 \
    --served-model-name Qwen3.8-Flash-Next \
    "$@"
