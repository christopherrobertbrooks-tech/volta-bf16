# TODO: leads found on 2026-11-08

Magpie, a local research tool, found these in llama.cpp's pull requests, and every link and quote below was then checked against
its source by hand. Nothing here has been run on the V100 yet.

## 1. Re-measure BF16 on current llama.cpp: the fallback the README lacked is now upstream
<https://github.com/ggml-org/llama.cpp/pull/28846> (merged 2026-09-13)

- **What changed (corrected twice on 2026-11-08):** the pull request's description says AMD only, but the merged code also
  covers NVIDIA: on Volta, a BF16 matrix multiply with more than 8 rows switches to F32 (`ggml-cuda.cu`, the
  `compute_type == GGML_TYPE_BF16 && !fast_bf16_hardware_available(cc)` branch). So there is nothing left to finish upstream.
- **But the README never measured it:** its benchmark ran on the PrismML fork (`prism` branch, commit `3ae4f51`, 2026-09-21; see
  `volta-bonsai/ENVIRONMENT.md`), whose `ggml-cuda.cu` has no BF16-to-F32 branch. The 1,538 t/s BF16 number is the old path.
- **First step:** rerun this README's `llama-bench` on current upstream llama.cpp, BF16 and F16 Qwen3-4B. If BF16 moves well toward
  F16, the headline ("BF16 costs a V100 77% of its prefill") needs updating for upstream builds.
- Found by a second Magpie run, which read the merged code rather than the description.

## 2. Test the Volta flash-attention crash fix
<https://github.com/ggml-org/llama.cpp/pull/31518> (open)

- **What it is:** a fix for a "no device code" crash on the V100 under CUDA 12.9 in some flash-attention cases: head size 128,
  F16 keys and values, a mask, and a KV length that isn't a multiple of 256.
- **Why here:** the benchmark above runs Qwen3-4B (head size 128) with `-fa 1`. Its 512-token prompt is a multiple of 256, so it
  doesn't hit the crash, but other prompt lengths can.
- **First step:** run `test-backend-ops -b CUDA0 -o FLASH_ATTN_EXT` on the V100 with and without the fix, and report the result
  on the pull request.

## 3. A caveat for "convert to F16"
<https://github.com/ggml-org/llama.cpp/pull/27016> (open)

- **What it is:** on Volta, some models overflow to NaN in F16. The pull request's case is Qwen3-Embedding-8B (issue #26044),
  from 64 input tokens on, on two V100 32 GB cards.
- **First step:** one line in "Practical": F16 is the fast path, but a model that returns NaN in F16 needs F32 for the affected
  layers until that fix lands.

Checked and set aside: #28037 (an estimated gain for a Volta flash-attention kernel nobody has built, measured on a Jetson; the
same write-up says the current kernel is already near its best) and #30672 (it checks for BF16 hardware first, so the V100 keeps
FP16).
