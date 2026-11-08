# TODO: leads found on 2026-11-08

Magpie, a local research tool, found these in llama.cpp's pull requests, and every link and quote below was then checked against
its source by hand. Nothing here has been run on the V100 yet.

## 1. ~~Finish llama.cpp's open TODO for older NVIDIA cards~~: already done upstream
<https://github.com/ggml-org/llama.cpp/pull/28846> (merged 2026-09-13)

- **Correction (2026-11-08):** this lead was wrong. The pull request's description says the change was made for AMD only, but
  the code that was merged also covers NVIDIA: on Volta, a BF16 matrix multiply with more than 8 rows already switches to F32
  (`ggml-cuda.cu`, the `compute_type == GGML_TYPE_BF16 && !fast_bf16_hardware_available(cc)` branch).
- **What it means here:** the README's measurement (BF16 pp512 1,538 t/s against 6,719 t/s in F16) already runs through that
  fallback, so it doesn't close the gap. Converting to F16 stays the fast path. Nothing is left to finish upstream.
- Found when a second Magpie run read the merged code instead of the description.

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
