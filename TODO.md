# TODO: leads found on 2026-11-08

Magpie, a local research tool, found these in llama.cpp's pull requests, and every link and quote below was then checked against
its source by hand. Nothing here has been run on the V100 yet.

## 1. Finish llama.cpp's open TODO for older NVIDIA cards
<https://github.com/ggml-org/llama.cpp/pull/28846> (merged 2026-09-13)

- **What it is:** a fallback to F32 for BF16 matrix work on GPUs without BF16 hardware. It was done for AMD only. Its author
  noted that NVIDIA cards before Ampere lack BF16 matrix hardware too, had no NVIDIA card to test on, and left a TODO for
  someone who does.
- **Why here:** this repo measures exactly that case: BF16 prefill on the V100 is 1,538 t/s against 6,719 t/s for the same
  weights in F16.
- **A hint, not proof:** <https://github.com/ggml-org/llama.cpp/pull/27016> measured on a V100 that forcing F32 for one cuBLAS
  path cost about 22% (pp64) and 38% (pp512) against F16. That's a different operation, so it only suggests an F32 fallback
  could land well above today's BF16 number.
- **First step:** a premise check. Add the F32 fallback for sm_70, rebuild, and rerun this README's `llama-bench` on the BF16
  Qwen3-4B. If BF16 prefill moves well toward F16, it's a pull request.
- **Scope:** it speeds up BF16 files only. Converting to F16 (below) stays the fastest path on this card.

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
