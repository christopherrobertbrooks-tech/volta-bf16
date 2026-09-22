# BF16 costs a V100 77% of its prefill — convert to F16 first

Volta has FP16 tensor cores but **no BF16 hardware**. llama.cpp knows this:

```c
static bool fast_bf16_hardware_available(const int cc) {
    return ... || (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE);
}
```

sm_70 fails that gate, and `ggml-cuda.cu:1629` takes an explicit fallback
branch. It does not crash — it just gets much slower, and only in one phase.

## Result

`Qwen3-4B`, the same weights as BF16 and as F16 (converted locally, so tensor
shapes and file size are identical at 7.49 GiB). `llama-bench -ngl 99 -fa 1
-p 512 -n 128 -r 3`, warm-up run discarded.

| card | dtype | pp512 | tg128 |
| :--- | :--- | ---: | ---: |
| Tesla V100 (sm_70) | BF16 | 1538.4 ± 22.5 | 83.50 ± 0.06 |
| Tesla V100 | **F16** | **6719.3 ± 190.4** | 83.21 ± 0.08 |
| RTX 4070 (sm_89) | BF16 | 4248.1 ± 83.9 | 56.24 ± 0.02 |
| RTX 4070 | F16 | 4919.1 ± 90.4 | 56.27 ± 0.01 |

**BF16 costs the V100 77% of prefill** — 4.37x slower than the identical model
in F16. The 4070, which has BF16 hardware, loses only 14%.

**Decode is unaffected on both cards.** 83.50 vs 83.21, and 56.24 vs 56.27.
Decode is memory-bound — the same 7.49 GiB is read either way — and the dtype
conversion is negligible at batch 1. The penalty is purely in the compute-bound
prefill phase, where tensor cores matter.

## The second finding: in F16 the V100 beats the 4070 at prefill

6719.3 against 4919.1 — the V100 is **37% faster**.

Every earlier measurement in these repos had the 4070 winning prefill, but all
of those used quantized models. Given data it can feed straight to its FP16
tensor cores, Volta's 112 TFLOPS FP16 shows up. The V100's weakness is not
"compute" in general; it is compute in formats it has no silicon for.

## Practical

**Convert any BF16 GGUF to F16 before running it on the V100.** The conversion
is a dtype cast and took **12 seconds** for a 4B model, at 38°C:

```
nice -n 19 ./build/bin/llama-quantize --allow-requantize in-BF16.gguf out-F16.gguf F16 8
```

Many modern models ship BF16 as their native format, so this is not an edge
case — it is the default trap for this card.

Only worth it if prompts are long enough for prefill to matter. Pure chat with
short prompts is decode-bound and sees no difference at all.

## Credit and scope

The BF16 hypothesis came from Gemini, which correctly identified that Volta
lacks BF16 hardware. Two of its other suggested tests did not survive contact
with the measurements already in these repos: FP8/FP4 "will crash or emulate
slowly" (gpt-oss MXFP4 runs at 143 t/s on the V100, because llama.cpp
dequantises in software with no architecture gate), and FlashAttention "will not
execute on Volta" (llama.cpp's own FA implementation runs there and *helps* two
of three models tested). Both of those assumed vLLM and the Dao-AILab
FlashAttention library rather than llama.cpp.

One model, one size. The magnitude of the prefill penalty may vary with
architecture and batch size.

## Environment

See `volta-bonsai/ENVIRONMENT.md`. Each card isolated via `CUDA_VISIBLE_DEVICES`.
