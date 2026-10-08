# hob: patch stack on upstream llama.cpp

`hob` is our unified llama.cpp fork. It lives on `Seraphiel102/llama.cpp`, where it builds the public
`hob-bNNNNN` releases, and is never upstreamed.

## Upstream base

`c06f84160` ("sync : ggml"), ggml-org/llama.cpp master as of 2026-10-05.
To check it: `git merge-base hob origin/master`.

## Patches (oldest first)

| # | Commit | Subject |
|---|--------|---------|
| 1 | `ea67b3c48` | graph : add SIGMOID_LOGIT_ADD expert gating |
| 2 | `ef45ded02` | convert : add Kolibri1ForCausalLM (Aleph-Alpha/Kolibri-1) |
| 3 | `42f3096c2` | model : add Kolibri 1 (kolibri1) |
| 4 | `97d40cbf7` | tests : cover kolibri1 in test-llama-archs |
| 5 | `594dfe5de` | model : add FlexOlmo (AllenAI BAR family) |

Each patch adds an architecture in the current upstream layout. The
architecture code lives in `src/models/<arch>.cpp` as a `llama_model_<arch>`
class declared in `src/models/models.h`. Registration happens in three
places in `src/llama-model.cpp`: `llama_model_mapping`,
`llama_model_rope_type`, and the arch name in `src/llama-arch.{h,cpp}`. The
converter class goes in `conversion/<family>.py` and is registered in
`conversion/__init__.py` (`TEXT_MODEL_MAP`). The GGUF arch, its name and its
tensor list go in `gguf-py/gguf/constants.py`.

FlexOlmo was originally written in April 2026 as `b9611feca` (branch
`flexolmo`), against the old monolithic `llama-model.cpp` and
`convert_hf_to_gguf.py`. Patch 5 re-applies it to the layout above. It
changes two things compared with the original:

- It adds the missing `MODEL_ARCH_NAMES` entry (`"flex_olmo"`). Without it,
  the April converter could not have written a GGUF from a clean tree.
- It excludes `flex_olmo` from `llm_arch_supports_sm_tensor`, as OLMo2 and
  OLMoE are, because q/k norm spans all heads.

## Fork-only CI and housekeeping

These are not architecture patches. They keep the fork's GitHub Actions green and must be carried across rebases.

| Commit | Subject | Why |
|--------|---------|-----|
| `8798b654e` | ci : drop the s390x release build | the `ubuntu-24.04-s390x` runner exists only upstream; it blocked the release |
| `9bb50fd25` | convert : regenerate pre-tokenizer hashes | the hand-placed kolibri1 hash sat out of generator order and failed "Check Pre-Tokenizer Hashes"; same hash, moved |
| `0f25b8c18` | ci : remove build-cann.yml | upstream left `jobs:` empty (all commented out), which GitHub reports as an invalid workflow on every push |
| `9f15f3d34` | ci : drop the musa and s390x docker targets | the musa base-image registry times out from GitHub runners; the s390x runner exists only upstream |

After a rebase, re-run `convert_hf_to_gguf_update.py --check-missing` and commit `conversion/base.py` if it changes.

## Verification

All checks were CPU-only (`-DGGML_CUDA=OFF`, `CUDA_VISIBLE_DEVICES=""`).
GPU backends have not been exercised for either architecture on this base.

**Kolibri 1 (patches 1-4)**
- `test-llama-archs`: kolibri1 passes on this base (2026-10-05).
- Tiny-checkpoint float64 PyTorch parity and real-weight logit parity
  (KL / top-1 against the reference) were run on the kolibri1 branch. See
  `orchestration_hub/research/kolibri/IMPLEMENTATION.md`.

**FlexOlmo (patch 5)**
- `test-llama-archs`: flex_olmo passes (MoE-mandatory). The full run
  reports "all 135 test(s) passed", kolibri1 included.
- Graph equivalence with the April build: we built a tiny random FlexOlmo
  GGUF (2 layers, n_embd 256, 2 experts, real BAR tokenizer) and compared
  the old and new builds on the same prompt, 16 greedy positions, top-5
  logprobs each.
  - F32: top-5 ids identical at all 16 positions, max |d logprob| = 2e-5.
  - Q4_K_M: top-1 identical at all 16 positions, max |d logprob| = 1e-5.
- Real model, `BAR-2x7B-Tool-Use.Q4_K_M.gguf` (the published April GGUF),
  temp 0, 32 tokens:
  - The plain prompt "The capital of France is" gives byte-identical output
    in the old and new builds, and identical top-5 logprobs.
  - A ChatML Fibonacci prompt (21 tokens) diverges at the first token
    ("Here" vs "def"; the two are 0.03-0.1 logprob apart). The cause is the
    upstream base, not the port. Gemma-2 Q6_K, an architecture the port does
    not touch, shows the same pattern between the two builds: the short
    prompt is identical and the multi-token prompt shifts by 0.1-0.4
    logprob. In addition, toggling `-fa off -nr` inside a single build moves
    BAR by about 0.1 on this prompt.

## Rebasing onto newer upstream

```sh
git fetch origin
git switch hob
git rebase origin/master   # replays patches 1-5
```

If a patch conflicts, it is usually because upstream reorganised
registration again. Re-apply the patch's intent by copying how a
neighbouring architecture is registered: OLMoE/OLMo2 for FlexOlmo,
Qwen3-MoE/AfMoE for Kolibri. Do not force the old hunks in.

Before porting, grep the tree for `flex_olmo`/`FlexOlmo`/`kolibri`. If
upstream has added any of these, drop our patch rather than duplicate it.

After rebasing:

```sh
cmake -B build -DGGML_CUDA=OFF
cmake --build build -j 6 --target llama-cli llama-server llama-quantize test-llama-archs
./build/bin/test-llama-archs -a 'flex_olmo|kolibri1'
```

Then run a short temp-0 generation on a real BAR GGUF and a Kolibri GGUF.
Copy the GGUF to local disk first, because loading over `/mnt/e` (9p) is
very slow. Compare against the previous `hob` build. If outputs differ,
repeat the comparison on an architecture we have not patched, to separate
upstream numeric drift from a porting bug before you blame the port.
