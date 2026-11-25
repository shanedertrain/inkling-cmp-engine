# engine/kernels (empty)

CUDA kernels of stage S2 (PLAN 6.1) go here, one directory per kernel group: `moe/`, `attn/`, `dense/`, `glue/`,
`head/`, `draft/`, `prefill/`. Each kernel enters as a torch extension swapped into the S1 reference
(`engine/ref/model.py` has one function per op), with its FP64 oracle test, bit lock and batch-invariance test in
`engine/tests/`. Build: `ink-ninfer-toolchain:cu1331` image, `-arch=sm_80`, inside docker only.

Nothing is written yet: S2 starts after the S1 gate passes.
