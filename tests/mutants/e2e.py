T = "src/omniscan/gpu/codec/turbo.py"
MUTANTS = [
    (
        T,
        "out[:, y : y + h, :].copy_(stage)",
        "out[:, y : y + h, :].copy_(stage, non_blocking=True)",
        "codec: async staging copy (the original race)",
    ),
    (
        T,
        "for i, (arr, y) in enumerate(zip(arrs, y_offsets, strict=True)):",
        "for i, (arr, y) in enumerate(zip(arrs[::-1], y_offsets, strict=True)):",
        "codec: pages placed in reverse order",
    ),
    (
        "src/omniscan/typeset/plan.py",
        "return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]",
        "return 0.0",
        "typeset: every bubble treated as dark",
    ),
]
