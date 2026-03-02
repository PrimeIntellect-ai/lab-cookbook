"""skills — Reusable building blocks for prime-cookbook environments.

Two sub-packages:

verifiers/
    Reward functions for use with vf.Rubric. All functions are async,
    accept completion/answer/info/prompt/state/parser/judge kwargs,
    and return float in [0.0, 1.0].

lab/
    Dataset construction (DatasetBuilder, load_jsonl, save_jsonl),
    search indexes (TFIDFSearchIndex / SimpleSearchIndex), and
    ground truth generation (generate_ground_truth, GroundTruth).
"""
