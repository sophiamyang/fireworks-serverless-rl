# Fireworks Serverless RL Colab

[`serverless_rl.ipynb`](serverless_rl.ipynb) is an explicit, cell-by-cell Colab
adaptation of the Fireworks cookbook
[`training/examples/serverless_rl`](https://github.com/fw-ai/cookbook/tree/main/training/examples/serverless_rl)
Countdown example.

The notebook makes the main learning loop visible instead of hiding it behind
`ServerlessCountdownRL.run()`:

1. save the current LoRA adapter and sample completion groups;
2. decode and score each completion;
3. remove zero-variance groups, calculate GRPO advantages, and build aligned
   importance-sampling datums;
4. run forward/backward and one Adam update; and
5. evaluate on the same held-out prompts without updating weights.

It includes the upstream reward restrictions, separate serverless and
control-plane URL handling, Router Replay, K1/K3 diagnostics, optional W&B,
decoded evaluation completions, resumable trainer-state checkpoints, and
defensive client cleanup.

## Run it in Google Colab

1. Open [`serverless_rl.ipynb`](serverless_rl.ipynb) in
   [Google Colab](https://colab.research.google.com/).
2. In Colab's left sidebar, open **Secrets** (the key icon).
3. Add `FIREWORKS_API_KEY`, paste your Fireworks API key, and enable notebook
   access for it.
4. Optional: add both `WANDB_API_KEY` and `WANDB_ENTITY` for tracking. Use
   `fireworks-ai-llm` for the Fireworks team or `sophia-yang` for your personal
   W&B account. W&B stays disabled unless both values are present.
5. Run from the top. A Colab GPU is not required because Fireworks performs the
   model training and sampling remotely.
6. Keep `RUN_PROFILE = "smoke"` for the first run. It uses the bundled sample,
   one optimizer step, two prompts, two completions per prompt, and a 512-token
   completion limit.
7. Read the four visible training stages and the evaluation function.
8. When ready to incur Fireworks usage, change `RUN_TRAINING = False` to
   `RUN_TRAINING = True` and run the paid cell.
9. Inspect the smoke-run outputs. Only then change `RUN_PROFILE` to `"full"`
   and rerun from the configuration cell.
10. Run the final cell to graph rewards and create a ZIP. Set
    `DOWNLOAD_RESULTS = True` if you want Colab to download it automatically.

## Profiles and cost boundary

`smoke` is the safe default. `full` matches the upstream Kimi K3 defaults:
20 steps, 16 prompt groups per step, 8 completions per group, a 4,096-token
completion limit, and held-out evaluation every 5 steps. That is 2,560 training
completions plus 640 evaluation completions, so switch deliberately.

All setup, reward tests, dataset work, and configuration checks are local.
Fireworks usage starts only in the guarded paid-run cell.

## Reproducibility and recovery

The notebook pins the tested cookbook revision instead of following a moving
`main` branch. Every execution gets a timestamped directory under
`/content/serverless_rl_runs` and records:

- the resolved configuration;
- the cookbook commit;
- the input dataset SHA-256;
- the deterministic held-out indices; and
- the serverless session, run, and checkpoint identifiers.

The `full` profile saves trainer state every five completed optimizer steps.
Copy the value from `resume_from.txt` into `RESUME_FROM`, rerun the setup cells,
and start the paid cell to continue from that optimizer state. On resume,
`STEPS` means the number of **additional** optimizer steps to run. The smoke
profile does not create DCP checkpoints by default.

## Outputs

Each run directory can contain:

- `config.json` — the resolved run settings;
- `cookbook_commit.txt` and `input_sha256.txt` — source provenance;
- `dataset_split.json` — reproducible held-out row indices;
- `metrics.jsonl` — one record per optimizer step;
- `eval_metrics.jsonl` — held-out aggregate metrics;
- `eval_completions/` — decoded, scored evaluation samples;
- `resume_from.txt` — latest full trainer-state checkpoint reference;
- `lifecycle.json` — session, run, model, and checkpoint identifiers;
- `final_checkpoint.txt` — final sampler checkpoint path;
- `reward_curve.png` — rollout and evaluation reward; and
- a ZIP archive beside the run directory.

Model promotion remains in the upstream command-line example. Promotion is a
separate mutation and should be performed only after verifying the exact final
checkpoint and desired output model ID.
