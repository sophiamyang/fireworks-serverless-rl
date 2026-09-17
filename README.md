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
5. periodically evaluate on the same held-out prompts without updating weights.

It also includes the upstream reward restrictions, separate serverless and
control-plane URL handling, Router Replay, K1/K3 diagnostics, W&B logging,
decoded evaluation completions, lifecycle metadata, and guaranteed client
cleanup if a run fails.

## Run it in Google Colab

1. Open [`serverless_rl.ipynb`](serverless_rl.ipynb) in
   [Google Colab](https://colab.research.google.com/).
2. In Colab's left sidebar, open **Secrets** (the key icon).
3. Add a secret named `FIREWORKS_API_KEY` and paste your Fireworks API key as
   its value. Enable notebook access for the secret.
4. Optional: add `WANDB_API_KEY` for experiment tracking.
5. Optional: add `WANDB_ENTITY`:
   - use `fireworks-ai-llm` for the Fireworks team;
   - use `sophia-yang` for your personal W&B account.
6. Run the notebook from the top. A Colab GPU is not required: Fireworks runs
   the model training and sampling remotely.
7. Keep `USE_CANONICAL_DATASET = True` for the upstream-style deterministic
   20,000-row dataset. Set it to `False` only for the bundled 32-row wiring
   sample; that sample is too small for a meaningful experiment.
8. Read through the four training stages and evaluation function.
9. When ready to incur Fireworks usage, change `RUN_TRAINING = False` to
   `RUN_TRAINING = True` and run the final training cell.
10. Run the last cell to graph rollout and held-out reward and list the saved
    artifacts.

The upstream defaults are retained: Kimi K3, LoRA rank 32/alpha 64, 20 steps,
16 prompt groups per step, 8 samples per group, and evaluation every 5 steps.
These settings can generate substantial remote sampling. Reduce them for an
initial smoke test, for example `STEPS = 1`, `PROMPT_GROUPS_PER_STEP = 2`,
`GROUP_SIZE = 2`, `EVAL_PROMPT_GROUPS = 2`, and `EVAL_GROUP_SIZE = 2`.

## Outputs

Artifacts are written to `/content/serverless_rl_run`:

- `metrics.jsonl` — one record per optimizer step;
- `eval_metrics.jsonl` — held-out aggregate metrics;
- `eval_completions/` — decoded, scored evaluation samples;
- `dataset_split.json` — reproducible held-out row indices;
- `lifecycle.json` — session, run, model, and checkpoint identifiers;
- `final_checkpoint.txt` — final sampler checkpoint path; and
- `reward_curve.png` — rollout and evaluation reward.

The notebook deliberately leaves full trainer-state resume (DCP) and checkpoint
promotion to the upstream command-line example; both add lifecycle operations
that are separate from understanding the core RL loop.
