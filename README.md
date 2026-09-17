# Fireworks Serverless RL Colab

[`serverless_rl.ipynb`](serverless_rl.ipynb) walks through the Fireworks cookbook
[`training/examples/serverless_rl`](https://github.com/fw-ai/cookbook/tree/main/training/examples/serverless_rl)
Countdown example **step by step in one notebook**.

Instead of calling `ServerlessCountdownRL(...).run()` as a black box, the notebook:

- installs the cookbook `training` package (SDK, renderers, Router Replay helpers);
- inlines the Countdown **`composite_reward`** logic from `countdown_rewards.py`;
- loads the bundled **`countdown_train.jsonl`** and carves out a fixed eval split;
- connects to **`/training/v1/serverless`** and creates a LoRA training client;
- defines explicit **`evaluate()`** and **`training_step()`** functions that mirror
  the cookbook’s `_evaluate` and `_step` (snapshot → sample → score → GRPO →
  importance sampling → Adam);
- keeps **`ROUTER_REPLAY = True`** (MoE models replay sampling routes; dense models skip automatically); and
- guards all paid API usage behind **`RUN_TRAINING = False`**.

## Run in Colab

1. Open `serverless_rl.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Run cells through hyperparameters, rewards, and data prep (no charges).
3. Add `FIREWORKS_API_KEY` in Colab **Secrets**.
4. Read the loop explanation and function definitions.
5. Set `RUN_TRAINING = True` only when you intend to run a paid training session.

Default hyperparameters match the upstream script (20 steps, 16 prompt groups × 8
samples, periodic held-out eval). With the 32-row sample file, half the rows are
held out for evaluation.

Artifacts land in `/content/serverless_rl_run` (`metrics.jsonl`, `eval_metrics.jsonl`,
`final_checkpoint.txt`, and an optional `reward_curve.png`).

For DCP resume, promotion, W&B, and shell launchers, see the
[cookbook README](https://github.com/fw-ai/cookbook/tree/main/training/examples/serverless_rl).
