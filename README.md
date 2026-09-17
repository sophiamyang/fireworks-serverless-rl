# Fireworks Serverless RL Colab

[`serverless_rl.ipynb`](serverless_rl.ipynb) is a detailed, single-notebook
entry point for the upstream Fireworks
[`training/examples/serverless_rl`](https://github.com/fw-ai/cookbook/tree/main/training/examples/serverless_rl)
Countdown example.

The notebook does not duplicate or simplify the RL implementation. It:

- sparse-checks out only `training/` from `fw-ai/cookbook`;
- runs `pip install --pre -e training`;
- imports the upstream `Config` and `ServerlessCountdownRL` directly from
  `training/examples/serverless_rl/countdown_rl.py`;
- uses the bundled `data/countdown_train.jsonl` sample;
- explicitly keeps Router Replay enabled; and
- guards construction and execution of the paid trainer behind
  `RUN_TRAINING = False`.

## Run in Colab

1. Upload or open `serverless_rl.ipynb` in
   [Google Colab](https://colab.research.google.com/).
2. Run the setup and configuration cells.
3. In Colab **Secrets**, add `FIREWORKS_API_KEY` and grant the notebook access.
4. Review the printed upstream commit and complete configuration.
5. Only when you intend to incur Fireworks usage, change
   `RUN_TRAINING = False` to `RUN_TRAINING = True` and run that cell.

The notebook keeps the upstream defaults except for selecting the bundled
32-row dataset, explicitly enabling Router Replay, and choosing a predictable
Colab output directory. The default run is not a cheap dry run: it uses 20
optimizer steps, 16 prompt groups per step, 8 samples per group, and periodic
held-out evaluation.

## Why the upstream commit is printed

Each fresh notebook run checks out the current upstream `main`. That guarantees
the code being executed is the upstream implementation, but it also means
behavior can change over time. Save the printed commit SHA with experiment
results. For repeatable experiments, replace `main` in the checkout cell with
a reviewed commit SHA.

Training artifacts are written to `/content/serverless_rl_run`. The final
notebook cell lists the generated metrics, evaluation completions, checkpoint
references, and reward curve without performing any training itself.
