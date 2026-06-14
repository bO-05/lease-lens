# Modal Evidence

This note records the public, judge-facing evidence from the Modal smoke run for Lease Lens.

## Run

- Command: `modal run training/finetune_legal_3b_modal_v2.py --smoke-test --no-push`
- Modal app/run: `https://modal.com/apps/bo-05/main/ap-Cas8EFv8gpZjG9s3eCXauZ`
- Hardware: NVIDIA A100-SXM4-40GB
- Model recipe: Llama-3.2-3B QLoRA with the v2.5 20% synthesized-NONE mix
- Data loaded: CUAD mirror, smoke split of 400 positives + 100 synthesized NONE negatives
- Training: 60 steps, effective batch size 16, 24,313,856 trainable parameters
- Result: run completed cleanly and saved merged 16-bit artifacts to `/outputs/merged-16bit`

## Key Metrics From Logs

- Training runtime: 160.35 seconds
- Train samples/sec: 5.987
- Train steps/sec: 0.374
- Final logged train loss: 1.6022
- Epoch reached: 1.9

## Notes

- The xFormers warning was non-fatal. Unsloth fell back to PyTorch attention and completed the smoke run.
- `--no-push` was used intentionally so this run verifies the Modal path without overwriting any published model.
- This is smoke-run evidence, not a replacement for the held-out CUAD metrics reported in the main README.
