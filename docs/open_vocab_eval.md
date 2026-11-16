# Open-vocabulary detector evaluation

## Scope

Evaluated the current `data/combined_v1/data.yaml` validation split on 2026-11-17: 962 images, 1,195 ground-truth boxes, and 8 classes. The split is provisional; rerun after the train/val/test split is locked. All runs used GPU 0, an NVIDIA RTX PRO 5000 Blackwell with 48 GB of memory. YOLO-World and OWLv2 used their default pretrained weights; Baseline v1 used `models/baseline_v1/best.pt`; Qwen used Ollama `qwen2.5vl:7b`.

## Prompt trials

Default wording is each class name with underscores changed to spaces. The same three prompt sets were tested for both open-vocabulary models on all 962 validation images. Prompt selection used mAP50-95.

| Trial | Wording overrides | YOLO-World mAP50-95 | OWLv2 mAP50-95 |
| --- | --- | ---: | ---: |
| Default | No overrides | 0.0508 | 0.1884 |
| v1 | `military_truck`: army truck; `missile_launcher`: rocket launcher vehicle; `warship`: navy warship | 0.0466 | 0.1924 |
| v2 | `drone`: unmanned aerial vehicle; `military_aircraft`: military airplane; `military_helicopter`: army helicopter; `military_truck`: army transport truck; `missile`: guided missile; `missile_launcher`: mobile rocket launcher; `tank`: battle tank; `warship`: naval vessel | 0.0468 | 0.2760 |

The highest-mAP wording was the default for YOLO-World and v2 for OWLv2. The selected files are [`open_vocab_yolo_world_best.yaml`](../configs/prompts/open_vocab_yolo_world_best.yaml) and [`open_vocab_owlv2_best.yaml`](../configs/prompts/open_vocab_owlv2_best.yaml). The trial files are [`open_vocab_v1.yaml`](../configs/prompts/open_vocab_v1.yaml) and [`open_vocab_v2.yaml`](../configs/prompts/open_vocab_v2.yaml).

## Results

All precision, recall, and F1 values below are from the selected operating confidence. Count MAE is the mean absolute error across the 7,696 image/class pairs; exact image/class share is the fraction of pairs with exactly the right count. Exact image share is included as an additional stricter count measure. Counts use the same confidence threshold shown in the table.

| Model | mAP50-95 | mAP50 | Confidence | Precision | Recall | F1 | Count MAE / image-class | Exact image-class | Exact image | Mean / median ms per image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline v1 | 0.4494 | 0.5987 | 0.482 | 0.6527 | 0.5966 | 0.6207 | 0.0456 | 96.56% | 79.42% | 4.49 / 4.28 |
| Qwen2.5-VL 7B | N/A | N/A | 0.25* | 0.4360 | 0.3410 | 0.3580 | 0.0672 | 94.83% | 68.30% | 554.31 / 487.80 |
| YOLO-World | 0.0508 | 0.0641 | 0.395 | 0.1105 | 0.1026 | 0.0805 | 0.1975 | 83.98% | 5.93% | 3.82 / 3.54 |
| OWLv2 | 0.2760 | 0.3426 | 0.458 | 0.3361 | 0.4195 | 0.3393 | 0.1650 | 85.47% | 27.55% | 431.25 / 425.17 |

\* Qwen assigns every box a score of 1.0 and has no calibrated confidence. Its count and P/R/F1 results use the evaluator's 0.25 threshold, which retains all returned boxes; mAP is not meaningful for this output.

YOLO-World and OWLv2 prediction latency excludes the script's untimed warm-up. Qwen latency is per-image Ollama request time. The corrected Qwen run returned 1,046 boxes and had 5 unparseable replies (0.5%).

## Visual check

The ten-image box checks were generated for both open-vocabulary models. Three shared examples (Finland military vehicle, RQ-11 drone, USS Harry S. Truman) were also rendered for all four models. Qwen's boxes are mapped from normalized 0–1000 coordinates to image pixels. Its carrier example detects the mast rather than the ship. Generated images, prediction JSON, and latency output remain local under `runs/` and are not part of this change.

Example images:

- Baseline v1: `runs/open_vocab/examples/baseline_v1/`
- YOLO-World: `runs/open_vocab/examples/yolo_world/`
- OWLv2: `runs/open_vocab/examples/owlv2/`
- Qwen2.5-VL: `runs/open_vocab/examples/qwen/`

The four-model contact sheet is `runs/open_vocab/examples/comparison.png`.

## Implementation notes

- `scripts/open_vocab_predict.py` converts a numeric device such as `0` to `cuda:0` for the Hugging Face OWLv2 pipeline.
- `scripts/vlm_predict.py` maps Qwen's normalized 0–1000 coordinates to the original image pixels and caps each response at 2,048 generated tokens by default.
- `scripts/count_evaluator.py` reports count MAE and exact count shares for the validation split.
- Qwen returns no confidence variation, so it is compared on P/R/F1 and count metrics rather than mAP.
