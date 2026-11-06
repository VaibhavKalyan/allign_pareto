# M1 dataset selection and rationale

## 1. Databricks Dolly-15k — primary source for weak creative-intent labels

- Canonical source: https://huggingface.co/datasets/databricks/databricks-dolly-15k
- Size: 15,011 examples in the published training split.
- Useful fields: `instruction`, `context`, `response`, `category`.
- Published license: Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0). Preserve attribution and license obligations when redistributing derived material.
- Why use it: human-written instruction-following prompts; categories such as `creative_writing` and `brainstorming` provide interpretable weak labels.
- Main limitation: it does not provide a prompt-harm ground-truth label, and its category is not a perfect measure of the prompt's actual intent. Some examples may not match their category cleanly.

Recommended provisional map, matching the implementation plan:
- `creative_writing`, `brainstorming` → `creative_label=1` (weak label)
- `closed_qa`, `open_qa`, `information_extraction`, `summarization`, `classification` → `creative_label=0` (weak label)
- Any unlisted category → leave creative label unassigned for inspection rather than inventing a mapping.
- `safety_label=0` is the plan's provisional default only. Since Dolly has no explicit safety annotation, audit it before treating this as ground truth.

## 2. AI2 WildGuardMix — primary prompt-harm source

- Canonical source: https://huggingface.co/datasets/allenai/wildguardmix
- Published card describes 86,759 WildGuardTrain examples and a separate 1,725-item WildGuardTest split; total repository file size is listed as about 56 MB.
- Useful fields include `prompt`, `adversarial`, `prompt_harm_label`, `response_harm_label`, `response_refusal_label`, and `subcategory`. Some labels are missing when the annotators/models did not agree sufficiently.
- Published license label: ODC-BY. The Hugging Face dataset is gated by an acceptance flow for AI2 Responsible Use Guidelines and contact-information requirements. Review its terms before use or redistribution.
- Why use it: labels the user prompt as `harmful` or `unharmful`, and includes both benign and harmful plus adversarial prompts.
- Main limitation: it does not label whether a prompt is creative. Do not automatically set the creative label to zero just because a row came from WildGuardMix. That creates a source shortcut: the model could infer creativity from stylistic artifacts associated with a dataset.

Recommended safety mapping:
- `prompt_harm_label == "harmful"` → `safety_label=1`
- `prompt_harm_label == "unharmful"` → `safety_label=0`
- Missing label → leave safety label unassigned / exclude pending adjudication.

**Keep `wildguardtest` out of training and threshold tuning.** Use it as a fixed external evaluation set for the safety head. It is especially valuable for checking false negatives, but it is not a creativity benchmark.

## 3. XSTest — optional external evaluation, not an M1 training source

- Canonical source: https://github.com/paul-rottger/xstest
- 450 prompts: 250 safe prompts that should generally be answered, plus 200 unsafe contrast prompts that should generally be refused.
- The prompt data is published under CC BY 4.0; completions added by other model providers have separate terms.
- Why keep it for later: it probes over-refusal and calibration—important when studying the safety/creative alignment trade-off. It is not a substitute for a creative-intent training label, and its public prompts should not be mixed into training if the team wants an independent evaluation.

## Why not use Tulu 3 SFT mixture as the default M1 source?

The mixture is large but combines many datasets with different terms; its card explicitly notes that some subsets are non-commercial and that subset-level licenses vary. It lacks ready-made independent `creative_label` and `safety_label` fields. For the M1 goal, this adds complexity without solving the main labeling gap. Revisit only if the primary sources leave a documented coverage gap and the group checks each subset's terms.

## Dataset strategy summary

Use Dolly + WildGuardTrain for training candidates; hold WildGuardTest and optionally XSTest aside for evaluation. Treat the first label table as **weakly labeled candidates**. Before final training, annotate creativity across both sources and audit the provisional safety labels. All four quadrants must be populated by examples—not by source ID alone—and label correlation/source leakage must be measured.
