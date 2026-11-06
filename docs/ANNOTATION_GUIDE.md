# M1 prompt-label annotation guide (draft for team agreement)

Annotate prompt intent, not the source dataset or the expected assistant response.

## `creative_label`

`1` when the user explicitly asks for an imaginative, expressive, or generative task where novelty/style is central: fiction, poems, scripts, character/world creation, story ideas, brainstorming or creative alternatives.

`0` when the primary intent is factual answering, explaining, extracting, classifying, summarizing, translating, or performing a conventional task where creative novelty is not central.

`uncertain` when either interpretation is plausible or the prompt is too underspecified. Keep the uncertain marker in the audit file; do not force it to 0.

## `safety_label`

For the first M1 pass, define `1` as a prompt marked `harmful` by WildGuardMix's `prompt_harm_label`, and `0` as marked `unharmful`. Missing labels are unknown, not safe.

Before labeling, the team must decide whether the intended concept is strictly **harmfulness** or broader **safety sensitivity**. For example, a benign prompt about a sensitive topic can need careful handling while still being unharmful. Do not mix these definitions in a single target label without an explicit policy.

## Independence requirement

Apply the creative rubric to prompts from both Dolly and WildGuardMix. If only Dolly rows receive `creative_label=1` and all WildGuard rows receive `creative_label=0`, the classifier may learn source/style differences rather than creative intent. Similarly, check whether prompt length, category, or adversarial status acts as a label shortcut.

## Suggested manual audit

1. Sample examples from each source × safety class × provisional creative class, where present.
2. Have two team members independently label a small overlapping sample.
3. Calculate agreement and discuss disagreements before labeling the rest.
4. Keep `annotator_1`, `annotator_2`, `adjudicated_label`, `label_method`, and `notes` in a separate annotation log.
5. Report counts for the four final label quadrants: (0,0), (0,1), (1,0), (1,1).
