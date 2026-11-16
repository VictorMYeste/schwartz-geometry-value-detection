# Human evaluation protocol

Fixed on 2026-11-16, after the annotation files were sent and before any answer was returned.
Scripts: `scripts/sample_human_eval_items.py` (items and workbooks) and
`scripts/analyze_human_eval.py` (analysis).

**Question.** For the same sentence, do people who do not know Schwartz's theory judge the
decoder's label set more plausible than the thresholded one?

**Items.** 150 test sentences sampled from the 346 sentences that the Schwartz decoder edits
(the BCE model trained with seed 42, frozen decoder setting). The sample is stratified by edit type: 81
additions, 66 removals and 3 mixed edits, drawn with sampling seed 11406. There are also 15
attention checks, each comparing the gold set with a random set from other broad values. Each
annotator first does 10 practice items, which are not counted.

**Task.** Blind pairwise preference:
- The annotator sees the sentence and two label sets, with the values listed alphabetically.
- The answer is: A fits better / B fits better / No clear difference.
- Item order and the A/B side are shuffled for each annotator.
- No system names, no gold labels and no circular order are shown.
- The value sheet gives one definition and one plain-language example per value.

**Annotators.** Exactly three unpaid volunteers, who are fluent in English, are not involved
in the work and are unfamiliar with the theory. Every item is annotated by all three. Two
reserve files exist only to replace an annotator who drops out, has not started by 4 December,
or fails the attention bar. A reserve never adds a fourth vote.

**Exclusion.** An annotator who picks the gold side on fewer than 10 of the 15 attention checks
is replaced. The bar is not 15/15 because the gold labels are noisy.

**Analysis.**
1. Each answer is mapped to the system it favours: decoded, thresholded or none.
2. Each item takes the majority vote of its three answers. Three different answers mean no
   preference.
3. *Primary:* the share of decisive items whose majority favours the decoded set. It is reported
   with a 95% percentile bootstrap CI over items (10,000 resamples) and an exact two-sided sign
   test against 0.5.
4. *Secondary, descriptive:*
   - the same share by edit type;
   - the share over all individual judgments;
   - per-annotator shares;
   - Krippendorff's α (nominal; ordinal as a sensitivity check);
   - accuracy on the attention checks;
   - the majority preference against the gold-based outcome of each edit (improved / worsened /
     unchanged in sample F1).

**Reading, fixed in advance.** With 150 items, about 60/40 or stronger among decisive items is
detectable. A weaker or null result is reported as "no clear preference".

The study has no random-order arm, so it does not show that a preference is due to the
Schwartz order. That question is answered by the gold-based random-order test
(`scripts/eval_decoder_edits_external.py`). The study also covers only one seed's edits.

**Release.** Anonymised judgments are released with the repository: annotator slot, dataset
IDs, sides and answers. Sentence text and free-text comments are not released.
