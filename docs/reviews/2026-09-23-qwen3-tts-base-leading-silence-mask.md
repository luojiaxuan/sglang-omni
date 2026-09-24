# External review: Qwen3-TTS Base x-vector-only leading-silence mask

- Date: 2026-09-23 PT (sent about 18:28 PDT, answer finished about 18:41 PDT)
- Channel: web ChatGPT, temporary chat (no custom instructions or memory)
- Model and tier actually selected: model picker entry "最新" (latest; the other entries listed were GPT-5.6 Sol and GPT-5.5, the latter marked for retirement on Oct 14), thinking-intensity slider at its rightmost position, labelled "6 Pro". The reply header reads "思考了 12m 1s" (thought for 12 min 1 s). The model ran web search during thinking (it cites GitHub, arXiv and Hugging Face pages besides the attachments).
- Attached files (all 7 uploaded to the message, 41 KB as counted by the uploader; scanned for credentials first, none found):
  - `fix_design.md`
  - `repro_README.md`
  - `onset_frame_histogram.txt`
  - `summary_peak.txt`
  - `onset_client.py`
  - `model_runner.py`
  - `prior_art_pr_description.txt`

## Prompt sent (verbatim, single line)

> Act as a hostile reviewer. Read all 7 attached files first; cite file+line/number. SETUP: Qwen3-TTS Base (12Hz, codebook-0 per 80ms frame) in sglang-omni. x-vector-only clone has variable leading silence. Repro (1440 req, H100): 1.7B xvec median onset 450ms, 90%>160ms; bimodal, frame 0 or after 4-7 silent frames; ICL onset tracks ref trailing pause. FIX (fix_design.md, Chinese): xvec-only, mask codebook-0 silence ids S to -inf for first N frames; S = ids the checkpoint's own codec encoder emits on digital silence and -80/-60 dBFS white noise; sweep N in {0,1,2,3,4,6} with onset + seed-tts-eval WER + WavLM SIM on 1.7B+0.6B; in model_runner.py apply_codec_suppress_tokens. Author bets N=1 suffices. ANSWER: (1) Principled? Better alternative? (2) Is encoder-derived S with -60 dBFS bound defensible; how to justify the level; failure modes (overlap with quiet speech tokens, checkpoint drift)? (3) Eval gaps (onset clipping/clicks, prosody, long utts, non-English, streaming, stats/sample size)? (4) Default-on if WER/SIM flat, or off? (5) Errors in repro analysis (-34 dBFS peak onset metric, ICL reading, bimodality inference)? FORMAT: Your reader is an AI agent, not a human. Be terse and information-dense: no preamble, no restating my setup, no hedging boilerplate, no motivational or summary paragraphs. Every sentence must carry a claim, a number, a citation, or an instruction. Prefer numbered claims and compact tables. Keep all technical content; drop everything else.

## Transcription notes

- Text taken from the page with `get_page_text` after scrolling to the end of the reply. Wording is unchanged.
- Formulas were KaTeX-rendered and came out of text extraction flattened; they are written back as LaTeX here with the same content.
- ChatGPT's citation chips are shown as `[cite: ...]`; "+1"/"+2" means the chip bundled that many further sources.
- In the table of section 3.1, the "Artifact integrity" row is cut off in ChatGPT's own rendering at "The client's resume key is only \`mode". The cut is ChatGPT's markdown table splitting on the `|` characters inside the key; the key in `onset_client.py` line 99 is `f"{mode}|{ref_id}|{prompt_id}|{seed}"`, so the point is that the resume key carries no model/codec revision, N, S hash or sampler settings.

---

## ChatGPT answer (verbatim)

### 1. Method: defensible constrained decoding; unproven silence correction

**1.1 Accept the experiment; reject the semantic equivalence.** `c0 ∉ S` means "exclude these token IDs," not "start with speech"; the latter requires demonstrating that `S` identifies generated non-speech without excluding legitimate initial phonemes. The prior-art author correctly calls this a behavioral override, not a strict bug fix. [cite: fix_design] [cite: prior_art_pr_description]

**1.2 The conditional-sampling justification needs three corrections.** At fixed history $h$, before support-changing sampling filters,

$$q(a\mid h)=\frac{p(a\mid h)\,\mathbf{1}[a\notin S]}{1-p(S\mid h)}$$

is the minimizer of $D_{\mathrm{KL}}(q\,\|\,p)$ subject to $q(S)=0$; its distortion is $-\log[1-p(S\mid h)]$. This establishes a local constrained-decoding principle, not the correctness of the constraint. For $N>1$, successive local renormalizations generally **do not** sample the original sequence distribution conditioned on avoiding `S` throughout the prefix: exact sequence conditioning additionally weights each choice by future constraint-survival probability. Replace "distribution unchanged after $N$" with "mask disabled after $N$; subsequent generation conditions on the altered history." [cite: fix_design]

**1.3 The deployed sampler weakens the claimed equivalence further.** With baseline `top_k=50`, masking before recomputing top-k can admit originally rank-$>50$ tokens; this is not simply conditioning the baseline top-50 distribution. External verification of upstream parent commit `481914c3` confirms that `apply_codec_suppress_tokens` precedes the sampler; document the exact ordering with temperature, repetition penalties, top-k/top-p, and EOS restrictions. [cite: repro_README] [cite: GitHub]

**1.4 Treat $N=1$ as an ablation, not a mechanistic conclusion.** Log actual generated codebook-0 IDs, their probability mass in `S`, residual codes, and speech-onset labels; estimate delayed-onset rates conditional on first-token identity. Perform controlled first-token interventions while regenerating the residual codebooks and continuation consistently; replacing a token in an already-generated sequence is not the same intervention. The runner explicitly generates residual codes after sampling layer 0, so layer-0 identity alone does not describe the whole first-frame intervention. [cite: fix_design +1] [cite: model_runner]

**1.5 The alternatives were dismissed too cheaply.**

| Alternative | Reviewer instruction |
|---|---|
| Finite silence penalty | Evaluate $q_\lambda \propto p\,e^{-\lambda \mathbf{1}_S}$, the minimizer of $D_{\mathrm{KL}}(q\,\|\,p)+\lambda\,q(S)$. Its parameter has a log-odds interpretation; "dimensionless" is not an objection. Hard masking is the $\lambda\to\infty$ limit. Specify whether the penalty is applied before temperature, which changes its effective strength. [cite: fix_design] |
| Onset-aware, bounded intervention | Target "speech begins by a budget," rather than "none of the first $N$ frames uses these IDs." Release the intervention once a validated speech-start detector fires; retain a frame cap. Validate the detector separately, because a false start can merely move the remaining silence later. |
| Conservative PCM trimming | Keep as a control: validated non-speech trimming with retained pre-roll preserves the generated speech trajectory, unlike resampling. It still consumes generation compute and risks removing weak onsets; compare actual audible latency and quality rather than rejecting it solely because it saves no talker steps. [cite: fix_design] |
| Structural correction | First compare the serving stack against the official implementation with matched conditioning and sampling; correct any prompt/codec-boundary discrepancy before compensating behaviorally. Onset-controlled training is the longer-term alternative when prompt-start behavior itself is unwanted. Mini-ICL changes conditioning, but so does the proposed override; neither wins by that objection alone. [cite: fix_design] |

**1.6 Implementation acceptance criteria:**

| Contract | Required check |
|---|---|
| Frame indexing | Use committed generated codec-frame count, with `t=0` at the first sampling prefill and eligibility `t<N`; exclude prompt/reference tokens and EOS. The supplied runner samples before both post hooks and appends only non-EOS output frames. [cite: model_runner +1] |
| Scope | Gate each row using resolved **Base + x-vector-only** mode, not missing reference text or batch-wide state; test mixed modes and different frame ages. The current suppression function applies to all active rows and contains no proposed silence-mode gate. [cite: model_runner] |
| Existing suppression | Preserve special-token suppression at `N=0`; keep the new silence helper independent of the existing reserved-range early return. Validate ID bounds, EOS exclusion, empty `S`, only-EOS support, and all-$-\infty$ interactions; define an explicit failure/fallback policy. [cite: model_runner] |
| Retraction | Force retraction before the first committed frame and around `1`, `N−1`, `N`, and `N+1`; verify history and eligibility survive. Persistence is asserted in the design, not demonstrated by the supplied scheduler code; this runner currently disables lookahead, so do not invent an existing async-overlap bug. [cite: fix_design] [cite: model_runner] |
| Graphs and identity | Test eager/graph execution, row reordering, padded rows, cancellation, and seeded batch changes; cache device masks without adding per-token synchronization. Require deterministic-path `N=0` output identity and no new encoder loading or RNG consumption on the disabled path. |

### 2. `S` and the −60 dBFS bound: candidate-generation heuristic, not a validated classifier

**2.1 Encoder provenance does not establish decoder semantics.** External architecture documentation identifies codebook 0 as semantic and the other 15 codebooks as acoustic; waveform reconstruction depends on both streams and causal context. "The encoder emitted ID $i$ on noise" therefore does not establish either "generated $i$ always decodes quietly" or "all generated quiet frames use these IDs." [cite: arXiv]

**2.2 Three one-second probes do not define the stated input class.** Zero, one −80 dBFS noise realization, and one −60 dBFS realization sample neither all amplitudes below −60 nor all non-speech spectra; quantizer assignments need not vary monotonically with level. At 80 ms/frame, the three probes provide only approximately 38–39 frame observations, with context dependence and padding effects. Separate BOS, steady-state, and padded-tail tokens; vary duration, noise seed, spectrum, and preceding context before declaring the set stable. [cite: fix_design] [cite: repro_README]

**2.3 −60 dBFS is defensible only as a calibrated operating point.** The design attributes that level to measured generated silence/Common Voice background noise, but the supplied client records onset, duration, and sample rate—not silence RMS, its distribution, or noise provenance. Neither "inaudible" nor "copied reference background" follows from this evidence. Digital full-scale level does not specify playback sound pressure. [cite: fix_design] [cite: repro_README] [cite: onset_client]

**2.4 Justify the level empirically, not rhetorically.** Label genuine generated non-speech and weak initial speech; measure their RMS distributions with specified windows and preprocessing. Sweep candidate probe levels—for example −100, −80, −70, −60, −50, −40 dBFS—and select the set/level on a latency-versus-speech-exclusion frontier using separate validation data. Generate precisely RMS-normalized noise, disable unintended AGC/normalization, and preserve production resampling/quantization; −60 dBFS **RMS** is not a −60 dBFS peak bound. The current −60 choice is a hypothesis for that sweep, not an established upper bound. [cite: fix_design]

**2.5 Measure overlap in both directions.** Publish `S`, per-ID counts, generated-silence coverage $\Pr(c_0\in S\mid \text{silence})$, speech exclusion $\Pr(c_0\in S\mid \text{speech})$, and context-stratified $\Pr(\text{silence}\mid c_0=i)$. Include whispered/breathy speech, unvoiced consonants, quiet gain-scaled speech, and frames spanning silence-to-phoneme transitions; an 80 ms frame can contain both. Prefer a conservative high-precision subset or finite penalties when overlap exists; adding IDs by union trades false negatives for potentially destructive false positives. The design already anticipates suppressing legitimate weak onsets but supplies no overlap measurements. [cite: fix_design]

**2.6 "Derived per checkpoint" does not solve drift.** Pin and hash codec encoder/decoder weights, configuration, preprocessing, dtype, probe generation, and resulting `S`; use a separate fixed probe RNG and verify replica agreement. Revalidate token semantics for each talker/checkpoint pair even when both talkers share a tokenizer; regenerate deterministically when the codec changes, and refuse silent policy expansion. The prior art reports 12 IDs for one checkpoint, explicitly leaves other checkpoints untested, and does not validate the proposed expanded noise-derived union. [cite: prior_art_pr_description +1]

### 3. Evaluation: onset + corpus WER + global SIM is insufficient

**3.1 Replace the current acceptance protocol with the following gates; retain its original onset metric only for comparability.** [cite: fix_design]

| Gap | Required measurement/control |
|---|---|
| Meaningful speech onset | Report original peak/RMS metrics, independently labeled first phonetic activity, and sustained-speech onset; include p50/p90/p95/p99, `>80`/`>160 ms`, and non-crossing/empty-output rates. Sweep detector thresholds and inspect disagreements. Detect an early click/breath followed by the original pause; do not reward merely relocating silence after frame 0 or after $N$. |
| Initial-phoneme damage and clicks | Score first-word and initial-phone errors separately from corpus WER; inspect initial consonant duration, breath/aspiration, attack envelope, start discontinuities, transient noise, and amplitude clipping. Use blinded, loudness-matched paired listening to the first second and the complete utterance; forced alignment or unchanged ASR text alone cannot establish intact phonetics. |
| Prosody and identity | Measure rate, durations, pauses, pitch/energy trajectories, emphasis, and breath retention after onset alignment. Report WavLM SIM with fixed preprocessing, plus a speech-region-matched diagnostic to expose changes caused merely by removing silence. Treat UTMOS as supplementary, not an artifact or prosody acceptance gate. |
| Length and language | Add very short outputs, phoneme-balanced beginnings, numbers/names, long paragraphs, and 30–120 s stress utterances; score completion, repetition, omissions, and termination. Evaluate non-English and cross-lingual cloning, explicit language versus `auto`, accented/noisy references, and whispered/breathy delivery; the supplied onset prompts are exclusively short English examples. [cite: onset_client] |
| Streaming and compute | Measure request-to-first-byte, request-to-first-meaningful-speech at client playout, chunk arrival times, buffering, underruns, and boundary artifacts across chunk sizes and concurrency. Record actual generated frame count, GPU/wall time, RTF, and throughput; eliminating $k$ 80 ms audio frames does not imply saving $80k$ ms of GPU time. Verify streaming/non-streaming codec sequences instead of assuming identity. [cite: repro_README +1] |
| Correct benchmark mode | Log resolved clone mode and mask applications in every run. The repro explicitly sends `x_vector_only_mode`; external benchmark payload code shows `--no-ref-text` merely omits transcript fields, leaving server inference of mode to be verified. Otherwise the quality benchmark can test the wrong path. [cite: onset_client] [cite: GitHub] |
| Selection and controls | Run matched `(reference, text, seed)` comparisons, repeat baseline runs, randomize execution order, and separate parameter-selection data from held-out confirmation. Test both checkpoints independently; include unchanged ICL/CustomVoice/VoiceDesign controls and hard-mask versus finite-penalty versus conservative-trim comparisons. Six $N$ settings × two checkpoints × 360 xvec cases already imply **4,320 onset requests**, excluding quality and robustness tests. [cite: fix_design +1] |
| Artifact integrity | Include model/codec revisions, `N`, `S` hash, sampler settings, and preprocessing in run identity and records. The client's resume key is only \`mode *(cut off here in ChatGPT's own rendering; see transcription notes)* |

**3.2 "Within baseline noise" is not a statistical criterion.** Predeclare non-inferiority margins and require the upper confidence bound on $\Delta\mathrm{WER}$ below its margin, the lower bound on $\Delta\mathrm{SIM}$ above its negative margin, and the artifact-rate upper bound below a separate risk budget. Failure to reject equality is not evidence of equivalence; one fixed-seed run cannot estimate seed variability. Bootstrap paired corpus error-count/word-count ratios and SIM differences while respecting reference/speaker and prompt clustering. [cite: fix_design]

**3.3 Count independent conditions, not merely requests.** Each reported cell contains 360 outputs but only three references and 12 prompts; broaden speakers and texts before multiplying seeds. Under an optimistic independent-Bernoulli model, zero observed defects in 360 tests leaves a one-sided 95% upper defect-rate bound of **0.829%**; establishing a bound below **0.1%** with zero defects requires **2,995 independent tests**. Clustering weakens those guarantees, and the current repro did not score onset defects at all. [cite: repro_README +1]

### 4. Default: off, even with flat aggregate WER/SIM

**4.1 Set general-purpose default `N=0`.** Flat WER/SIM does not establish preserved initial phonetics, prosody, intended hesitation, or multilingual behavior; this changes a model's output policy rather than implementing a distribution-preserving performance optimization. The prior art also defaults off and explicitly characterizes the change as behavioral. [cite: fix_design] [cite: prior_art_pr_description]

**4.2 Permit default-on only inside an explicit, versioned low-latency policy after the additional gates pass.** Select the smallest $N$ meeting a predefined onset target and quality constraints on held-out data, separately per checkpoint—not merely the apparent plateau. Provide request-level opt-out or a separate unmodified endpoint; an engine-only toggle is adequate for opt-in deployment, not transparent per-request choice under a shared default-on service. [cite: fix_design +1]

### 5. Repro-analysis corrections

**5.1 The −34 dBFS arithmetic is correct; the interpretation is not.** $20\log_{10}(0.02)=-33.9794$ dBFS, but the detector measures first threshold-crossing peak, not speech onset. It rewards transient noise and strong attacks while delaying weak consonants; its returned 5 ms bin start can precede the crossing by almost 5 ms, and the RMS detector reports a 10 ms window's start. [cite: onset_client]

**Executed counterexample using the supplied functions:** a 24 kHz waveform with unchanged speech beginning at 450 ms reports **450 ms peak / 445 ms RMS**; adding one `0.2`-full-scale sample at time zero changes both reports to **0 / 0 ms**. That sample's 10 ms RMS is `0.01291 > 0.01`, so even agreement between both detectors does not establish earlier speech. The reported ≤25 ms difference between aggregate medians also says nothing about per-utterance disagreements or tails. [cite: onset_client] [cite: repro_README]

**5.2 "Reference-independent" is unsupported on the report's own metric.** For 1.7B xvec, reference-specific means are **336 / 478 / 475 ms**, and `>160 ms` rates are **74% / 98% / 98%**. Similar-looking medians of 400–485 ms do not establish independence; some observed dependence could also reflect reference-conditioned loudness affecting the detector. [cite: repro_README] [cite: summary_peak]

**5.3 "Mainly seed" overstates the variance decomposition.** Within-reference×prompt variability includes seed effects, interactions, and unmeasured execution variability; variance of ten-seed cell means still contains sampling noise and combines reference, prompt, and their interaction. The 0.6B prompt medians also range from **40/65 ms** for two prompts to **455/468 ms** for others. Repeat identical seeds under controlled batching and use stratified/two-part models for immediate versus delayed onset; do not treat nondeterministic concurrent execution as proven harmless. [cite: summary_peak +1] [cite: repro_README]

**5.4 The histogram supports a descriptive pattern, not a first-token causal mechanism.**

| Xvec checkpoint | Bin 0 | Bins 1–3 | Bins 4–7 | Bins ≥8 |
|---|---|---|---|---|
| 1.7B | 34/360 = **9.4%** | 13/360 = **3.6%** | 287/360 = **79.7%** | 26/360 = **7.2%** |
| 0.6B | 81/360 = **22.5%** | 59/360 = **16.4%** | 195/360 = **54.2%** | 25/360 = **6.9%** |

These are threshold-onset bins, not observed silent-token runs; "bins 1–3 are rare" fits 1.7B much better than 0.6B. Bin 0 means onset in `[0,80)` ms, not literally at sample zero. [cite: onset_frame_histogram]

The pattern argues against a simple homogeneous memoryless onset model, but does not distinguish first-token causation from time-dependent onset hazards, full-frame acoustic choices, prompt mixtures, or detector effects. External tokenizer configuration confirms **1,920/24,000 = 80 ms = 12.5 Hz**; the "12Hz" model name does not make the report's 80 ms bins erroneous. [cite: repro_README] [cite: Hugging Face]

**5.5 ICL association is not demonstrated pause imitation.** Three different references jointly change speaker, transcript, phonetic ending, duration, level, and trailing pause; **135/170/300 ms → 105/342/755 ms** is neither an isolated intervention nor one-to-one copying. Trim/pad the same reference at controlled durations while holding its spoken content, transcript, and speaker embedding fixed; audit reference-code/PCM crop boundaries to distinguish generated delay from retained reference audio or decoder context. Keep the supported conclusion—ICL is not uniformly faster or more stable—and withdraw the claim that its delay is therefore justified prosody. [cite: repro_README] [cite: fix_design]

**5.6 The prior-art scope is misreported.** The README says the reference implementation restricts the fix to xvec, but the supplied PR snippet has no mode gate and its closing note offers xvec scoping as a future change. "ICL byte-identical" on tested cases is an observation, not isolation by construction. [cite: repro_README] [cite: prior_art_pr_description +2]

**5.7 Prior-art evidence does not validate $N=1$, portability, or retraction.** It evaluates $N=3$, specifies eight prompts × ten seeds, omits one control's baseline, leaves other checkpoints and eager execution untested, and explicitly failed to induce preemption. Treat it as a feasibility result, not an acceptance test. [cite: prior_art_pr_description +1]

**5.8 The attached evidence cannot independently verify several advertised findings.** The 1,440 total consists of **360 outputs per model×mode**, not 1,440 xvec observations. Raw WAVs/JSONL, `summarize.py`, silence-level measurements, reference-tail measurement code, and per-utterance RMS comparisons are absent from these seven attachments; the README points to a private archived revision instead. The supplied summaries establish reported aggregate threshold-onset behavior, not token causality, perceptual silence labels, or noise provenance. [cite: repro_README +1]
