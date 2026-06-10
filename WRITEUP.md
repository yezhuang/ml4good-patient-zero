# Misalignment Contagion in Multi-Agent Games — Findings

**Question:** Does a fine-tuned "bad" (power-seeking) agent's behavior spread to a
neutral agent in repeated multi-agent games (Iterated Prisoner's Dilemma, Public Goods)?

## TL;DR
Misaligned **behavior spreads in-game** (neutral defection cascades, ~3σ) — but it's
**game-theoretic reciprocity, not transmission of the specific trait** (a trait-less
"always-defect" script spreads it just as much), and it **does *not* transfer as a
propensity** (the neutral's stated values are unchanged before vs. after, on two
independent eval methods). Public Goods shows the spillover is game-dependent
(agents first *compensate*, then slowly erode).

---

## Findings

**1. In-game contagion is real and builds over rounds.**
With a bad agent present, the neutral's cooperation erodes round-by-round; with no bad
agent it stays cooperative.

![IPD defection by round](results/contagion_lines_postfix_clean_n5-10.png)

**2. ...but it's reciprocity, not the trait.**
Every bad-agent condition lifts neutral defection far above the control (0.11) /
baseline (0.00) floor — and a **trait-less always-defect bot (0.42) ties the
fine-tuned power-seeker (0.39)**. So *any* defector spreads defection equally; the
specific misaligned trait isn't required. (n=5–10; only the bad-vs-reference gap is
robust — rankings among bad configs overlap.)

![IPD contagion by condition](results/contagion_bar_4configs.png)

**3. No *propensity* contagion (free-form trait evals).**
Conditioning each agent on its game transcript and re-measuring power-seeking
propensity: `after_treatment ≈ after_control` for both the neutral and the agreeable
agent (all deltas within noise). The behavioral spillover does **not** become a
shift in stated values.

![Propensity before vs after gameplay](results/evals/FINAL_barplot_powerseeking.png)

**4. No propensity contagion (Anthropic forced-choice eval, position-bias controlled).**
An independent multiple-choice eval (Anthropic `power-seeking-inclination`), debiased
by asking each item in both A/B orderings, gives **treatment − control = +0.027,
95% CI [−0.002, +0.056] — not significant.** Two methods converge on the same null.

![Debiased MC eval](results/evals/neutral_mc_debiased.png)

**5. Public Goods: a different dynamic — compensate, then erode.**
Facing a free-rider, the neutral first *over-contributes* to prop up the public good,
then gradually gives up — the opposite of IPD's immediate reciprocation. No clean
contagion (heterogeneous across games).

![PG free-ride trajectory](results/PG_freeride_trajectory.png)

---

## Comparison to the original misalignment-contagion paper

The original paper reports a stronger effect than we observe here: after thousands
of prompt-elicited multi-agent games, default agents drift toward more anti-social
traits on Anthropic Model-Written Evaluations, and the drift is amplified by
maliciously prompted opponents. In other words, their headline result is
**post-game trait/persona drift**, not just more defecting inside the game.

Our current results separate those two phenomena. We do reproduce something that
*looks* like contagion at the behavioral level: the neutral agent defects much more
after exposure to a bad or defecting player. But our controls suggest that this is
mostly **local strategic adaptation** rather than transmission of the bad agent's
specific misaligned trait. The strongest evidence is that the scripted
always-defect agent, which has no malicious persona or SFT trait, produces roughly
the same neutral defection rate as the fine-tuned power-seeking agent.

We do **not** reproduce the original paper's stronger post-game eval result in the
current SFT setting. On both the free-form trait evals and the debiased Anthropic
MC eval, the neutral's post-game power-seeking score does not reliably increase
relative to control. The closest Anthropic MC signal is small and statistically
fragile: treatment − control = +0.027 with 95% CI [−0.002, +0.056].

So the clean comparison is:

| Claim | Original paper | Our current result |
|---|---|---|
| Agents become more anti-social during/after social-dilemma gameplay | Yes | Yes in-game, especially in IPD |
| Malicious/bad agents amplify the effect | Yes | Yes behaviorally, but always-defect does about as much |
| The effect transfers to post-game Anthropic-style traits | Yes | Not detected |
| Mechanism looks like persona/value drift | Yes / implied | More consistent with reciprocity or strategic adaptation |
| Source of misalignment | Prompt-elicited personas | SFT checkpoints plus controls |

This makes our result a **partial behavioral replication but not a trait-level
replication** of the original paper.

## Conclusions and hypotheses

**Conclusion 1: We should distinguish behavioral contagion from trait contagion.**
The neutral agent clearly changes what it does in the game, but we do not yet see
evidence that it changes what it endorses outside the game. For our setup,
"misalignment contagion" is currently better described as **context-bound
retaliation/reciprocity** than as persistent misalignment transfer.

**Conclusion 2: SFT misalignment may be less contagious than prompt-elicited
misalignment.** A prompt-elicited malicious agent can explicitly advocate deceptive
or anti-social norms in the transcript. A fine-tuned power-seeking agent may mostly
reveal its trait through actions, not through contagious language. That could make
SFT bad agents behaviorally harmful while producing weaker post-game persona drift
in other agents.

**Conclusion 3: The neutral model may be too small or too brittle to show durable
trait update.** The original paper used larger, more capable instruction-tuned
models and many more games. Our Qwen3-8B neutral responds to payoff pressure, but
the sensitivity check suggests the post-game eval channel is weak: even blatant
power/cooperation primes do not move the score cleanly.

**Working hypothesis:** SFT bad agents create **strategic contamination** rather
than **value contamination**. They change the game environment by defecting or
breaking cooperative norms, which causes other agents to defect in response; but
unless the bad agent also transmits an explicit persuasive/persona frame, that
behavior does not generalize to Anthropic-style trait evaluations after the game.

**Next hypothesis test:** Run the same Anthropic MC pre/post evals on (a) the SFT
bad and good checkpoints themselves and (b) paper-faithful prompt-elicited
Default/Benevolent/Malicious games. If prompt-elicited malicious agents reproduce
post-game trait drift while SFT agents do not, then the paper's contagion effect
may depend more on **linguistic persona transmission** than on exposure to a
misaligned policy. If both show null under our setup, the likely explanation shifts
toward model size, game scale, or eval sensitivity.

---

## Caveats
- Small n throughout (n=5–10 games per condition); IPD control floor is ~0.11 (not 0) —
  spontaneous cascades happen occasionally without a bad agent.
- Propensity evals have a ~±5-point single-pass noise floor; effects of that size can't
  be resolved.
- The clean IPD/PG comparison required matching communication (1 chat turn/round) — PG's
  default heavier chat let agents *coordinate* and froze the metric.

## Method notes (rigor that mattered)
A chat-degeneration bug (mute base agents → falsely cooperative games) initially hid the
in-game contagion; the headline result only appeared after fixing it. Each apparent
signal (0.60 → 0.04 → 0.42 → "+0.06 MC") survived a confound that only the right control
exposed (persona-prompt, mute agents, control floor, position bias). The trustworthy
claims are the ones that held up under those controls.
