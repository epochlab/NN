# NN

*Design overview: a computational and cognitive neuroscience framework for a 3D embodied agent*

## 1. The loop

The architecture is organised as a continuous, recurrent loop rather than a single forward pass. Environment feeds Vision, Vision feeds Processing, Processing feeds Decision, Decision feeds Execution, and execution updates the Environment again. Every component described below sits somewhere on this loop.

## 2. Full architecture overview

- Environment
  - → Retina → Visual cortex → Neocortex → Basal ganglia → Motor output
  - → Reward → Basal ganglia
- Motor output → updates Environment (closes the loop) and Hippocampus (writes new memory)

Six anatomically-grounded stages carry the main loop. The retina samples the scene; the visual cortex extracts features; the neocortex reasons over them; the basal ganglia learns action values and filters between alternatives; motor output selects the action that updates the environment.

A direct reward channel from the environment to the basal ganglia was a gap identified during design and is now included explicitly. Without it, the basal ganglia has nothing to learn from.

## 3. Component reference

| Stage | Computational mechanism | Biological analogue |
|---|---|---|
| Retina | Ray-cast tiered-colour + depth sampling (`docs/agent.md` §3) + short frame buffer (2–4 frames) | Photoreceptor sampling; direction-selective retinal ganglion cells for motion |
| Visual cortex | CNN feature extraction (Sobel now lives upstream as a `docs/agent.md` §3 retina tier, not an alternative here) | Hierarchical visual cortex (V1–V4), edge and feature detection |
| Hippocampus | Hopfield / attractor network; Hebbian (outer-product) storage | CA3 pattern completion and separation; episodic memory |
| Proprioception | Encoded body-state vector (position, velocity) via a small MLP | Somatosensory cortex, projecting into association areas |
| Neocortex | Transformer (pretrained), plus a sliding context window over recent per-step encodings | Association cortex; persistent activity underlying working memory |
| Dopaminergic gate | Scalar or vector weighting of the cortical signal, driven by TD-error | SNc / VTA dopamine modulating corticostriatal synapses |
| Basal ganglia | DQN / PPO with prioritised experience replay; cross-inhibition decision filter | Striatum; direct / indirect pathway action selection |
| Motor output | Ring attractor | Head-direction / oculomotor ring attractor circuits |
| Reward channel | Scalar reward from the environment | Primary reward signal reaching dopaminergic nuclei |

## 4. Weights and training regimes

Three distinct learning mechanisms operate side by side in this architecture, not two:

| Component | Weight origin |
|---|---|
| Visual cortex (CNN) | Pretrained |
| Neocortex (transformer) | Pretrained |
| Hippocampus (Hopfield network) | Neither: stored via a Hebbian / outer-product rule, not gradient descent |
| Basal ganglia (RL policy / value) | Learned online, from experience in the 3D environment |

## 5. Open questions

Parked against the section that owns them, to be resolved when that mechanism is specified in detail:

| Section | Open question |
|---|---|
| Hippocampus | Does encoding (writing new memories) happen on every step, or is it gated by novelty / salience? |
| Dopaminergic gate | Scalar (single trust value) or vector (per-feature, attention-like) weighting? |
| Neocortex | Early fusion (concatenate memory into the token stream) or cross-attention (memory as a separate key/value stream)? |
| Basal ganglia | Exploration mechanism: decoupled epsilon-greedy, δ-modulated, or a separate noradrenergic-style gate? |
| Basal ganglia | Model-free RL only, or augmented with MCTS-based lookahead for action planning at decision time? |
| Efference copy / forward model | Should a forward model be added to predict the sensory consequences of the agent's own actions (corollary discharge), so self-caused sensory change can be distinguished from externally caused change? |

## 6. Future considerations

| Consideration | Note |
|---|---|
| Mushroom body / Kenyon cells | Insect mushroom-body architecture (sparse Kenyon-cell coding fanning onto a small number of output neurons) is a candidate alternative/complement to the Hopfield hippocampus (§3) for associative memory. It offers sparse-coding pattern separation instead of Hopfield's dense attractor dynamics. Not yet integrated; flagged for future comparison. |
| Place cells via the Hopfield network | The Hopfield hippocampus (§3) could additionally encode spatial position, i.e. place-cell-like attractors keyed on location. This would be a second way of representing space, alongside (not replacing) the ring attractor's heading representation (§3, and `docs/agent.md` §2's "Agent spatial state"). Whether the two spatial codes should be unified, kept separate, or one subsumes the other is open. |

## 7. References

- LeCun, Y. et al. (1989). Backpropagation applied to handwritten zip code recognition.
- Vaswani, A. et al. (2017). Attention is all you need. arXiv:1706.03762.
- Horgan, D. et al. (2020). Distributed prioritised experience replay. arXiv:2003.13350.
- Cross-inhibition, value-based multi-alternative decision making: worldscientific.com/doi/abs/10.1142/S0219493701000102.
- Phenomenal interface theory: a model for basal ganglia function. Philosophical Transactions of the Royal Society B, 380(1939).
- Ring attractor networks for action/state representation: arXiv:2410.03119.
- Montague, P.R., Dayan, P. & Sejnowski, T.J. (1996); Schultz, W., Dayan, P. & Montague, P.R. (1997). Foundational work establishing dopamine as a reward-prediction-error signal.
- Frank, M.J. (2005). Computational account of the basal ganglia's D1 / D2, direct / indirect pathway architecture.
- Kakade, S. & Dayan, P. (2002). Dopamine: generalisation and bonuses. Neural Networks.
- Aston-Jones, G. & Cohen, J.D. (2005). An integrative theory of locus coeruleus-norepinephrine function: adaptive gain and optimal performance. Annual Review of Neuroscience.
- Behrens, T.E.J. et al. (2007). Learning the value of information in an uncertain world. Nature Neuroscience.
- O'Keefe, J., Dostrovsky, J. (1971). The hippocampus as a spatial map: preliminary evidence from unit activity in the freely-moving rat. Brain Research.
- Caron, S.J.C., Ruta, V., Abbott, L.F., Axel, R. (2013). Random convergence of olfactory inputs in the Drosophila mushroom body. Nature.
