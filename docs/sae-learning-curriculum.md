# Learning curriculum: how to tell if a sparse autoencoder is doing well

## How to use this document

- This document is self-contained.
  - You can copy it to any repository or session and start from it.
- The goal is to learn how to judge a sparse autoencoder (SAE).
  - You learn by building small SAEs, breaking them on purpose, and measuring what happens.
- The work uses a small language model.
  - Runs are cheap, so you can make many mistakes.
- A final stage hands the results to a large model (Marin 32B).
- Rules for working through the stages:
  - Write your predictions before you run an experiment.
  - Keep a lab notebook. Use the template at the end.
  - Keep the failures in the notebook. They teach the most.
  - Decide the pass criteria before you see the results.

## Part 1: Background

### What an SAE is

- A language model has an internal vector for each token.
  - We call it an activation.
- Many researchers think that an activation is a sparse sum of directions.
  - Each direction can stand for one idea, for example "the word is a month".
- An SAE tries to find these directions without labels.
  - The encoder maps an activation `x` to a long vector of latent values `f`.
  - The decoder maps `f` back to a rebuilt activation `x_hat`.
  - Most values in `f` must be zero. This is the sparsity rule.
- The two functions in short form:
  - `f = activation(W_enc (x - b_dec) + b_enc)`
  - `x_hat = W_dec f + b_dec`
- Each column of `W_dec` is a **latent direction**.
  - We call the learned directions "latents", not "features".
  - A feature is an idea in the model. A latent is a direction that the SAE found.

### Terms

- **Residual stream**: the vector that each layer reads and writes. We train on it.
- **Dictionary size** (width): the number of latents.
- **L0**: the average number of non-zero latents for one token.
- **k**: the L0 that a TopK SAE forces.
- **FVU** (fraction of variance unexplained): rebuild error divided by the error of always guessing the mean. Lower is better.
- **Spliced-in loss**: the language model's next-token loss when the SAE output replaces the activation.
- **Delta loss**: spliced-in loss minus the normal loss.
- **Loss recovered**: how much of the damage of a crude replacement the SAE fixes. See Part 3.
- **Dead latent**: a latent that never fires in a long window.
- **Dense latent**: a latent that fires on a large share of tokens.
- **Firing frequency**: the share of tokens on which a latent is non-zero.
- **Feature splitting**: one idea becomes several narrow latents in a wider SAE.
- **Feature absorption**: a narrow latent takes tokens from a broad latent, so the broad one has gaps.

### The standard parts of an SAE

- The encoder and decoder matrices.
- A decoder bias `b_dec`. The code subtracts it before the encoder ("pre-encoder bias").
- An encoder bias `b_enc`.
- A sparsity rule. The common choices are:
  - ReLU plus an L1 penalty. This is the oldest rule.
  - TopK. It keeps the `k` largest values for each token.
  - BatchTopK. It keeps the largest values across the whole batch, so L0 can differ by token.
  - JumpReLU. Each latent has its own threshold, and the code trains it with an L0 penalty.
- A unit-norm decoder.
  - The code resets each decoder column to length 1 after every update.
  - It also removes the part of the gradient that is parallel to that column.
- One fixed number that scales all activations before training.
- A learning-rate warmup.
- Optional: AuxK, a second loss that lets dead latents rebuild the error, so they can start again.
- Optional: Matryoshka training. The loss also asks the first N latents alone to rebuild the input, for several N.

### Reference recipes

- **Gemma Scope** (Lieberum et al., arXiv 2408.05147). This paper was read in full.
  - JumpReLU SAEs. L0 penalty with straight-through estimators. Bandwidth 0.001.
  - The sparsity coefficient rises from 0 over the first 10,000 steps.
  - Adam with betas 0 and 0.999. Epsilon 1e-8. Batch size 4,096.
  - Learning rate 0.00007 for every run, chosen with sweeps.
  - Cosine warmup from 0.1 times the rate to the full rate over 1,000 steps.
  - The decoder starts with He-uniform values, then each column gets unit norm.
  - The encoder starts as the transpose of the decoder. The two are not tied after that.
  - Both biases start at 0. JumpReLU thresholds start at 0.001.
  - Activations are scaled by one number so that the mean squared norm is 1.
  - BOS, EOS and padding tokens are excluded. Activations are shuffled in buckets of about 1 million.
  - 4 billion tokens for width 16K, 8 billion for most widths, 16 billion for width 1M.
  - 32-bit precision.
  - Main metric: delta loss on 2,048 sequences of 1,024 tokens, special tokens masked. FVU is second.
- **TopK SAEs** (Gao et al., 2024, arXiv 2406.04093). This is recalled from memory. Check it before you rely on it.
  - TopK sparsity, plus an AuxK loss for dead latents.
  - A learning rate that scales with the inverse square root of the width.

### Seeds, in plain words

- A seed fixes the random start of the weights and the order of the data.
- Two runs with the same recipe and different seeds give two different SAEs.
- Two questions follow:
  - How much do the metrics change between seeds? That is the **noise floor**.
  - Do the same latents appear in both SAEs? That is **latent stability**.
- Without a noise floor, you cannot tell a real improvement from luck.
- I recall a 2025 paper that found SAEs with different seeds learn partly different latents. Look it up before you cite it.

## Part 2: The lab setup

### Model, site and data

- Default model: GPT-2 small. It has 12 layers and 768 values in each activation.
  - Faster option: Pythia-70m, with 6 layers and 512 values.
- Default site: the residual stream after block 6, with blocks counted from 0.
- Default width: 24,576 latents, which is 32 times the activation size.
  - Smaller width for quick tests: 6,144.
- Default sparsity: TopK with k of 64 at first. You will test other values.
- Default data: general web text.
  - Sequence length 1,024 tokens, with a BOS token at the start of each document.
  - Drop BOS activations. BOS has a very large norm.
  - Hold out text for evaluation. Use a different seed or different files.
- Data sources that were readable in a test in November 2026:
  - Web text: `HuggingFaceFW/fineweb`, config `sample-10BT`.
  - Math: `HuggingFaceTB/finemath`, config `finemath-3plus`.
  - Encyclopedia: `wikimedia/wikipedia`, config `20231101.en`.
  - Science papers: `common-pile/arxiv_papers`.
  - Code: `bigcode/starcoderdata`, folder `python`. You must accept its terms on Hugging Face.
  - Templated instructions: `allenai/dolmino-mix-1124`, files `data/flan/*.json.gz`.
  - These datasets may change. Check each one before you depend on it.
- Shuffle the activations.
  - Keep a pool of several hundred thousand rows from many documents.
  - Take each batch at random from the pool.
  - Refill the pool as it empties.

### Compute

- By arithmetic, one 100-million-token run of a 24,576-latent SAE on GPT-2 small costs about 4e16 operations.
  - This is roughly 15 to 60 minutes on a modern data-center GPU, and a few hours on an older one.
  - This is an estimate. Measure the real speed in the first 500 steps.
- Do not cache activations to disk for GPT-2 small unless you need to.
  - Generating them again is cheap.

### Tools

- Build your own small SAE in Stages 1 and 2, in about 100 lines of PyTorch.
  - Building it teaches you what every part does.
- Use an existing library after that, to compare. Examples are SAELens and EleutherAI's sparsify.
  - Library interfaces change. Read the current documentation first.

## Part 3: Metrics

### Training health

- Loss curve for training and for held-out data.
- L0 at test time, compared with the target.
- Dead fraction over time.
- Decoder column norms. They must be 1 if the constraint works.
- Speed in tokens each second, and memory use.

### Reconstruction and faithfulness

- **FVU at matched L0.**
  - Always compare two SAEs at the same L0.
  - Show a curve of FVU against L0, not one point.
- **Spliced-in loss.**
  - Run the model with the SAE output in place of the activation.
  - Compute three losses on the same text:
    - `clean`: the normal model.
    - `ablated`: the activation replaced by its mean over the data.
    - `spliced`: the activation replaced by the SAE output.
  - `loss recovered = (ablated - spliced) / (ablated - clean)`.
  - A value of 1 means a perfect SAE. A value of 0 means no better than the mean.
- **Baselines.**
  - An SAE with random weights and no training.
  - PCA: keep the top components so that the count equals your L0.
  - Report each metric for these baselines. A number only has meaning next to them.
- **Breakdowns.**
  - By position in the sequence.
  - By data source.

### Latent statistics

- Histogram of firing frequency, on a log scale.
- Share of alive, dead and dense latents.
  - You choose the line for "dense". A common choice is more than 10 percent of tokens.
- Near-duplicate latents. For each decoder column, find the largest cosine with another column.

### Reading latents

- Top examples: the contexts where a latent fires most.
- Manual rating.
  - Pick 50 random alive latents.
  - For each one, read the top examples.
  - Give a label: yes (a clear idea), maybe, or no (no idea you can state).
  - Report the share of each label.
- Token specificity.
  - For each latent, compute the share of its activation mass that falls on its single most common token.
  - Many latents with a high share mean the SAE found word identity and not meaning.

### Stability

- **Latent stability between two runs.**
  - For each latent in run A, find the largest cosine with any decoder column in run B.
  - Report the share of latents above 0.9, and the share weighted by firing frequency.
- **Checkpoint stability.**
  - Do the same match between two checkpoints of one run.
  - Latents that still change fast mean the SAE is still learning.

## Part 4: The curriculum

Each stage has a goal, steps, a prediction to write down, and a finish line.
Times are rough, for part-time work.

### Stage 0: Read and set up (2 to 3 days)

- Goal: know the vocabulary and the reference recipes.
- Steps:
  - Read Part 1 of this document.
  - Read the Gemma Scope paper, sections 2 to 4 and Appendix A.
  - Run GPT-2 small, and print one activation shape.
- Finish line: you can explain each term in Part 1 to someone else.

### Stage 1: A toy SAE with known answers (2 to 3 days)

- Goal: see what a perfect answer looks like.
- Steps:
  - Make synthetic data from 50 known random directions in 20 dimensions.
  - Make each data point a sum of 2 or 3 active directions.
  - Train a small SAE on it.
  - Measure recovery. For each true direction, find the best cosine with a decoder column.
- Predict first:
  - How many true directions will the SAE recover with a cosine above 0.95?
- Break it, one change at a time:
  - Remove the decoder norm constraint.
  - Make the width the same as the number of true directions.
  - Make the width 10 times larger.
  - Use too few data points.
  - Add a large constant to all data points, and then remove the pre-encoder bias.
- Finish line: you can say which change hurts the most and why.

### Stage 2: First real SAE on GPT-2 small (3 to 4 days)

- Goal: a baseline that trains without errors.
- Steps:
  - Build the pipeline: tokens, model, hook, scale, shuffled pool, SAE, loop.
  - Scale activations with one number. Pick the number from the first 100,000 rows.
  - Train a TopK SAE of width 24,576 and k of 64 on 100 million tokens.
  - Log the training-health metrics.
- Predict first:
  - What FVU will you reach?
  - What share of latents will be dead at the end?
- Traps from earlier work on a large model:
  - A bug that shows only on a GPU, such as a tensor on the wrong device. CPU tests cannot find it.
  - A crashed run that stays alive because a data library keeps threads running.
  - A test suite that never imports the main entrypoint.
  - Scaling each neuron by its own number. It is non-standard and changes the geometry.
  - A large constant part in the residual stream. It makes the mean almost as large as a typical row.
  - The BOS token has a huge norm.
- Finish line: a run that finishes, with a loss curve that flattens.

### Stage 3: Measure it properly (3 to 4 days)

- Goal: a measurement tool that you trust.
- Steps:
  - Build the evaluation script for Part 3.
  - Compute every metric for your Stage 2 SAE.
  - Compute the same metrics for the random SAE and for PCA.
  - Read the top examples of 30 random latents.
- Predict first:
  - How much better than PCA will your SAE be at the same L0?
  - What loss recovered will you get?
- Finish line: one table with your SAE and the two baselines side by side.

### Stage 4: The noise floor (2 to 3 days)

- Goal: know how big a difference must be to matter.
- Steps:
  - Train 3 seeds of the Stage 2 recipe. Use 5 for any comparison you plan to publish.
  - Report the mean and the spread of each metric.
  - Compute latent stability between each pair of seeds.
- Predict first:
  - What share of latents will match above 0.9 between two seeds?
- Finish line: a rule of thumb, for example "differences under X in FVU are noise".
- Cost note: later stages reuse this rule. Do not skip this stage.

### Stage 5: How much data is enough (3 to 5 days)

- Goal: learn what a low-data SAE looks like and how to spot it.
- Define low data with the **tokens per latent** ratio.
  - Tokens per latent is the number of training tokens divided by the width.
  - Gemma Scope is about 15,000 to 240,000 tokens per latent, from its Table 1.
- Steps:
  - Train at 5, 20, 50, 100 and 300 million tokens, with the same width.
  - Keep the learning rate constant, so the points compare cleanly.
  - Repeat at width 6,144.
  - Use one seed for the ladder, and three at the most important point.
- Predict first:
  - How will the dead fraction change with more data?
  - Will early latents be about single words?
  - How will loss recovered change?
- Look at:
  - FVU against tokens, on a log scale.
  - Dead fraction, loss recovered and top examples at each point.
  - Checkpoint stability between neighboring points.
- You have enough data when all of these flatten:
  - FVU improves by only a small amount for each doubling of tokens. You choose the line.
  - The dead fraction is stable.
  - Most latents match above 0.9 against the previous checkpoint.
- Finish line: a plot and a written answer to "what is low data for this width".

### Stage 6: Bad mixtures (3 to 5 days)

- Goal: learn what a poor data mix does.
- Steps:
  - Train on narrow data: code only, math only, templated instructions only, and web only.
  - Use 20 to 50 million tokens each.
  - Evaluate each SAE on every domain. This gives a matrix of FVU and loss recovered.
  - Train on the full mix minus one source, and compare per-source FVU.
  - Train once with equal weights for all sources and once with weights by tokens.
  - Train once with many repeated documents.
- Predict first:
  - Which cells of the matrix will be bad?
  - Will latents for a missing domain stay dead?
- Look at:
  - Per-source FVU and per-source dead share.
  - For each latent, the source where it fires most.
- A good mix has a small spread of FVU across sources.
- Finish line: a short list of what each kind of bad mix does to the output.

### Stage 7: The parts of a vanilla SAE (5 to 7 days)

- Goal: learn the job of each part by removing it.
- Steps:
  - Change one thing at a time. Use the Stage 2 recipe as the base.
  - Run each change with at least 2 seeds.
  - Predict first, for every row of the table.

| Part | Change | Watch |
|---|---|---|
| Pre-encoder bias | Remove it | FVU, dense latents |
| `b_dec` start | Zeros, or the data mean | Early loss, dead fraction |
| Unit-norm decoder | Remove the norm step, or the gradient step | Decoder norms, FVU |
| Encoder tied to decoder | Tie them for the whole run | FVU, encoder-decoder cosine |
| Sparsity rule | TopK, BatchTopK, JumpReLU, ReLU with L1 | FVU at matched L0 |
| k or L0 | 16, 32, 64, 128, 256 | The full frontier and loss recovered |
| AuxK | Weight 0 and 1/32 | Dead fraction over time |
| Activation scale | One number, none, per dimension, per vector | Loss recovered |
| Shuffling | A tiny pool, and a large pool | Loss curve noise |
| Learning rate | 7e-5, 2e-4, 6e-4 | Final FVU, dead fraction |
| Adam betas | (0, 0.999) and (0.9, 0.999) | Same |
| Width | 6K, 12K, 24K, 49K | The frontier, splitting, absorption |
| Precision | 32-bit and 16-bit | FVU difference |

- Finish line: one page that says what each part does and what breaks without it.

### Stage 8: Newer methods (3 to 5 days)

- Goal: learn what Matryoshka training changes.
- Steps:
  - Train a Matryoshka SAE with nested sizes, for example 1.5K, 6K, 12K and 24K.
  - Train a vanilla SAE of the same width with the same seed.
  - Evaluate both at every nested size.
  - Measure feature splitting and absorption in both.
- Predict first:
  - Will the vanilla SAE reconstruct well from its first 1.5K latents?
  - Will the Matryoshka SAE lose much at full width?
- Finish line: a fair comparison at matched L0, with the noise floor from Stage 4.

### Stage 9: Judge the latents (3 to 4 days)

- Goal: decide if the latents mean something.
- Steps:
  - Rate 50 random alive latents. Use the yes, maybe, no labels.
  - Compute token specificity for all latents.
  - Pick 5 latents and test them with a causal check.
    - Set one latent to zero in the SAE output.
    - Check that the model's output changes as the examples suggest.
  - Optionally, rate the same latents with a language model as a second judge.
- Finish line: a share of "yes" labels, and a list of the kinds of bad latents you saw.

### Stage 10: Scorecard and write-up (2 to 3 days)

- Goal: one test that tells you if an SAE is good enough.
- Steps:
  - Fill in the scorecard below, using the numbers you learned.
  - Apply it to your best SAE and to two bad ones.
  - Check that the scorecard separates them.
  - Write what you learned, including what surprised you.
- Finish line: a scorecard that you would trust on a new SAE.

### Stage 11: Hand-off to a large model (a few days of setup, then runs)

- Goal: test the top results on Marin 32B, layer 8 of the residual stream.
- Facts about that model:
  - 64 layers, 5,120 values in each activation, 27,648 neurons in each MLP.
  - The residual stream at layer 8 has a very large constant part. The mean is almost as large as a typical row.
  - It needs about 10 GB of GPU memory to run layers 0 to 8.
- Plan:
  - Run 3 or 4 large runs only, chosen from what the lab taught you.
  - Use 2 seeds at most.
  - Start with a vanilla SAE and one token budget picked from Stage 5.
- A 100-million-token run on one A100 is about 2 to 3 hours. This is an estimate. A measured run of two SAEs together gave about 9,500 rows each second.

## Part 5: The scorecard

- Decide each pass line before you run the test. Then write it down.
- Items, with the line left for you to choose:

| Check | Pass line |
|---|---|
| FVU against PCA at the same L0 | better by at least ___ |
| Loss recovered | at least ___ |
| Dead fraction | at most ___ |
| Dense latent share | at most ___ |
| Share of "yes" ratings on 50 latents | at least ___ |
| Latent stability between seeds | at least ___ above cosine 0.9 |
| Per-source FVU spread | at most ___ |
| Train versus held-out FVU gap | at most ___ |

- How to fill the blanks, as a plan for you and a helper to do together:
  - Start from what the noise floor and the baselines say is possible.
  - Use the Gemma Scope curves (their Figures 2, 4 and 13) as a reference. Read the values from the figures.
  - Set the lines, then check that a good SAE passes and a bad SAE fails.
  - Change a line only with a written reason.

## Part 6: Decisions to make together

- The "scorecard" in Part 5.
  - It is a list of pass or fail checks that you write before you look at results.
  - It stops you from moving the target after you see the numbers.
- The manual rating in Stages 3 and 9.
  - You read the best examples of a latent and decide if they show one clear idea.
  - A language model can do a second rating to cross-check you.
- How many seeds.
  - 3 for most small-model tests, 5 for results you want to publish.
- Which stages to skip if time is short.
  - Keep Stages 1 to 5 and 7. Skip Stage 8 or 11 first.

## Appendix A: Lab notebook template

- Date and stage:
- Question:
- Setup (model, site, width, sparsity rule, k, data, tokens, seed, learning rate):
- Prediction (write before the run):
- Result (numbers, with units):
- What matched, and what did not:
- What went wrong (errors, surprises, mistakes):
- Next step:

## Appendix B: Prediction table template

| Stage | Question | My prediction | Result | Gap and reason |
|---|---|---|---|---|
| | | | | |

## Appendix C: Short checklist for any new SAE run

- Is the data held out from training?
- Is BOS excluded from the activations?
- Is the activation scale one number for all values?
- Is the pre-encoder bias in use?
- Are the decoder columns unit norm?
- Is the L0 at test time near the target?
- Are you comparing two SAEs at the same L0?
- Did you measure spliced-in loss, and not only FVU?
- Did you run the baselines?
- Did you run at least two seeds?
- Did you write the prediction before the run?
