# Differences from Gemma Scope

## Purpose

- This document compares our SAE recipe with the recipe of Gemma Scope.
  - Source: Lieberum et al., "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2", arXiv 2408.05147.
  - Page numbers in this document are the page numbers of the paper.
- The document lists facts only.
  - It does not change any code or setting.
- Our run is a pilot.
  - It compares a vanilla SAE with a Matryoshka SAE on layer 8 of Marin 32B.

## What is the same

- The decoder has unit-norm columns after every update.
  - The gradient part that is parallel to each column is removed first (p4).
- The encoder starts as the transpose of the decoder.
  - The two matrices are not tied after that (p4).
- The code subtracts the decoder bias before the encoder (p4).
- The batch size is 4,096 (p4).
- One scalar scales all activations (p3).
- The code does not use BOS tokens as training data (p3).
- The shuffle bucket is about 1,000,000 activations (p3).
  - The authors did not test a perfect shuffle.
- The site is the residual stream after a full block (p3-4).
  - Layers are counted from 0.
- The training data and the evaluation data come from the same type of text (p3, p6).

## What is different

### Sparsity method

- Gemma Scope uses JumpReLU (p2-3).
  - Each latent has its own learned threshold.
  - An L0 penalty is trained with straight-through estimators.
  - The kernel bandwidth is 0.001.
- Our run uses BatchTopK with an average k of 64.
  - The AuxK loss gives dead latents a gradient.
- Gemma Scope sweeps the sparsity coefficient and releases several L0 values for each width (Table 1).
  - Our run has one k, so it has one operating point.
- Gemma Scope raises the sparsity coefficient from 0 over the first 10,000 steps (p4).
  - Our run has no equivalent.
- Gemma Scope uses one threshold for each latent at test time.
  - Our run uses one global threshold.
  - The threshold is a moving average of the batch cut.

### Loss

- Gemma Scope uses the squared error plus the L0 penalty (p3).
- Our run uses the FVU plus the AuxK term.
  - The Matryoshka SAE averages the FVU over four prefixes.

### Optimizer

- Gemma Scope uses a learning rate of 0.00007 for all runs (p4).
  - The authors chose it with sweeps across layers and sites.
- Our run uses 0.0002.
  - We did not run a sweep.
- Gemma Scope warms up with a cosine curve from 0.1 times the rate to the full rate over 1,000 steps (p4).
  - The text does not mention a decay at the end.
- Our run warms up linearly over 500 steps.
  - The rate is flat after that.
  - The rate decays to 0 over the last 20 percent of the steps.
- Gemma Scope uses Adam with betas of 0 and 0.999 (p4).
  - Beta 1 of 0 means there is no momentum.
- Our run uses Adam with betas of 0.9 and 0.999.

### Scale of the activations

- Gemma Scope scales activations so that the mean squared norm is 1 (p3).
  - After training, the authors fold the scale into the weights (p17).
  - Inference then needs no scaling.
- Our run scales activations so that the mean norm is the square root of 5120.
  - This follows Conerly et al., which the paper cites.
  - The scale stays outside the weights.
  - A checkpoint must carry the scale.
- The bandwidth of 0.001 in Gemma Scope assumes the unit scale (p3).
  - A JumpReLU SAE in our setup would need a new bandwidth.

### Start values

- Gemma Scope starts the decoder bias and the encoder bias at 0 (p4).
  - The JumpReLU thresholds start at 0.001.
- Our run starts the decoder bias at the mean of the data.
  - The mean has almost the same norm as a typical row.
  - Our smoke run measured a mean norm of 70.97.

### Precision

- Gemma Scope trains in 32-bit precision (p10).
  - A test shows that bfloat16 inference has a very small effect (p10-11).
- Our run keeps the weights in float32.
  - The matrix products run in bfloat16.
  - The activation buffer holds bfloat16 values.
  - We did not measure the effect of this.

### Amount of training

- Gemma Scope trains on 4B tokens for width 16K (p3, Table 1).
  - It uses 8B tokens for most widths.
  - It uses 16B tokens for width 1M.
- Our run is planned for 300M tokens.
  - This is 13 to 53 times fewer tokens.

### Size and sites

- Gemma Scope has widths from 16K to 1M (Table 1).
  - A separate suite of widths from 16K to 524K studies feature splitting (p8).
- Our run has one width of 65,536.
- Gemma Scope trains three sites on every layer (p3-4).
  - The sites are the attention output before the output projection, the MLP output and the residual stream.
- Our run trains one site on one layer.

### Evaluation

- Gemma Scope uses the change in the language-model loss as the main metric (p6).
  - The authors splice the SAE into the model.
  - FVU is the second metric.
- Our run has the in-run FVU only.
  - The spliced loss is written but was never run.
- Gemma Scope evaluates 2,048 sequences of 1,024 tokens (p6).
- Our run plans 256 sequences for the spliced loss.
  - The in-run FVU uses about 490 sequences.
- Gemma Scope also reports these results:
  - the loss by position in the sequence (p7)
  - the loss by Pile subset (p10)
  - bfloat16 inference (p10)
  - transfer to instruction-tuned models (p9)
  - ratings of how interpretable the latents are (p8-9)
- Our run has none of these.

### Methods that Gemma Scope does not use

- Matryoshka training.
- BatchTopK.
- AuxK.

### Infrastructure

- Gemma Scope uses TPUs (p4).
  - It stores activations on disk and shares them across many training jobs (p5-6).
  - The authors say that storing is cheaper than regenerating when many SAEs share one site.
- Our run generates activations during training on one GPU.
  - This is not a difference in quality.

## What the paper does not say

- The length of the training sequences. The paper gives only the evaluation length.
- The value of the sparsity coefficient for each SAE.
- The number of dead latents.
  - Appendix B (p18) calls JumpReLU, Gated and TopK architectures that prevent dead features.
  - That sentence is a claim about the architecture. It is not a measurement.
- A decay of the learning rate at the end of training.
- The size of the hidden dimension of each Gemma 2 model.

## Open points for our pilot

- Learning rate and Adam betas.
  - Our values differ from the values that Gemma Scope chose with sweeps.
  - We have not tested whether this matters.
- Matched L0.
  - Both of our SAEs target a k of 64.
  - At test time each SAE uses its own threshold, so the L0 values can differ.
  - Compare the FVU of the two SAEs only at the same L0.
- Spliced loss.
  - It is the main metric of Gemma Scope.
  - FVU alone can be misleading.
- Scale of the run.
  - 300M tokens is far less than the Gemma Scope runs.
  - Do not compare absolute values with the paper.
  - Compare the vanilla SAE with the Matryoshka SAE only.
- Scale of the activations.
  - Our scale moves the bias step sizes by about 71 times, compared with the unit scale of Gemma Scope.
  - We expect a small effect with Adam. We have not measured it.
