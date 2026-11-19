# marin-sae

## Purpose

- This project trains sparse autoencoders (SAEs) on one layer of the Marin 32B model.
  - The model is `marin-community/marin-32b-base`.
  - The SAE input is the residual stream after decoder layer 8.
    - Layers are counted from 0.
    - The vector has 5120 values for each token.
- The project trains two SAEs at the same time.
  - The two SAEs have the same size and the same start weights.
  - Only the loss is different.
    - The vanilla SAE learns to rebuild the input with all its latents.
    - The Matryoshka SAE also learns to rebuild the input with the first latents only.
- The goal is to find out if the Matryoshka loss gives better SAEs than the vanilla loss.

## Status

- Done:
  - The code is written and the CPU tests pass.
  - A short run on a real GPU trained both SAEs on real Marin activations.
    - The speed was about 9,500 rows each second.
    - The loss went down for both SAEs.
- Not done:
  - The full run of 300 million tokens did not start.
  - No final results exist.
  - The spliced-in loss and the top examples were not measured on a trained SAE.
- Known problems:
  - The Colab machine was lost during the short run.
    - The cause is not known.
    - The logs show no evaluation, no checkpoint and no Drive upload.
    - The speed of the Drive upload is not measured.
  - See `experiments/layer8-pilot.md` for the full log.

## Terms

- **Latent**: one learned direction of the SAE. The SAE has 65,536 latents.
- **Residual stream**: the vector that each layer reads and writes.
- **Row**: the activation of one token. One row has 5120 values.
- **BOS**: the first token of a document. Its activation is very large, so the code drops it.
- **k**: the average number of latents that are active for one token. The value is 64.
- **FVU**: the part of the variance that the SAE does not explain. A lower value is better.
- **Prefix**: the first N latents of the dictionary. The Matryoshka loss uses four prefixes.
- **Spliced-in loss**: the next-token loss of the model when the SAE output replaces the layer 8 output.

## How the pipeline works

- Step 1: The data stream makes batches of tokens.
  - The code is in `src/marin_sae/data/`.
  - A scheduler picks the source of each document.
    - It keeps the share of tokens of each source close to the weight in the config.
  - A packer joins documents into sequences of 1024 tokens.
    - Each document starts with a BOS token.
- Step 2: The model prefix makes the activations.
  - The code is in `src/marin_sae/model/prefix.py`.
  - The code builds only layers 0 to 8.
    - This needs about 10 GB of GPU memory.
    - The full model needs about 65 GB.
  - A hook reads the output of layer 8.
- Step 3: The code normalizes the activations.
  - The code is in `src/marin_sae/activations/normalisation.py`.
  - One number scales all values.
    - After scaling, the mean norm of a row is the square root of 5120.
  - The code does not scale each neuron by itself.
- Step 4: The activation buffer mixes the rows.
  - The code is in `src/marin_sae/activations/buffer.py`.
  - The buffer holds 1,000,000 rows from many documents.
  - The code shuffles the rows and uses half of them.
  - Then the code refills the buffer and shuffles again.
- Step 5: The SAE trains.
  - The code is in `src/marin_sae/dictionary/` and `src/marin_sae/train/`.
  - Both SAEs use the same batch in each step.
    - The model runs only one time for both SAEs.
  - The SAE keeps the largest latent values of the whole batch.
    - The number kept is 64 times the batch size.
    - This method is called BatchTopK.
  - The decoder adds only the rows of the latents that are active.
  - The AuxK loss gives dead latents a gradient so that they can start again.

## The two losses

- Vanilla:
  - The loss is the FVU of the full reconstruction.
- Matryoshka:
  - The loss is the mean of four FVU values.
  - The four values use the first 2,048, 8,192, 32,768 and 65,536 latents.
  - The code computes the four reconstructions with one pass over the active latents.
- Both losses add the AuxK term with a weight of 1/32.

## Data mix

- The mix is fixed in `configs/layer8_mix_300m.yaml`.
  - The weights are percent of tokens.
- Web text: FineWeb sample, 62.
- Science papers: peS2o 7, arXiv 7.
- Math: FineMath 3-plus 6, Dolmino math 2.
- Question and answer: StackExchange 5.
- Instructions: Flan 4.
- Code: StarCoderData Python 4.
- Encyclopedia: Wikipedia 3.
- The Marin cooldown mix has Nemotron-CC as its web source.
  - That dataset needs access, and the access request was refused.
  - FineWeb replaces it.
  - The other sources follow the shape of the cooldown mix.

## Training settings

- Training length: 300,000,000 tokens.
- Batch size: 4,096 rows.
- Dictionary size: 65,536 latents.
- Learning rate: 0.0002.
  - The first 500 steps warm up.
  - The last 20 percent of the steps decay to zero.
- AuxK: 512 dead latents, weight 1/32.
  - A latent is dead after 5,000,000 rows without firing.
- Precision: weights in float32 and matrix products in bfloat16.

## Evaluation

- In-run evaluation uses 500,000 held-out rows.
  - The code scores each SAE at each of the four prefixes.
  - The code also reports the dead fraction and the mean number of active latents.
- Post-hoc evaluation needs the full model.
  - The code is in `src/marin_sae/posthoc/`.
  - Spliced-in loss: the code replaces the layer 8 output with the SAE output.
    - It compares this loss with the clean loss and with a mean-ablated loss.
  - Top examples: the code shows the best contexts for random latents.

## How to run

- Install the package.
  - Run `uv sync` in this folder.
- Run the tests.
  - Run `uv run pytest`.
  - The tests use only the CPU.
- Train.
  - Run `python -m marin_sae.train --config configs/layer8_mix_300m.yaml`.
  - You need a GPU with 80 GB of memory and the model files on disk.
- Change a setting.
  - Add a flag, for example `--train.lr 0.0001`.
  - Flags override the YAML file.
- Evaluate a finished run.
  - Run `python -m marin_sae.posthoc --config configs/layer8_mix_300m.yaml --posthoc.weights-tar <tar file>`.

## Checkpoints

- Each checkpoint is one tar file.
  - Many small files are slow on Google Drive.
- `resume.tar` has the weights and the optimizer state.
  - Each new checkpoint replaces it.
- `weights_<N>M.tar` has the weights only.
  - The code keeps one every 100 million tokens.
- The copy to the archive folder runs in a background thread.
  - Training does not wait for the copy.

## Design decisions

- The code uses one number to scale the activations.
  - A scale for each neuron would change the geometry that the SAE must learn.
- The code drops BOS rows.
  - The BOS activation is very large and would change the scale.
- The code uses BatchTopK and not a TopK for each token.
  - Some tokens need more latents than others.
- A threshold replaces the batch cut at test time.
  - The code tracks the threshold during training.
- Both SAEs start from the same weights.
  - A difference in the result then comes from the loss only.
- Settings come from a YAML file and typed classes.
  - Unknown keys cause an error.
  - Environment variables are ignored.

## Known limits

- The eval split uses a different seed from the train split.
  - Some documents can appear in both.
  - The sources are much larger than the data that the run reads, so the overlap is small.
- Each document is cut to 4,096 tokens.
- The mix is a stand-in for the real cooldown mix.
  - Do not use the results to make claims about the cooldown mix.
- The run code needs the real model and a GPU.
  - Only the Colab run tests that code.
