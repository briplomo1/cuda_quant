# Parallel post-training quantization

```sh
pip install -r requirements.txt
python3 prepare.py --images 4096             # once, locally: writes data/ (gitignored)
./cmake-build-release/calib data/layer01_act.bin --images 512                  # P1, CUDA
mpirun -np 8 ./cmake-build-release/calib_mpi data/layer01_act.bin --images 512 # P2, MPI
```

The pretrained ResNet-20 is from `akamaster/pytorch_resnet_cifar10`
(`resnet20-12fca82f.th`, reported 91.73%). There are 20 layers, 00 to 19. Layer 00's
input is the image, so it is not calibrated; layer 19 is the FC layer. Use a
Release build for timing. Copy only `data/layer*.bin` to the server.

| Path | What |
|---|---|
| `cuda/calib.cu` | P1: Algorithm 1 in CUDA |
| `mpi/calib_mpi.cpp` | P2: Algorithm 1 in MPI |
| `common/calib_common.h` | Shared by P1 and P2: file reading and the one-thread version of every step |
| `cuda/gptq.cu` | Algorithm 2 (in progress) |
| `scripts/run_experiments.sh` | Every Algorithm 1 experiment, on the server; logs to `results/raw` |
| `scripts/summarize.py` | Logs to `results/summary.md`, `results/plots` and `results/thresholds` |
| `scripts/eval_accuracy.ps1` | Accuracy for each thresholds file, locally |
| `docs/algorithm1_breakdown.md` | How each kernel and MPI step works |

## Corrections to the project plan

- **KL zero fix:** set `hist[0] = hist[1]` before scoring, or the ReLU spike pins
  every layer to candidate 128.
- **KL epsilon:** replace only `q == 0` with eps, not `fmaxf(q, eps)`.
- **MSE threshold:** use `T = i * width`. The `+ 0.5` applies to KL only.
- **GPTQ sweep:** `d` is often not a multiple of `B`, so the sweep needs a runtime
  column count or it runs past the end of each row.
- **Layer shapes:** stride is 2 for the first conv of stages 2 and 3, otherwise 1;
  padding is 1. Layer 19 is the FC layer.
- **Weak scaling:** needs up to 4,096 images: `python3 prepare.py --images 4096`.
- **Timing:** the search kernels dominated the histogram on synthetic CPU runs, so
  include them in the speedup runs.

## Optional extension: histogram smoothing

Far-tail bins hold a few values, mostly by chance, so a threshold search landing
there can move with the calibration images. Smoothing turns those spikes and gaps
into a slope. It runs between the histogram and the search:

```
absmax -> histogram -> smooth -> KL, MSE
                    -> percentile (raw histogram)
```

- **Method:** one symmetric convolution, using Gaussian weights with
  sigma = radius / 2. Try sigma = 1, 2 and 4 bins: `calib FILE --radius 2`, `4`
  and `8`. The default, `--radius 0`, is unsmoothed.
- **Bin 0:** never smooth it, and keep it out of bin 1's window. The ReLU spike is
  real, and KL's `hist[0] = hist[1]` fix would otherwise inherit the leak.
- **Edges:** reflect at bins 1 and 2047 so the total doesn't change.
- **Percentile and max:** leave them on the raw data. A percentile reads a running
  total, which already averages out bin noise, and max has no histogram.
- **MPI:** allreduce the raw histogram, then every rank smooths its own copy. This
  adds no communication.

### Experiment

`prepare.py` already shuffles the images, so consecutive images in
`layerNN_act.bin` form a random subset. Export once with `--images 4096` (about
3 GB of activations) and pick disjoint subsets with `--first` and `--images`, for
example `calib data/layer19_act.bin --first 64 --images 32 --radius 4` for the
third subset of 32. Each run ends with a `result:` line holding the settings and
every T.

1. For sizes 32, 64, 128 and 512, run several disjoint subsets of each size, with
   and without smoothing (radius 0, 2, 4, 8).
2. Compare T as a value, not a bin index. Each subset has its own `max`, so bin
   widths differ between subsets.
3. Report the spread of T per layer, and accuracy from `prepare.py --eval`.
4. Keep smoothing out of the speedup runs.

The finding is worth reporting if smoothing makes T more stable at small sizes
without lowering accuracy. Expect the most effect on layer 19, the FC layer, which
gets only 64 values per image: 2,048 values at 32 images, against 524,288 for layer
01.
