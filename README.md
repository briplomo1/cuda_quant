# Parallel post-training quantization

```sh
pip install -r requirements.txt              # server: ~/miniforge3/envs/ptq
python3 prepare.py                           # once: writes ~/ptq
./cmake-build-release/calib ~/ptq/layer01_act.bin
```

The pretrained ResNet-20 is from `akamaster/pytorch_resnet_cifar10`
(`resnet20-12fca82f.th`, reported 91.73%). Layer 00's input is the image, so it
is not calibrated. Use a Release build for timing.

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

- **Method:** one symmetric convolution, using Gaussian weights. A moving average
  is the same kernel with equal weights. Try sigma = 1, 2 and 4 bins.
- **Bin 0:** never smooth it, and keep it out of bin 1's window. The ReLU spike is
  real, and KL's `hist[0] = hist[1]` fix would otherwise inherit the leak.
- **Edges:** reflect at bins 1 and 2047 so the total doesn't change.
- **Percentile and max:** leave them on the raw data. A percentile reads a running
  total, which already averages out bin noise, and max has no histogram.
- **MPI:** allreduce the raw histogram, then every rank smooths its own copy. This
  adds no communication.

### Experiment

`prepare.py` already shuffles the images, so consecutive images in
`layerNN_act.bin` form a random subset. Export once with `--images 4096` and split
each file into disjoint subsets.

1. For sizes 32, 64, 128 and 512, run several disjoint subsets of each size, with
   and without smoothing.
2. Compare T as a value, not a bin index. Each subset has its own `max`, so bin
   widths differ between subsets.
3. Report the spread of T per layer, and accuracy from `prepare.py --eval`.
4. Keep smoothing out of the speedup runs.

The finding is worth reporting if smoothing makes T more stable at small sizes
without lowering accuracy. Expect the most effect on layer 19, the FC layer, which
gets only 64 values per image: 2,048 values at 32 images, against 524,288 for layer
01.
