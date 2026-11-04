"""Everything that isn't CUDA: get a pretrained ResNet-20 and dump its data.

  pip install -r requirements.txt
  python3 prepare.py                                 # export to data/ (gitignored)
  python3 prepare.py --eval data/thresholds.txt      # accuracy with INT8 inputs

Binary files: int32 ndim, int32 dims, then raw float32.
thresholds.txt: one T per line per layer, 0 to leave that layer unquantized.
"""
import argparse
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

URL = ("https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/"
       "pretrained_models/resnet20-12fca82f.th")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # what it was trained with


class Block(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.pad = c_out // 4 if stride != 1 or c_in != c_out else 0

    def forward(self, x):
        y = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        if self.pad:
            x = F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.pad, self.pad))
        return F.relu(y + x)


class ResNet20(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self.stage(16, 16, 1)
        self.layer2 = self.stage(16, 32, 2)
        self.layer3 = self.stage(32, 64, 2)
        self.linear = nn.Linear(64, 10)

    @staticmethod
    def stage(c_in, c_out, stride):
        return nn.Sequential(Block(c_in, c_out, stride), Block(c_out, c_out, 1),
                             Block(c_out, c_out, 1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.linear(F.avg_pool2d(x, x.size(3)).flatten(1))


@torch.no_grad()
def fold(conv, bn):
    """Lesson 2: w' = w * gamma / sigma, b' = beta - gamma * mu / sigma."""
    gain = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight.mul_(gain.view(-1, 1, 1, 1))
    conv.bias = nn.Parameter(bn.bias - gain * bn.running_mean)


def load_model(path, device):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    model = ResNet20()
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})

    fold(model.conv1, model.bn1)
    model.bn1 = nn.Identity()
    for m in model.modules():
        if isinstance(m, Block):
            fold(m.conv1, m.bn1)
            fold(m.conv2, m.bn2)
            m.bn1 = m.bn2 = nn.Identity()
    return model.to(device).eval()


def layers_of(model):
    return [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def write_tensor(path, array):
    with open(path, "wb") as f:
        np.array([array.ndim, *array.shape], dtype="<i4").tofile(f)
        array.astype("<f4").tofile(f)


def accuracy(model, data, device):
    loader = torch.utils.data.DataLoader(data, 512)
    with torch.no_grad():
        correct = sum((model(x.to(device)).argmax(1).cpu() == y).sum().item()
                      for x, y in loader)
    return 100 * correct / len(data)


parser = argparse.ArgumentParser()
parser.add_argument("--dir", default=str(Path(__file__).resolve().parent / "data"))
parser.add_argument("--images", type=int, default=512)
parser.add_argument("--eval", help="thresholds.txt to evaluate instead of exporting")
args = parser.parse_args()

root = Path(args.dir).expanduser()
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
dataset = lambda train: torchvision.datasets.CIFAR10(root / "cifar10", train, transform,
                                                     download=True)
model = load_model(root / "resnet20.th", device)
layers = layers_of(model)
test_set = dataset(False)

if args.eval:
    thresholds = [float(v) for v in Path(args.eval).expanduser().read_text().split()]
    for layer, t in zip(layers, thresholds):
        if t > 0:
            s = t / 127
            layer.register_forward_pre_hook(
                lambda _m, x, s=s: (torch.clamp(torch.round(x[0] / s), -127, 127) * s,))
    print(f"accuracy with INT8 activations: {accuracy(model, test_set, device):.2f}%")
else:
    print(f"FP32 accuracy: {accuracy(model, test_set, device):.2f}% (expect about 91.73)")

    order = np.random.default_rng(0).permutation(50000)[:args.images]
    subset = torch.utils.data.Subset(dataset(True), order.tolist())
    inputs = [[] for _ in layers]
    for i, layer in enumerate(layers):
        layer.register_forward_pre_hook(lambda _m, x, i=i: inputs[i].append(x[0].cpu().numpy()))
    with torch.no_grad():
        for x, _ in torch.utils.data.DataLoader(subset, 256):
            model(x.to(device))

    for i, layer in enumerate(layers):
        act = np.concatenate(inputs[i])
        write_tensor(root / f"layer{i:02d}_act.bin", act)
        write_tensor(root / f"layer{i:02d}_w.bin", layer.weight.detach().cpu().numpy())
        print(f"layer {i:02d}: act {act.shape}, weights {tuple(layer.weight.shape)}")
