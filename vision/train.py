"""Train the thermal-image classifier (healthy vs faulty bearing) and export it to ONNX.

Run from the repo root (with .venv active, torch + torchvision + onnx installed):
    python vision/train.py

Model: MobileNetV3-Small (ImageNet-pretrained, ~1.5 M parameters), fine-tuned.
Evaluation: leave-one-load-out. Images of the same load are near-duplicates of each other, so a
random split would inflate accuracy. We train on two loads and test on the third, three times.

Outputs: vision/models/{thermal.onnx, meta.json, metrics.json}
         vision/demo_images/  (small neutral-named sample set used by the camera simulator)
"""
import csv
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torchvision.transforms import v2

MANIFEST = Path("data/processed/thermal_manifest.csv")
OUT = Path("vision/models")
DEMO = Path("vision/demo_images")
SIZE = 224
LABELS = ["healthy", "faulty"]
EPOCHS = int(os.environ.get("VISION_EPOCHS", "8"))
PRETRAINED = os.environ.get("VISION_PRETRAINED", "1") == "1"
DEVICE = torch.device(os.environ.get("VISION_DEVICE", "cpu"))
LOADS = ["no", "medium", "full"]
DEMO_PER_CONDITION = 10
MASK_TOP = float(os.environ.get("VISION_MASK_TOP", "0"))   # fraction of image height blanked (camera text overlay)

AUGMENT = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomResizedCrop(SIZE, scale=(0.8, 1.0), antialias=True),
    v2.ColorJitter(brightness=0.1, contrast=0.1),   # keep it mild: colour encodes temperature
])


def seed_all(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def prepare(img):
    """Blank the top MASK_TOP of the frame (FLIR draws a temperature readout and logo there), then resize.
    The same step runs in the service, so training and runtime see identical inputs."""
    img = img.convert("RGB")
    if MASK_TOP > 0:
        a = np.array(img); a[: int(a.shape[0] * MASK_TOP)] = 0; img = Image.fromarray(a)
    return img.resize((SIZE, SIZE), Image.BILINEAR)


def load_images(rows):
    imgs = [np.asarray(prepare(Image.open(r["path"]))) for r in rows]
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()      # uint8 N,3,H,W
    y = torch.tensor([LABELS.index(r["label"]) for r in rows])
    return x, y


class Net(nn.Module):
    """Takes float images in 0..1 (N,3,224,224) and returns class probabilities.
    Normalisation lives inside the model so the runtime only has to resize and divide by 255."""
    def __init__(self):
        super().__init__()
        w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if PRETRAINED else None
        self.backbone = models.mobilenet_v3_small(weights=w)
        self.backbone.classifier[3] = nn.Linear(self.backbone.classifier[3].in_features, 2)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def logits(self, x):
        return self.backbone((x - self.mean) / self.std)

    def forward(self, x):
        return F.softmax(self.logits(x), dim=1)


def train_model(x, y):
    seed_all()
    model = Net().to(DEVICE)
    counts = torch.bincount(y, minlength=2).float()
    weight = (counts.sum() / (2 * counts)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = EPOCHS * ((len(x) + 31) // 32)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), 32):
            idx = perm[i:i + 32]
            xb = torch.stack([AUGMENT(img) for img in x[idx]]).float().div(255).to(DEVICE)
            loss = F.cross_entropy(model.logits(xb), y[idx].to(DEVICE), weight=weight)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total += loss.item() * len(idx)
        print(f"    epoch {epoch + 1}/{EPOCHS} loss {total / len(x):.4f}")
    recalibrate_bn(model, x)
    return model.eval()


@torch.no_grad()
def recalibrate_bn(model, x):
    """Recompute BatchNorm running statistics over the training set (no gradient steps).
    Fine-tuning on a few hundred images leaves the running averages out of step with the weights,
    which made eval-mode predictions differ from train-mode ones in testing."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats(); m.momentum = None      # None = plain cumulative average
    model.train()
    for i in range(0, len(x), 32):
        model(x[i:i + 32].float().div(255).to(DEVICE))
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.1


@torch.no_grad()
def predict(model, x):
    out = []
    for i in range(0, len(x), 64):
        out.append(model(x[i:i + 64].float().div(255).to(DEVICE)).cpu())
    return torch.cat(out)


def score(y, prob):
    pred = prob.argmax(1)
    cm = [[int(((y == t) & (pred == p)).sum()) for p in range(2)] for t in range(2)]
    n = len(y)
    return {"n": n, "accuracy": round(float((pred == y).float().mean()), 3),
            "healthy_recall": round(cm[0][0] / max(sum(cm[0]), 1), 3),
            "faulty_recall": round(cm[1][1] / max(sum(cm[1]), 1), 3),
            "confusion[true healthy,true faulty][pred healthy,pred faulty]": cm}


def export_onnx(model, path):
    import onnx
    model = model.cpu().eval()
    dummy = torch.rand(1, 3, SIZE, SIZE)
    torch.onnx.export(model, dummy, str(path), input_names=["image"], output_names=["probs"],
                      dynamic_axes={"image": {0: "batch"}, "probs": {0: "batch"}},
                      opset_version=17, dynamo=False)
    m = onnx.load(str(path))
    m.ir_version = 9           # older onnxruntime (the container's) rejects newer IR versions
    onnx.save(m, str(path))


def write_demo_images(rows):
    """A small, neutral-named sample set so the repo runs without the raw dataset."""
    if DEMO.exists():
        shutil.rmtree(DEMO)
    DEMO.mkdir(parents=True)
    index, n = [], 0
    for cond in sorted({r["condition"] for r in rows}):
        group = [r for r in rows if r["condition"] == cond]
        picks = [group[int(i)] for i in np.linspace(0, len(group) - 1, min(DEMO_PER_CONDITION, len(group)))]
        for r in picks:
            name = f"img_{n:04d}.jpg"; n += 1
            Image.open(r["path"]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR).save(DEMO / name, quality=90)
            index.append({"file": name, "label": r["label"], "load": r["load"]})
    with (DEMO / "index.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "load"]); w.writeheader(); w.writerows(index)
    print(f"wrote {len(index)} demo images to {DEMO}")


def main():
    rows = list(csv.DictReader(MANIFEST.open()))
    print(f"{len(rows)} images; pretrained={PRETRAINED}, mask_top={MASK_TOP}, epochs={EPOCHS}, device={DEVICE}")
    x, y = load_images(rows)
    loads = np.array([r["load"] for r in rows])

    report = {"leave_one_load_out": {}}
    all_true, all_prob = [], []
    for held in LOADS:
        te = torch.from_numpy(loads == held)
        print(f"\nHold out load '{held}': train on {int((~te).sum())}, test on {int(te.sum())}")
        model = train_model(x[~te], y[~te])
        prob = predict(model, x[te])
        report["leave_one_load_out"][held] = score(y[te], prob)
        print("  ->", report["leave_one_load_out"][held])
        all_true.append(y[te]); all_prob.append(prob)
    report["pooled_held_out"] = score(torch.cat(all_true), torch.cat(all_prob))
    print("\nPooled held-out:", report["pooled_held_out"])

    print("\nFinal model on all images")
    model = train_model(x, y)
    OUT.mkdir(parents=True, exist_ok=True)
    export_onnx(model, OUT / "thermal.onnx")
    (OUT / "meta.json").write_text(json.dumps({"labels": LABELS, "size": SIZE, "mask_top": MASK_TOP, "input": "float32 0..1 NCHW RGB"}))
    report["note"] = "leave-one-load-out numbers are the honest estimate; the exported model uses all images"
    (OUT / "metrics.json").write_text(json.dumps(report, indent=1))

    # sanity: ONNX must reproduce the torch output
    import onnxruntime as ort
    sess = ort.InferenceSession(str(OUT / "thermal.onnx"), providers=["CPUExecutionProvider"])
    sample = x[:8].float().div(255)
    diff = np.abs(sess.run(None, {"image": sample.numpy()})[0] - model.cpu()(sample).detach().numpy()).max()
    print(f"ONNX vs torch max difference: {diff:.2e}")
    write_demo_images(rows)
    print("saved", OUT / "thermal.onnx")


if __name__ == "__main__":
    main()