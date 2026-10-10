 import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    model_name = ckpt["model_name"]
    img_size = ckpt["img_size"]
    idx_to_class = ckpt["idx_to_class"]
    n_classes = len(idx_to_class)

    model = timm.create_model(model_name, pretrained=False, num_classes=n_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, img_size, idx_to_class


@torch.no_grad()
def predict_image(model, image_path, transform, idx_to_class, device):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    outputs = model(x)
    probs = torch.softmax(outputs, dim=1)[0]
    pred_idx = probs.argmax().item()

    pred_class = idx_to_class[str(pred_idx)] if str(pred_idx) in idx_to_class else idx_to_class[pred_idx]
    confidence = probs[pred_idx].item()

    return pred_class, confidence


@torch.no_grad()
def evaluate_folder(model, data_dir, transform, device):
    ds = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    return acc, total, ds.classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    parser.add_argument("--image", help="Single image path")
    parser.add_argument("--data-dir", help="Folder to evaluate, like datasets/.../val")
    parser.add_argument("--device", default="", help="cuda or cpu")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model, img_size, idx_to_class = load_model(args.checkpoint, device)
    transform = build_transform(img_size)

    if args.image:
        pred, conf = predict_image(model, args.image, transform, idx_to_class, device)
        print(f"Predicted action: {pred}")
        print(f"Confidence: {conf:.4f}")

    if args.data_dir:
        acc, total, classes = evaluate_folder(model, args.data_dir, transform, device)
        print(f"Evaluated folder: {args.data_dir}")
        print(f"Classes: {classes}")
        print(f"Images: {total}")
        print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()