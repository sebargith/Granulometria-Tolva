from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_DATASET = PROJECT_DIR / "outputs" / "patches" / "grid96_batch001_ai_prompt_v2"
DEFAULT_OUTPUT = SCRIPT_DIR / "runs"
DEFAULT_CLASSES = ("fino", "medio", "grueso")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena una CNN ResNet18 RGB para clasificar patches granulometricos. "
            "El split se hace por source_image para evitar fuga entre frames."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        nargs="+",
        default=[DEFAULT_DATASET],
        help="Una o mas carpetas con patches.csv, labels.csv e images/.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="resnet18_rgb_grid96")
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument(
        "--confidence",
        nargs="+",
        default=["alta", "media"],
        help="Niveles de confianza admitidos desde labels.csv.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument(
        "--min-class-holdout",
        type=int,
        default=3,
        help="Minimo de patches por clase exigido en val y test.",
    )
    parser.add_argument(
        "--reuse-split",
        type=Path,
        default=None,
        help="group_split.csv anterior. Los grupos nuevos se asignan a train.",
    )
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--freeze-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--weights",
        default="DEFAULT",
        help=(
            "Pesos torchvision para transfer learning. Usar DEFAULT para ImageNet "
            "o none para inicializar desde cero."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cpu, cuda, cuda:0, etc. Si se omite, usa CUDA si esta disponible.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida dataset y split por frame, pero no entrena ni guarda modelo.",
    )
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=json_default)


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def next_run_dir(output_dir: Path, run_name: str | None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = run_name or "resnet18_rgb"
    candidate = output_dir / base_name
    if candidate.exists():
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = output_dir / f"{base_name}_{suffix}"
    candidate.mkdir(parents=True)
    return candidate


def resolve_patch_file(dataset_dir: Path, value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return dataset_dir / path


def load_dataset(dataset_dir: Path, classes: list[str], confidence: list[str]) -> pd.DataFrame:
    dataset_dir = dataset_dir.expanduser().resolve()
    manifest_path = dataset_dir / "patches.csv"
    labels_path = dataset_dir / "labels.csv"
    if not manifest_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(
            f"El dataset requiere patches.csv y labels.csv: {dataset_dir}"
        )

    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", dtype={"patch_id": str})
    labels = pd.read_csv(labels_path, encoding="utf-8-sig", dtype={"patch_id": str})
    if "confidence" not in labels.columns:
        labels["confidence"] = "alta"
    if "labeled_at" not in labels.columns:
        labels["labeled_at"] = ""

    label_columns = ["patch_id", "label", "confidence", "labeled_at"]
    labels = labels[[column for column in label_columns if column in labels.columns]]
    labels = labels.drop_duplicates("patch_id", keep="last")

    rows = manifest.merge(labels, on="patch_id", how="inner")
    rows["label"] = rows["label"].astype(str).str.strip().str.lower()
    rows["confidence"] = rows["confidence"].astype(str).str.strip().str.lower()
    rows = rows[rows["label"].isin(classes) & rows["confidence"].isin(confidence)].copy()
    rows["dataset_dir"] = str(dataset_dir)
    rows["dataset_name"] = dataset_dir.name
    rows["patch_file"] = rows["patch_path"].map(
        lambda value: str(resolve_patch_file(dataset_dir, str(value)).resolve())
    )
    rows["group"] = rows["source_image"].astype(str)

    missing = [path for path in rows["patch_file"] if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Faltan {len(missing)} patches. Ejemplo: {missing[0]}")
    return rows


def split_score(
    rows: pd.DataFrame,
    assignments: np.ndarray,
    classes: list[str],
    target_fraction: dict[str, float],
    min_class_holdout: int,
) -> float:
    total_distribution = rows["label"].value_counts(normalize=True)
    score = 0.0
    for split in ("train", "val", "test"):
        subset = rows[assignments == split]
        if set(subset["label"]) != set(classes):
            return float("inf")
        if split in {"val", "test"}:
            counts = subset["label"].value_counts()
            if any(counts.get(label, 0) < min_class_holdout for label in classes):
                return float("inf")
        score += abs(len(subset) / len(rows) - target_fraction[split]) * 3.0
        distribution = subset["label"].value_counts(normalize=True)
        score += sum(
            abs(distribution.get(label, 0.0) - total_distribution[label])
            for label in classes
        )
    return score


def create_group_split(
    rows: pd.DataFrame,
    classes: list[str],
    seed: int,
    val_size: float,
    test_size: float,
    min_class_holdout: int,
) -> np.ndarray:
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("--val-size y --test-size deben ser positivos y sumar menos de 1.")

    best_assignments = None
    best_score = float("inf")
    holdout_size = val_size + test_size
    target_fraction = {
        "train": 1.0 - holdout_size,
        "val": val_size,
        "test": test_size,
    }
    for offset in range(1000):
        outer = GroupShuffleSplit(
            n_splits=1, test_size=holdout_size, random_state=seed + offset
        )
        train_indices, holdout_indices = next(
            outer.split(rows, rows["label"], groups=rows["group"])
        )
        holdout = rows.iloc[holdout_indices]
        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size / holdout_size,
            random_state=seed + 10_000 + offset,
        )
        val_relative, test_relative = next(
            inner.split(holdout, holdout["label"], groups=holdout["group"])
        )
        assignments = np.full(len(rows), "train", dtype=object)
        assignments[holdout_indices[val_relative]] = "val"
        assignments[holdout_indices[test_relative]] = "test"
        score = split_score(
            rows,
            assignments,
            classes,
            target_fraction,
            min_class_holdout,
        )
        if score < best_score:
            best_assignments = assignments.copy()
            best_score = score

    if best_assignments is None:
        raise RuntimeError(
            "No fue posible crear train/val/test con todas las clases. "
            "Se necesitan mas frames independientes por clase."
        )
    return best_assignments


def reuse_group_split(rows: pd.DataFrame, split_file: Path) -> np.ndarray:
    previous = pd.read_csv(split_file.expanduser().resolve(), encoding="utf-8-sig")
    if not {"group", "split"}.issubset(previous.columns):
        raise ValueError("--reuse-split debe contener columnas group y split.")
    mapping = previous.drop_duplicates("group").set_index("group")["split"].to_dict()
    return rows["group"].map(mapping).fillna("train").to_numpy(dtype=object)


class PatchDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        class_to_index: dict[str, int],
        transform,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        image = Image.open(row.patch_file).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = self.class_to_index[row.label]
        return image, torch.tensor(target, dtype=torch.long), row.patch_id


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def resolve_weights(weights_name: str):
    if weights_name.lower() in {"none", "false", "no"}:
        return None
    try:
        enum = models.ResNet18_Weights
    except AttributeError:
        return True
    if weights_name.upper() == "DEFAULT":
        return enum.DEFAULT
    return enum[weights_name]


def build_model(num_classes: int, weights_name: str) -> nn.Module:
    weights = resolve_weights(weights_name)
    try:
        model = models.resnet18(weights=weights)
    except TypeError:
        model = models.resnet18(pretrained=bool(weights))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def choose_device(raw_device: str | None) -> torch.device:
    if raw_device:
        return torch.device(raw_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(rows: pd.DataFrame, classes: list[str], device: torch.device) -> torch.Tensor:
    counts = rows["label"].value_counts()
    total = sum(counts.get(label, 0) for label in classes)
    values = [
        total / (len(classes) * max(counts.get(label, 0), 1))
        for label in classes
    ]
    return torch.tensor(values, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
) -> dict:
    model.train()
    losses = []
    y_true = []
    y_pred = []
    for images, targets, _patch_ids in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item() * len(targets))
        predictions = torch.argmax(logits.detach(), dim=1)
        y_true.extend(targets.detach().cpu().numpy().tolist())
        y_pred.extend(predictions.cpu().numpy().tolist())

    return {
        "loss": float(sum(losses) / max(len(loader.dataset), 1)),
        "accuracy": accuracy_score(y_true, y_pred) if y_true else 0.0,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
) -> dict:
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    probabilities = []
    patch_ids = []
    for images, targets, batch_patch_ids in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        probs = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)

        losses.append(loss.item() * len(targets))
        y_true.extend(targets.cpu().numpy().tolist())
        y_pred.extend(predictions.cpu().numpy().tolist())
        probabilities.extend(probs.cpu().numpy().tolist())
        patch_ids.extend(list(batch_patch_ids))

    return {
        "loss": float(sum(losses) / max(len(loader.dataset), 1)),
        "accuracy": accuracy_score(y_true, y_pred) if y_true else 0.0,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0,
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": np.asarray(probabilities, dtype=np.float32),
        "patch_ids": patch_ids,
    }


def make_loader(
    rows: pd.DataFrame,
    class_to_index: dict[str, int],
    transform,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = PatchDataset(rows, class_to_index, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def prediction_table(rows: pd.DataFrame, result: dict, classes: list[str]) -> pd.DataFrame:
    metadata_columns = [
        "patch_id",
        "dataset_name",
        "patch_file",
        "source_image",
        "source_relpath",
        "group",
        "label",
        "confidence",
        "split",
        "x",
        "y",
        "width",
        "height",
    ]
    table = rows[[column for column in metadata_columns if column in rows.columns]].copy()
    table["target_index"] = result["y_true"]
    table["prediction"] = [classes[index] for index in result["y_pred"]]
    for index, label in enumerate(classes):
        table[f"prob_{label}"] = result["probabilities"][:, index]
    return table


def save_confusion_matrix(
    path_csv: Path,
    path_png: Path,
    y_true: list[int],
    y_pred: list[int],
    classes: list[str],
    title: str,
) -> None:
    labels = list(range(len(classes)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        path_csv, encoding="utf-8-sig"
    )
    display = ConfusionMatrixDisplay(matrix, display_labels=classes)
    display.plot(cmap="Blues", colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path_png, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    classes = [label.strip().lower() for label in dict.fromkeys(args.classes)]
    confidence = [label.strip().lower() for label in args.confidence]

    frames = [load_dataset(path, classes, confidence) for path in args.dataset]
    rows = pd.concat(frames, ignore_index=True)
    rows = rows.drop_duplicates(["dataset_dir", "patch_id"], keep="last").reset_index(drop=True)
    if rows.empty:
        raise ValueError("No quedaron patches con las clases y confianza seleccionadas.")

    if args.reuse_split:
        assignments = reuse_group_split(rows, args.reuse_split)
    else:
        assignments = create_group_split(
            rows,
            classes,
            args.seed,
            args.val_size,
            args.test_size,
            args.min_class_holdout,
        )
    rows["split"] = assignments

    for split in ("train", "val", "test"):
        if set(rows.loc[rows["split"] == split, "label"]) != set(classes):
            raise ValueError(f"El split {split} no contiene todas las clases.")

    split_summary = {
        split: {
            "patches": int((rows["split"] == split).sum()),
            "frames": int(rows.loc[rows["split"] == split, "group"].nunique()),
            "classes": dict(Counter(rows.loc[rows["split"] == split, "label"])),
        }
        for split in ("train", "val", "test")
    }
    if args.dry_run:
        print(f"Muestras utiles: {len(rows)}")
        print(f"Frames: {rows['group'].nunique()}")
        print(f"Clases: {dict(Counter(rows['label']))}")
        print(json.dumps(split_summary, indent=2, ensure_ascii=False, default=json_default))
        return

    run_dir = next_run_dir(args.output_dir.expanduser().resolve(), args.run_name)
    device = choose_device(args.device)
    class_to_index = {label: index for index, label in enumerate(classes)}
    index_to_class = {index: label for label, index in class_to_index.items()}

    train_transform, eval_transform = build_transforms(args.image_size)
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    val_rows = rows[rows["split"].eq("val")].reset_index(drop=True)
    test_rows = rows[rows["split"].eq("test")].reset_index(drop=True)

    train_loader = make_loader(
        train_rows,
        class_to_index,
        train_transform,
        args.batch_size,
        args.num_workers,
        shuffle=True,
    )
    val_loader = make_loader(
        val_rows,
        class_to_index,
        eval_transform,
        args.batch_size,
        args.num_workers,
        shuffle=False,
    )
    test_loader = make_loader(
        test_rows,
        class_to_index,
        eval_transform,
        args.batch_size,
        args.num_workers,
        shuffle=False,
    )

    model = build_model(len(classes), args.weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_rows, classes, device))

    print(f"Muestras utiles: {len(rows)}")
    print(f"Frames: {rows['group'].nunique()}")
    print(f"Clases: {dict(Counter(rows['label']))}")
    print(f"Device: {device}")
    print(f"Salida: {run_dir}")

    history = []
    best_state = None
    best_epoch = None
    best_phase = None
    best_val_f1 = -1.0

    phases = [
        ("frozen_head", args.freeze_epochs, args.lr, False),
        ("finetune", args.finetune_epochs, args.finetune_lr, True),
    ]
    global_epoch = 0
    for phase_name, epochs, lr, trainable_backbone in phases:
        if epochs <= 0:
            continue
        set_backbone_trainable(model, trainable_backbone)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=lr,
            weight_decay=args.weight_decay,
        )
        for local_epoch in range(1, epochs + 1):
            global_epoch += 1
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            row = {
                "phase": phase_name,
                "epoch": global_epoch,
                "phase_epoch": local_epoch,
                "lr": lr,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_f1_macro": train_metrics["f1_macro"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
            }
            history.append(row)
            print(
                f"{phase_name} epoch {local_epoch}/{epochs} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_f1={val_metrics['f1_macro']:.3f} "
                f"val_acc={val_metrics['accuracy']:.3f}"
            )
            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = global_epoch
                best_phase = phase_name

    if best_state is None:
        raise RuntimeError("No se entreno ningun epoch. Revisa --freeze-epochs/--finetune-epochs.")
    model.load_state_dict(best_state)

    val_result = evaluate(model, val_loader, criterion, device)
    test_result = evaluate(model, test_loader, criterion, device)
    labels = list(range(len(classes)))
    validation_report = classification_report(
        val_result["y_true"],
        val_result["y_pred"],
        labels=labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    test_report = classification_report(
        test_result["y_true"],
        test_result["y_pred"],
        labels=labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    rows.to_csv(run_dir / "split.csv", index=False, encoding="utf-8-sig")
    rows[["group", "split"]].drop_duplicates().sort_values("group").to_csv(
        run_dir / "group_split.csv", index=False, encoding="utf-8-sig"
    )
    prediction_table(val_rows, val_result, classes).to_csv(
        run_dir / "predictions_val.csv", index=False, encoding="utf-8-sig"
    )
    prediction_table(test_rows, test_result, classes).to_csv(
        run_dir / "predictions_test.csv", index=False, encoding="utf-8-sig"
    )
    save_confusion_matrix(
        run_dir / "confusion_matrix_val.csv",
        run_dir / "confusion_matrix_val.png",
        val_result["y_true"],
        val_result["y_pred"],
        classes,
        "ResNet18 RGB - conjunto val",
    )
    save_confusion_matrix(
        run_dir / "confusion_matrix_test.csv",
        run_dir / "confusion_matrix_test.png",
        test_result["y_true"],
        test_result["y_pred"],
        classes,
        "ResNet18 RGB - conjunto test",
    )

    dataset_summary = {
        "datasets": [str(path.expanduser().resolve()) for path in args.dataset],
        "classes": classes,
        "confidence": confidence,
        "total_patches": int(len(rows)),
        "total_frames": int(rows["group"].nunique()),
        "class_distribution": dict(Counter(rows["label"])),
        "splits": split_summary,
        "seed": args.seed,
        "min_class_holdout": args.min_class_holdout,
        "reuse_split": str(args.reuse_split) if args.reuse_split else None,
    }
    metrics = {
        "architecture": "resnet18",
        "input": "rgb",
        "image_size": args.image_size,
        "weights": args.weights,
        "best_epoch": best_epoch,
        "best_phase": best_phase,
        "best_val_f1_macro": best_val_f1,
        "validation_report": validation_report,
        "test_report": test_report,
        "test_f1_macro": f1_score(
            test_result["y_true"],
            test_result["y_pred"],
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "test_accuracy": accuracy_score(test_result["y_true"], test_result["y_pred"]),
    }
    save_json(run_dir / "dataset_summary.json", dataset_summary)
    save_json(run_dir / "metrics.json", metrics)
    save_json(
        run_dir / "class_mapping.json",
        {
            "class_to_index": class_to_index,
            "index_to_class": index_to_class,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": "resnet18",
        "classes": classes,
        "class_to_index": class_to_index,
        "image_size": args.image_size,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "weights": args.weights,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_summary": dataset_summary,
        "metrics": metrics,
    }
    torch.save(checkpoint, run_dir / "model.pt")

    print(f"Mejor epoch: {best_epoch} ({best_phase})")
    print(f"Test F1 macro: {metrics['test_f1_macro']:.3f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.3f}")
    print(f"Modelo guardado: {run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
