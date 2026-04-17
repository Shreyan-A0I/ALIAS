"""
GPU-optimized UNI feature extraction.

Reads PNG patches from donor subdirectories, runs them through the UNI backbone,
and writes one cached feature file per donor.
"""

import argparse
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import timm
from torchvision import transforms


DONOR_IDS = [
    "Br2743_ant", "Br3942_ant", "Br6423_ant", "Br8492_ant",
    "Br6471_ant", "Br6522_ant", "Br8325_ant", "Br8667_ant",
]


class PatchDataset(Dataset):
    def __init__(self, patch_dir, transform):
        self.patch_dir = Path(patch_dir)
        self.transform = transform
        self.patch_files = sorted(self.patch_dir.glob("*.png"))

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, index):
        patch_path = self.patch_files[index]
        with Image.open(patch_path) as img:
            tensor = self.transform(img.convert("RGB"))
        return tensor, patch_path.stem


def parse_args():
    parser = argparse.ArgumentParser(description="Cache UNI features with GPU acceleration.")
    parser.add_argument("--checkpoint", default="uni.bin", help="Path to the UNI checkpoint.")
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Base directory containing donor patch folders. Defaults to data/patches when present, else data/raw_data.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/cached_features_uni",
        help="Directory where cached donor feature files will be written.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, e.g. cuda, cuda:0, mps, or cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size. Defaults to 256 on CUDA and 64 otherwise.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker count. Defaults to min(8, CPU count).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="How many batches each worker preloads ahead of time.",
    )
    return parser.parse_args()


def resolve_device(device_arg=None):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_input_dir(input_dir_arg=None):
    candidates = []
    if input_dir_arg:
        candidates.append(Path(input_dir_arg))
    candidates.extend([Path("data/patches"), Path("data/raw_data")])

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    looked_in = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find an input patch directory. Looked in: {looked_in}"
    )


def configure_runtime(device):
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def get_default_batch_size(device):
    return 256 if device.type == "cuda" else 64


def get_default_num_workers():
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count))


def unwrap_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        return checkpoint_obj["state_dict"]
    return checkpoint_obj


def infer_uni_config(state_dict):
    embed_dim = state_dict["cls_token"].shape[-1]
    patch_size = state_dict["patch_embed.proj.weight"].shape[-1]
    reg_tokens = state_dict["reg_token"].shape[1] if "reg_token" in state_dict else 0
    depth = max(
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("blocks.")
    ) + 1

    if embed_dim == 1536 and patch_size == 14 and depth == 24 and reg_tokens == 8:
        return {
            "variant": "UNI2-h",
            "model_name": "vit_giant_patch14_224",
            "timm_kwargs": {
                "img_size": 224,
                "patch_size": 14,
                "depth": 24,
                "num_heads": 24,
                "init_values": 1e-5,
                "embed_dim": 1536,
                "mlp_ratio": 2.66667 * 2,
                "num_classes": 0,
                "no_embed_class": True,
                "mlp_layer": timm.layers.SwiGLUPacked,
                "act_layer": torch.nn.SiLU,
                "reg_tokens": 8,
                "dynamic_img_size": True,
            },
            "transform": transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            ),
        }

    return {
        "variant": "UNI",
        "model_name": "vit_large_patch16_224",
        "timm_kwargs": {
            "img_size": 224,
            "patch_size": 16,
            "init_values": 1e-5,
            "num_classes": 0,
            "dynamic_img_size": True,
        },
        "transform": None,
    }


def load_uni_model(checkpoint_path, device):
    print(f"Loading UNI model from {checkpoint_path} on {device}...")

    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
    state_dict = unwrap_state_dict(checkpoint_obj)
    model_config = infer_uni_config(state_dict)

    print(f"Detected checkpoint variant: {model_config['variant']}")

    model = timm.create_model(
        model_config["model_name"],
        pretrained=False,
        **model_config["timm_kwargs"],
    )

    model.load_state_dict(state_dict, strict=True)

    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)
    model.eval()

    if model_config["transform"] is not None:
        transform = model_config["transform"]
    else:
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)

    return model, transform


def build_dataloader(dataset, batch_size, num_workers, device, prefetch_factor):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def get_amp_context(device):
    if device.type != "cuda":
        return nullcontext()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def collect_donor_patch_counts(input_dir):
    donor_patch_counts = {}
    for donor_id in DONOR_IDS:
        donor_patch_dir = input_dir / donor_id
        if not donor_patch_dir.exists():
            donor_patch_counts[donor_id] = None
            continue
        donor_patch_counts[donor_id] = len(list(donor_patch_dir.glob("*.png")))
    return donor_patch_counts


def cache_features_for_donor(
    donor_id,
    model,
    transform,
    input_dir,
    output_dir,
    device,
    batch_size,
    num_workers,
    prefetch_factor,
    donor_index,
    donor_total,
    overall_progress,
):
    donor_patch_dir = input_dir / donor_id
    if not donor_patch_dir.exists():
        print(f"Directory {donor_patch_dir} not found. Skipping.")
        return

    dataset = PatchDataset(donor_patch_dir, transform)
    if len(dataset) == 0:
        print(f"No PNG patches found in {donor_patch_dir}. Skipping.")
        return

    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        prefetch_factor=prefetch_factor,
    )

    print(
        f"[{donor_index}/{donor_total}] Processing {len(dataset)} patches for donor {donor_id} "
        f"(batch_size={batch_size}, workers={num_workers})..."
    )

    donor_start = time.time()
    features_list = []
    barcodes = []
    with torch.inference_mode():
        for batch_tensors, batch_barcodes in tqdm(
            dataloader,
            desc=f"{donor_id}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        ):
            batch_tensors = batch_tensors.to(device, non_blocking=device.type == "cuda")
            with get_amp_context(device):
                features = model(batch_tensors)

            features_list.append(features.float().cpu())
            barcodes.extend(batch_barcodes)
            overall_progress.update(len(batch_barcodes))

    all_features = torch.cat(features_list, dim=0)
    output_path = output_dir / f"{donor_id}.pt"

    torch.save(
        {
            "barcodes": barcodes,
            "features": all_features,
            "model": "UNI2-h" if all_features.shape[1] == 1536 else "UNI",
            "source_dir": str(donor_patch_dir),
        },
        output_path,
    )

    donor_elapsed = time.time() - donor_start
    donor_rate = all_features.shape[0] / donor_elapsed if donor_elapsed > 0 else 0.0
    print(
        f"Finished {donor_id}: saved {all_features.shape[0]} features to {output_path} "
        f"in {donor_elapsed:.1f}s ({donor_rate:.1f} patches/s)"
    )


def main():
    overall_start = time.time()
    args = parse_args()
    device = resolve_device(args.device)
    configure_runtime(device)

    input_dir = resolve_input_dir(args.input_dir)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size or get_default_batch_size(device)
    num_workers = args.num_workers if args.num_workers is not None else get_default_num_workers()

    donor_patch_counts = collect_donor_patch_counts(input_dir)
    available_donors = [donor_id for donor_id, count in donor_patch_counts.items() if count]
    total_patches = sum(count for count in donor_patch_counts.values() if count)

    model, transform = load_uni_model(args.checkpoint, device=device)

    print(f"Using input patches from: {input_dir}")
    print(f"Writing cached features to: {output_dir}")
    print(
        f"Run config: device={device}, batch_size={batch_size}, "
        f"num_workers={num_workers}, prefetch_factor={args.prefetch_factor}"
    )
    print(
        f"Found {len(available_donors)}/{len(DONOR_IDS)} donors with patches, "
        f"{total_patches} total patches."
    )
    for donor_id in DONOR_IDS:
        count = donor_patch_counts[donor_id]
        if count is None:
            print(f"  - {donor_id}: missing donor directory")
        else:
            print(f"  - {donor_id}: {count} patches")

    with tqdm(
        total=total_patches,
        desc="Overall",
        unit="patch",
        dynamic_ncols=True,
    ) as overall_progress:
        for donor_index, donor_id in enumerate(DONOR_IDS, start=1):
            cache_features_for_donor(
                donor_id=donor_id,
                model=model,
                transform=transform,
                input_dir=input_dir,
                output_dir=output_dir,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_factor=args.prefetch_factor,
                donor_index=donor_index,
                donor_total=len(DONOR_IDS),
                overall_progress=overall_progress,
            )

    total_elapsed = time.time() - overall_start
    total_rate = total_patches / total_elapsed if total_elapsed > 0 else 0.0
    print(
        f"\nFeature extraction complete in {total_elapsed:.1f}s "
        f"({total_rate:.1f} patches/s overall)."
    )


if __name__ == "__main__":
    main()
