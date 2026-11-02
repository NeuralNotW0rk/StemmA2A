"""
Copies a model that is already on the engine host into its model store and registers it as a shared
model, so clients skip uploading it. UIDs are computed by the model's adapter, the same way a client
computes them, so a client project that uses this model finds its files already here.

Run it where the engine service runs, with paths as that process sees them. For the container:

    docker compose exec execution-engine python scripts/ingest_model.py \\
        --adapter stable_audio_tools --name "Stable Audio 3 Medium" \\
        --checkpoint /data/import/model.ckpt --encoder /data/import/encoder \\
        --config /data/import/base_model_config.json

The source files must be inside a mounted volume. They are copied, never moved or linked.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Ensure we can import from the backend directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.engine import ADAPTER_REGISTRY
from engine.model_store import FILE_REFS, ModelStore, data_root_from_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, choices=sorted(ADAPTER_REGISTRY))
    parser.add_argument("--name", required=True, help="Name shown for the shared model")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--encoder", type=Path, help="Encoder file or directory, if the model has one")
    parser.add_argument("--config", type=Path, help="Model config JSON, stored inline in the registry")
    parser.add_argument("--checkpoint-uid", help="Use this UID instead of hashing the checkpoint")
    parser.add_argument("--encoder-uid", help="Use this UID instead of hashing the encoder")
    parser.add_argument("--dry-run", action="store_true", help="Compute and print UIDs without copying anything")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for path in (args.checkpoint, args.encoder, args.config):
        if path and not path.exists():
            sys.exit(f"Error: {path} does not exist")
    if args.encoder and args.encoder.is_file() and not args.encoder_uid:
        # The adapter derives a single-file encoder's UID from its path, which differs between machines
        sys.exit("Error: a single-file encoder needs --encoder-uid, copied from the client's project")

    config = None
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

    print(f"Computing UIDs with the '{args.adapter}' adapter (this loads the checkpoint and can take a while)...")
    adapter = ADAPTER_REGISTRY[args.adapter]()
    model = adapter.register_model(
        name=args.name,
        model_type=args.adapter,
        config=config,
        checkpoint_path=str(args.checkpoint),
        checkpoint_uid=args.checkpoint_uid,
        encoder_path=str(args.encoder) if args.encoder else None,
        encoder_uid=args.encoder_uid,
    )

    entry = {"name": args.name, "adapter": args.adapter, "model_type": args.adapter}
    if config is not None:
        entry["config"] = config
    sources = {}
    for field_name, asset in model._iter_assets():
        if field_name not in FILE_REFS or not asset.uid or not asset.path:
            continue
        entry[field_name] = {k: v for k, v in (("uid", asset.uid), ("size", asset.size)) if v}
        sources[asset.uid] = Path(asset.path)
        print(f"  {field_name}: {asset.uid} ({(asset.size or 0) / 1024 ** 3:.2f} GB) from {asset.path}")
    print(f"  model id: {model.id}")

    if args.dry_run:
        print("Dry run: nothing copied or registered.")
        return

    store = ModelStore(data_root_from_env())
    for uid, source in sources.items():
        print(f"Copying {source} into the model store...")
        if not store.ingest(uid, source):
            print(f"  '{uid}' is already in the store or cache; skipped.")

    store.promote(entry)
    print(f"Registered '{args.name}' in {store.registry_path}.")
    print(
        "Jobs that use these files find them immediately. Restart the engine service for the model "
        "to appear in its shared models list."
    )


if __name__ == "__main__":
    main()
