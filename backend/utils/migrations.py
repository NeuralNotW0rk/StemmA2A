import os
from pathlib import Path

# Feature flag to enable/disable all filesystem/compatibility migrations
ENABLE_MIGRATIONS = os.environ.get("ENABLE_MIGRATIONS", "true").lower() == "true"


def migrate_sharded_cache(root_dir: Path) -> None:
    """
    Migrates any old-style 2-character sharded cache folders
    from root_dir to root_dir / 'cache'.
    """
    import shutil
    import re

    if not root_dir or not root_dir.is_dir():
        return
    
    cache_dir = root_dir / "cache"
    hex_pattern = re.compile(r"^[0-9a-fA-F]{2}$")
    
    try:
        for path in root_dir.iterdir():
            if path.is_dir() and hex_pattern.match(path.name):
                dest = cache_dir / path.name
                cache_dir.mkdir(exist_ok=True)
                
                if dest.exists():
                    for item in path.iterdir():
                        item_dest = dest / item.name
                        if not item_dest.exists():
                            shutil.move(str(item), item_dest)
                    try:
                        path.rmdir()
                    except OSError:
                        pass
                else:
                    shutil.move(str(path), dest)
                print(f"Migrated sharded cache folder {path.name} to {dest}")
    except Exception as e:
        print(f"Failed to migrate sharded cache in {root_dir}: {e}")


def run_global_migrations(data_cache_root: Path) -> None:
    """
    Runs all startup global data cache migrations.
    """
    if not ENABLE_MIGRATIONS:
        return
    
    print("Running global migrations...")
    migrate_sharded_cache(data_cache_root)


def run_project_migrations(project_path: Path) -> None:
    """
    Runs all project-level migrations when a project is loaded.
    """
    if not ENABLE_MIGRATIONS:
        return
    
    print(f"Running project migrations for {project_path.name}...")
    migrate_sharded_cache(project_path)
