import hashlib
from pathlib import Path

def hash_dataset(dataset_dir: str, algo: str = "blake2b", chunk_size: int = 1024 * 1024) -> str:
    root = Path(dataset_dir).absolute()
    print(root)
    h = hashlib.new(algo)

    # Alle Dateien rekursiv sammeln und deterministisch sortieren
    files = sorted(
        (p for p in root.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(root).as_posix()
    )

    for i, file_path in enumerate(files):
        rel_path = file_path.relative_to(root).as_posix().encode("utf-8")
        h.update(rel_path + b"\0")

        with file_path.open("rb") as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)

        if i % 100 == 0:
            print(f"{i:6d}/{len(files)}")

    return h.hexdigest()

if __name__ == "__main__":
    print(hash_dataset("./data/Anti-UAV410"))