import argparse

from config import PREPARE_IDS

EXAMPLES = """examples:
  python src/main.py -p 410 100                         extract/prepare datasets from data/*.zip
  python src/main.py -l                                 (re)build the PGDA target library in data/library/
  python src/main.py -c configs/exp.yaml -t             train with a run config
  python src/main.py -c configs/exp.yaml -t --dry-run   quick end-to-end check (CPU, tiny subsets)
  python src/main.py -c configs/exp.yaml -t -v          same, with detailed DEBUG log
"""


def cli_handler(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drone detection in thermal IR video: dataset preparation, PGDA target library and YOLO training.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-c", "--config", metavar="YAML",
                        help="run config (copy configs/template.yaml per experiment)")

    parser.add_argument("-p", "--prepare", type=int, nargs="+", choices=sorted(PREPARE_IDS), metavar="ID",
                        help="extract/prepare dataset zips in data/ (ids: "
                             + ", ".join(f"{k}={v}" for k, v in sorted(PREPARE_IDS.items())) + ")")

    parser.add_argument("-l", "--build-library", action="store_true",
                        help="(re)build the PGDA target library in data/library/ from Anti-UAV410/train "
                             "with the default LibraryConfig; an existing library is replaced")

    parser.add_argument("-t", "--train", action="store_true",
                        help="train YOLO with the run config given by -c")

    parser.add_argument("--dry-run", action="store_true",
                        help="with -t: tiny subsets, 1 CPU epoch, augmented previews; output in output/dryrun/")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="detailed DEBUG log (default: INFO)")

    args = parser.parse_args(argv)

    if args.train and not args.config:
        parser.error("-t/--train needs a run config: -c configs/<exp>.yaml")
    if args.dry_run and not args.train:
        parser.error("--dry-run only works together with -t/--train")
    if not (args.prepare or args.build_library or args.train):
        parser.error("nothing to do: give -p, -l and/or -t (see examples below --help)")
    return args
