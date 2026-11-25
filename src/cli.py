import argparse

from config import PREPARE_IDS

EXAMPLES = """examples:
  python src/main.py -p 410 100                         extract/prepare datasets from data/*.zip
  python src/main.py -l                                 (re)build the PGDA target library in data/library/
  python src/main.py -c configs/template.yaml --freeze-test   write the fixed test set configs/test_set.csv (once)
  python src/main.py -c configs/exp.yaml -t             train with a run config
  python src/main.py -c configs/exp.yaml -t --dry-run   quick end-to-end check (CPU, tiny subsets)
  python src/main.py -c configs/exp.yaml -t -v          same, with detailed DEBUG log
  python src/main.py -r weights/20261124-210547         continue an interrupted run from its last.pt
  python src/main.py -e weights/20261124-210547         robustness test of a finished run (configs/robustness.yaml)
  python src/main.py -e weights/20261124-210547 --suite configs/other_suite.yaml
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
                        help="(re)build the PGDA target library in data/library/ from the train split of all datasets "
                             "with the default LibraryConfig; an existing library is replaced")

    parser.add_argument("--freeze-test", action="store_true",
                        help="with -c: sample the test split once (datasets + sampling.test of the config) and write "
                             "it to run.test_set (default configs/test_set.csv); every run is evaluated on these "
                             "frames. An existing file is never overwritten")

    parser.add_argument("-r", "--resume", metavar="RUN_DIR",
                        help="continue an interrupted run from RUN_DIR/weights/last.pt with the run's own "
                             "config.yaml (same folder, same frames), then evaluate it on the test set")

    parser.add_argument("-e", "--evaluate", metavar="RUN_DIR",
                        help="robustness evaluation of a finished run: RUN_DIR/weights/best.pt on the frozen test set, "
                             "clean and per corruption test of the suite -> RUN_DIR/robustness_metrics.json")

    parser.add_argument("--suite", metavar="YAML",
                        help="with -e: robustness suite (default: run.robustness_suite of the run, else "
                             "configs/robustness.yaml)")

    parser.add_argument("-t", "--train", action="store_true",
                        help="train YOLO with the run config given by -c")

    parser.add_argument("--dry-run", action="store_true",
                        help="with -t: tiny subsets, 1 CPU epoch, augmented previews; output in output/dryrun/. "
                             "With -e: only the 32 test frames of a dry run, on the CPU")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="detailed DEBUG log (default: INFO)")

    args = parser.parse_args(argv)

    if args.train and not args.config:
        parser.error("-t/--train needs a run config: -c configs/<exp>.yaml")
    if args.freeze_test and not args.config:
        parser.error("--freeze-test needs a run config: -c configs/<exp>.yaml")
    if args.dry_run and not (args.train or args.evaluate):
        parser.error("--dry-run only works together with -t/--train or -e/--evaluate")
    if args.suite and not args.evaluate:
        parser.error("--suite only works together with -e/--evaluate")
    if args.evaluate and args.config:
        parser.error("-e/--evaluate uses the run's own config.yaml: do not combine it with -c")
    if args.resume and (args.train or args.config):
        parser.error("-r/--resume uses the run's own config.yaml: do not combine it with -t or -c")
    if not (args.prepare or args.build_library or args.freeze_test or args.train or args.resume or args.evaluate):
        parser.error("nothing to do: give -p, -l, --freeze-test, -t, -r and/or -e (see examples below --help)")
    return args
