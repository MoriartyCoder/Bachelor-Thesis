"""Single entry point for every action. See `python src/main.py --help` and README.md.

Order when several actions are given: prepare (-p), then build the target library (-l), then train (-t).
"""
import logging
import sys
import time

from cli import cli_handler
from config import PREPARE_IDS
from log_setup import get_logger, setup_logging

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    args = cli_handler(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    start = time.perf_counter()
    logger.info("=" * 15 + " START " + "=" * 15)
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.debug(f"Parsed arguments: {vars(args)}")

    try:
        if args.prepare:
            from data_handler.prepare_dataset import prepare_dataset
            for pid in args.prepare:
                prepare_dataset(PREPARE_IDS[pid])

        if args.build_library:
            from augmentations.target_library import LibraryConfig, build_library
            build_library(LibraryConfig(), overwrite=True)

        if args.train:
            from training.run import run  # heavy imports (torch/ultralytics) only when training
            run(args.config, dry_run=args.dry_run)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    except Exception as e:
        if not getattr(e, "already_logged", False):
            logger.exception("Aborted with an error:")
        return 1
    finally:
        logger.info("=" * 15 + f" END ({time.perf_counter() - start:.1f} s) " + "=" * 15)
    return 0


if __name__ == "__main__":
    sys.exit(main())
