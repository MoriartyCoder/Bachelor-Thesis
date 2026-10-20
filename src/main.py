from cli import cli_handler
from setup import setup
import logging
from data_handler.prepare_dataset import prepare_anti_uav_300

def main():
    logger = logging.getLogger("main")
    logging.info("\n" + "="*15 + "\n" + " "*5+ "START" + "\n" + 15 * "=")
    args = cli_handler()

    if args.prepare is not None:
        prepare_anti_uav_300()


if __name__ == "__main__":
    setup()
    main()

    logging.info("\n" + "="*15 + "\n" + " "*5+ "END" + "\n" + 15 * "=")
