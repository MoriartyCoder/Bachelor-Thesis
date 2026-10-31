from cli import cli_handler
from setup import setup
import logging
from data_handler.prepare_dataset import prepare_anti_uav_300, prepare_anti_uav_410, prepare_anti_uav_cst


def main():
    logger = logging.getLogger("main")
    logging.info("\n" + "="*15 + "\n" + " "*5 + "START" + "\n" + 15 * "=")
    args = cli_handler()

    if args.prepare is not None:
        if args.prepare == 300:
            prepare_anti_uav_300()
        elif args.prepare == 410:
            prepare_anti_uav_410()
        elif args.prepare == 100:
            prepare_anti_uav_cst()
        return

if __name__ == "__main__":
    #setup()
    main()

    logging.info("\n" + "="*15 + "\n" + " "*5+ "END" + "\n" + 15 * "=")
