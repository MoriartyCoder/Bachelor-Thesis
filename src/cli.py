import argparse

def cli_handler():
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="verbose output")

    parser.add_argument("-p", "--prepare",  type=int, choices=[300, 410, 600])
    
    return parser.parse_args()