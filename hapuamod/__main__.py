import argparse

from .core import main

# Parse arguments from commandline
Parser = argparse.ArgumentParser(description='Hapua model')
Parser.add_argument('ModelConfigFile', type=str, help='Model config file (*.cnf)')
Parser.add_argument('-o', '--Overwrite', action='store_true', help='Optional flag to overwrite output file without checking')
ArgsIn = Parser.parse_args()
# TODO: Add verbosity and overwrite commandline arguments

main(ArgsIn.ModelConfigFile, ArgsIn.Overwrite)