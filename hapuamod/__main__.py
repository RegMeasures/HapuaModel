import argparse

from .core import main

# Parse arguments from commandline
Parser = argparse.ArgumentParser(description='HapuaMod is a model which simulates morphological evolution of hapua type rivermouth lagoons. ' +
                                             'For a full description of the model and how to use it see https://github.com/RegMeasures/HapuaModel/wiki')
Parser.add_argument('ModelConfigFile', type=str, help='Model config file (*.cnf)')
Parser.add_argument('-o', '--Overwrite', action='store_true', help='Optional flag to overwrite output file without checking')
ArgsIn = Parser.parse_args()
# TODO: Add verbosity commandline argument

main(ArgsIn.ModelConfigFile, ArgsIn.Overwrite)