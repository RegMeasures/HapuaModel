import argparse
import logging
import os

from .core import main

#%% Parse arguments from commandline
Parser = argparse.ArgumentParser(description='HapuaMod is a model which simulates morphological evolution of hapua type rivermouth lagoons. ' +
                                             'For a full description of the model and how to use it see https://github.com/RegMeasures/HapuaModel/wiki')
Parser.add_argument('ModelConfigFile', type=str, 
                    help='Model config file (*.cnf)')
Parser.add_argument('-o', '--Overwrite', action='store_true', 
                    help='Optional flag to overwrite output file without checking')
Parser.add_argument('-v', '--verbose', action='store_true',
                    help='Turn on debug outputs to log file and console')
Parser.add_argument('-q', '--quiet', action='store_true',
                    help='Turn off info outputs log file and console (i.e. warnings and errors only)')
ArgsIn = Parser.parse_args()

#%% Set up logging
RootLogger = logging.getLogger()
RootLogger.setLevel(logging.DEBUG)

LogFile = os.path.splitext(os.path.split(ArgsIn.ModelConfigFile)[1])[0] + '_log.txt'

ConsoleHandler = logging.StreamHandler()
FileHandler = logging.FileHandler(LogFile)

if ArgsIn.quiet:
    ConsoleHandler.setLevel(logging.WARNING)
    FileHandler.setLevel(logging.WARNING)
elif ArgsIn.verbose:
    ConsoleHandler.setLevel(logging.DEBUG)
    FileHandler.setLevel(logging.DEBUG)
else:
    ConsoleHandler.setLevel(logging.INFO)
    FileHandler.setLevel(logging.INFO)

LogFormatter = logging.Formatter('%(levelname)-7s: %(message)s')
FileHandler.setFormatter(LogFormatter)

RootLogger.addHandler(ConsoleHandler)
RootLogger.addHandler(FileHandler)

main(ArgsIn.ModelConfigFile, ArgsIn.Overwrite)