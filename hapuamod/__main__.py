import argparse

from hapuamod import core

# Parse arguments from commandline
Parser = argparse.ArgumentParser(description='Hapua model')
Parser.add_argument('ModelConfigFile', type=str, help='Model config file (*.cnf)')
ArgsIn = Parser.parse_args()

core.run(ArgsIn.ModelConfigFile)