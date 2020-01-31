import logging
from hapuamod.core import main

ModelConfigFile = 'inputs\HurunuiModel.cnf'

RootLogger = logging.getLogger()
RootLogger.setLevel(logging.DEBUG)
ConsoleHandler = logging.StreamHandler()
ConsoleHandler.setLevel(logging.INFO)
RootLogger.addHandler(ConsoleHandler)

main(ModelConfigFile, True)
