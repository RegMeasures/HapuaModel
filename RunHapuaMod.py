import logging
import os
from hapuamod.core import main

#ModelConfigFile = 'inputs\HurunuiModel.cnf'
ModelConfigFile = 'inputs\SimplifiedSynthetic.cnf'

RootLogger = logging.getLogger()
RootLogger.setLevel(logging.DEBUG)

ConsoleHandler = logging.StreamHandler()
ConsoleHandler.setLevel(logging.INFO)
RootLogger.addHandler(ConsoleHandler)

LogFile = os.path.splitext(os.path.split(ModelConfigFile)[1])[0] + '_log.txt'
FileHandler = logging.FileHandler(LogFile)
RootLogger.addHandler(FileHandler)

main(ModelConfigFile, True)
