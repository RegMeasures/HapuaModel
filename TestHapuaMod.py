import hapuamod as hm

ModelConfigFile = 'inputs\HurunuiModel.cnf'
Config = hm.load.readConfig(ModelConfigFile)
(FlowTs, WaveTs, SeaLevelTs, Origin, ShoreNormalDir, ShoreX, ShoreY) = hm.load.loadModel(Config)
