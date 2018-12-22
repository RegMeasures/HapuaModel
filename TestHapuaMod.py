import hapuamod as hm

ModelConfigFile = 'inputs\HurunuiModel.cnf'
(FlowTs, WaveTs, SeaLevelTs, Origin, ShoreNormalDir, ShoreX, ShoreY) = hm.loadModel(ModelConfigFile)
