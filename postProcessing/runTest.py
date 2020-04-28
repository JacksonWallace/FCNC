import os

from importlib import import_module
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor   import PostProcessor
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel       import Collection
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop       import Module

from PhysicsTools.NanoAODTools.postprocessing.modules.tW_scattering.ObjectSelection import *
from PhysicsTools.NanoAODTools.postprocessing.modules.tW_scattering.GenAnalyzer import *
from PhysicsTools.NanoAODTools.postprocessing.modules.tW_scattering.lumiWeightProducer import *

#json support to be added

modules = [\
    lumiWeightProd(0.5), #some dummy value
    selector2018(),
    GenAnalyzer(),
    ]

# apply PV requirement
cut  = 'PV_ndof>4 && sqrt(PV_x*PV_x+PV_y*PV_y)<=2 && abs(PV_z)<=24'
# loose skim
cut += '&& nJet>2&&(nElectron+nMuon)>0'

testFile = '/hadoop/cms/store/user/dspitzba/tW_scattering/tW_scattering/nanoAOD/tW_scattering_nanoAOD_100.root'
#testFile = '/hadoop/cms/store/user/dspitzba/nanoAOD/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1/86A2DA79-0EFC-EE49-B3B5-AF104D44D05A.root'

p = PostProcessor('./', [testFile], cut=cut, modules=modules, maxEntries=100,\
    branchsel=os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoAODTools/python/postprocessing/modules/tW_scattering/keep_and_drop_in.txt'),\
    outputbranchsel=os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoAODTools/python/postprocessing/modules/tW_scattering/keep_and_drop.txt') )

p.run()
