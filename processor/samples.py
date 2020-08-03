import os
import glob
from Tools.helpers import *
cfg = loadConfig()

version = cfg['meta']['version']
tag = version.replace('.','p')

data_path = os.path.join(cfg['meta']['localSkim'], tag)


# this could still be automatized better
fileset = {
        'tW_scattering': glob.glob(data_path+"/tW_scattering__nanoAOD/merged/*.root"),
        "TTX":           glob.glob(data_path+"/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1/merged/*.root"),
        "TTW":           glob.glob(data_path+"/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1/merged/*.root") \
                        + glob.glob(data_path+"/TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20-v1/merged/*.root"),
        "ttbar":        [] \
                        + glob.glob(data_path+"/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20-v1/merged/*.root") \
                        + glob.glob(data_path+"/TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20-v1/merged/*.root") \
                        + glob.glob(data_path+"/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20-v1/merged/*.root"),
        "wjets":    [] \
                    + glob.glob(data_path+"/W1JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20-v1/merged/*.root") \
                    + glob.glob(data_path+"/W2JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20-v1/merged/*.root") \
                    + glob.glob(data_path+"/W3JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20-v1/merged/*.root") \
                    + glob.glob(data_path+"/W4JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20-v1/merged/*.root"),
    }
