# Measuring tW scattering

Prerequisite: if you haven't, add this line to your `~/.profile`:
```
source /cvmfs/cms.cern.ch/cmsset_default.sh
```

Currently lives within CMSSW_10_2_9. Set up in a fresh directory, recipe as follows:
```
cmsrel CMSSW_10_2_9
cd CMSSW_10_2_9/src
cmsenv
git cms-init

git clone --branch tW_scattering https://github.com/danbarto/nanoAOD-tools.git NanoAODTools

cd $CMSSW_BASE/src

git clone --recursive https://github.com/danbarto/tW_scattering.git

scram b -j 8
cmsenv

```

Then you can set up the tools to run coffea, deactivate the environment again and recompile.
```
cd tW_scattering
source setup_environment.sh
deactivate
scram b -j 8
```

Every time you want to use coffea you need to activate the environment *this has changed in order to disentangle coffea from CMSSW*
```
source activate_environment.sh
```

To deactivate the coffea environment, just type `deactivate`


Use available nanoAOD tools to quickly process samples.

### Use jupyter notebooks

To install jupyter inside the coffeaEnv do the following (now part of the setup script too):
```
python -m ipykernel install --user --name=coffeaEnv
jupyter nbextension install --py widgetsnbextension --user
jupyter nbextension enable widgetsnbextension --user --py
```

To start the server:
```
jupyter notebook --no-browser --port=8893
```

On your local machine do the following to connect to uaf
```
ssh -N -f -L localhost:8893:localhost:8893 uaf-10.t2.ucsd.edu
```

Then just paste the jupyter link into your browser and start working.

### Get combine (for later)
Latest recommendations at https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/#setting-up-the-environment-and-installation
```
cd $CMSSW_BASE/src
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit
git fetch origin
git checkout v8.0.1
scramv1 b clean; scramv1 b # always make a clean build
```

### for combineTools (for later)
```
cd $CMSSW_BASE/src
wget https://raw.githubusercontent.com/cms-analysis/CombineHarvester/master/CombineTools/scripts/sparse-checkout-https.sh; source sparse-checkout-https.sh
scram b -j 8
```
