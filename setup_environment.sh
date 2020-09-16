#!/usr/bin/env bash

cd /cvmfs/cms.cern.ch/slc6_amd64_gcc700/cms/cmssw/CMSSW_10_2_9/ ; cmsenv ; cd -

[ -d virtualenv ] || pip install --target=`pwd`/virtualenv virtualenv
[ -d coffeaEnv ] || virtualenv -p `which python3` coffeaEnv
source coffeaEnv/bin/activate
pip3 install --upgrade numpy 
pip3 install --upgrade matplotlib
pip3 install uproot coffea jupyter tqdm pandas backports.lzma pyyaml klepto
pip3 install --upgrade tqdm
pip3 install lbn
pip install -U memory_profiler
python -m ipykernel install --user --name=coffeaEnv
jupyter nbextension install --py widgetsnbextension --user
jupyter nbextension enable widgetsnbextension --user --py
