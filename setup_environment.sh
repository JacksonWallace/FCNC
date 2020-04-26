#!/usr/bin/env bash

[ -d virtualenv ] || pip install --target=`pwd`/virtualenv virtualenv
[ -d myenv ] || virtualenv -p `which python3` coffeaEnv
source coffeaEnv/bin/activate
pip3 install --upgrade numpy 
pip3 install --upgrade matplotlib
pip3 install uproot coffea jupyter tqdm pandas backports.lzma pyyaml
