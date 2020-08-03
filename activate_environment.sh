#!/usr/bin/env bash

cd /cvmfs/cms.cern.ch/slc6_amd64_gcc700/cms/cmssw/CMSSW_10_2_9/ ; cmsenv ; cd -

source coffeaEnv/bin/activate
export TWHOME=$PWD
export PYTHONPATH=${PYTHONPATH}:$PWD
