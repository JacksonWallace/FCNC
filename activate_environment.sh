#!/usr/bin/env bash

cd /cvmfs/cms.cern.ch/slc6_amd64_gcc700/cms/cmssw/CMSSW_10_2_9/ ; eval `scramv1 runtime -sh` ; cd -

source coffeaEnv/bin/activate
expor tTWHOME=$PWD
export PYTHONPATH=${PYTHONPATH}:$PWD:$PWD/coffea/:$PWD/postProcessing/ProjectMetis/
