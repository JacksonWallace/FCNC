#!/bin/bash

# This is nanoAOD based sample making condor executable for CondorTask of ProjectMetis. Passed in arguments are:
# arguments = [outdir, outname_noext, inputs_commasep, index, cmssw_ver, scramarch, self.arguments]

OUTPUTDIR=$1
OUTPUTNAME=$2
INPUTFILENAMES=$3
IFILE=$4
CMSSW_VERSION=$5
SCRAM_ARCH=$6

VERSION=$7
SUMWEIGHT=$8

OUTPUTNAME=$(echo $OUTPUTNAME | sed 's/\.root//')

echo -e "\n--- begin header output ---\n" #                     <----- section division
echo "OUTPUTDIR: $OUTPUTDIR"
echo "OUTPUTNAME: $OUTPUTNAME"
echo "INPUTFILENAMES: $INPUTFILENAMES"
echo "IFILE: $IFILE"
echo "CMSSW_VERSION: $CMSSW_VERSION"
echo "SCRAM_ARCH: $SCRAM_ARCH"

echo "hostname: $(hostname)"
echo "uname -a: $(uname -a)"
echo "time: $(date +%s)"
echo "args: $@"

echo -e "\n--- end header output ---\n" #                       <----- section division
ls -ltrha
echo ----------------------------------------------

# Setup Enviroment
export SCRAM_ARCH=$SCRAM_ARCH
source /cvmfs/cms.cern.ch/cmsset_default.sh
#pushd /cvmfs/cms.cern.ch/$SCRAM_ARCH/cms/cmssw/$CMSSW_VERSION/src/ > /dev/null
#eval `scramv1 runtime -sh`
#popd > /dev/null
scramv1 project CMSSW $CMSSW_VERSION
cd $CMSSW_VERSION/src
eval `scramv1 runtime -sh`

# The output name is the sample name for stop baby
SAMPLE_NAME=$OUTPUTNAME
NEVENTS=-1

# checkout the package
git clone --branch $VERSION --depth 1  https://github.com/danbarto/nanoAOD-tools.git PhysicsTools/NanoAODTools

scram b


echo "Running PhysicsTools/NanoAODTools/scripts/nano_postproc.py:"
#echo "Running BabyMakera:"

echo "Input:"
echo $INPUTFILENAMES

OUTFILE=$(python -c "print('$INPUTFILENAMES'.split('/')[-1].split('.root')[0]+'_Skim.root')")

echo $OUTFILE

python << EOL
from importlib import import_module
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor   import PostProcessor
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel       import Collection
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop       import Module

from PhysicsTools.NanoAODTools.postprocessing.modules.tW_scattering.ObjectSelection import *
from PhysicsTools.NanoAODTools.postprocessing.modules.tW_scattering.GenAnalyzer import *
from PhysicsTools.NanoAODTools.postprocessing.modules.tW_scattering.lumiWeightProducer import *

#json support to be added

modules = [\
    lumiWeightProd("$SUMWEIGHT"),
    genAnalyzer(),
    selector2018(),
    ]

# apply PV requirement
cut  = 'PV_ndof>4 && sqrt(PV_x*PV_x+PV_y*PV_y)<=2 && abs(PV_z)<=24'
# loose skim
cut += '&& (Sum\$(Electron_pt>30&&abs(Electron_eta)<2.4&&Electron_miniPFRelIso_all<0.1&&Electron_cutBased>=3)+Sum\$(Muon_pt>25&&abs(Muon_eta)<2.4&&Muon_mediumId>0&&Muon_miniPFRelIso_all<0.1))>0'
cut += '&& ( (Sum\$(Jet_pt>25&&abs(Jet_eta)<2.4)>=4) || (Sum\$(Jet_pt>25&&abs(Jet_eta)<2.4)>=2 && (Sum\$(Electron_pt>10&&abs(Electron_eta)<2.4)+Sum\$(Muon_pt>10&&abs(Muon_eta)<2.4&&Muon_mediumId>0))>=3) )'


p = PostProcessor('./', ["$INPUTFILENAMES"], cut=cut, modules=modules,\
    branchsel='PhysicsTools/NanoAODTools/python/postprocessing/modules/tW_scattering/keep_and_drop_in.txt',\
    outputbranchsel='PhysicsTools/NanoAODTools/python/postprocessing/modules/tW_scattering/keep_and_drop.txt' )

p.run()
EOL

#python PhysicsTools/NanoAODTools/scripts/nano_postproc.py ./ $INPUTFILENAMES \
#    --branch-selection PhysicsTools/NanoAODTools/python/postprocessing/modules/tW_scattering/keep_and_drop.txt \
#    --cut='nJet>0&&(nElectron+nMuon)>0' \
#    -I PhysicsTools.NanoAODTools.postprocessing.modules.tW_scattering.ObjectSelection selector2018,\
#       PhysicsTools.NanoAODTools.postprocessing.modules.tW_scattering.lumiWeightProducer lumiWeightProd(0.5)


mv $OUTFILE ${OUTPUTNAME}_${IFILE}.root

# Rigorous sweeproot which checks ALL branches for ALL events.
# If GetEntry() returns -1, then there was an I/O problem, so we will delete it
python << EOL
import ROOT as r
import os
import traceback
foundBad = False
try:
    f1 = r.TFile("${OUTPUTNAME}_${IFILE}.root")
    t = f1.Get("Events")
    nevts = t.GetEntries()
    print "[SweepRoot] ntuple has %i events." % t.GetEntries()
    if int(t.GetEntries()) <= 0:
        foundBad = True
    for i in range(0,t.GetEntries(),1):
        if t.GetEntry(i) < 0:
            foundBad = True
            print "[RSR] found bad event %i" % i
            break
except Exception as ex:
    msg = traceback.format_exc()
    print "Encounter error during SweepRoot:"
    print msg
    foundBad = True
if foundBad:
    print "[RSR] removing output file because it does not deserve to live"
    os.system("rm ${OUTPUTNAME}_${IFILE}.root")
else:
    print "[RSR] passed the rigorous sweeproot"
EOL

echo -e "\n--- end running ---\n" #                             <----- section division

# Copy back the output file

if [[ $(hostname) == "uaf"* ]]; then
    mkdir -p ${OUTPUTDIR}
    echo cp ${OUTPUTNAME}_${IFILE}.root ${OUTPUTDIR}/${OUTPUTNAME}_${IFILE}.root
    cp ${OUTPUTNAME}_${IFILE}.root ${OUTPUTDIR}/${OUTPUTNAME}_${IFILE}.root
    if [ ! -z $EXTRAOUT ]; then
        echo cp ${EXTRAOUT}_${IFILE}.root ${OUTPUTDIR}/${EXTRAOUT}/${EXTRAOUT}_${IFILE}.root
        cp ${EXTRAOUT}_${IFILE}.root ${OUTPUTDIR}/${EXTRAOUT}/${EXTRAOUT}_${IFILE}.root
    fi
else
    export LD_PRELOAD=/usr/lib64/gfal2-plugins//libgfal_plugin_xrootd.so # needed in cmssw versions later than 9_3_X
    gfal-copy -p -f -t 4200 --verbose file://`pwd`/${OUTPUTNAME}_${IFILE}.root gsiftp://gftp.t2.ucsd.edu${OUTPUTDIR}/${OUTPUTNAME}_${IFILE}.root --checksum ADLER32
    if [ ! -z $EXTRAOUT ]; then
        gfal-copy -p -f -t 4200 --verbose file://`pwd`/${EXTRAOUT}_${IFILE}.root gsiftp://gftp.t2.ucsd.edu${OUTPUTDIR}/${EXTRAOUT}/${EXTRAOUT}_${IFILE}.root --checksum ADLER32
    fi
fi


echo -e "\n--- cleaning up ---\n" #                             <----- section division
cd ../../
rm -r $CMSSW_VERSION/


