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

python PhysicsTools/NanoAODTools/scripts/run_processor.py $INPUTFILENAMES $SUMWEIGHT 

mv tree.root ${OUTPUTNAME}_${IFILE}.root

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
    echo ${OUTPUTDIR}/${OUTPUTNAME}_${IFILE}.root
    export LD_PRELOAD=/usr/lib64/gfal2-plugins//libgfal_plugin_xrootd.so # needed in cmssw versions later than 9_3_X
    env -i X509_USER_PROXY=${X509_USER_PROXY} gfal-copy -p -f -t 4200 --verbose file://`pwd`/${OUTPUTNAME}_${IFILE}.root gsiftp://gftp.t2.ucsd.edu${OUTPUTDIR}/${OUTPUTNAME}_${IFILE}.root --checksum ADLER32
    if [ ! -z $EXTRAOUT ]; then
        env -i X509_USER_PROXY=${X509_USER_PROXY} gfal-copy -p -f -t 4200 --verbose file://`pwd`/${EXTRAOUT}_${IFILE}.root gsiftp://gftp.t2.ucsd.edu${OUTPUTDIR}/${EXTRAOUT}/${EXTRAOUT}_${IFILE}.root --checksum ADLER32
    fi
fi


echo -e "\n--- cleaning up ---\n" #                             <----- section division
cd ../../
rm -r $CMSSW_VERSION/


