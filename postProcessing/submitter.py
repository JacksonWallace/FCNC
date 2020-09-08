
import time

from metis.Sample import DirectorySample, DBSSample
from metis.CondorTask import CondorTask
from metis.StatsParser import StatsParser
from metis.Utils import do_cmd

from Tools.helpers import *

# load samples
import yaml
from yaml import Loader, Dumper

import os


data_path = os.path.expandvars('$TWHOME/data/')
with open(data_path+'samples.yaml') as f:
    samples = yaml.load(f, Loader=Loader)

# load config
cfg = loadConfig()

print ("Loaded version %s from config."%cfg['meta']['version'])

import argparse

argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--version', action='store', default=None, help="Define a new version number")
argParser.add_argument('--newVersion', action='store_true', default=None, help="Create a version and tag automatically?")
argParser.add_argument('--dryRun', action='store_true', default=None, help="Don't submit?")
args = argParser.parse_args()

version = str(cfg['meta']['version'])

# if no version is defined, increase last version number by one
if args.newVersion:
    tag_str = str(cfg['meta']['version'])
    version = '.'.join(tag_str.split('.')[:-1]+[str(int(tag_str.split('.')[-1])+1)])
    cfg['meta']['version'] = version
elif args.version:
    version = args.version
    cfg['meta']['version'] = version
    # should check that the format is the same

tag = version.replace('.','p')

## create a new tag of nanoAOD-tools on the fly
if args.newVersion or args.version:
    print ("Commiting and creating new tag: %s"%tag)
    import subprocess
    subprocess.call("cd $CMSSW_BASE/src/PhysicsTools/NanoAODTools/; git commit -am 'latest'; git tag %s; git push ownFork --tags; cd"%tag, shell=True)
    dumpConfig(cfg)
    
    # Dumpong the config

# example
sample = DirectorySample(dataset='TTWJetsToLNu_Autumn18v4', location='/hadoop/cms/store/user/dspitzba/nanoAOD/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1/')


#metisSamples = []
#for sample in samples.keys():   

#outDir = os.path.join(version, tag)
outDir = os.path.join(cfg['meta']['localSkim'], tag)

print ("Output will be here: %s"%outDir)

maker_tasks = []
merge_tasks = []

#raise NotImplementedError

#samples = {'/hadoop/cms/store/user/dspitzba/tW_scattering/tW_scattering/nanoAOD/': samples['/hadoop/cms/store/user/dspitzba/tW_scattering/tW_scattering/nanoAOD/']}

#if True:
for s in samples.keys():
    if samples[s]['path'] is not None:
        sample = DirectorySample(dataset = samples[s]['name'], location = samples[s]['path'])
    else:
        sample = DBSSample(dataset = s) # should we make use of the files??

    lumiWeightString = 1000*samples[s]['xsec']/samples[s]['sumWeight']

    #tag = str(cfg['meta']['version']).replace('.','p')
    
    maker_task = CondorTask(
        sample = sample,
            #'/hadoop/cms/store/user/dspitzba/nanoAOD/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1/',
        # open_dataset = True, flush = True,
        executable = "executable.sh",
        arguments = "%s %s"%(tag, lumiWeightString),
        #tarfile = "merge_scripts.tar.gz",
        files_per_output = 3,
        output_dir = os.path.join(outDir, samples[s]['name']),
        output_name = "nanoSkim.root",
        output_is_tree = True,
        # check_expectedevents = True,
        tag = tag,
        condor_submit_params = {"sites":"T2_US_UCSD,UAF"},
        cmssw_version = "CMSSW_10_2_9",
        scram_arch = "slc6_amd64_gcc700",
        # recopy_inputs = True,
        # no_load_from_backup = True,
        min_completion_fraction = 0.99,
    )
    
    maker_tasks.append(maker_task)



##if False:
#    merge_task = CondorTask(
#        sample = DirectorySample(
#            dataset="merge_"+samples[s]['name'],
#            location=maker_task.get_outputdir(),
#        ),
#        # open_dataset = True, flush = True,
#        executable = "merge_executable.sh",
#        arguments = "%s %s"%(tag, lumiWeightString),
#        #tarfile = "merge_scripts.tar.gz",
#        files_per_output = 10,
#        output_dir = maker_task.get_outputdir() + "/merged",
#        output_name = "nanoSkim.root",
#        output_is_tree = True,
#        # check_expectedevents = True,
#        tag = tag,
#        # condor_submit_params = {"sites":"T2_US_UCSD"},
#        # cmssw_version = "CMSSW_9_2_8",
#        # scram_arch = "slc6_amd64_gcc530",
#        condor_submit_params = {"sites":"T2_US_UCSD,UAF"},
#        cmssw_version = "CMSSW_10_2_9",
#        scram_arch = "slc6_amd64_gcc700",
#        # recopy_inputs = True,
#        # no_load_from_backup = True,
#        min_completion_fraction = 0.99,
#    )
#
#    merge_tasks.append(merge_task)

if not args.dryRun:
    for i in range(100):
        total_summary = {}
    
        #for maker_task, merge_task in zip(maker_tasks,merge_tasks):
        for maker_task in maker_tasks:
            maker_task.process()
    
            frac = maker_task.complete(return_fraction=True)
            #if frac >= maker_task.min_completion_fraction:
            ## if maker_task.complete():
            #    do_cmd("mkdir -p {}/merged".format(maker_task.get_outputdir()))
            #    do_cmd("mkdir -p {}/skimmed".format(maker_task.get_outputdir()))
            #    merge_task.reset_io_mapping()
            #    merge_task.update_mapping()
            #    merge_task.process()
    
            total_summary[maker_task.get_sample().get_datasetname()] = maker_task.get_task_summary()
            #total_summary[merge_task.get_sample().get_datasetname()] = merge_task.get_task_summary()
 
        print (frac)
   
        # parse the total summary and write out the dashboard
        StatsParser(data=total_summary, webdir="~/public_html/dump/metis_tW_scattering/").do()
    
        # 15 min power nap
        time.sleep(15.*60)



