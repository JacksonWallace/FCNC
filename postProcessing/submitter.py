
import time

from metis.Sample import DirectorySample
from metis.CondorTask import CondorTask
from metis.StatsParser import StatsParser
from metis.Utils import do_cmd

# example
sample = DirectorySample(dataset='TTWJetsToLNu_Autumn18v4', location='/hadoop/cms/store/user/dspitzba/nanoAOD/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1/')

outDir = "/hadoop/cms/store/user/dspitzba/tW_scattering_babies/"

maker_tasks = []
merge_tasks = []


if True:
    maker_task = CondorTask(
        sample = sample,
        #DirectorySample(
        #    dataset='TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1',
        #    location='/hadoop/cms/store/user/dspitzba/nanoAOD/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1/',
        #),
        # open_dataset = True, flush = True,
        executable = "executable.sh",
        #tarfile = "merge_scripts.tar.gz",
        files_per_output = 1,
        output_dir = outDir+sample.get_datasetname(),
        output_name = sample.get_datasetname() + ".root",
        output_is_tree = True,
        # check_expectedevents = True,
        tag = 'v0p1',
        condor_submit_params = {"sites":"T2_US_UCSD,UAF"},
        cmssw_version = "CMSSW_10_2_9",
        scram_arch = "slc6_amd64_gcc700",
        # recopy_inputs = True,
        # no_load_from_backup = True,
        min_completion_fraction = 0.99,
    )
    
    maker_tasks.append(maker_task)



if False:
    merge_task = CondorTask(
        sample = DirectorySample(
            dataset="merge_"+babyname,
            location=maker_task.get_outputdir(),
        ),
        # open_dataset = True, flush = True,
        executable = "merge_executable.sh",
        tarfile = "merge_scripts.tar.gz",
        files_per_output = 100000,
        output_dir = maker_task.get_outputdir() + "/merged",
        output_name = babyname + ".root",
        output_is_tree = True,
        # check_expectedevents = True,
        tag = tag,
        # condor_submit_params = {"sites":"T2_US_UCSD"},
        # cmssw_version = "CMSSW_9_2_8",
        # scram_arch = "slc6_amd64_gcc530",
        condor_submit_params = {"sites":"T2_US_UCSD,UAF"},
        cmssw_version = "CMSSW_10_2_14",
        scram_arch = "slc6_amd64_gcc700",
        # recopy_inputs = True,
        # no_load_from_backup = True,
        min_completion_fraction = minfrac,
    )

    merge_tasks.append(merge_task)

if True:
    for i in range(100):
        total_summary = {}
    
        #for maker_task, merge_task in zip(maker_tasks,merge_tasks):
        for maker_task in maker_tasks:
            maker_task.process()
    
            frac = maker_task.complete(return_fraction=True)
            if frac >= maker_task.min_completion_fraction:
            # if maker_task.complete():
                do_cmd("mkdir -p {}/merged".format(maker_task.get_outputdir()))
                do_cmd("mkdir -p {}/skimmed".format(maker_task.get_outputdir()))
            #    merge_task.reset_io_mapping()
            #    merge_task.update_mapping()
            #    merge_task.process()
    
            total_summary[maker_task.get_sample().get_datasetname()] = maker_task.get_task_summary()
            #total_summary[merge_task.get_sample().get_datasetname()] = merge_task.get_task_summary()
    
        # parse the total summary and write out the dashboard
        StatsParser(data=total_summary, webdir="~/public_html/dump/metis_tW_scattering/").do()
    
        # 1 hr power nap
        time.sleep(15.*60)



