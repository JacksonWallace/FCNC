from metis.CMSSWTask import CMSSWTask
from metis.Sample import DirectorySample,DummySample
from metis.Path import Path
from metis.StatsParser import StatsParser
import time

#lhe = CMSSWTask(
#        sample = DirectorySample(
#            location="/hadoop/cms/store/user/dspitzba/tW_scattering/test/",
#            #globber="*seed6*.lhe",
#            #dataset="/stop-stop/procv2/LHE",
#            ),
#        events_per_output = 20,
#        total_nevents = 100,
#        pset = "cfgs/pset_gensim.py",
#        cmssw_version = "CMSSW_10_2_7",
#        scram_arch = "slc6_amd64_gcc700",
#        #split_within_files = True,
#        )

gen = CMSSWTask(
        sample = DummySample(N=1, dataset="/ttWq/privateMC_102x/GENSIM"),
        events_per_output = 10,
        total_nevents = 100,
        pset = "cfgs/pset_gensim.py",
        cmssw_version = "CMSSW_10_2_7",
        scram_arch = "slc6_amd64_gcc700",
        tag = 'v0',
        split_within_files = True,
        )


for task in tasks:
    task.process()
    summary = task.get_task_summary()
    total_summary[task.get_sample().get_datasetname()] = summary

StatsParser(data=total_summary, webdir="~/public_html/dump/metis/").do()
