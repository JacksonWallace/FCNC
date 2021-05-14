"""
Helpers for the use of raw NanoAOD
"""
#from yaml import load, dump
#try:
#    from yaml import CLoader as Loader, CDumper as Dumper
#except ImportError:
#    from yaml import Loader, Dumper
from metis.Sample import DBSSample
from Tools.helpers import get_samples
from Tools.config_helpers import redirector_ucsd, load_yaml, data_path

import uproot


def nano_mapping(year=2018):
    print(data_path+'nano_mapping_'+str(year)+'.yaml' )
    nano_mapping = load_yaml(data_path+'nano_mapping_'+str(year)+'.yaml')
    return nano_mapping

def make_fileset(datasets, samples, year=2018, redirector=redirector_ucsd, small=False, n_max=1):
    fileset = {}
    nano_mappings = nano_mapping(year)
    for dataset in datasets:
        for nano_sample in nano_mappings[dataset]:
            dbs_files = DBSSample(dataset=nano_sample).get_files()
            files = [ redirector+x.name for x in dbs_files ]
            if not small:
                fileset.update({nano_sample: files})
            else:
                fileset.update({nano_sample: files[:n_max]})

    return fileset


if __name__ == '__main__':

    samples = get_samples()
    samples.update(get_samples('samples_QCD.yaml'))

    fileset = make_fileset(['TTW', 'TTZ', 'QCD'], samples)
