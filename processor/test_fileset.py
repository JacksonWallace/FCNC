import uproot
from Tools.helpers import get_samples
from Tools.config_helpers import redirector_ucsd, redirector_ucsd_mini, redirector_fnal
from Tools.nano_mapping import make_fileset, nano_mapping


fileset = make_fileset(['Data', 'DY', 'TT/TW', 'WZ/ZZ'], 2017, redirector=redirector_ucsd, small=False)
good = []
bad = []

for sample in list(fileset.keys()):
    for f_in in fileset[sample]:
        print (f_in)
        try:
            tree = uproot.open(f_in)["Events"]
            good.append(f_in)
        except OSError:
            print ("XRootD Error")
            bad.append(f_in)
        
