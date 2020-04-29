'''
small script that reades histograms from an archive and saves figures in a public space

'''


from coffea import hist
import pandas as pd
#import uproot_methods

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from tW_scattering.Tools.helpers import *
from klepto.archives import dir_archive

def saveFig( ax, path, name, scale='linear' ):
    outdir = os.path.join(path,scale)
    finalizePlotDir(outdir)
    ax.set_yscale(scale)
    if scale == 'log':
        ax.set_ylim(0.001,1)
    ax.figure.savefig(os.path.join(outdir, "{}.pdf".format(name)))
    #ax.clear()

# load the configuration
cfg = loadConfig()

# load the results
cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cfg['caches']['simpleProcessor']), serialized=True)
cache.load()

histograms = cache.get('histograms')
output = cache.get('simple_output')
plotDir = os.path.expandvars(cfg['meta']['plots']) + '/simplePlots/'
finalizePlotDir(plotDir)

if not histograms:
    print ("Couldn't find histograms in archive. Quitting.")
    exit()

print ("Plots will appear here:", plotDir )

for name in histograms:
    skip = False
    histogram = output[name]
    if name == 'MET_pt':
        # rebin
        new_met_bins = hist.Bin('MET_pt', r'$E_T^{miss} \ (GeV)$', 20, 0, 200)
        histogram = histogram.rebin('MET_pt', new_met_bins)
    elif name == 'W_pt_notFromTop':
        # rebin
        new_pt_bins = hist.Bin('W_pt', r'$p_{T}(W) \ (GeV)$', 25, 0, 500)
        histogram = histogram.rebin('W_pt', new_pt_bins)
    else:
        skip = True

    if not skip:
        ax = hist.plot1d(histogram,overlay="dataset", stack=True) # make density plots because we don't care about x-sec differences
        for l in ['linear', 'log']:
            saveFig(ax, plotDir, name, scale=l)
        ax.clear()

        ax = hist.plot1d(histogram,overlay="dataset", density=True, stack=False) # make density plots because we don't care about x-sec differences
        for l in ['linear', 'log']:
            saveFig(ax, plotDir, name+'_shape', scale=l)
        ax.clear()

