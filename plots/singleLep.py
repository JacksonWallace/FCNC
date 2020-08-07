'''
small script that reades histograms from an archive and saves figures in a public space

ToDo:
[x] Cosmetics (labels etc)
[x] ratio pad!
  [x] pseudo data
    [ ] -> move to processor to avoid drawing toys every time!
[x] uncertainty band
[ ] fix shapes
'''


from coffea import hist
import pandas as pd
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from klepto.archives import dir_archive

# import all the colors and tools for plotting
from Tools.helpers import loadConfig
from helpers import *

# load the configuration
cfg = loadConfig()

# load the results
cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cfg['caches']['singleLep']), serialized=True)
cache.load()

histograms = cache.get('histograms')
output = cache.get('simple_output')
plotDir = os.path.expandvars(cfg['meta']['plots']) + '/plots1l/'
finalizePlotDir(plotDir)

if not histograms:
    print ("Couldn't find histograms in archive. Quitting.")
    exit()

print ("Plots will appear here:", plotDir )

bins = {\
    'MET_pt':   {'axis': 'pt',              'bins': hist.Bin('pt', r'$E_T^{miss} \ (GeV)$', 20, 0, 200)},
    'MT':       {'axis': 'pt',              'bins': hist.Bin('pt', r'$M_T \ (GeV)$', 20, 0, 200)},
    'N_jet':    {'axis': 'multiplicity',    'bins': hist.Bin('multiplicity', r'$N_{jet}$', 15, -0.5, 14.5)},
    'N_spec':   {'axis': 'multiplicity',    'bins': hist.Bin('multiplicity', r'$N_{jet, fwd}$', 6, -0.5, 5.5)},
    'N_b':      {'axis': 'multiplicity',    'bins': hist.Bin('multiplicity', r'$N_{b-jet}$', 5, -0.5, 4.5)},
    'pt_spec_max': {'axis': 'pt',           'bins': hist.Bin('pt', r'$p_{T, fwd jet}\ (GeV)$', 20, 0, 400)},
    'mbj_max':  {'axis': 'mass',            'bins': hist.Bin('mass', r'$M(b, light) \ (GeV)$', 25, 0, 1500)},
    'mjj_max':  {'axis': 'mass',            'bins': hist.Bin('mass', r'$M(b, light) \ (GeV)$', 25, 0, 1500)},
    'mlb_min':  {'axis': 'mass',            'bins': hist.Bin('mass', r'$M(b, light) \ (GeV)$', 15, 0, 300)},
    'mlb_max':  {'axis': 'mass',            'bins': hist.Bin('mass', r'$M(b, light) \ (GeV)$', 25, 0, 500)},
    'mlj_min':  {'axis': 'mass',            'bins': hist.Bin('mass', r'$M(b, light) \ (GeV)$', 15, 0, 300)},
    'mlj_max':  {'axis': 'mass',            'bins': hist.Bin('mass', r'$M(b, light) \ (GeV)$', 25, 0, 1000)},
    'HT':       {'axis': 'ht',              'bins': hist.Bin('ht', r'$M(b, light) \ (GeV)$', 30, 0, 1500)},
    'ST':       {'axis': 'ht',              'bins': hist.Bin('ht', r'$M(b, light) \ (GeV)$', 30, 0, 1500)},
    'FWMT1':    {'axis': 'norm',            'bins': hist.Bin('norm', r'FWMT1', 25, 0, 1)},
    'FWMT2':    {'axis': 'norm',            'bins': hist.Bin('norm', r'FWMT2', 25, 0, 1)},
    }

for name in histograms:
    print (name)
    skip = False
    histogram = output[name]
    
    if not name in bins.keys():
        continue

    axis = bins[name]['axis']
    print (name, axis)
    histogram = histogram.rebin(axis, bins[name]['bins'])

    y_max = histogram.sum("dataset").values(overflow='over')[()].max()
    y_over = histogram.sum("dataset").values(overflow='over')[()][-1]

    # get pseudo data
    bin_values = histogram.axis(axis).centers(overflow='over')
    poisson_means = histogram.sum('dataset').values(overflow='over')[()]
    values = np.repeat(bin_values, np.random.poisson(np.maximum(np.zeros(len(poisson_means)), poisson_means)))
    if axis == 'pt':
        histogram.fill(dataset='pseudodata', pt=values)
    elif axis == 'mass':
        histogram.fill(dataset='pseudodata', mass=values)
    elif axis == 'multiplicity':
        histogram.fill(dataset='pseudodata', multiplicity=values)
    elif axis == 'ht':
        histogram.fill(dataset='pseudodata', ht=values)
    elif axis == 'norm':
        histogram.fill(dataset='pseudodata', norm=values)

    
    import re
    notdata = re.compile('(?!pseudodata)')

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)

    # get axes
    hist.plot1d(histogram[notdata],overlay="dataset", ax=ax, stack=True, overflow='over', clear=False, line_opts=None, fill_opts=fill_opts, error_opts=error_opts, order=['tW_scattering', 'TTX', 'TTW','ttbar','wjets']) #error_opts??
    hist.plot1d(histogram['pseudodata'], overlay="dataset", ax=ax, overflow='over', error_opts=data_err_opts, clear=False)

    # build ratio
    hist.plotratio(
        num=histogram['pseudodata'].sum("dataset"),
        denom=histogram[notdata].sum("dataset"),
        ax=rax,
        error_opts=data_err_opts,
        denom_fill_opts={},
        guide_opts={},
        unc='num',
        overflow='over'
    )


    for l in ['linear', 'log']:
        saveFig(fig, ax, rax, plotDir, name, scale=l, shape=False, y_max=y_max)
    fig.clear()
    rax.clear()
    ax.clear()

    
    try:
        fig, ax = plt.subplots(1,1,figsize=(7,7))
        notdata = re.compile('(?!pseudodata|wjets|diboson)')
        hist.plot1d(histogram[notdata],overlay="dataset", density=True, stack=False, overflow='over', ax=ax) # make density plots because we don't care about x-sec differences
        for l in ['linear', 'log']:
            saveFig(fig, ax, None, plotDir, name+'_shape', scale=l, shape=True)
        fig.clear()
        ax.clear()
    except ValueError:
        print ("Can't make shape plot for a weird reason")

    fig.clear()
    ax.clear()

    plt.close()


print ()
print ("Plots are here: http://uaf-10.t2.ucsd.edu/~%s/"%os.path.expandvars('$USER')+str(plotDir.split('public_html')[-1]) )
