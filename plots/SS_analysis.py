import os
try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea import processor, hist

import numpy as np

from Tools.config_helpers import loadConfig, make_small

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from plots.helpers import makePlot

from klepto.archives import dir_archive


if __name__ == '__main__':

    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--small', action='store_true', default=None, help="Run on a small subset?")
    argParser.add_argument('--verysmall', action='store_true', default=None, help="Run on a small subset?")
    argParser.add_argument('--year', action='store', default='2018', help="Which year to run on?")
    args = argParser.parse_args()

    small       = args.small
    verysmall   = args.verysmall
    if verysmall:
        small = True
    year        = int(args.year)

    cfg = loadConfig()

    plot_dir = os.path.expandvars(cfg['meta']['plots'])

    cacheName = 'SS_analysis_%s'%year
    if small: cacheName += '_small'
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)
    

    cache.load()

    output = cache.get('simple_output')
    
    # defining some new axes for rebinning.
    N_bins = hist.Bin('multiplicity', r'$N$', 10, -0.5, 9.5)
    N_bins_red = hist.Bin('multiplicity', r'$N$', 5, -0.5, 4.5)
    mass_bins = hist.Bin('mass', r'$M\ (GeV)$', 20, 0, 200)
    pt_bins = hist.Bin('pt', r'$p_{T}\ (GeV)$', 30, 0, 300)
    pt_bins_coarse = hist.Bin('pt', r'$p_{T}\ (GeV)$', 10, 0, 300)
    pt_bins_coarse_red = hist.Bin('pt', r'$p_{T}\ (GeV)$', 10, 0, 100)
    pt_bins_ext = hist.Bin('pt', r'$p_{T}\ (GeV)$', 10, 0, 1000)
    eta_bins = hist.Bin('eta', r'$\eta $', 25, -5.0, 5.0)
    score_bins = hist.Bin("score",          r"N", 25, 0, 1)
 

    my_labels = {
        'topW_v3': 'top-W scat.',
        'topW_EFT_cp8': 'EFT, cp8',
        'topW_EFT_mix': 'EFT mix',
        'TTZ': r'$t\bar{t}Z$',
        'TTW': r'$t\bar{t}W$',
        'TTH': r'$t\bar{t}H$',
        'diboson': 'VV/VVV',
        'rare': 'rare',
        'ttbar': r'$t\bar{t}$',
        'np_obs_mc': 'nonprompt (MC true)',
        'np_est_mc': 'nonprompt (MC est)',
        'cf_obs_mc': 'charge flip (MC true)',
        'cf_est_mc': 'charge flip (MC est)',
        'np_est_data': 'nonprompt (est)',
        'cf_est_data': 'charge flip (est)',
    }
    
    my_colors = {
        'topW_v3': '#FF595E',
        'topW_EFT_cp8': '#000000',
        'topW_EFT_mix': '#0F7173',
        'TTZ': '#FFCA3A',
        'TTW': '#8AC926',
        'TTH': '#34623F',
        'diboson': '#525B76',
        'rare': '#EE82EE',
        'ttbar': '#1982C4',
        'np_obs_mc': '#1982C4',
        'np_est_mc': '#1982C4',
        'np_est_data': '#1982C4',
        'cf_obs_mc': '#0F7173',
        'cf_est_mc': '#0F7173',
        'cf_est_data': '#0F7173',
    }
   

    #for k in my_labels.keys():

    ## DATA DRIVEN BKG ESTIMATES


    all_processes = [ x[0] for x in output['node'].values().keys() ]

    if True:

        sub_dir = '/SS/v21/dd/'

        data    = ['DoubleMuon', 'MuonEG', 'EGamma']
        order   = ['np_est_data', 'TTW', 'TTH', 'TTZ','rare', 'diboson', 'cf_est_data', 'topW_v3']
        signals = []
        omit    = [ x for x in all_processes if (x not in signals and x not in order and x not in data) ]

        makePlot(output, 'MET', 'pt',
             data=data,
             bins=pt_bins_coarse, log=False, normalize=True, axis_label=r'$p_{T}^{miss}$',
             new_colors=my_colors, new_labels=my_labels,
             order=order,
             signals=signals,
             omit=omit,
             save=os.path.expandvars(plot_dir+sub_dir+'MET_pt'),
            )

        makePlot(output, 'fwd_jet', 'pt',
             data=data,
             bins=pt_bins_coarse, log=False, normalize=True, axis_label=r'$p_{T}$',
             new_colors=my_colors, new_labels=my_labels,
             order=order,
             signals=signals,
             omit=omit,
             save=os.path.expandvars(plot_dir+sub_dir+'fwd_jet_pt'),
            )

        makePlot(output, 'node', 'multiplicity',
             data=data,
             bins=N_bins_red, log=False, normalize=False, axis_label='best node',
             new_colors=my_colors, new_labels=my_labels,
             order=order,
             signals=signals,
             omit=omit,
             save=os.path.expandvars(plot_dir+sub_dir+'best_node'),
            )

        makePlot(output, 'node0_score', 'score',
             data=[],
             bins=score_bins, log=False, normalize=False, axis_label='Score',
             new_colors=my_colors, new_labels=my_labels,
             order=order,
             signals=signals,
             omit=omit+data,
             save=os.path.expandvars(plot_dir+sub_dir+'node0_score'),
            )

        makePlot(output, 'node1_score', 'score',
             data=[],
             bins=score_bins, log=False, normalize=False, axis_label='Score',
             new_colors=my_colors, new_labels=my_labels,
             order=order,
             signals=signals,
             omit=omit+data,
             save=os.path.expandvars(plot_dir+sub_dir+'node1_score'),
            )

        makePlot(output, 'node2_score', 'score',
             data=[],
             bins=score_bins, log=False, normalize=False, axis_label='Score',
             new_colors=my_colors, new_labels=my_labels,
             order=order,
             signals=signals,
             omit=omit+data,
             save=os.path.expandvars(plot_dir+sub_dir+'node2_score'),
            )

        makePlot(output, 'node3_score', 'score',
             data=data,
             bins=score_bins, log=False, normalize=False, axis_label='Score',
             new_colors=my_colors, new_labels=my_labels,
             order=order,
             signals=signals,
             omit=omit,
             save=os.path.expandvars(plot_dir+sub_dir+'node3_score'),
            )

        makePlot(output, 'node4_score', 'score',
             data=data,
             bins=score_bins, log=False, normalize=False, axis_label='Score',
             new_colors=my_colors, new_labels=my_labels,
             order=order,
             signals=signals,
             omit=omit,
             save=os.path.expandvars(plot_dir+sub_dir+'node4_score'),
            )

        makePlot(output, 'node0_score_incl', 'score',
             data=[],
             bins=score_bins, log=False, normalize=False, axis_label='Score',
             new_colors=my_colors, new_labels=my_labels,
             order=order,
             signals=signals,
             omit=omit+data,
             save=os.path.expandvars(plot_dir+sub_dir+'node0_score_incl'),
            )

        makePlot(output, 'node1_score_incl', 'score',
             data=[],
             bins=score_bins, log=False, normalize=False, axis_label='Score',
             new_colors=my_colors, new_labels=my_labels,
             order=order,
             signals=signals,
             omit=omit+data,
             save=os.path.expandvars(plot_dir+sub_dir+'node1_score_incl'),
            )

        makePlot(output, 'node2_score_incl', 'score',
             data=[],
             bins=score_bins, log=False, normalize=False, axis_label='Score',
             new_colors=my_colors, new_labels=my_labels,
             order=order,
             signals=signals,
             omit=omit+data,
             save=os.path.expandvars(plot_dir+sub_dir+'node2_score_incl'),
            )

        makePlot(output, 'node3_score_incl', 'score',
             data=data,
             bins=score_bins, log=False, normalize=False, axis_label='Score',
             new_colors=my_colors, new_labels=my_labels,
             order=order,
             signals=signals,
             omit=omit,
             save=os.path.expandvars(plot_dir+sub_dir+'node3_score_incl'),
            )

        makePlot(output, 'node4_score_incl', 'score',
             data=data,
             bins=score_bins, log=False, normalize=False, axis_label='Score',
             new_colors=my_colors, new_labels=my_labels,
             order=order,
             signals=signals,
             omit=omit,
             save=os.path.expandvars(plot_dir+sub_dir+'node4_score_incl'),
            )

    ## MC DRIVEN BKG ESTIMATES

    sub_dir = '/SS/v21/mc/'

    if True:

        data    = ['DoubleMuon', 'MuonEG', 'EGamma']
        order   = ['rare', 'diboson', 'TTW', 'TTH', 'TTZ', 'np_obs_mc', 'cf_obs_mc', 'topW_v3', 'ttbar']
        signals = []
        omit    = [ x for x in all_processes if (x not in signals and x not in order and x not in data) ]

        makePlot(output, 'MET', 'pt',
             data=data,
             bins=pt_bins_coarse, log=False, normalize=True, axis_label=r'$p_{T}^{miss}$',
             new_colors=my_colors, new_labels=my_labels,
             order=order,
             signals=signals,
             omit=omit,
             save=os.path.expandvars(plot_dir+sub_dir+'MET_pt'),
            )

        makePlot(output, 'fwd_jet', 'pt',
             data=data,
             bins=pt_bins_coarse, log=False, normalize=True, axis_label=r'$p_{T}$',
             new_colors=my_colors, new_labels=my_labels,
             order=order,
             signals=signals,
             omit=omit,
             save=os.path.expandvars(plot_dir+sub_dir+'fwd_jet_pt'),
            )



    data    = []
    order   = ['np_est_mc', 'TTW', 'TTH', 'TTZ', 'rare', 'diboson', 'cf_est_mc', 'topW_v3']
    signals = []
    omit    = [ x for x in all_processes if (x not in signals and x not in order and x not in data) ]

    makePlot(output, 'node', 'multiplicity',
         data=[],
         bins=N_bins_red, log=False, normalize=False, axis_label='best node',
         new_colors=my_colors, new_labels=my_labels,
         order=order,
         signals=signals,
         omit=omit+data,
         save=os.path.expandvars(plot_dir+sub_dir+'best_node'),
        )

    data    = []
    order   = ['np_obs_mc', 'TTW', 'TTH', 'TTZ', 'rare', 'diboson', 'cf_obs_mc', 'topW_v3']
    signals = []
    omit    = [ x for x in all_processes if (x not in signals and x not in order and x not in data) ]

    makePlot(output, 'node', 'multiplicity',
         data=[],
         bins=N_bins_red, log=False, normalize=False, axis_label='best node',
         new_colors=my_colors, new_labels=my_labels,
         order=order,
         signals=signals,
         omit=omit+data,
         save=os.path.expandvars(plot_dir+sub_dir+'best_node_MC'),
        )


    data    = []
    order   = ['np_est_mc', 'TTW', 'TTH', 'TTZ', 'rare', 'diboson', 'cf_est_mc', 'topW_v3']
    signals = []
    omit    = [ x for x in all_processes if (x not in signals and x not in order and x not in data) ]

    makePlot(output, 'node0_score', 'score',
         data=[],
         bins=score_bins, log=False, normalize=False, axis_label='Score',
         new_colors=my_colors, new_labels=my_labels,
         order=order,
         signals=signals,
         omit=omit+data,
         save=os.path.expandvars(plot_dir+sub_dir+'node0_score'),
        )

    data    = []
    order   = ['np_est_mc', 'TTW', 'TTH', 'TTZ', 'topW_v3']
    signals = []
    omit    = [ x for x in all_processes if (x not in signals and x not in order and x not in data) ]

    makePlot(output, 'node0_score', 'score',
         data=[],
         bins=score_bins, log=False, normalize=False, axis_label='Score',
         shape=True, ymax=0.25,
         new_colors=my_colors, new_labels=my_labels,
         order=order,
         signals=signals,
         omit=omit+data,
         save=os.path.expandvars(plot_dir+sub_dir+'node0_score_shape'),
        )


    data    = []
    order   = ['np_est_mc', 'TTW', 'TTH', 'TTZ', 'rare', 'diboson', 'cf_est_mc', 'topW_v3']
    signals = []
    omit    = [ x for x in all_processes if (x not in signals and x not in order and x not in data) ]

    makePlot(output, 'node0_score_incl', 'score',
         data=[],
         bins=score_bins, log=True, normalize=False, axis_label='Score',
         new_colors=my_colors, new_labels=my_labels,
         order=order,
         signals=signals,
         omit=omit+data,
         save=os.path.expandvars(plot_dir+sub_dir+'node0_score_incl'),
        )

    data    = []
    order   = ['np_est_mc', 'TTW', 'TTH', 'TTZ', 'topW_v3']
    signals = []
    omit    = [ x for x in all_processes if (x not in signals and x not in order and x not in data) ]

    makePlot(output, 'node0_score_incl', 'score',
         data=[],
         bins=score_bins, log=False, normalize=False, axis_label='Score',
         shape=True, ymax=0.35,
         new_colors=my_colors, new_labels=my_labels,
         order=order,
         signals=signals,
         omit=omit+data,
         save=os.path.expandvars(plot_dir+sub_dir+'node0_score_incl_shape'),
        )


    ### NP estimate closure test for inputs ###

    data    = []
    order   = ['np_est_mc', 'np_obs_mc']
    signals = []
    omit    = [ x for x in all_processes if (x not in signals and x not in order and x not in data) ]

    makePlot(output, 'HT', 'ht',
         data=[],
         bins=pt_bins_ext, log=False, normalize=False, axis_label=r'$H_{T}\ (GeV)$',
         shape=True, ymax=0.6,
         new_colors={'np_est_mc': my_colors['ttbar'], 'np_obs_mc': my_colors['TTW']},
         new_labels=my_labels,
         order=order,
         signals=signals,
         omit=omit+data,
         save=os.path.expandvars(plot_dir+sub_dir+'np_closure_ht_shape'),
        )

    makePlot(output, 'ST', 'ht',
         data=[],
         bins=pt_bins_ext, log=False, normalize=False, axis_label=r'$S_{T}\ (GeV)$',
         shape=True, ymax=0.6,
         new_colors={'np_est_mc': my_colors['ttbar'], 'np_obs_mc': my_colors['TTW']},
         new_labels=my_labels,
         order=order,
         signals=signals,
         omit=omit+data,
         save=os.path.expandvars(plot_dir+sub_dir+'np_closure_st_shape'),
        )

    makePlot(output, 'lead_lep', 'pt',
         data=[],
         bins=pt_bins_coarse, log=False, normalize=False, axis_label=r'$p_{T}\ (GeV)$',
         shape=True, ymax=0.6,
         new_colors={'np_est_mc': my_colors['ttbar'], 'np_obs_mc': my_colors['TTW']},
         new_labels=my_labels,
         order=order,
         signals=signals,
         omit=omit+data,
         save=os.path.expandvars(plot_dir+sub_dir+'np_closure_lead_lep_pt_shape'),
        )

    makePlot(output, 'lead_lep', 'pt',
         data=[],
         bins=pt_bins_coarse, log=False, normalize=False, axis_label=r'$p_{T}\ (GeV)$',
         shape=True, ymax=0.6,
         new_colors={'np_est_mc': my_colors['ttbar'], 'np_obs_mc': my_colors['TTW']},
         new_labels=my_labels,
         order=order,
         signals=signals,
         omit=omit+data,
         save=os.path.expandvars(plot_dir+sub_dir+'np_closure_lead_lep_pt_shape'),
        )

    makePlot(output, 'trail_lep', 'pt',
         data=[],
         bins=pt_bins_coarse_red, log=False, normalize=False, axis_label=r'$p_{T}\ (GeV)$',
         shape=True, ymax=0.6,
         new_colors={'np_est_mc': my_colors['ttbar'], 'np_obs_mc': my_colors['TTW']},
         new_labels=my_labels,
         order=order,
         signals=signals,
         omit=omit+data,
         save=os.path.expandvars(plot_dir+sub_dir+'np_closure_trail_lep_pt_shape'),
        )

    makePlot(output, 'lead_lep', 'eta',
         data=[],
         bins=eta_bins, log=False, normalize=False, axis_label=r'$\eta$',
         shape=True, ymax=0.2,
         new_colors={'np_est_mc': my_colors['ttbar'], 'np_obs_mc': my_colors['TTW']},
         new_labels=my_labels,
         order=order,
         signals=signals,
         omit=omit+data,
         save=os.path.expandvars(plot_dir+sub_dir+'np_closure_lead_lep_eta_shape'),
        )

    makePlot(output, 'MET', 'pt',
         data=[],
         bins=pt_bins_coarse, log=False, normalize=False, axis_label=r'$p_{T}\ (GeV)$',
         shape=True, ymax=0.6,
         new_colors={'np_est_mc': my_colors['ttbar'], 'np_obs_mc': my_colors['TTW']},
         new_labels=my_labels,
         order=order,
         signals=signals,
         omit=omit+data,
         save=os.path.expandvars(plot_dir+sub_dir+'np_closure_MET_pt_shape'),
        )

    makePlot(output, 'fwd_jet', 'pt',
         data=[],
         bins=pt_bins_coarse, log=False, normalize=False, axis_label=r'$p_{T}\ (GeV)$',
         shape=True, ymax=0.6,
         new_colors={'np_est_mc': my_colors['ttbar'], 'np_obs_mc': my_colors['TTW']},
         new_labels=my_labels,
         order=order,
         signals=signals,
         omit=omit+data,
         save=os.path.expandvars(plot_dir+sub_dir+'np_closure_fwd_jet_pt_shape'),
        )

    makePlot(output, 'N_fwd', 'multiplicity',
         data=[],
         bins=N_bins_red, log=False, normalize=False, axis_label=r'$N_{fwd\ jet}$',
         shape=True, ymax=0.6,
         new_colors={'np_est_mc': my_colors['ttbar'], 'np_obs_mc': my_colors['TTW']},
         new_labels=my_labels,
         order=order,
         signals=signals,
         omit=omit+data,
         save=os.path.expandvars(plot_dir+sub_dir+'np_closure_N_fwd_shape'),
        )

    makePlot(output, 'N_jet', 'multiplicity',
         data=[],
         bins=N_bins, log=False, normalize=False, axis_label=r'$N_{jet}$',
         shape=True, ymax=0.6,
         new_colors={'np_est_mc': my_colors['ttbar'], 'np_obs_mc': my_colors['TTW']},
         new_labels=my_labels,
         order=order,
         signals=signals,
         omit=omit+data,
         save=os.path.expandvars(plot_dir+sub_dir+'np_closure_N_jet_shape'),
        )


    raise NotImplementedError


    makePlot(output, 'nGenL', 'multiplicity',
         data=[],
         bins=N_bins_red, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson'],
         signals=[],
         omit=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         overlay=output['nGenL']['topW_v3'],
         save=os.path.expandvars(plot_dir+'/SS/nGenL_test'),
        )

    makePlot(output, 'nGenL', 'multiplicity',
         data=[],
         bins=N_bins_red, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/nGenL'),
        )

    makePlot(output, 'nGenTau', 'multiplicity',
         data=[],
         bins=N_bins_red, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/nGenTau'),
        )

    makePlot(output, 'nLepFromW', 'multiplicity',
         data=[],
         bins=N_bins_red, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/nLepFromW'),
        )

    makePlot(output, 'nLepFromZ', 'multiplicity',
         data=[],
         bins=N_bins_red, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/nLepFromZ'),
        )

    makePlot(output, 'nLepFromTau', 'multiplicity',
         data=[],
         bins=N_bins_red, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/nLepFromTau'),
        )

    makePlot(output, 'nLepFromTop', 'multiplicity',
         data=[],
         bins=N_bins_red, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/nLepFromTop'),
        )

    makePlot(output, 'chargeFlip_vs_nonprompt', 'n1',
         data=[],
         bins=None, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/nChargeFlip'),
        )

    makePlot(output, 'chargeFlip_vs_nonprompt', 'n2',
         data=[],
         bins=None, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/nNonprompt'),
        )

    makePlot(output, 'chargeFlip_vs_nonprompt', 'n2',
         data=[],
         bins=None, log=True, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/nNonprompt_log'),
        )

    makePlot(output, 'node', 'multiplicity',
         data=['DoubleMuon', 'MuonEG', 'EGamma'],
         bins=N_bins_red, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/ML_node'),
        )

    makePlot(output, 'node0_score', 'score',
         data=[],
         bins=score_bins, log=False, normalize=False, axis_label=r'score',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         omit=['DoubleMuon', 'MuonEG', 'EGamma'],
         save=os.path.expandvars(plot_dir+'/SS/ML_node0_score'),
        )

    makePlot(output, 'lead_lep', 'pt',
         data=[],
         bins=pt_bins_coarse, log=True, normalize=False, axis_label=r'$p_{T}\ lead \ lep\ (GeV)$',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         omit=['DoubleMuon', 'MuonEG', 'EGamma'],
         save=os.path.expandvars(plot_dir+'/SS/lead_lep_pt'),
        )

    makePlot(output, 'lead_lep', 'pt',
         data=[],
         bins=pt_bins_coarse, log=False, normalize=False, axis_label=r'$p_{T}\ lead \ lep\ (GeV)$',
         new_colors=my_colors, new_labels=my_labels,
         order=['TTW'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         omit=['DoubleMuon', 'MuonEG', 'EGamma', 'diboson', 'TTH', 'TTZ', 'ttbar'],
         save=os.path.expandvars(plot_dir+'/SS/lead_lep_pt_signals'),
        )

    makePlot(output, 'node1_score', 'score',
         data=[],
         bins=score_bins, log=False, normalize=False, axis_label=r'score',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         omit=['DoubleMuon', 'MuonEG', 'EGamma'],
         save=os.path.expandvars(plot_dir+'/SS/ML_node1_score'),
        )

    makePlot(output, 'node2_score', 'score',
         data=[],
         bins=score_bins, log=False, normalize=False, axis_label=r'score',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         omit=['DoubleMuon', 'MuonEG', 'EGamma'],
         save=os.path.expandvars(plot_dir+'/SS/ML_node2_score'),
        )

    makePlot(output, 'node3_score', 'score',
         data=[],
         bins=score_bins, log=False, normalize=False, axis_label=r'score',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         omit=['DoubleMuon', 'MuonEG', 'EGamma'],
         save=os.path.expandvars(plot_dir+'/SS/ML_node3_score'),
        )

    makePlot(output, 'node4_score', 'score',
         data=['DoubleMuon', 'MuonEG', 'EGamma'],
         #data=[],
         bins=score_bins, log=False, normalize=True, axis_label=r'score',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         #omit=['DoubleMuon', 'MuonEG', 'EGamma'],
         omit=[],
         save=os.path.expandvars(plot_dir+'/SS/ML_node4_score'),
        )

    makePlot(output, 'MET', 'pt',
         data=['DoubleMuon', 'MuonEG', 'EGamma'],
         bins=pt_bins_coarse, log=False, normalize=True, axis_label=r'$p_{T}^{miss}$',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3'],
         omit=['topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/MET_pt'),
        )

    makePlot(output, 'MET', 'pt',
         data=[],
         normalize=False,
         bins=pt_bins_coarse, log=False, shape=True, axis_label=r'$p_{T}^{miss}$',
         new_colors=my_colors, new_labels=my_labels,
         ymax=0.4,
         order=['TTW', 'topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         #signals=['topW_v3'],
         omit=['ttbar', 'diboson', 'TTZ', 'TTH', 'DoubleMuon', 'MuonEG', 'EGamma'],
         save=os.path.expandvars(plot_dir+'/SS/MET_pt_shape'),
        )

    makePlot(output, 'MET', 'pt',
         data=[],
         normalize=False,
         bins=pt_bins_coarse, log=False, axis_label=r'$p_{T}^{miss}$',
         new_colors=my_colors, new_labels=my_labels,
         order=['TTW'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         omit=['ttbar', 'diboson', 'TTZ', 'TTH', 'DoubleMuon', 'MuonEG', 'EGamma'],
         save=os.path.expandvars(plot_dir+'/SS/MET_pt_signals'),
        )

    makePlot(output, 'ST', 'pt',
         data=[],
         normalize=False,
         bins=pt_bins_ext, log=False, axis_label=r'$S_{T}$',
         new_colors=my_colors, new_labels=my_labels,
         order=['TTW'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         omit=['ttbar', 'diboson', 'TTZ', 'TTH', 'DoubleMuon', 'MuonEG', 'EGamma'],
         save=os.path.expandvars(plot_dir+'/SS/ST_signals'),
        )

    makePlot(output, 'PV_npvsGood', 'multiplicity',
         data=['DoubleMuon', 'MuonEG', 'EGamma'],
         bins=None, log=False, normalize=True, axis_label=r'$N_{PV}$',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar', 'topW_v3'],
         signals=[],
         omit=['topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/PV_npvsGood'),
        )


    ## shapes

    makePlot(output, 'node0_score', 'score',
         data=[],
         bins=score_bins, log=False, normalize=False, axis_label=r'score', shape=True, ymax=0.35,
         new_colors=my_colors, new_labels=my_labels,
         order=['TTW'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         omit=['DoubleMuon', 'MuonEG', 'EGamma', 'diboson', 'ttbar', 'TTH', 'TTZ'],
         save=os.path.expandvars(plot_dir+'/SS/ML_node0_score_shape'),
        )


    fig, ax  = plt.subplots(1,1,figsize=(10,10) )
    ax = hist.plot2d(
        output['chargeFlip_vs_nonprompt']['ttbar'].sum('n_ele').sum('dataset'),
        xaxis='n1',
        ax=ax,
        text_opts={'format': '%.3g'},
        patch_opts={},
    )
    ax.set_xlabel(r'$N_{charge flips}$')
    ax.set_ylabel(r'$N_{nonprompt}$')
    fig.savefig(plot_dir+'/SS/nChargeFlip_vs_nNonprompt_ttbar')

