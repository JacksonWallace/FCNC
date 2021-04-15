try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea import processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection


# this is all very bad practice
from Tools.objects import *
from Tools.basic_objects import *
from Tools.cutflow import *
from Tools.config_helpers import *
from Tools.helpers import build_weight_like
from Tools.triggers import *
from Tools.btag_scalefactors import *
from Tools.lepton_scalefactors import *
from Tools.charge_flip import *

class charge_flip_check(processor.ProcessorABC):
    def __init__(self, year=2018, variations=[], accumulator={}):
        self.variations = variations
        self.year = year
        
        self.btagSF = btag_scalefactor(year)
        
        #self.leptonSF = LeptonSF(year=year)
        
        self.charge_flip_ratio = charge_flip('histos/chargeflipfull.pkl.gz')
        
        self._accumulator = processor.dict_accumulator( accumulator )

    @property
    def accumulator(self):
        return self._accumulator

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        
        output = self.accumulator.identity()
        
        # we can use a very loose preselection to filter the events. nothing is done with this presel, though
        presel = ak.num(events.Jet)>0
        
        ev = events[presel]
        dataset = ev.metadata['dataset']

        # load the config - probably not needed anymore
        cfg = loadConfig()
        
        output['totalEvents']['all'] += len(events)
        output['skimmedEvents']['all'] += len(ev)
        
        ## Muons
        muon     = ev.Muon
        
        ## Electrons
        electron     = Collections(ev, "Electron", "tight").get()
        electron = electron[(electron.miniPFRelIso_all < 0.12) & (electron.pt > 20) & (abs(electron.eta) < 2.4)]

        gen_matched_electron = electron[electron.genPartIdx >= 0]
        
        is_flipped = (gen_matched_electron.matched_gen.pdgId*(-1) == gen_matched_electron.pdgId)

        ## Merge electrons and muons - this should work better now in ak1
        gen_matched_electron = electron[electron.genPartIdx >= 0]
        
        #is_flipped = (abs(ev.GenPart[gen_matched_electron.genPartIdx].pdgId) == abs(gen_matched_electron.pdgId))&(ev.GenPart[gen_matched_electron.genPartIdx].pdgId/abs(ev.GenPart[gen_matched_electron.genPartIdx].pdgId) != gen_matched_electron.pdgId/abs(gen_matched_electron.pdgId))
        flipped_electron = gen_matched_electron[is_flipped]
        n_flips = ak.num(flipped_electron)
        
        dielectron = choose(electron, 2)
        SSelectron = ak.any((dielectron['0'].charge * dielectron['1'].charge)>0, axis=1)
         
        leading_electron_idx = ak.singletons(ak.argmax(electron.pt, axis=1))
        leading_electron = electron[leading_electron_idx]
        
        ## MET -> can switch to puppi MET
        met_pt  = ev.MET.pt
        met_phi = ev.MET.phi

        # setting up the various weights
        weight = Weights( len(ev) )
        weight2 = Weights( len(ev))
        
        if not dataset=='MuonEG':
            # generator weight
            weight.add("weight", ev.genWeight)
            weight2.add("weight", ev.genWeight)
            
        weight2.add("charge flip", self.charge_flip_ratio.flip_ratio(dielectron['0'], dielectron['1']))
                                   
                      
        #selections    
        filters   = getFilters(ev, year=self.year, dataset=dataset)
        dilep     = ((ak.num(electron) + ak.num(muon))==2)
        electr = ((ak.num(electron) == 2))
        ss = (SSelectron)
        flip = (n_flips >= 1)
        
        
        selection = PackedSelection()
        selection.add('filter',        (filters) )
        selection.add('electr',        electr  )
        selection.add('ss',        ss)
        selection.add('flip',          flip)
        
        bl_reqs = ['filter', 'electr']

        bl_reqs_d = { sel: True for sel in bl_reqs }
        baseline = selection.require(**bl_reqs_d)
        
        s_reqs = bl_reqs + ['ss']
        s_reqs_d = { sel: True for sel in s_reqs }
        ss_sel = selection.require(**s_reqs_d)
        
        f_reqs = bl_reqs + ['flip']
        f_reqs_d = { sel: True for sel in f_reqs }
        flip_sel = selection.require(**f_reqs_d)
   
        
        #outputs
        output['N_ele'].fill(dataset=dataset, multiplicity=ak.num(electron)[ss_sel], weight=weight.weight()[ss_sel])
        output['electron_flips'].fill(dataset=dataset, multiplicity=n_flips[ss_sel], weight=weight.weight()[ss_sel])
                      
        output['N_ele2'].fill(dataset=dataset, multiplicity=ak.num(electron)[baseline], weight=weight2.weight()[baseline])
        output['electron_flips2'].fill(dataset=dataset, multiplicity=n_flips[baseline], weight=weight2.weight()[baseline])


        
        output["electron"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_electron[baseline].pt)),
            eta = ak.to_numpy(ak.flatten(abs(leading_electron[baseline].eta))),
            #phi = ak.to_numpy(ak.flatten(leading_electron[baseline].phi)),
            weight = weight.weight()[ss_sel]
        )

        output["gen_matched_electron"].fill(
            dataset = dataset,
            pt  = ak.flatten(gen_matched_electron.pt),
            eta = abs(ak.flatten(gen_matched_electron.eta)),
            weight = build_weight_like(weight.weight(), (ak.num(gen_matched_electron)>0), gen_matched_electron.pt),
            #weight = ak.flatten(weight.weight() * ak.ones_like(gen_matched_electron.pt)),
        )

        output["flipped_electron"].fill(
            dataset = dataset,
            pt  = ak.flatten(gen_matched_electron[is_flipped].pt),
            eta = abs(ak.flatten(gen_matched_electron[is_flipped].eta)),
            weight = build_weight_like(weight.weight(), (ak.num(gen_matched_electron[is_flipped])>0), gen_matched_electron[is_flipped].pt),
            #weight = ak.flatten(weight.weight() * ak.ones_like(gen_matched_electron.pt)),
        )
        
        output["electron2"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_electron[baseline].pt)),
            eta = ak.to_numpy(ak.flatten(abs(leading_electron[baseline].eta))),
            #phi = ak.to_numpy(ak.flatten(leading_electron[baseline].phi)),
            weight = weight2.weight()[baseline]
        )

        return output

    def postprocess(self, accumulator):
        return accumulator


if __name__ == '__main__':

    from klepto.archives import dir_archive
    from processor.default_accumulators import desired_output, add_processes_to_output, add_files_to_output, dataset_axis

    from Tools.helpers import get_samples
    from Tools.config_helpers import redirector_ucsd, redirector_fnal
    from Tools.nano_mapping import make_fileset, nano_mapping

    from processor.meta_processor import get_sample_meta
    overwrite = True
    local = True
    
    # load the config and the cache
    cfg = loadConfig()
    
    cacheName = 'nano_analysis'
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)
    histograms = sorted(list(desired_output.keys()))
    
    year = 2018
    
    samples = get_samples()

    #fileset = make_fileset(['TTW', 'TTZ'], samples, redirector=redirector_ucsd, small=True, n_max=5)  # small, max 5 files per sample
    #fileset = make_fileset(['DY'], samples, redirector=redirector_ucsd, small=True, n_max=10)
    fileset = make_fileset(['top'], samples, redirector=redirector_ucsd, small=True, n_max=1)

    add_processes_to_output(fileset, desired_output)

    if local:

        exe_args = {
            'workers': 16,
            'function_args': {'flatten': False},
             "schema": NanoAODSchema,
        }
        exe = processor.futures_executor
    meta = get_sample_meta(fileset, samples)
    else:
        from Tools.helpers import get_scheduler_address
        from dask.distributed import Client, progress

        scheduler_address = get_scheduler_address()
        c = Client(scheduler_address)

        exe_args = {
            'client': c,
            'function_args': {'flatten': False},
            "schema": NanoAODSchema,
        }
        exe = processor.dask_executor
    
    if not overwrite:
        cache.load()
    
    if cfg == cache.get('cfg') and histograms == cache.get('histograms') and cache.get('simple_output'):
        output = cache.get('simple_output')
    
    else:
        print ("I'm running now")
        
        output = processor.run_uproot_job(
            fileset,
            "Events",
            nano_analysis(year=year, variations=[], accumulator=desired_output),
            exe,
            exe_args,
            chunksize=250000,
        )
        
        cache['fileset']        = fileset
        cache['cfg']            = cfg
        cache['histograms']     = histograms
        cache['simple_output']  = output
        cache.dump()

    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    
    # load the functions to make a nice plot from the output histograms
    # and the scale_and_merge function that scales the individual histograms
    # to match the physical cross section
    
    from plots.helpers import makePlot, scale_and_merge
    
    # define a few axes that we can use to rebin our output histograms
    N_bins_red     = hist.Bin('multiplicity', r'$N$', 5, -0.5, 4.5)
    
    # define nicer labels and colors
    
    my_labels = {
        nano_mapping['TTW'][0]: 'ttW',
        nano_mapping['TTZ'][0]: 'ttZ',
        nano_mapping['DY'][0]: 'DY',
        nano_mapping['top'][0]: 't/tt+jets',
    }
    
    my_colors = {
        nano_mapping['TTW'][0]: '#8AC926',
        nano_mapping['TTZ'][0]: '#FFCA3A',
        nano_mapping['DY'][0]: '#6A4C93',
        nano_mapping['top'][0]: '#1982C4',
    }

    # take the N_ele histogram out of the output, apply the x-secs from samples to the samples in fileset
    # then merge the histograms into the categories defined in nano_mapping

    print ("Total events in output histogram N_ele: %.2f"%output['N_ele'].sum('dataset').sum('multiplicity').values(overflow='all')[()])
    
    my_hists = {}
    #my_hists['N_ele'] = scale_and_merge(output['N_ele'], samples, fileset, nano_mapping)
    my_hists['N_ele'] = scale_and_merge(output['N_ele'], meta, fileset, nano_mapping)
    print ("Total scaled events in merged histogram N_ele: %.2f"%my_hists['N_ele'].sum('dataset').sum('multiplicity').values(overflow='all')[()])
    
    # Now make a nice plot of the electron multiplicity.
    # You can have a look at all the "magic" (and hard coded monstrosities) that happens in makePlot
    # in plots/helpers.py
    
    makePlot(my_hists, 'N_ele', 'multiplicity',
             data=[],
             bins=N_bins_red, log=True, normalize=False, axis_label=r'$N_{electron}$',
             new_colors=my_colors, new_labels=my_labels,
             #order=[nano_mapping['DY'][0], nano_mapping['TTZ'][0]],
             save=os.path.expandvars(cfg['meta']['plots'])+'/nano_analysis/N_ele_test.png'
            )
