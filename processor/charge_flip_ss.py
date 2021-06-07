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
from Tools.gen import find_first_parent, get_charge_parent

class charge_flip_ss(processor.ProcessorABC):
    def __init__(self, year=2018, variations=[], accumulator={}):
        self.variations = variations
        self.year = year
        
        self.btagSF = btag_scalefactor(year)
        
        #self.leptonSF = LeptonSF(year=year)
        
        self.charge_flip_ratio = charge_flip('../histos/chargeflipfull2018.pkl.gz')
        
        self._accumulator = processor.dict_accumulator( accumulator )

    @property
    def accumulator(self):
        return self._accumulator

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        
        output = self.accumulator.identity()
        
        # we can use a very loose preselection to filter the events. nothing is done with this presel, though
        presel = ak.num(events.Jet)>=2
        
        ev = events[presel]
        dataset = ev.metadata['dataset']

        # load the config - probably not needed anymore
        cfg = loadConfig()
        
        output['totalEvents']['all'] += len(events)
        output['skimmedEvents']['all'] += len(ev)
        
        
        ## Electrons
        electron = Collections(ev, "Electron", "tightFCNC", 0, self.year).get()
        electron = electron[(electron.pt > 20) & (np.abs(electron.eta) < 2.4)]

        electron = electron[(electron.genPartIdx >= 0)]
        electron = electron[(np.abs(electron.matched_gen.pdgId)==11)]  #from here on all leptons are gen-matched
        electron = electron[( (electron.genPartFlav==1) | (electron.genPartFlav==15) )] #and now they are all prompt
     
        
        leading_electron_idx = ak.singletons(ak.argmax(electron.pt, axis=1))
        leading_electron = electron[leading_electron_idx]
        
        trailing_electron_idx = ak.singletons(ak.argmin(electron.pt, axis=1))
        trailing_electron = electron[trailing_electron_idx]
        
        leading_parent = find_first_parent(leading_electron.matched_gen)
        trailing_parent = find_first_parent(trailing_electron.matched_gen)
        
       
        is_flipped = ( ( (electron.matched_gen.pdgId*(-1) == electron.pdgId) | (find_first_parent(electron.matched_gen)*(-1) == electron.pdgId) ) & (np.abs(electron.pdgId) == 11) )
        
        
        flipped_electron = electron[is_flipped]
        flipped_electron = flipped_electron[(ak.fill_none(flipped_electron.pt, 0)>0)]
        flipped_electron = flipped_electron[~(ak.is_none(flipped_electron))]
        n_flips = ak.num(flipped_electron)
                
        ##Muons
        muon     = Collections(ev, "Muon", "tightFCNC").get()
        muon = muon[(muon.pt > 20) & (np.abs(muon.eta) < 2.4)]
        
        muon = muon[(muon.genPartIdx >= 0)]
        muon = muon[(np.abs(muon.matched_gen.pdgId)==13)] #from here, all muons are gen-matched
        muon = muon[( (muon.genPartFlav==1) | (muon.genPartFlav==15) )] #and now they are all prompt
       
        
        ##Leptons

        lepton   = ak.concatenate([muon, electron], axis=1)
        SSlepton = (ak.sum(lepton.charge, axis=1) != 0) & (ak.num(lepton)==2)
        OSlepton = (ak.sum(lepton.charge, axis=1) == 0) & (ak.num(lepton)==2)
        
        emulepton = (ak.num(electron) == 1) & (ak.num(muon) == 1)
        no_mumu = (ak.num(muon) <= 1)
        
        
        leading_lepton_idx = ak.singletons(ak.argmax(lepton.pt, axis=1))
        leading_lepton = lepton[leading_lepton_idx]
        
        trailing_lepton_idx = ak.singletons(ak.argmin(lepton.pt, axis=1))
        trailing_lepton = lepton[trailing_lepton_idx]
        
        
        
        #jets
        jet       = getJets(ev, minPt=40, maxEta=2.4, pt_var='pt')
        jet       = jet[ak.argsort(jet.pt, ascending=False)] # need to sort wrt smeared and recorrected jet pt
        jet       = jet[~match(jet, muon, deltaRCut=0.4)] # remove jets that overlap with muons
        jet       = jet[~match(jet, electron, deltaRCut=0.4)] 
        
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
            
        weight2.add("charge flip", self.charge_flip_ratio.flip_weight(electron))
                                   
                      
        #selections    
        filters   = getFilters(ev, year=self.year, dataset=dataset)
        ss = (SSlepton)
        os = (OSlepton)
        jet_all = (ak.num(jet) >= 2)
        diele = (ak.num(electron) == 2)
        emu = (emulepton)
        flips = (n_flips == 1)
        no_flips = (n_flips == 0)
        nmm = no_mumu
        
        
        selection = PackedSelection()
        selection.add('filter',      (filters) )
        selection.add('ss',          ss )
        selection.add('os',          os )
        selection.add('jet',         jet_all )
        selection.add('ee',          diele)
        selection.add('emu',         emu)
        selection.add('flip',        flips)
        selection.add('nflip',       no_flips)
        selection.add('no_mumu',     nmm)
        
        bl_reqs = ['filter', 'jet']

        bl_reqs_d = { sel: True for sel in bl_reqs }
        baseline = selection.require(**bl_reqs_d)
        
        f_reqs = bl_reqs + ['flip'] + ['ss'] + ['no_mumu']
        f_reqs_d = {sel: True for sel in f_reqs}
        flip_sel = selection.require(**f_reqs_d)
        
        nf_reqs = bl_reqs + ['nflip'] + ['os'] + ['no_mumu']
        nf_reqs_d = {sel: True for sel in nf_reqs}
        n_flip_sel = selection.require(**nf_reqs_d)
        
        nf2_reqs = bl_reqs + ['nflip'] + ['ss'] + ['no_mumu']
        nf2_reqs_d = {sel: True for sel in nf2_reqs}
        n_flip_sel2 = selection.require(**nf2_reqs_d)
        
        s_reqs = bl_reqs + ['ss'] + ['no_mumu']
        s_reqs_d = { sel: True for sel in s_reqs }
        ss_sel = selection.require(**s_reqs_d)
        
        o_reqs = bl_reqs + ['os'] + ['no_mumu']
        o_reqs_d = {sel: True for sel in o_reqs }
        os_sel = selection.require(**o_reqs_d)
        
        ees_reqs = s_reqs + ['ee']
        ees_reqs_d = { sel: True for sel in ees_reqs }
        eess_sel = selection.require(**ees_reqs_d)
        
        eeo_reqs = o_reqs + ['ee']
        eeo_reqs_d = { sel: True for sel in eeo_reqs }
        eeos_sel = selection.require(**eeo_reqs_d)
        
        ems_reqs = s_reqs + ['emu']
        ems_reqs_d = { sel: True for sel in ems_reqs }
        emss_sel = selection.require(**ems_reqs_d)
        
        emo_reqs = o_reqs + ['emu']
        emo_reqs_d = { sel: True for sel in emo_reqs }
        emos_sel = selection.require(**emo_reqs_d)
       
        #outputs
        output['N_jet'].fill(dataset=dataset, multiplicity=ak.num(jet)[baseline], weight=weight.weight()[baseline])
        
        output['N_ele'].fill(dataset=dataset, multiplicity=ak.num(lepton)[ss_sel], weight=weight.weight()[ss_sel])
                      
        output['N_ele2'].fill(dataset=dataset, multiplicity=ak.num(lepton)[os_sel], weight=weight2.weight()[os_sel])
        
        output['electron_flips'].fill(dataset=dataset, multiplicity = n_flips[ss_sel], weight=weight.weight()[ss_sel])

        output['electron_flips2'].fill(dataset=dataset, multiplicity = n_flips[os_sel], weight=weight2.weight()[os_sel])

        output["electron"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_lepton[ss_sel].pt)),
            eta = np.abs(ak.to_numpy(ak.flatten(leading_lepton[ss_sel].eta))),
            weight = weight.weight()[ss_sel]
        )
        
        output["electron2"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_lepton[os_sel].pt)),
            eta = np.abs(ak.to_numpy(ak.flatten(leading_lepton[os_sel].eta))),
            weight = weight2.weight()[os_sel]
        )
        
        output["flipped_electron"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_lepton[flip_sel].pt)),
            eta = np.abs(ak.to_numpy(ak.flatten(leading_lepton[flip_sel].eta))),
            weight = weight.weight()[flip_sel]
        )
        
        output["flipped_electron2"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_lepton[n_flip_sel].pt)),
            eta = np.abs(ak.to_numpy(ak.flatten(leading_lepton[n_flip_sel].eta))),
            weight = weight2.weight()[n_flip_sel]
        )
        
        output["lepton_parent"].fill(
            dataset = dataset,
            pdgID = np.abs(ak.to_numpy(ak.flatten(leading_parent[ss_sel]))),
            weight = weight.weight()[ss_sel]
        )
        
        output["lepton_parent2"].fill(
            dataset = dataset,
            pdgID = np.abs(ak.to_numpy(ak.flatten(trailing_parent[ss_sel]))),
            weight = weight.weight()[ss_sel]
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
    
    cacheName = 'charge_flip_check'
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)
    histograms = sorted(list(desired_output.keys()))
    
    year = 2018
    
    samples = get_samples(2018)

    #fileset = make_fileset(['TTW', 'TTZ'], samples, redirector=redirector_ucsd, small=True, n_max=5)  # small, max 5 files per sample
    #fileset = make_fileset(['DY'], samples, redirector=redirector_ucsd, small=True, n_max=10)
    fileset = make_fileset(['top', 'DY',], redirector=redirector_ucsd, small=False)
   
    add_processes_to_output(fileset, desired_output)

    #meta = get_sample_meta(fileset, samples)
   
    if local:

        exe_args = {
            'workers': 16,
            'function_args': {'flatten': False},
            'schema': NanoAODSchema,
            'skipbadfiles': True,
        }
        exe = processor.futures_executor
   
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
            charge_flip_ss(year=year, variations=[], accumulator=desired_output),
            exe,
            exe_args,
            chunksize=500000,
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
    
    nano_mappings = nano_mapping(year)
    
    my_labels = {
        nano_mappings['TTW'][0]: 'ttW',
        nano_mappings['TTZ'][0]: 'ttZ',
        nano_mappings['DY'][0]: 'DY',
        nano_mappings['top'][0]: 't/tt+jets',
    }
    
    my_colors = {
        nano_mappings['TTW'][0]: '#8AC926',
        nano_mappings['TTZ'][0]: '#FFCA3A',
        nano_mappings['DY'][0]: '#6A4C93',
        nano_mappings['top'][0]: '#1982C4',
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
             order=[nano_mapping['DY'][0], nano_mapping['TTZ'][0]],
             #save=os.path.expandvars(cfg['meta']['plots'])+'/nano_analysis/N_ele_test.png'
            )
