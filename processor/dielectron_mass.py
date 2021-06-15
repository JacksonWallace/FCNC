try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea import processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiData, LumiMask, LumiList


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


class dielectron_mass(processor.ProcessorABC):
    def __init__(self, year=2018, variations=[], accumulator={}):
        self.variations = variations
        self.year = year
        
        self.btagSF = btag_scalefactor(year)
        
        #self.leptonSF = LeptonSF(year=year)
                
        self._accumulator = processor.dict_accumulator( accumulator )

    @property
    def accumulator(self):
        return self._accumulator

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        
        output = self.accumulator.identity()
        
        # we can use a very loose preselection to filter the events. nothing is done with this presel, though
        presel = ak.num(events.Jet)>0
        
        lumimask = LumiMask('../processor/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt')
        
        if self.year == 2018:
            triggers = events.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL
        elif self.year == 2017:
            triggers = events.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL
        elif self.year == 2016:
            triggers == Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ
        
        ev = events[presel]
        dataset = ev.metadata['dataset']
        dataset_weight = ev.metadata['dataset']

        # load the config - probably not needed anymore
        cfg = loadConfig()
        
        output['totalEvents']['all'] += len(events)
        output['skimmedEvents']['all'] += len(ev)
        
      
        ## Electrons
        electron = Collections(ev, "Electron", "tightFCNC", 0, self.year).get()
        electron = electron[(electron.pt > 20) & (np.abs(electron.eta) < 2.4)]
        
        SSelectron = (ak.sum(electron.charge, axis=1) != 0) & (ak.num(electron)==2)
        OSelectron = (ak.sum(electron.charge, axis=1) == 0) & (ak.num(electron)==2)
        
        dielectron = choose(electron, 2)
        dielectron_mass = (dielectron['0']+dielectron['1']).mass
        dielectron_pt = (dielectron['0']+dielectron['1']).pt
        
        leading_electron_idx = ak.singletons(ak.argmax(electron.pt, axis=1))
        leading_electron = electron[leading_electron_idx]
        
        trailing_electron_idx = ak.singletons(ak.argmin(electron.pt, axis=1))
        trailing_electron = electron[trailing_electron_idx]
        
 
        ## MET -> can switch to puppi MET
        met_pt  = ev.MET.pt
        met_phi = ev.MET.phi
        
        #triggers
                                   
                      
        #selections    
        filters   = getFilters(ev, year=self.year, dataset=dataset)
        mask = lumimask(ev.run, ev.luminosityBlock)
        ss = (SSelectron)
        os = (OSelectron)
        mass = ()
        
        
        selection = PackedSelection()
        selection.add('filter',      (filters) )
        selection.add('mask',        (mask) )
        selection.add('ss',          ss)
        selection.add('os',          os)
        selection.add('mass',        mass)
        selection.add('triggers',    triggers)
        
        bl_reqs = ['filter'] + ['mass'] + ['mask'] + ['triggers']

        bl_reqs_d = { sel: True for sel in bl_reqs }
        baseline = selection.require(**bl_reqs_d)
        
        s_reqs = bl_reqs + ['ss']
        s_reqs_d = { sel: True for sel in s_reqs }
        ss_sel = selection.require(**s_reqs_d)
        
        os_reqs = bl_reqs + ['os']
        os_reqs_d = { sel: True for sel in os_reqs }
        os_sel = selection.require(**os_reqs_d)
   
        
        #outputs
        
        output["electron_data1"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_electron[os_sel].pt)),
            eta = ak.to_numpy(ak.flatten(leading_electron[os_sel].eta)),
            phi = ak.to_numpy(ak.flatten(leading_electron[os_sel].phi)),
        )
        
        output["electron_data2"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(trailing_electron[os_sel].pt)),
            eta = ak.to_numpy(ak.flatten(trailing_electron[os_sel].eta)),
            phi = ak.to_numpy(ak.flatten(trailing_electron[os_sel].phi)),
        )
        
        output["dilep_mass"].fill(
            dataset = dataset,
            mass = ak.to_numpy(ak.flatten(dielectron_mass[os_sel])),
            pt = ak.to_numpy(ak.flatten(dielectron_pt[os_sel])),
        )

        return output

    def postprocess(self, accumulator):
        return accumulator