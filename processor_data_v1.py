import awkward as ak

from coffea import processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiData, LumiMask, LumiList


# this is slightly better practice
import numpy as np
from Tools.objects2 import Collections, choose, cross, match
from Tools.basic_objects import getJets
from Tools.config_helpers import loadConfig
from Tools.helpers import build_weight_like
from Tools.triggers import getFilters

class dielectron_mass(processor.ProcessorABC):
    def __init__(self, year=2018, variations=[], accumulator={}):
        self.variations = variations
        self.year = year
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

        #output['totalEvents']['all'] += len(events)
        #output['skimmedEvents']['all'] += len(ev)

        if self.year == 2018:
            triggers = ev.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL
        elif self.year == 2017:
            triggers = ev.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL
        elif self.year == 2016:
            triggers = ev.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ

        if self.year == 2018:
            lumimask = LumiMask('processors/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt')

        ## Electrons
        electron = Collections(ev, "Electron", "tight").get()
        electron = electron[(electron.pt > 25) & (np.abs(electron.eta) < 2.4)]

        loose_electron = Collections(ev, "Electron", "veto").get()
        loose_electron = loose_electron[(loose_electron.pt > 25) & (np.abs(loose_electron.eta) < 2.4)]

        SSelectron = (ak.sum(electron.charge, axis=1) != 0) & (ak.num(electron)==2)
        OSelectron = (ak.sum(electron.charge, axis=1) == 0) & (ak.num(electron)==2)

        dielectron = choose(electron, 2)
        dielectron_mass = (dielectron['0']+dielectron['1']).mass
        dielectron_pt = (dielectron['0']+dielectron['1']).pt

        leading_electron_idx = ak.singletons(ak.argmax(electron.pt, axis=1))
        leading_electron = electron[(leading_electron_idx)]
        leading_electron = leading_electron[(leading_electron.pt > 30)]

        trailing_electron_idx = ak.singletons(ak.argmin(electron.pt, axis=1))
        trailing_electron = electron[trailing_electron_idx]

        ##Muons

        loose_muon = Collections(ev, "Muon", "veto").get()
        loose_muon = loose_muon[(loose_muon.pt > 20) & (np.abs(loose_muon.eta) < 2.4)]

        #jets
        jet       = getJets(ev, minPt=40, maxEta=2.4, pt_var='pt', UL = False)
        jet       = jet[ak.argsort(jet.pt, ascending=False)] # need to sort wrt smeared and recorrected jet pt
        jet       = jet[~match(jet, loose_muon, deltaRCut=0.4)] # remove jets that overlap with muons
        jet       = jet[~match(jet, electron, deltaRCut=0.4)] # remove jets that overlap with electrons

        ## MET -> can switch to puppi MET
        met_pt  = ev.MET.pt
        met_phi = ev.MET.phi

        #selections
        filters   = getFilters(ev, year=self.year, dataset=dataset)
        mask = lumimask(ev.run, ev.luminosityBlock)
        ss = (SSelectron)
        os = (OSelectron)
        mass = (ak.min(np.abs(dielectron_mass-91.2), axis = 1) < 15)
        lead_electron = (ak.min(leading_electron.pt, axis = 1) > 30)
        jet1 = (ak.num(jet) >= 1)
        jet2 = (ak.num(jet) >= 2)
        num_loose = ( (ak.num(loose_electron) == 2) & (ak.num(loose_muon) == 0) )

        selection = PackedSelection()
        selection.add('filter',      (filters) )
        selection.add('mask',        (mask) )
        selection.add('ss',          ss)
        selection.add('os',          os)
        selection.add('mass',        mass)
        selection.add('leading',     lead_electron)
        selection.add('triggers',    triggers)
        selection.add('one jet',     jet1)
        selection.add('two jets',    jet2)
        selection.add('num_loose',   num_loose)

        bl_reqs = ['filter'] + ['mass'] + ['mask'] + ['triggers'] + ['leading'] + ['num_loose']
        #bl_reqs = ['filter'] + ['mass'] + ['triggers'] + ['leading'] + ['num_loose']

        bl_reqs_d = { sel: True for sel in bl_reqs }
        baseline = selection.require(**bl_reqs_d)

        s_reqs = bl_reqs + ['ss']
        s_reqs_d = { sel: True for sel in s_reqs }
        ss_sel = selection.require(**s_reqs_d)

        o_reqs = bl_reqs + ['os']
        o_reqs_d = { sel: True for sel in o_reqs }
        os_sel = selection.require(**o_reqs_d)

        j1s_reqs = s_reqs + ['one jet']
        j1s_reqs_d = { sel: True for sel in j1s_reqs }
        j1ss_sel = selection.require(**j1s_reqs_d)

        j1o_reqs = o_reqs + ['one jet']
        j1o_reqs_d = {sel: True for sel in j1o_reqs }
        j1os_sel = selection.require(**j1o_reqs_d)

        j2s_reqs = s_reqs + ['two jets']
        j2s_reqs_d = { sel: True for sel in j2s_reqs }
        j2ss_sel = selection.require(**j2s_reqs_d)

        j2o_reqs = o_reqs + ['two jets']
        j2o_reqs_d = {sel: True for sel in j2o_reqs }
        j2os_sel = selection.require(**j2o_reqs_d)

        output["N_jet"].fill(
            dataset = dataset,
            multiplicity = ak.num(jet)[os_sel],
        )

        return output

    def postprocess(self, accumulator):
        return accumulator


if __name__ == '__main__':
    from Tools.config_helpers import redirector_ucsd
    from Tools.nano_mapping import make_fileset
    from processors.default_accumulators import desired_output

    year = 2018

    fileset = make_fileset(['Data'], year, redirector=redirector_ucsd, small=False)

    exe_args = {
        'workers': 8,
        'function_args': {'flatten': False},
        "schema": NanoAODSchema,
        "skipbadfiles": True,
    }
    exe = processor.futures_executor

    output = processor.run_uproot_job(
            fileset,
            "Events",
            dielectron_mass(year=year, variations=[], accumulator=desired_output),
            exe,
            exe_args,
            chunksize=250000,
    )
