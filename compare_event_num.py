from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection

from Tools.config_helpers import redirector_ucsd
from Tools.nano_mapping import make_fileset
from Tools.basic_objects import getJets, getBTagsDeepFlavB
from Tools.objects import Collections, choose



import awkward as ak
import numpy as np

year = 2018

fileset = make_fileset(['hut'], year, redirector=redirector_ucsd, small=False)

# load events
for sample in [2]:
    for file in range(len(fileset[list(fileset.keys())[sample]])):
        events = NanoEventsFactory.from_root(
            fileset[list(fileset.keys())[sample]][file],
            schemaclass = NanoAODSchema,
        ).events()
        
        ## Electrons
        electron = Collections(events, "Electron", "tightFCNC", year, 0).get()
        electron = electron[(electron.pt > 20) & (np.abs(electron.eta) < 2.4)] #matches skim 

        loose_electron = Collections(events, "Electron", "fakeFCNC", year, 0).get()
        loose_electron = loose_electron[((loose_electron.pt > 15) | (loose_electron.conePt > 15) ) & (np.abs(loose_electron.eta) < 2.4)] #matches skim 

        ##Muons
        muon = Collections(events, "Muon", "tightFCNC", year, 0).get()
        muon = muon[(muon.pt > 20) & (np.abs(muon.eta) < 2.4)] #matches skim

        loose_muon = Collections(events, "Muon", "fakeFCNC", year, 0).get()
        loose_muon = loose_muon[((loose_muon.pt > 15) | (loose_muon.conePt > 15)) & (np.abs(loose_muon.eta) < 2.4)] #matches skim 

        ##Leptons
        lepton   = ak.concatenate([muon, electron], axis=1) #tight leptons, matches skim

        lepton = lepton[ak.argsort(lepton.pt, ascending = False)]
        leading_lepton = lepton[:,0:1]

        loose_lepton = ak.concatenate([loose_muon, loose_electron], axis=1) #matches skim
        
        met_pt = events.MET.pt
        
        skim = (((ak.num(loose_lepton) == 2) & (ak.sum(loose_lepton.charge, axis=1) != 0)) | (ak.num(loose_lepton) > 2))
        tight = (ak.num(lepton) == 2)
        leading = (ak.min(leading_lepton.pt, axis = 1) > 25)
        met =  ( met_pt > 50 )
        SSlepton = ( ak.sum(lepton.charge, axis=1) != 0 )
        
        test_events = events[skim & met & tight & leading & SSlepton]
        
        with open('Jackson_event_nums.txt', 'a+') as f:
            for event in test_events:
                if event is not None:
                    print(str(event.event), file=f)
    

with open('/home/users/ewallace/CMSSW_10_2_9/src/tW_scattering3/Jackson_event_nums.txt') as file1:
    for l1 in file1:
        with open('/home/users/ksalyer/FranksFCNC/ana/analysis/debug/signal_tuh.log') as file2:
            for l2 in file2:
                if l1 == l2:
                    cont = True
                    break
                else:
                    cont = False
                    continue
            if not cont:
                with open('no_match_event_nums.txt', 'a+') as f:
                    print(l1, file=f)