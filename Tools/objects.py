'''
Standardized object selection, based on SS(++) analysis
Trigger-safe requirements for electrons missing!
'''

import uproot
import awkward
import numpy as np
from uproot_methods import TLorentzVectorArray

from coffea.processor import LazyDataFrame
from coffea.analysis_objects import JaggedCandidateArray

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


with open('../data/objects.yaml') as f:
    obj_def = load(f, Loader=Loader)

class Collections:
    def __init__(self, df, obj, wp, verbose=0):
        self.obj = obj
        self.wp = wp
        if self.wp == None:
            self.selection_dict = {}
        else:
            self.selection_dict = obj_def[self.obj][self.wp]

        self.v = verbose
        #self.year = df['year'][0] ## to be implemented in next verison of babies
        self.year = 2018
        
        if self.obj == "Muon":
            self.cand = JaggedCandidateArray.candidatesfromcounts(
                df['nMuon'],
                pt               = df['Muon_pt'].content,
                eta              = df['Muon_eta'].content,
                phi              = df['Muon_phi'].content,
                mass             = df['Muon_mass'].content,
                pdgId            = df['Muon_pdgId'].content, 
                charge           = (df['Muon_pdgId']/abs(df['Muon_pdgId'])).content, #NanoAOD charge might be bullshit?
                mediumId         = df['Muon_mediumId'].content,
                looseId          = df['Muon_looseId'].content,
                dxy              = df['Muon_dxy'].content,
                dz               = df['Muon_dz'].content,
                sip3d            = df['Muon_sip3d'].content,
                miniPFRelIso_all = df['Muon_miniPFRelIso_all'].content,
                ptErrRel         = (df['Muon_ptErr']/df['Muon_pt']).content,
                mvaTTH           = df['Muon_mvaTTH'].content,
                genPartIdx       = df['Muon_genPartIdx'].content,
                jetRelIso        = df['Muon_jetRelIso'].content,
                jetPtRelv2       = df['Muon_jetPtRelv2'].content,
            )
        
        elif self.obj == "Electron":
            self.cand = JaggedCandidateArray.candidatesfromcounts(
                df['nElectron'],
                pt               = df['Electron_pt'].content,
                eta              = df['Electron_eta'].content,
                phi              = df['Electron_phi'].content,
                mass             = df['Electron_mass'].content,
                pdgId            = df['Electron_pdgId'].content,
                charge           = (df['Electron_pdgId']/abs(df['Electron_pdgId'])).content, #NanoAOD charge might be bullshit?
                dxy              = df['Electron_dxy'].content,
                dz               = df['Electron_dz'].content,
                sip3d            = df['Electron_sip3d'].content,
                miniPFRelIso_all = df['Electron_miniPFRelIso_all'].content,
                mvaFall17V2noIso = df['Electron_mvaFall17V2noIso'].content,
                mvaTTH           = df['Electron_mvaTTH'].content,
                genPartIdx       = df['Electron_genPartIdx'].content,
                etaSC            = (df['Electron_eta'] + df['Electron_deltaEtaSC']).content, # verify this
                jetRelIso        = df['Electron_jetRelIso'].content,
                jetPtRelv2       = df['Electron_jetPtRelv2'].content,
                convVeto         = df['Electron_convVeto'].content,
                lostHits         = df['Electron_lostHits'].content,
                tightCharge      = df['Electron_tightCharge'].content,
            )
            
        self.getSelection()
        
        if self.obj == "Electron" and self.wp == "tight":
            self.selection = self.selection & self.getElectronMVAID() & self.getIsolation(0.07, 0.78, 8.0)
        if self.obj == "Muon" and self.wp == "tight":
            self.selection = self.selection & self.getIsolation(0.11, 0.74, 6.8)
        
    def getValue(self, var):
        #return np.nan_to_num(getattr(self.cand, var), -999)
        return getattr(self.cand, var)
    
    def getSelection(self):
        self.selection = (self.cand.pt>0)
        if self.wp == None: return
        if self.v>0: print ("## %s selection for WP %s ##"%(self.obj, self.wp))
        for var in obj_def[self.obj][self.wp].keys():
            #print (var)
            if type(obj_def[self.obj][self.wp][var]) == type(1):
                if self.v>0: print (" - %s == %s"%(var, obj_def[self.obj][self.wp][var]))
                self.selection = self.selection & ( self.getValue(var) == obj_def[self.obj][self.wp][var])
            else:
                extra = obj_def[self.obj][self.wp][var].get('extra')
                if extra=='abs':
                    try:
                        self.selection = self.selection & (abs(self.getValue(var)) >= obj_def[self.obj][self.wp][var][self.year]['min'])
                        if self.v>0: print (" - abs(%s) >= %s"%(var, obj_def[self.obj][self.wp][var][self.year]['min']))
                    except:
                        pass
                    try:
                        self.selection = self.selection & (abs(self.getValue(var)) >= obj_def[self.obj][self.wp][var]['min'])
                        if self.v>0: print (" - abs(%s) >= %s"%(var, obj_def[self.obj][self.wp][var]['min']))
                    except:
                        pass
                    try:
                        self.selection = self.selection & (abs(self.getValue(var)) <= obj_def[self.obj][self.wp][var][self.year]['max'])
                        if self.v>0: print (" - abs(%s) <= %s"%(var, obj_def[self.obj][self.wp][var][self.year]['max']))
                    except:
                        pass
                    try:
                        self.selection = self.selection & (abs(self.getValue(var)) <= obj_def[self.obj][self.wp][var]['max'])
                        if self.v>0: print (" - abs(%s) <= %s"%(var, obj_def[self.obj][self.wp][var]['max']))
                    except:
                        pass
                else:
                    try:
                        self.selection = self.selection & (self.getValue(var) >= obj_def[self.obj][self.wp][var][self.year]['min'])
                        if self.v>0: print (" - %s >= %s"%(var, obj_def[self.obj][self.wp][var][self.year]['min']))
                    except:
                        pass
                    try:
                        self.selection = self.selection & (self.getValue(var) >= obj_def[self.obj][self.wp][var]['min'])
                        if self.v>0: print (" - %s >= %s"%(var, obj_def[self.obj][self.wp][var]['min']))
                    except:
                        pass
                    try:
                        self.selection = self.selection & (self.getValue(var) <= obj_def[self.obj][self.wp][var][self.year]['max'])
                        if self.v>0: print (" - %s <= %s"%(var, obj_def[self.obj][self.wp][var][self.year]['max']))
                    except:
                        pass
                    try:
                        self.selection = self.selection & (self.getValue(var) <= obj_def[self.obj][self.wp][var]['max'])
                        if self.v>0: print (" - %s <= %s"%(var, obj_def[self.obj][self.wp][var]['max']))
                    except:
                        pass
                    
        if self.v>0: print ()
                    
    def get(self):
        return self.cand[self.selection]
        #return selection
        
    def getCuts(self, etaSC, pt):
        if etaSC<0.8: return 2.597
        
    def getMVAscore(self):
        MVA = np.minimum(np.maximum(self.cand.mvaFall17V2noIso, -1.0 + 1.e-6), 1.0 - 1.e-6)
        return -0.5*np.log(2/(MVA+1)-1)
    
    ## some more involved cuts from SS analysis
    def getElectronMVAID(self):
        # this should be year specific, only 2018 for now
        lowEtaCuts  = 2.597, 4.277, 2.597
        midEtaCuts  = 2.252, 3.152, 2.252
        highEtaCuts = 1.054, 2.359, 1.054
        lowEta      = ( abs(self.cand.etaSC) < 0.8 )
        midEta      = ( (abs(self.cand.etaSC) <= 1.479) & (abs(self.cand.etaSC) >= 0.8) )
        highEta     = ( abs(self.cand.etaSC) > 1.479 )
        lowPt       = ( self.cand.pt < 10 )
        midPt       = ( (self.cand.pt <= 25) & (self.cand.pt >= 10) )
        highPt      = (self.cand.pt > 25)
        
        MVA = self.getMVAscore()
        
        ll = ( lowEta & lowPt & (MVA > lowEtaCuts[2] ) )
        lm = ( lowEta & midPt & (MVA > (lowEtaCuts[0]+(lowEtaCuts[1]-lowEtaCuts[0])/15*(self.cand.pt-10)) ) )
        lh = ( lowEta & highPt & (MVA > lowEtaCuts[1] ) )

        ml = ( midEta & lowPt & (MVA > midEtaCuts[2] ) )
        mm = ( midEta & midPt & (MVA > (midEtaCuts[0]+(midEtaCuts[1]-midEtaCuts[0])/15*(self.cand.pt-10)) ) )
        mh = ( midEta & highPt & (MVA > midEtaCuts[1] ) )

        hl = ( highEta & lowPt & (MVA > highEtaCuts[2] ) )
        hm = ( highEta & midPt & (MVA > (highEtaCuts[0]+(highEtaCuts[1]-highEtaCuts[0])/15*(self.cand.pt-10)) ) )
        hh = ( highEta & highPt & (MVA > highEtaCuts[1] ) )
        
        return ( ll | lm | lh | ml | mm | mh | hl | hm | hh )
    
    ## SS isolation
    def getIsolation(self, mini, jet, jetv2 ):
        # again, this is only for 2018 so far
        jetRelIso = 1/(self.cand.jetRelIso+1)
        return ( (self.cand.miniPFRelIso_all < mini) & ( (jetRelIso>jet) | (self.cand.jetPtRelv2>jetv2) ) )
        