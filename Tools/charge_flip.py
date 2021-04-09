try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from Tools.helpers import yahist_2D_lookup
 

def flip_ratio(h, lepton1, lepton2):
    """takes a dilepton event and weights it based on the 
    odds that one of the leptons has a charge flip"""
    
    flip1 = yahist_2D_lookup(h, lepton1.pt, abs(lepton1.eta))
    flip2 = yahist_2D_lookup(h, lepton2.pt, abs(lepton2.eta))
    
    flip_rate1 = (ak.prod(flip1, axis = 1) * ak.prod(1-flip2, axis = 1)) + (ak.prod(flip2, axis = 1) * ak.prod(1-flip1, axis = 1))
    
    return flip_rate1