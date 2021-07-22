import awkward as ak

from coffea.lookup_tools import extractor
from Tools.helpers import yahist_2D_lookup
import gzip
import pickle
 
class charge_flip:
    def __init__(self, path):
        self.path = path
        with gzip.open(self.path) as fin:
            self.ratio= pickle.load(fin)
    
    def flip_ratio(self, lepton1, lepton2):
        """takes a dilepton event and weights it based on the 
        odds that one of the leptons has a charge flip"""

        flip1 = yahist_2D_lookup(self.ratio, lepton1.pt, abs(lepton1.eta))
        flip2 = yahist_2D_lookup(self.ratio, lepton2.pt, abs(lepton2.eta))

        flip_rate1 = (ak.prod(flip1, axis = 1) * ak.prod(1/(1-flip1), axis = 1) * ak.prod(1-flip2/(1-flip2), axis = 1)) + (ak.prod(flip2, axis = 1) * ak.prod(1/(1-flip2), axis = 1) * ak.prod(1-flip1/(1-flip1), axis = 1))

        return flip_rate1
    
    def flip_weight(self, electron):

        #f_1 = self.evaluator['el'](electron.pt[:,0:1], abs(electron.eta[:,0:1]))
        #f_2 = self.evaluator['el'](electron.pt[:,1:2], abs(electron.eta[:,1:2]))

        # For custom measurements
        f_1 = yahist_2D_lookup(self.ratio, electron.pt[:,0:1], abs(electron.eta[:,0:1]))
        f_2 = yahist_2D_lookup(self.ratio, electron.pt[:,1:2], abs(electron.eta[:,1:2]))

        # I'm using ak.prod and ak.sum to replace empty arrays by 1 and 0, respectively
        weight = ak.sum(f_1/(1-f_1), axis=1)*ak.prod(1-f_2/(1-f_2), axis=1) + ak.sum(f_2/(1-f_2), axis=1)*ak.prod(1-f_1/(1-f_1), axis=1)

        return weight


