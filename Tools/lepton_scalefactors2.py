import numpy as np

try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from Tools.helpers import yahist_2D_lookup

from yahist import Hist1D, Hist2D

class LeptonSF2:
    
    def __init__(self, year=2018):
        self.year = year
        
        if self.year ==2018:
            muonScaleFactor_RunABCD_bins = [np.array([20.,25.,30.,40.,50.,100.]),
                                            np.array([0,0.9,1.2,2.1,2.4]),]

            muonScaleFactor_RunABCD_counts = np.array([[0.9824, 1.0271, 0.9948, 0.9960, 0.9990],
                                               [0.9784, 1.0110, 0.9906, 0.9949, 0.9970],
                                               [1.0153, 0.9855, 1.0042, 1.0010, 1.0010],
                                               [1.0511, 0.9913, 1.0103, 1.0041, 1.0030]])

            self.h_muonScaleFactor_RunABCD = Hist2D.from_bincounts(muonScaleFactor_RunABCD_counts, muonScaleFactor_RunABCD_bins) 

            muonScaleFactor_Medium_bins = [np.array([20.,25.,30.,40.,50.,60.,100.]),
                                           np.array([0,0.9,1.2,2.1,2.4]),]

            muonScaleFactor_Medium_counts = np.array([[0.9916, 0.9951, 1.0004, 0.9980, 0.9965, 0.9989],
                                                 [1.0018, 0.9962, 0.9994, 0.9971, 0.9945, 0.9985],
                                                 [1.0031, 0.9935, 0.9981, 0.9960, 0.9939, 0.9957],
                                                 [0.9889, 0.9733, 0.9786, 0.9762, 0.9720, 0.9806]])


            self.h_muonScaleFactor_Medium = Hist2D.from_bincounts(muonScaleFactor_Medium_counts, muonScaleFactor_Medium_bins) 

            electronScaleFactor_RunABCD_bins = [np.array([10, 20, 35, 50, 100, 200, 300]),
                                                np.array([-2.500, -2.000, -1.566, -1.444, -0.800, 0.000, 0.800, 1.444, 1.566, 2.000, 2.500]),]

            electronScaleFactor_RunABCD_counts = np.array([[1.3737, 1.0673, 0.9891, 0.9433, 0.9245, 0.9371],
                                                      [1.0453, 0.9401, 0.9352, 0.9310, 0.9250, 0.9500],
                                                      [1.3240, 0.9614, 0.9598, 0.9751, 0.9432, 0.8901],
                                                      [0.9262, 0.8841, 0.9237, 0.9311, 0.9448, 0.9460],
                                                      [0.8536, 0.8877, 0.9294, 0.9367, 0.9443, 0.9635],
                                                      [0.9133, 0.8955, 0.9346, 0.9417, 0.9626, 0.9709],
                                                      [0.9344, 0.8932, 0.9231, 0.9354, 0.9644, 0.8999],
                                                      [1.2237, 0.9316, 0.9421, 0.9530, 1.0114, 0.9646],
                                                      [1.0047, 0.9295, 0.9343, 0.9366, 0.8989, 0.9169],
                                                      [1.3372, 1.0471, 0.9709, 0.9212, 0.8736, 1.0113],])

            self.h_electronScaleFactor_RunABCD = Hist2D.from_bincounts(electronScaleFactor_RunABCD_counts, electronScaleFactor_RunABCD_bins)

            electronScaleFactorReco_RunABCD_bins = [np.array([10, 20, 45, 75, 100, 200]),
                                                np.array([-2.500, -2.000, -1.566, -1.444, -1.000, -0.500, 0.000, 0.500, 1.000, 1.444, 1.566, 2.000, 2.500]),]

            electronScaleFactorReco_RunABCD_counts = np.array([[1.0115, 0.9886, 0.9846, 1.0010, 1.0072],
                                                           [0.9724, 0.9908, 0.9908, 1.0061, 0.9919],
                                                           [1.4158, 0.9815, 0.9591, 1.0467, 0.9837],
                                                           [1.0163, 0.9875, 0.9887, 1.0051, 1.0010],
                                                           [0.9095, 0.9897, 0.9908, 1.0020, 1.0010],
                                                           [1.0000, 0.9856, 0.9887, 1.0061, 0.9869],
                                                           [1.0000, 0.9835, 0.9866, 1.0061, 0.9869],
                                                           [0.9095, 0.9866, 0.9887, 1.0020, 1.0010],
                                                           [1.0163, 0.9844, 0.9824, 1.0051, 1.0010],
                                                           [1.4158, 0.9848, 0.9727, 1.0467, 0.9837],
                                                           [0.9724, 0.9887, 0.9908, 1.0061, 0.9919],
                                                           [1.0115, 0.9918, 0.9857, 1.0010, 1.0072]])
            self.h_electronScaleFactorReco_RunABCD = Hist2D.from_bincounts(electronScaleFactorReco_RunABCD_counts, electronScaleFactorReco_RunABCD_bins)
    
    def get(self, ele, mu):
        if self.year == 2018:
            muonScaleFactor_RunABCD = yahist_2D_lookup(self.h_muonScaleFactor_RunABCD, mu.pt, np.abs(mu.eta))
            muonScaleFactor_Medium = yahist_2D_lookup(self.h_muonScaleFactor_Medium, mu.pt, np.abs(mu.eta))
            electronScaleFactor_RunABCD = yahist_2D_lookup(self.h_electronScaleFactor_RunABCD, ele.pt, ele.eta)
            electronScaleFactorReco_RunABCD = yahist_2D_lookup(self.h_electronScaleFactorReco_RunABCD, ele.pt, ele.eta)
            
            sf = ak.prod(muonScaleFactor_RunABCD, axis=1)*ak.prod(muonScaleFactor_Medium, axis=1)*ak.prod(electronScaleFactor_RunABCD, axis=1)*ak.prod(electronScaleFactorReco_RunABCD, axis=1)
            
            return sf
    
    
     