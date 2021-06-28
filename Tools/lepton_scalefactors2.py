import numpy as np

try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from Tools.helpers import yahist_1D_lookup, yahist_2D_lookup

from yahist import Hist1D, Hist2D

class LeptonSF2:
    
    def __init__(self, year=2018):
        self.year = year
        
        if self.year == 2016:
            
            muonScaleFactor_bins = [np.array([10., 20., 25., 30., 40., 50., 60., 100.]),
                                    np.array([0., 0.9, 1.2, 2.1, 2.4]),]
            
            muonScaleFactor_counts = np.array([[0.8964, 0.9560, 0.9732, 0.9833, 0.9837, 0.9672, 0.9812],
                                               [0.8871, 0.9518, 0.9622, 0.9720, 0.9769, 0.9727, 0.9797],
                                               [0.9727, 0.9736, 0.9794, 0.9839, 0.9855, 0.9801, 0.9920],
                                               [0.9059, 0.9197, 0.9318, 0.9388, 0.9448, 0.9355, 0.9481],])
            
            self.h_muonScaleFactor = Hist2D.from_bincounts(muonScaleFactor_counts, muonScaleFactor_bins)
            
            trackingSF_bins = np.array([-2.4, -2.1, -1.6, -1.2, -0.9, -0.6, -0.3, -0.2, 0.2, 0.3, 0.6, 0.9, 1.2, 1.6, 2.1, 2.4])
            
            trackingSF_counts = np.array([0.991237, 0.994853, 0.996413, 0.997157, 0.997512, 0.99756, 0.996745, 0.996996, 0.99772, 0.998604, 0.998321, 0.997682, 0.995252, 0.994919, 0.987334])
            
            self.h_trackingSF = Hist1D.from_bincounts(trackingSF_counts, trackingSF_bins)
            
            electronScaleFactor_legacy_bins = [np.array([10., 20., 35., 50., 100., 200., 300.]),
                                               np.array([-2.500, -2.000, -1.566, -1.444, -0.800, 0.000, 0.800, 1.444, 1.566, 2.000, 2.500]),]
            
            electronScaleFactor_legacy_counts = np.array([[1.0634, 0.9979, 0.9861, 0.9949, 1.0214, 0.7836],
                                                          [0.9313, 0.9300, 0.9431, 0.9619, 0.9882, 0.9927],
                                                          [1.0123, 0.9336, 0.9259, 0.9366, 0.9216, 1.0170],
                                                          [0.9516, 0.9176, 0.9339, 0.9334, 0.9215, 0.9272],
                                                          [0.9588, 0.9277, 0.9336, 0.9351, 0.9382, 0.9366],
                                                          [0.9735, 0.9426, 0.9471, 0.9480, 0.9537, 0.9238],
                                                          [0.9210, 0.8963, 0.9155, 0.9185, 0.9074, 0.9028],
                                                          [1.0384, 0.8672, 0.9029, 0.9167, 0.8905, 0.8326],
                                                          [0.9239, 0.9107, 0.9310, 0.9569, 0.9314, 0.9973],
                                                          [0.9779, 0.9431, 0.9554, 0.9662, 1.0301, 1.0125],])
            
            self.h_electronScaleFactor_legacy = Hist2D.from_bincounts(electronScaleFactor_legacy_counts, electronScaleFactor_legacy_bins)
            
            electronScaleFactorReco_legacy_bins = [np.array([10., 20., 45., 75., 100., 200.]),
                                                   np.array([-2.500, -2.000, -1.566, -1.444, -1.000, -0.500, 0.000, 0.500, 1.000, 1.444, 1.566, 2.000, 2.500]),]
            
            electronScaleFactorReco_legacy_counts = np.array([[1.0423, 1.0164, 1.0021, 1.0180, 0.9843],
                                                              [0.9744, 0.9979, 0.9969, 1.0154, 1.0000],
                                                              [1.4277, 0.9908, 0.9622, 1.0329, 1.0022],
                                                              [0.9894, 0.9918, 0.9918, 1.0081, 0.9869],
                                                              [0.9820, 0.9867, 0.9878, 1.0051, 0.9939],
                                                              [0.9820, 0.9836, 0.9868, 0.9969, 0.9858],
                                                              [0.9820, 0.9836, 0.9868, 0.9969, 0.9858],
                                                              [0.9820, 0.9867, 0.9878, 1.0051, 0.9939],
                                                              [0.9894, 0.9918, 0.9918, 1.0081, 0.9869],
                                                              [1.4277, 0.9908, 0.9622, 1.0329, 1.0022],
                                                              [0.9744, 0.9979, 0.9969, 1.0154, 1.0000],
                                                              [1.0423, 1.0164, 1.0021, 1.0180, 0.9843],])
                                #can later rebin these by abs eta
            self.h_electronScaleFactorReco_legacy = Hist2D.from_bincounts(electronScaleFactorReco_legacy_counts, electronScaleFactorReco_legacy_bins)
        
        if self.year == 2017:
            muonScaleFactor_Medium_bins = [np.array([20., 25., 30., 40., 50., 60., 100.]),
                                           np.array([0, 0.9, 1.2, 2.1, 2.4]),]
            
            muonScaleFactor_Medium_counts = np.array([[0.9946, 0.9943, 0.9985, 0.9967, 0.9938, 0.9973],
                                                      [1.0046, 0.9963, 0.9995, 0.9975, 0.9967, 0.9957],
                                                      [0.9955, 0.9935, 0.9982, 0.9959, 0.9946, 0.9964],
                                                      [0.9760, 0.9746, 0.9791, 0.9764, 0.9711, 0.9810]])
            
            self.h_muonScaleFactor_Medium = Hist2D.from_bincounts(muonScaleFactor_Medium_counts, muonScaleFactor_Medium_bins)
            
            muonScaleFactor_RunBCDEF_bins = [np.array([20.,25.,30.,40.,50.,60.]),
                                           np.array([0,0.9,1.2,2.1,2.4]),]
            
            muonScaleFactor_RunBCDEF_counts = np.array([[0.9920, 0.9957, 0.9969, 0.9980, 0.9990],
                                                        [0.9932, 0.9913, 0.9948, 0.9970, 0.9990],
                                                        [0.9945, 0.9957, 0.9979, 0.9990, 0.9990],
                                                        [0.9904, 0.9958, 0.9980, 0.9990, 1.0030]])
            
            self.h_muonScaleFactor_RunBCDEF = Hist2D.from_bincounts(muonScaleFactor_RunBCDEF_counts, muonScaleFactor_RunBCDEF_bins)
            
            electronScaleFactor_RunBCDEF_bins = [np.array([10., 20., 35., 50., 100., 200., 300.]),
                                                 np.array([-2.500, -2.000, -1.566, -1.444, -0.800, 0.000, 0.800, 1.444, 1.566, 2.000, 2.500]),]
            
            electronScaleFactor_RunBCDEF_counts = np.array([[0.8110, 0.7964, 0.8392, 0.8922, 0.9225, 0.8807],
                                                            [0.8088, 0.8230, 0.8713, 0.9195, 0.9652, 1.0230],
                                                            [1.0723, 0.8844, 0.9108, 0.9390, 0.9500, 0.9424],
                                                            [0.9052, 0.8773, 0.9025, 0.9183, 0.9436, 0.9426],
                                                            [0.9279, 0.9041, 0.9245, 0.9358, 0.9560, 0.9518],
                                                            [0.9836, 0.9071, 0.9288, 0.9388, 0.9581, 0.9547],
                                                            [0.9607, 0.8810, 0.9032, 0.9211, 0.9373, 0.9231],
                                                            [0.9872, 0.8527, 0.9018, 0.9243, 1.0209, 0.7952],
                                                            [0.8348, 0.8121, 0.8671, 0.9230, 0.9889, 0.8837],
                                                            [0.8348, 0.7849, 0.8309, 0.8838, 0.9965, 0.9757],])
            
            self.h_electronScaleFactor_RunBCDEF = Hist2D.from_bincounts(electronScaleFactor_RunBCDEF_counts, electronScaleFactor_RunBCDEF_bins)
            
            electronScaleFactorReco_RunBCDEF_bins = [np.array([10, 20, 45, 75, 100, 200]),
                                                np.array([-2.500, -2.000, -1.566, -1.444, -1.000, -0.500, 0.000, 0.500, 1.000, 1.444, 1.566, 2.000, 2.500]),]
            
            electronScaleFactorReco_RunBCDEF_counts = np.array([[0.9823, 0.9775, 0.9837, 0.9970, 0.9899],
                                                                [0.9886, 0.9816, 0.9817, 0.9970, 0.9899],
                                                                [1.0000, 0.9484, 0.9706, 1.0032, 1.0096],
                                                                [0.9765, 0.9693, 0.9755, 0.9959, 0.9847],
                                                                [0.9693, 0.9766, 0.9797, 0.9919, 0.9879],
                                                                [0.9693, 0.9704, 0.9797, 0.9919, 0.9939],
                                                                [0.9693, 0.9703, 0.9776, 0.9919, 0.9939],
                                                                [0.9693, 0.9724, 0.9787, 0.9919, 0.9879],
                                                                [0.9765, 0.9699, 0.9774, 0.9959, 0.9847],
                                                                [1.0000, 0.9580, 0.9639, 1.0032, 1.0096],
                                                                [0.9886, 0.9796, 0.9827, 0.9970, 0.9899],
                                                                [0.9823, 0.9796, 0.9837, 0.9970, 0.9899]])
            
            self.h_electronScaleFactorReco_RunBCDEF = Hist2D.from_bincounts(electronScaleFactorReco_RunBCDEF_counts, electronScaleFactorReco_RunBCDEF_bins)
            
            
        if self.year == 2018:
            muonScaleFactor_RunABCD_bins = [np.array([20., 25., 30., 40., 50., 100.]),
                                            np.array([0, 0.9, 1.2, 2.1, 2.4]),]

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
        if self.year == 2016:
            muonScaleFactor = yahist_2D_lookup(self.h_muonScaleFactor, mu.pt, np.abs(mu.eta))
            trackingSF = yahist_1D_lookup(self.h_trackingSF, mu.eta)
            electronScaleFactor_legacy = yahist_2D_lookup(self.h_electronScaleFactor_legacy, ele.pt, ele.eta)
            electronScaleFactorReco_legacy = yahist_2D_lookup(self.h_electronScaleFactorReco_legacy, ele.pt, ele.eta)
            
            sf = ak.prod(muonScaleFactor, axis=1)*ak.prod(trackingSF, axis=1)*ak.prod(electronScaleFactor_legacy, axis=1)*ak.prod(electronScaleFactorReco_legacy, axis=1)
            
            return sf
        if self.year == 2017:
            muonScaleFactor_Medium = yahist_2D_lookup(self.h_muonScaleFactor_Medium, mu.pt, np.abs(mu.eta))
            muonScaleFactor_RunBCDEF = yahist_2D_lookup(self.h_muonScaleFactor_RunBCDEF, mu.pt, np.abs(mu.eta))
            electronScaleFactor_RunBCDEF = yahist_2D_lookup(self.h_electronScaleFactor_RunBCDEF, ele.pt, ele.eta)
            electronScaleFactorReco_RunBCDEF = yahist_2D_lookup(self.h_electronScaleFactorReco_RunBCDEF, ele.pt, ele.eta)
            
            sf = ak.prod(muonScaleFactor_Medium, axis=1)*ak.prod(muonScaleFactor_RunBCDEF, axis=1)*ak.prod(electronScaleFactor_RunBCDEF, axis=1)*ak.prod(electronScaleFactorReco_RunBCDEF, axis=1)
            
            return sf
        
        if self.year == 2018:
            muonScaleFactor_RunABCD = yahist_2D_lookup(self.h_muonScaleFactor_RunABCD, mu.pt, np.abs(mu.eta))
            muonScaleFactor_Medium = yahist_2D_lookup(self.h_muonScaleFactor_Medium, mu.pt, np.abs(mu.eta))
            electronScaleFactor_RunABCD = yahist_2D_lookup(self.h_electronScaleFactor_RunABCD, ele.pt, ele.eta)
            electronScaleFactorReco_RunABCD = yahist_2D_lookup(self.h_electronScaleFactorReco_RunABCD, ele.pt, ele.eta)
            
            sf = ak.prod(muonScaleFactor_RunABCD, axis=1)*ak.prod(muonScaleFactor_Medium, axis=1)*ak.prod(electronScaleFactor_RunABCD, axis=1)*ak.prod(electronScaleFactorReco_RunABCD, axis=1)
            
            return sf
    
    
     