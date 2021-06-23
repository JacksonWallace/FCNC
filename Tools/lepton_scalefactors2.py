import numpy as np

class LeptonSF2:
    
    def __init__(self, year=2018):
        self.year = year
        
    
    def get(self, ele, mu):
        if self.year == 2018:
            sf = ak.prod(muonScaleFactor_RunABCD(mu.pt, mu.eta), axis=1)*ak.prod(muonScaleFactor_Medium(mu.pt, mu.eta), axis=1)*ak.prod(electronScaleFactor_RunABCD(electron.pt, electron.eta), axis=1)*ak.prod(electronScaleFactorReco_RunABCD(electron.pt, electron.eta),axis=1)
            return sf
    
    
    def muonScaleFactor_RunABCD(pt, eta):
        if (pt >= 20 and pt < 25 and np.abs(eta) >= 0.000 and np.abs(eta) < 0.900):
            return 0.9824
        if (pt >= 20 and pt < 25 and np.abs(eta) >= 0.900 and np.abs(eta) < 1.200):
            return 0.9784
        if (pt >= 20 and pt < 25 and np.abs(eta) >= 1.200 and np.abs(eta) < 2.100):
            return 1.0153
        if (pt >= 20 and pt < 25 and np.abs(eta) >= 2.100 and np.abs(eta) < 2.400):
            return 1.0511
        if (pt >= 25 and pt < 30 and np.abs(eta) >= 0.000 and np.abs(eta) < 0.900):
            return 0.9913
        if (pt >= 25 and pt < 30 and np.abs(eta) >= 0.900 and np.abs(eta) < 1.200):
            return 0.9855
        if (pt >= 25 and pt < 30 and np.abs(eta) >= 1.200 and np.abs(eta) < 2.100):
            return 1.0110
        if (pt >= 25 and pt < 30 and np.abs(eta) >= 2.100 and np.abs(eta) < 2.400):
            return 1.0271
        if (pt >= 30 and pt < 40 and np.abs(eta) >= 0.000 and np.abs(eta) < 0.900):
            return 0.9948
        if (pt >= 30 and pt < 40 and np.abs(eta) >= 0.900 and np.abs(eta) < 1.200):
            return 0.9906
        if (pt >= 30 and pt < 40 and np.abs(eta) >= 1.200 and np.abs(eta) < 2.100):
            return 1.0042
        if (pt >= 30 and pt < 40 and np.abs(eta) >= 2.100 and np.abs(eta) < 2.400):
            return 1.0103
        if (pt >= 40 and pt < 50 and np.abs(eta) >= 0.000 and np.abs(eta) < 0.900):
            return 0.9960
        if (pt >= 40 and pt < 50 and np.abs(eta) >= 0.900 and np.abs(eta) < 1.200):
            return 0.9949
        if (pt >= 40 and pt < 50 and np.abs(eta) >= 1.200 and np.abs(eta) < 2.100):
            return 1.0010
        if (pt >= 40 and pt < 50 and np.abs(eta) >= 2.100 and np.abs(eta) < 2.400):
            return 1.0041
        if (pt >= 50 and np.abs(eta) >= 0.000 and np.abs(eta) < 0.900):
            return 0.9990
        if (pt >= 50 and np.abs(eta) >= 0.900 and np.abs(eta) < 1.200):
            return 0.9970
        if (pt >= 50 and np.abs(eta) >= 1.200 and np.abs(eta) < 2.100):
            return 1.0010
        if (pt >= 50 and np.abs(eta) >= 2.100 and np.abs(eta) < 2.400):
            return 1.0030
        return 1.0
    
    def muonScaleFactor_Medium(pt, eta):
        if (pt >= 20 and pt < 25 and np.abs(eta) >= 0.000 and np.abs(eta) < 0.900):
            return 0.9916
        if (pt >= 20 and pt < 25 and np.abs(eta) >= 0.900 and np.abs(eta) < 1.200):
            return 1.0018
        if (pt >= 20 and pt < 25 and np.abs(eta) >= 1.200 and np.abs(eta) < 2.100):
            return 1.0031
        if (pt >= 20 and pt < 25 and np.abs(eta) >= 2.100 and np.abs(eta) < 2.400):
            return 0.9889
        if (pt >= 25 and pt < 30 and np.abs(eta) >= 0.000 and np.abs(eta) < 0.900):
            return 0.9951
        if (pt >= 25 and pt < 30 and np.abs(eta) >= 0.900 and np.abs(eta) < 1.200):
            return 0.9962
        if (pt >= 25 and pt < 30 and np.abs(eta) >= 1.200 and np.abs(eta) < 2.100):
            return 0.9935
        if (pt >= 25 and pt < 30 and np.abs(eta) >= 2.100 and np.abs(eta) < 2.400):
            return 0.9733
        if (pt >= 30 and pt < 40 and np.abs(eta) >= 0.000 and np.abs(eta) < 0.900):
            return 1.0004
        if (pt >= 30 and pt < 40 and np.abs(eta) >= 0.900 and np.abs(eta) < 1.200):
            return 0.9994
        if (pt >= 30 and pt < 40 and np.abs(eta) >= 1.200 and np.abs(eta) < 2.100):
            return 0.9981
        if (pt >= 30 and pt < 40 and np.abs(eta) >= 2.100 and np.abs(eta) < 2.400):
            return 0.9786
        if (pt >= 40 and pt < 50 and np.abs(eta) >= 0.000 and np.abs(eta) < 0.900):
            return 0.9980
        if (pt >= 40 and pt < 50 and np.abs(eta) >= 0.900 and np.abs(eta) < 1.200):
            return 0.9971
        if (pt >= 40 and pt < 50 and np.abs(eta) >= 1.200 and np.abs(eta) < 2.100):
            return 0.9960
        if (pt >= 40 and pt < 50 and np.abs(eta) >= 2.100 and np.abs(eta) < 2.400):
            return 0.9762
        if (pt >= 50 and pt < 60 and np.abs(eta) >= 0.000 and np.abs(eta) < 0.900):
            return 0.9965
        if (pt >= 50 and pt < 60 and np.abs(eta) >= 0.900 and np.abs(eta) < 1.200):
            return 0.9945
        if (pt >= 50 and pt < 60 and np.abs(eta) >= 1.200 and np.abs(eta) < 2.100):
            return 0.9939
        if (pt >= 50 and pt < 60 and np.abs(eta) >= 2.100 and np.abs(eta) < 2.400):
            return 0.9720
        if (pt >= 60 and np.abs(eta) >= 0.000 and np.abs(eta) < 0.900):
            return 0.9989
        if (pt >= 60 and np.abs(eta) >= 0.900 and np.abs(eta) < 1.200):
            return 0.9985
        if (pt >= 60 and np.abs(eta) >= 1.200 and np.abs(eta) < 2.100):
            return 0.9957
        if (pt >= 60 and np.abs(eta) >= 2.100 and np.abs(eta) < 2.400):
            return 0.9806
        return 1.0
    
    
    def electronScaleFactor_RunABCD(pt, eta):
        if (pt >= 10 and pt < 20 and eta >= -2.500 and eta < -2.000):
            return 1.3737
        if (pt >= 10.0 and pt < 20.0 and eta >= -2.000 and eta < -1.566):
            return 1.0453
        if (pt >= 10 and pt < 20 and eta >= -1.566 and eta < -1.444):
            return 1.3240
        if (pt >= 10 and pt < 20 and eta >= -1.444 and eta < -0.800):
            return 0.9262
        if (pt >= 10 and pt < 20 and eta >= -0.800 and eta < 0.000):
            return 0.8536
        if (pt >= 10 and pt < 20 and eta >= 0.000 and eta < 0.800):
            return 0.9133
        if (pt >= 10 and pt < 20 and eta >= 0.800 and eta < 1.444):
            return 0.9344
        if (pt >= 10 and pt < 20 and eta >= 1.444 and eta < 1.566):
            return 1.2237
        if (pt >= 10 and pt < 20 and eta >= 1.566 and eta < 2.000):
            return 1.0047
        if (pt >= 10 and pt < 20 and eta >= 2.000 and eta < 2.500):
            return 1.3372
        if (pt >= 20 and pt < 35 and eta >= -2.500 and eta < -2.000):
            return 1.0673
        if (pt >= 20 and pt < 35 and eta >= -2.000 and eta < -1.566):
            return 0.9401
        if (pt >= 20 and pt < 35 and eta >= -1.566 and eta < -1.444):
            return 0.9614
        if (pt >= 20 and pt < 35 and eta >= -1.444 and eta < -0.800):
            return 0.8841
        if (pt >= 20 and pt < 35 and eta >= -0.800 and eta < 0.000):
            return 0.8877
        if (pt >= 20 and pt < 35 and eta >= 0.000 and eta < 0.800):
            return 0.8955
        if (pt >= 20 and pt < 35 and eta >= 0.800 and eta < 1.444):
            return 0.8932
        if (pt >= 20 and pt < 35 and eta >= 1.444 and eta < 1.566):
            return 0.9316
        if (pt >= 20 and pt < 35 and eta >= 1.566 and eta < 2.000):
            return 0.9295
        if (pt >= 20 and pt < 35 and eta >= 2.000 and eta < 2.500):
            return 1.0471
        if (pt >= 35 and pt < 50 and eta >= -2.500 and eta < -2.000):
            return 0.9891
        if (pt >= 35 and pt < 50 and eta >= -2.000 and eta < -1.566):
            return 0.9352
        if (pt >= 35 and pt < 50 and eta >= -1.566 and eta < -1.444):
            return 0.9598
        if (pt >= 35 and pt < 50 and eta >= -1.444 and eta < -0.800):
            return 0.9237
        if (pt >= 35 and pt < 50 and eta >= -0.800 and eta < 0.000):
            return 0.9294
        if (pt >= 35 and pt < 50 and eta >= 0.000 and eta < 0.800):
            return 0.9346
        if (pt >= 35 and pt < 50 and eta >= 0.800 and eta < 1.444): 
            return 0.9231
        if (pt >= 35 and pt < 50 and eta >= 1.444 and eta < 1.566):
            return 0.9421
        if (pt >= 35 and pt < 50 and eta >= 1.566 and eta < 2.000):
            return 0.9343
        if (pt >= 35 and pt < 50 and eta >= 2.000 and eta < 2.500): 
            return 0.9709
        if (pt >= 50 and pt < 100 and eta >= -2.500 and eta < -2.000):
            return 0.9433
        if (pt >= 50 and pt < 100 and eta >= -2.000 and eta < -1.566):
            return 0.9310
        if (pt >= 50 and pt < 100 and eta >= -1.566 and eta < -1.444):
            return 0.9751
        if (pt >= 50 and pt < 100 and eta >= -1.444 and eta < -0.800):
            return 0.9311
        if (pt >= 50 and pt < 100 and eta >= -0.800 and eta < 0.000):
            return 0.9367
        if (pt >= 50 and pt < 100 and eta >= 0.000 and eta < 0.800):
            return 0.9417
        if (pt >= 50 and pt < 100 and eta >= 0.800 and eta < 1.444):
            return 0.9354
        if (pt >= 50 and pt < 100 and eta >= 1.444 and eta < 1.566):
            return 0.9530
        if (pt >= 50 and pt < 100 and eta >= 1.566 and eta < 2.000):
            return 0.9366
        if (pt >= 50 and pt < 100 and eta >= 2.000 and eta < 2.500):
            return 0.9212
        if (pt >= 100 and pt < 200 and eta >= -2.500 and eta < -2.000):
            return 0.9245
        if (pt >= 100 and pt < 200 and eta >= -2.000 and eta < -1.566):
            return 0.9250
        if (pt >= 100 and pt < 200 and eta >= -1.566 and eta < -1.444):
            return 0.9432
        if (pt >= 100 and pt < 200 and eta >= -1.444 and eta < -0.800):
            return 0.9448
        if (pt >= 100 and pt < 200 and eta >= -0.800 and eta < 0.000):
            return 0.9443
        if (pt >= 100 and pt < 200 and eta >= 0.000 and eta < 0.800):
            return 0.9626
        if (pt >= 100 and pt < 200 and eta >= 0.800 and eta < 1.444):
            return 0.9644
        if (pt >= 100 and pt < 200 and eta >= 1.444 and eta < 1.566):
            return 1.0114
        if (pt >= 100 and pt < 200 and eta >= 1.566 and eta < 2.000):
            return 0.8989
        if (pt >= 100 and pt < 200 and eta >= 2.000 and eta < 2.500):
            return 0.8736
        if (pt >= 200 and eta >= -2.500 and eta < -2.000):
            return 0.9371
        if (pt >= 200 and eta >= -2.000 and eta < -1.566):
            return 0.9500
        if (pt >= 200 and eta >= -1.566 and eta < -1.444):
            return 0.8901
        if (pt >= 200 and eta >= -1.444 and eta < -0.800):
            return 0.9460
        if (pt >= 200 and eta >= -0.800 and eta < 0.000):
            return 0.9635
        if (pt >= 200 and eta >= 0.000 and eta < 0.800):
            return 0.9709
        if (pt >= 200 and eta >= 0.800 and eta < 1.444):
            return 0.8999
        if (pt >= 200 and eta >= 1.444 and eta < 1.566):
            return 0.9646
        if (pt >= 200 and eta >= 1.566 and eta < 2.000): 
            return 0.9169
        if (pt >= 200 and eta >= 2.000 and eta < 2.500): 
            return 1.0113
        return 0.0
    
    def electronScaleFactorReco_RunABCD(pt, eta):
        if (pt >= 10 and pt < 20 and eta >= -2.500 and eta < -2.000):
            return 1.0115
        if (pt >= 10 and pt < 20 and eta >= -2.000 and eta < -1.566):
            return 0.9724
        if (pt >= 10 and pt < 20 and eta >= -1.566 and eta < -1.444):
            return 1.4158
        if (pt >= 10 and pt < 20 and eta >= -1.444 and eta < -1.000):
            return 1.0163
        if (pt >= 10 and pt < 20 and eta >= -1.000 and eta < -0.500):
            return 0.9095
        if (pt >= 10 and pt < 20 and eta >= -0.500 and eta < 0.000):
            return 1.0000
        if (pt >= 10 and pt < 20 and eta >= 0.000 and eta < 0.500):
            return 1.0000
        if (pt >= 10 and pt < 20 and eta >= 0.500 and eta < 1.000):
            return 0.9095
        if (pt >= 10 and pt < 20 and eta >= 1.000 and eta < 1.444):
            return 1.0163
        if (pt >= 10 and pt < 20 and eta >= 1.444 and eta < 1.566):
            return 1.4158
        if (pt >= 10 and pt < 20 and eta >= 1.566 and eta < 2.000):
            return 0.9724
        if (pt >= 10 and pt < 20 and eta >= 2.000 and eta < 2.500):
            return 1.0115
        if (pt >= 20 and pt < 45 and eta >= -2.500 and eta < -2.000):
            return 0.9886
        if (pt >= 20 and pt < 45 and eta >= -2.000 and eta < -1.566):
            return 0.9908
        if (pt >= 20 and pt < 45 and eta >= -1.566 and eta < -1.444):
            return 0.9815
        if (pt >= 20 and pt < 45 and eta >= -1.444 and eta < -1.000):
            return 0.9875
        if (pt >= 20 and pt < 45 and eta >= -1.000 and eta < -0.500):
            return 0.9897
        if (pt >= 20 and pt < 45 and eta >= -0.500 and eta < 0.000):
            return 0.9856
        if (pt >= 20 and pt < 45 and eta >= 0.000 and eta < 0.500):
            return 0.9835
        if (pt >= 20 and pt < 45 and eta >= 0.500 and eta < 1.000):
            return 0.9866
        if (pt >= 20 and pt < 45 and eta >= 1.000 and eta < 1.444):
            return 0.9844
        if (pt >= 20 and pt < 45 and eta >= 1.444 and eta < 1.566):
            return 0.9848
        if (pt >= 20 and pt < 45 and eta >= 1.566 and eta < 2.000):
            return 0.9887
        if (pt >= 20 and pt < 45 and eta >= 2.000 and eta < 2.500):
            return 0.9918
        if (pt >= 45 and pt < 75 and eta >= -2.500 and eta < -2.000):
            return 0.9846
        if (pt >= 45 and pt < 75 and eta >= -2.000 and eta < -1.566):
            return 0.9908
        if (pt >= 45 and pt < 75 and eta >= -1.566 and eta < -1.444):
            return 0.9591
        if (pt >= 45 and pt < 75 and eta >= -1.444 and eta < -1.000):
            return 0.9887
        if (pt >= 45 and pt < 75 and eta >= -1.000 and eta < -0.500):
            return 0.9908
        if (pt >= 45 and pt < 75 and eta >= -0.500 and eta < 0.000):
            return 0.9887
        if (pt >= 45 and pt < 75 and eta >= 0.000 and eta < 0.500):
            return 0.9866
        if (pt >= 45 and pt < 75 and eta >= 0.500 and eta < 1.000):
            return 0.9887
        if (pt >= 45 and pt < 75 and eta >= 1.000 and eta < 1.444):
            return 0.9824
        if (pt >= 45 and pt < 75 and eta >= 1.444 and eta < 1.566):
            return 0.9727
        if (pt >= 45 and pt < 75 and eta >= 1.566 and eta < 2.000):
            return 0.9908
        if (pt >= 45 and pt < 75 and eta >= 2.000 and eta < 2.500):
            return 0.9857
        if (pt >= 75 and pt < 100 and eta >= -2.500 and eta < -2.000):
            return 1.0010
        if (pt >= 75 and pt < 100 and eta >= -2.000 and eta < -1.566):
            return 1.0061
        if (pt >= 75 and pt < 100 and eta >= -1.566 and eta < -1.444):
            return 1.0467
        if (pt >= 75 and pt < 100 and eta >= -1.444 and eta < -1.000):
            return 1.0051
        if (pt >= 75 and pt < 100 and eta >= -1.000 and eta < -0.500):
            return 1.0020
        if (pt >= 75 and pt < 100 and eta >= -0.500 and eta < 0.000):
            return 1.0061
        if (pt >= 75 and pt < 100 and eta >= 0.000 and eta < 0.500):
            return 1.0061
        if (pt >= 75 and pt < 100 and eta >= 0.500 and eta < 1.000):
            return 1.0020
        if (pt >= 75 and pt < 100 and eta >= 1.000 and eta < 1.444):
            return 1.0051
        if (pt >= 75 and pt < 100 and eta >= 1.444 and eta < 1.566):
            return 1.0467
        if (pt >= 75 and pt < 100 and eta >= 1.566 and eta < 2.000):
            return 1.0061
        if (pt >= 75 and pt < 100 and eta >= 2.000 and eta < 2.500):
            return 1.0010
        if (pt >= 100  and eta >= -2.500 and eta < -2.000):
            return 1.0072
        if (pt >= 100  and eta >= -2.000 and eta < -1.566):
            return 0.9919
        if (pt >= 100  and eta >= -1.566 and eta < -1.444):
            return 0.9837
        if (pt >= 100  and eta >= -1.444 and eta < -1.000):
            return 1.0010
        if (pt >= 100  and eta >= -1.000 and eta < -0.500):
            return 1.0010
        if (pt >= 100  and eta >= -0.500 and eta < 0.000):
            return 0.9869
        if (pt >= 100  and eta >= 0.000 and eta < 0.500):
            return 0.9869
        if (pt >= 100  and eta >= 0.500 and eta < 1.000):
            return 1.0010
        if (pt >= 100  and eta >= 1.000 and eta < 1.444):
            return 1.0010
        if (pt >= 100  and eta >= 1.444 and eta < 1.566):
            return 0.9837
        if (pt >= 100  and eta >= 1.566 and eta < 2.000):
            return 0.9919
        if (pt >= 100  and eta >= 2.000 and eta < 2.500):
            return 1.0072
        return 0.0