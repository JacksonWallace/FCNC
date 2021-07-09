import awkward as ak
from coffea import processor, hist
import numpy as np


def add_processes_to_output(fileset, output):
    for sample in fileset:
        if sample not in output:
            output.update({sample: processor.defaultdict_accumulator(int)})
            
def add_files_to_output(fileset, output):
    for sample in fileset:
        for f in fileset[sample]:
            output.update({f: processor.defaultdict_accumulator(int)})


dataset_axis            = hist.Cat("dataset",       "Primary dataset")
pt_axis                 = hist.Bin('pt',            r'$p_{T}\ (GeV)$', np.array([15, 40, 60, 80, 100, 200, 300]))
pt_axis2                = hist.Bin('pt',            r'$p_{T}\ (GeV)$', np.array([0, 5, 10, 25, 100, 200, 300]))
pt_fine_axis            = hist.Bin('pt',            r'$p_{T}\ (GeV)$', 500, 0, 500)
p_axis                  = hist.Bin("p",             r"$p$ (GeV)", int(2500/5), 0, 2500) # 5 GeV is fine enough
ht_axis                 = hist.Bin("ht",            r"$H_{T}$ (GeV)", 500, 0, 5000)
mass_axis               = hist.Bin("mass",          r"M (GeV)", 1000, 0, 2000)
eta_axis                = hist.Bin('eta',           r'$\eta $', np.array([0, 0.8, 1.479, 2.5]))
etaSC_axis              = hist.Bin('eta',         r'$\eta\ SC$', np.array([0, 0.8, 1.479, 2.5]))
eta_fine_axis           = hist.Bin('eta',           r'$\eta $', 25, -2.5, 2.5)  
phi_axis                = hist.Bin("phi",           r"$\phi$", 64, -3.2, 3.2)
delta_axis              = hist.Bin("delta",         r"$\delta$", 100,0,10 )
multiplicity_axis       = hist.Bin("multiplicity",  r"N", 5, -0.5, 4.5)
n1_axis                 = hist.Bin("n1",            r"N", 4, -0.5, 3.5)
n2_axis                 = hist.Bin("n2",            r"N", 4, -0.5, 3.5)
n_ele_axis              = hist.Bin("n_ele",         r"N", 4, -0.5, 3.5) # we can use this as categorization for ee/emu/mumu
ext_multiplicity_axis   = hist.Bin("multiplicity",  r"N", 100, -0.5, 99.5) # e.g. for PV
norm_axis               = hist.Bin("norm",          r"N", 25, 0, 1)
score_axis              = hist.Bin("score",         r"N", 100, 0, 1)
pdgID_axis              = hist.Bin("pdgID",         r"N", 26, 0, 25)
mva_id_axis             = hist.Bin("mva_id",        r"mva ID", np.array([-1, -0.01, 0.11, 0.48, 0.52, 0.56, 0.77, 100]))
isolation1_axis         = hist.Bin("isolation1",    r"Iso1", np.array([0, 1/0.80-1, 1]))
isolation2_axis         = hist.Bin("isolation2",    r"Iso2", np.array([0, 7.2, 16]))

variations = ['pt_jesTotalUp', 'pt_jesTotalDown']
nb_variations = ['centralUp', 'centralDown', 'upCentral', 'downCentral']

desired_output = {
            
            "PV_npvs" :         hist.Hist("PV_npvs", dataset_axis, ext_multiplicity_axis),
            "PV_npvsGood" :     hist.Hist("PV_npvsGood", dataset_axis, ext_multiplicity_axis),
            "PV_npvsGood2" :     hist.Hist("PV_npvsGood", dataset_axis, ext_multiplicity_axis),
            "PV_npvsGood3" :     hist.Hist("PV_npvsGood", dataset_axis, ext_multiplicity_axis),
            "PV_npvsGood4" :     hist.Hist("PV_npvsGood", dataset_axis, ext_multiplicity_axis),
            "PV_npvsGood5" :     hist.Hist("PV_npvsGood", dataset_axis, ext_multiplicity_axis),
            "PV_npvsGood6" :     hist.Hist("PV_npvsGood", dataset_axis, ext_multiplicity_axis),
            
            "MET" :             hist.Hist("Counts", dataset_axis, pt_fine_axis),
            "MET2" :             hist.Hist("Counts", dataset_axis, pt_fine_axis),
            "MET3" :             hist.Hist("Counts", dataset_axis, pt_fine_axis),
            "MET4" :             hist.Hist("Counts", dataset_axis, pt_fine_axis),
            "MET5" :             hist.Hist("Counts", dataset_axis, pt_fine_axis),
            "MET6" :             hist.Hist("Counts", dataset_axis, pt_fine_axis),
            
            "lead_gen_lep":     hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis),
            "trail_gen_lep":    hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis),
            "j1":               hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis),
            "j2":               hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis),
            "j3":               hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis),

            "b1":               hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis),
            "b2":               hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis),

            "chargeFlip_vs_nonprompt": hist.Hist("Counts", dataset_axis, n1_axis, n2_axis, n_ele_axis),
            
            "high_p_fwd_p":     hist.Hist("Counts", dataset_axis, p_axis),
            
    
            "electron":              hist.Hist("Counts", dataset_axis, pt_fine_axis, eta_axis),
            "electron2":             hist.Hist("Counts", dataset_axis, pt_fine_axis, eta_axis),
            "electron3":             hist.Hist("Counts", dataset_axis, pt_fine_axis, eta_axis),
            "electron4":             hist.Hist("Counts", dataset_axis, pt_fine_axis, eta_axis),
            "electron_data1":        hist.Hist("Counts", dataset_axis, pt_fine_axis, eta_fine_axis, phi_axis),
            "electron_data2":        hist.Hist("Counts", dataset_axis, pt_fine_axis, eta_fine_axis, phi_axis),
            "electron_data3":        hist.Hist("Counts", dataset_axis, pt_fine_axis, eta_fine_axis, phi_axis),
            "electron_data4":        hist.Hist("Counts", dataset_axis, pt_fine_axis, eta_fine_axis, phi_axis),
            "electron_data5":        hist.Hist("Counts", dataset_axis, pt_fine_axis, eta_fine_axis, phi_axis),
            "electron_data6":        hist.Hist("Counts", dataset_axis, pt_fine_axis, eta_fine_axis, phi_axis),
            "electron_data7":        hist.Hist("Counts", dataset_axis, pt_fine_axis, eta_fine_axis, phi_axis),
            "electron_data8":        hist.Hist("Counts", dataset_axis, pt_fine_axis, eta_fine_axis, phi_axis),
            "electron_data9":        hist.Hist("Counts", dataset_axis, pt_fine_axis, eta_fine_axis, phi_axis),
            "electron_data10":        hist.Hist("Counts", dataset_axis, pt_fine_axis, eta_fine_axis, phi_axis),
            "electron_data11":        hist.Hist("Counts", dataset_axis, pt_fine_axis, eta_fine_axis, phi_axis),
            "electron_data12":        hist.Hist("Counts", dataset_axis, pt_fine_axis, eta_fine_axis, phi_axis),
            "flipped_electron":      hist.Hist("Counts", dataset_axis, pt_fine_axis, eta_axis),
            "flipped_electron2":     hist.Hist("Counts", dataset_axis, pt_fine_axis, eta_axis),
            "flipped_electron3":     hist.Hist("Counts", dataset_axis, pt_fine_axis, eta_axis),
            "flipped_electron4":     hist.Hist("Counts", dataset_axis, pt_fine_axis, eta_axis),
            "electron_flips":        hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "electron_flips2":       hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "electron_flips3":        hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "electron_flips4":       hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "muon":                  hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis),
            "lead_lep":              hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis),
            "trail_lep":             hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis),
            "fwd_jet":               hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis), 

            "N_b" :           hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_central" :     hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_ele" :         hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_ele2" :        hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_ele3" :        hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_ele4" :        hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_mu" :          hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_jet" :         hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_jet2" :        hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_jet3" :        hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_jet4" :         hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_jet5" :        hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_jet6" :        hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_fwd" :         hist.Hist("Counts", dataset_axis, multiplicity_axis),

            "nLepFromTop" :       hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "nLepFromW" :         hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "nLepFromTau" :       hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "nLepFromZ" :         hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "nGenTau" :           hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "nGenL" :             hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "GenL" :              hist.Hist("Counts", pt_axis, multiplicity_axis),
            "lepton_parent":      hist.Hist("Counts", dataset_axis, pdgID_axis),
            "lepton_parent2":     hist.Hist("Counts", dataset_axis, pdgID_axis),
    
            "dilep_mass1":        hist.Hist("Counts", dataset_axis, mass_axis, pt_fine_axis),
            "dilep_mass2":        hist.Hist("Counts", dataset_axis, mass_axis, pt_fine_axis),
            "dilep_mass3":        hist.Hist("Counts", dataset_axis, mass_axis, pt_fine_axis),
            "dilep_mass4":        hist.Hist("Counts", dataset_axis, mass_axis, pt_fine_axis),
            "dilep_mass5":        hist.Hist("Counts", dataset_axis, mass_axis, pt_fine_axis),
            "dilep_mass6":        hist.Hist("Counts", dataset_axis, mass_axis, pt_fine_axis),
            
            "mva_id":            hist.Hist("Counts", dataset_axis, mva_id_axis, etaSC_axis),
            "mva_id2":            hist.Hist("Counts", dataset_axis, mva_id_axis, pt_axis2),
            "isolation":         hist.Hist("Counts", dataset_axis, isolation1_axis, isolation2_axis),


            'skimmedEvents':    processor.defaultdict_accumulator(int),
            'totalEvents':      processor.defaultdict_accumulator(int),

}

outputs_with_vars = ['j1', 'j2', 'j3', 'b1', 'b2', 'N_jet', 'fwd_jet', 'N_b', 'N_fwd', 'N_central', 'MET']
for out in outputs_with_vars:
    desired_output.update( { out+'_'+var: desired_output[out].copy() for var in variations } )
    
outputs_with_nb_vars = ['N_b']
for out in outputs_with_nb_vars:
    desired_output.update( { out+'_'+var: desired_output[out].copy() for var in nb_variations } )
