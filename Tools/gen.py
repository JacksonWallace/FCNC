import awkward as ak

def find_first_parent(particle, maxgen=10):
    tmp_mother = particle.parent
    out_mother_pdg = tmp_mother.pdgId
    found = ak.zeros_like(particle.pdgId)

    for i in range(maxgen):
        update = ak.fill_none((tmp_mother.pdgId!=particle.pdgId) * (found==0), False)
        out_mother_pdg = update*ak.fill_none(tmp_mother.pdgId, 0) + (~update) * out_mother_pdg
        found=found+update
        tmp_mother = tmp_mother.parent
        
    return out_mother_pdg

def get_lepton_fromW(events):
    gp = events.GenPart
    gp_lep = gp[((abs(gp.pdgId)==11)|(abs(gp.pdgId)==13)|(abs(gp.pdgId)==15))]
    gp_lep_fromW = gp_lep[abs(gp_lep.parent.pdgId)==24]
    gp_leptonic_W_parent = find_first_parent(gp_lep_fromW.parent, maxgen=19)
    gp_lep_fromW_noTop = gp_lep_fromW[abs(gp_leptonic_W_parent)!=6]
    return gp_lep_fromW_noTop

def get_neutrino_fromW(events):
    gp = events.GenPart
    gp_lep = gp[((abs(gp.pdgId)==12)|(abs(gp.pdgId)==14)|(abs(gp.pdgId)==16))]
    gp_lep_fromW = gp_lep[abs(gp_lep.parent.pdgId)==24]
    gp_leptonic_W_parent = find_first_parent(gp_lep_fromW.parent, maxgen=19)
    gp_lep_fromW_noTop = gp_lep_fromW[abs(gp_leptonic_W_parent)!=6]
    return gp_lep_fromW_noTop

def get_lepton_filter(events):
    gp = events.GenPart
    gp_lep = gp[((abs(gp.pdgId)==11)|(abs(gp.pdgId)==13)|(abs(gp.pdgId)==15))]
    gp_lep_fromW = gp_lep[abs(gp_lep.parent.pdgId)==24]
    gp_leptonic_W_parent = find_first_parent(gp_lep_fromW.parent, maxgen=19)
    gp_lep_fromW_noTop = gp_lep_fromW[abs(gp_leptonic_W_parent)!=6]
    return ak.num(gp_lep_fromW_noTop)>0

def get_lepton_from(events, parent_pdg=6):
    gp = events.GenPart
    gp_lep = gp[((abs(gp.pdgId)==11)|(abs(gp.pdgId)==13)|(abs(gp.pdgId)==15))]
    gp_lep_fromW = gp_lep[abs(gp_lep.parent.pdgId)==24]
    gp_leptonic_W_parent = find_first_parent(gp_lep_fromW.parent, maxgen=19)
    return gp_lep_fromW[gp_leptonic_W_parent==parent_pdg]

def get_charge_parent(particle):
    parent = find_first_parent(particle)
    if parent == 11 or parent == 13 or parent == 15 or parent == 17 or parent == -24 or parent == -37:
        charge = -1
    elif parent == -11 or parent == -13 or parent == -15 or parent == -17 or parent == 24 or parent == 37:
        charge = 1
    elif parent == 2 or parent == 4 or parent == 6 or parent == 8:
        charge = 2/3
    elif parent == -2 or parent == -4 or parent == -6 or parent == -8:
        charge = -2/3
    elif parent == 1 or parent == 3 or parent == 5 or parent == 7:
        charge = -1/3
    elif parent == -1 or parent == -3 or parent == -5 or parent == -7:
        charge = 1/3
    elif parent == 12 or parent == 14 or parent == 16 or parent == 18 or parent == 9 or parent == 21 or parent == 22 or parent == 23 or parent == 25:
        charge = 0
    return charge
    
