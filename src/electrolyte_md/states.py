def coordination_state(n_polymer_contacts:int,n_anion_contacts:int)->str:
    if n_polymer_contacts>0 and n_anion_contacts==0: return "P"
    if n_polymer_contacts>0 and n_anion_contacts>0: return "PT"
    if n_polymer_contacts==0 and n_anion_contacts>0: return "T"
    return "F"