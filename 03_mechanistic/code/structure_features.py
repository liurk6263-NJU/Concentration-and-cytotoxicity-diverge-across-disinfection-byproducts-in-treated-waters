from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
FEATURES = ["ketone_groups", "aliphatic_carbocycles", "aliphatic_halogens", "noncarboxylic_co", "alkyl_halides", "halogen_atom_fraction"]

def feature_values(mol: Chem.Mol) -> dict[str, float]:
    halogens = [a for a in mol.GetAtoms() if a.GetSymbol() in {"F", "Cl", "Br", "I"}]
    aliphatic_halogens = sum(
        not any(neighbor.GetIsAromatic() for neighbor in atom.GetNeighbors())
        for atom in halogens
    )
    heavy_atoms = mol.GetNumHeavyAtoms()
    return {
        "ketone_groups": float(Descriptors.fr_ketone(mol)),
        "aliphatic_carbocycles": float(rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)),
        "aliphatic_halogens": float(aliphatic_halogens),
        "noncarboxylic_co": float(Descriptors.fr_C_O_noCOO(mol)),
        "alkyl_halides": float(Descriptors.fr_alkyl_halide(mol)),
        "halogen_atom_fraction": float(len(halogens) / heavy_atoms) if heavy_atoms else 0.0,
        "heavy_atoms": float(heavy_atoms),
        "mol_wt": float(Descriptors.MolWt(mol)),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "fr_aldehyde": float(Descriptors.fr_aldehyde(mol)),
        "fr_carboxylic_acid": float(Descriptors.fr_COO(mol)),
        "fr_phenol": float(Descriptors.fr_phenol(mol)),
        "fr_ether": float(Descriptors.fr_ether(mol)),
        "fr_nitro": float(Descriptors.fr_nitro(mol)),
    }