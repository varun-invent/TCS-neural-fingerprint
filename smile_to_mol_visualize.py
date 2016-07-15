from mol_graph import graph_from_smiles_tuple, degrees,graph_from_smiles
from build_convnet import array_rep_from_smiles
import numpy as np
import pdb
from rdkit import Chem
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem import Draw

import matplotlib.pyplot as plt
# from visualization import draw_molecule_with_highlights

highlight_color = (30.0/255.0, 100.0/255.0, 255.0/255.0)  # A nice light blue.
figsize = (100, 100)

def draw_molecule_with_highlights(filename, smiles, highlight_atoms):
    drawoptions = DrawingOptions()
    drawoptions.selectColor = highlight_color
    drawoptions.elemDict = {}   # Don't color nodes based on their element.
    drawoptions.bgColor=None

    mol = Chem.MolFromSmiles(smiles)
    fig = Draw.MolToMPL(mol, highlightAtoms=highlight_atoms, size=figsize, options=drawoptions,fitImage=False)

    fig.gca().set_axis_off()
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def main():
	print "This is the file"
	mol_smile = [('OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O'),('c1ccsc1')]
	# mol_smile = mol_smile[1]
	# m = Chem.MolFromSmiles(mol_smile)
	# alist = m.GetAtoms()
	# a = alist[0]
	print mol_smile
	molgraph = graph_from_smiles_tuple(mol_smile)
	arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'), # List of lists.
                'rdkit_ix'      : molgraph.rdkit_ix_array()}  # For plotting only.
	for degree in degrees:
        # import pdb; pdb.set_trace()
		arrayrep[('atom_neighbors', degree)] = \
		    np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)    #V: The degree of an atom is defined to be its number of directly-bonded neighbors.
		arrayrep[('bond_neighbors', degree)] = \
			np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
	# pdb.set_trace()
	filename = 'varun_visualize_atoms'
	molecule_idx = 1
	highlight_atom_nodes = arrayrep['atom_list'][molecule_idx]
	highlight_list_rdkit = [arrayrep['rdkit_ix'][our_ix] for our_ix in highlight_atom_nodes]
	pdb.set_trace()
	draw_molecule_with_highlights(filename, mol_smile[molecule_idx],highlight_list_rdkit[0:3])

if __name__ == '__main__':
    main()