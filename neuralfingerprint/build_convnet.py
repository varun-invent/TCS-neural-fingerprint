import autograd.numpy as np
from autograd.scipy.misc import logsumexp

from features import num_atom_features, num_bond_features
from util import memoize, WeightsParser
from mol_graph import graph_from_smiles_tuple, degrees
from build_vanilla_net import build_fingerprint_deep_net, relu, batch_normalize


def fast_array_from_list(xs):
    # import pdb; pdb.set_trace()
    # print 'I am in fast_array_from_list'
    return np.concatenate([np.expand_dims(x, axis=0) for x in xs], axis=0)

def sum_and_stack(features, idxs_list_of_lists):
    # import pdb; pdb.set_trace()
    # print 'I am in sum_and_stack'
    return fast_array_from_list([np.sum(features[idx_list], axis=0) for idx_list in idxs_list_of_lists])

def softmax(X, axis=0):
    return np.exp(X - logsumexp(X, axis=axis, keepdims=True))

def matmult_neighbors(array_rep, atom_features, bond_features, get_weights):
    # import pdb; pdb.set_trace()
    activations_by_degree = []
    for degree in degrees:
        atom_neighbors_list = array_rep[('atom_neighbors', degree)]
        bond_neighbors_list = array_rep[('bond_neighbors', degree)]
        if len(atom_neighbors_list) > 0:
            # import pdb; pdb.set_trace()
            neighbor_features = [atom_features[atom_neighbors_list],
                                 bond_features[bond_neighbors_list]]
            # dims of stacked_neighbors are [atoms, neighbors, atom and bond features]
            stacked_neighbors = np.concatenate(neighbor_features, axis=2) # V: Concatenates the atom and bond features 62 + 6 = 68 features for 'degree' = degree and outputs (no_atoms x 1 x 68)  
            summed_neighbors = np.sum(stacked_neighbors, axis=1) # V: removes the singleton dimension and outputs (no_atoms x 68)
            activations = np.dot(summed_neighbors, get_weights(degree))  # V: 336 x 20 for 1st iteration
            activations_by_degree.append(activations)
    # This operation relies on atoms being sorted by degree,
    # in Node.graph_from_smiles_tuple()
    # import pdb; pdb.set_trace()
    return np.concatenate(activations_by_degree, axis=0) # V: outputs a matrix of size 1370 x 20 for first call to matmult_neighbors

def weights_name(layer, degree):
    return "layer " + str(layer) + " degree " + str(degree) + " filter"

def build_convnet_fingerprint_fun(num_hidden_features=[100, 100], fp_length=512,
                                  normalize=True, activation_function=relu,
                                  return_atom_activations=False):
    """Sets up functions to compute convnets over all molecules in a minibatch together."""
    #import pdb; pdb.set_trace()
    # Specify weight shapes.
    parser = WeightsParser()
    all_layer_sizes = [num_atom_features()] + num_hidden_features  # """ V:Concatinating 2 lists OUT: [62,20,20, 20, 20] """
    print("num_atom_features ",num_atom_features())
    for layer in range(len(all_layer_sizes)):
        parser.add_weights(('layer output weights', layer), (all_layer_sizes[layer], fp_length))
        parser.add_weights(('layer output bias', layer),    (1, fp_length))

    in_and_out_sizes = zip(all_layer_sizes[:-1], all_layer_sizes[1:]) #""" V :OUT: [(62,20), (20,20), (20,20), (20,20)]"""
    print("in_and_out_sizes ",in_and_out_sizes)
    for layer, (N_prev, N_cur) in enumerate(in_and_out_sizes):
        parser.add_weights(("layer", layer, "biases"), (1, N_cur))
        parser.add_weights(("layer", layer, "self filter"), (N_prev, N_cur))
        for degree in degrees:  ################## V: I Dont know what a degree is ##########################   degrees = [0, 1, 2, 3, 4, 5]
            parser.add_weights(weights_name(layer, degree), (N_prev + num_bond_features(), N_cur))

    def update_layer(weights, layer, atom_features, bond_features, array_rep, normalize=False):
        # import pdb; pdb.set_trace()  
        def get_weights_func(degree):
            return parser.get(weights, weights_name(layer, degree))
        layer_bias         = parser.get(weights, ("layer", layer, "biases"))
        layer_self_weights = parser.get(weights, ("layer", layer, "self filter"))
        self_activations = np.dot(atom_features, layer_self_weights)           
        neighbour_activations = matmult_neighbors(   
            array_rep, atom_features, bond_features, get_weights_func)             
        # import pdb; pdb.set_trace()
        total_activations = neighbour_activations + self_activations + layer_bias    ### DOUBT : if i check the atom features here for visualisation, it is 1370
        # import pdb; pdb.set_trace() 
        # print("Total activations", np.shape(total_activations))           
        if normalize:
            total_activations = batch_normalize(total_activations)
        return activation_function(total_activations)

    def output_layer_fun_and_atom_activations(weights, smiles):  # V: Came here from line # 108 def output_layer_fun(weights, smiles)
        """Computes layer-wise convolution, and returns a fixed-size output."""
        import pdb; pdb.set_trace()
        array_rep = array_rep_from_smiles(tuple(smiles))
        atom_features = array_rep['atom_features']  # V: (1370,62)
        bond_features = array_rep['bond_features'] # V: (1416,6)

        all_layer_fps = []
        atom_activations = []
        def write_to_fingerprint(atom_features, layer):
            # import pdb; pdb.set_trace()
            cur_out_weights = parser.get(weights, ('layer output weights', layer))
            cur_out_bias    = parser.get(weights, ('layer output bias', layer))
            # import pdb; pdb.set_trace()
            atom_outputs = softmax(cur_out_bias + np.dot(atom_features, cur_out_weights), axis=1)  #V: Smooth all the atom features and then find the softmax, i.e the FP
            atom_activations.append(atom_outputs)   # V: Not needed for neural fingerprint, needed for visualization in neural FP
            # Sum over all atoms within a moleclue:
            layer_output = sum_and_stack(atom_outputs, array_rep['atom_list'])  #V: array_rep['atom_list'] stores the indexes of atoms in each smile size: (100,)
            all_layer_fps.append(layer_output)

        num_layers = len(num_hidden_features) #V: (num_layers = 4) , num_hidden_features = [20, 20, 20, 20]
        for layer in xrange(num_layers):
            write_to_fingerprint(atom_features, layer)
            atom_features = update_layer(weights, layer, atom_features, bond_features, array_rep,
                                         normalize=normalize)
        write_to_fingerprint(atom_features, num_layers)
        return sum(all_layer_fps), atom_activations, array_rep

    def output_layer_fun(weights, smiles):  # V: Came here from line # 80 in build_vanilla_net.py
        #import pdb; pdb.set_trace()
        output, _, _ = output_layer_fun_and_atom_activations(weights, smiles)
        return output

    def compute_atom_activations(weights, smiles):
        _, atom_activations, array_rep = output_layer_fun_and_atom_activations(weights, smiles)
        return atom_activations, array_rep

    if return_atom_activations:
        #import pdb; pdb.set_trace()
        return output_layer_fun, parser, compute_atom_activations
    else:
        #import pdb; pdb.set_trace()
        return output_layer_fun, parser

@memoize
def array_rep_from_smiles(smiles):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    molgraph = graph_from_smiles_tuple(smiles)
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
    # import pdb; pdb.set_trace()
    return arrayrep

def build_conv_deep_net(conv_params, net_params, fp_l2_penalty=0.0):
    """Returns loss_fun(all_weights, smiles, targets), pred_fun, combined_parser."""
    conv_fp_func, conv_parser = build_convnet_fingerprint_fun(**conv_params)   # V: conv_fp_func is the func that computes the fingerprints, conv_parser is the weights parser for fp net
    return build_fingerprint_deep_net(net_params, conv_fp_func, conv_parser, fp_l2_penalty) # V: creates and returns loss_fun, pred_fun, combined_parser of build_vanilla_net.py, loss_fun and pred_func is of the combined net i.e. fp+vanilla net

