import numpy as np
import scipy.special
import scipy.stats
import code

def cleanup_w_communities(arr,n_vertices,N_COMS):
	""" Given an array on n_vertices with split into N_COMS, tidy up the labelling of nodes and communities to be sequential. """
	unq = np.unique(arr)
	map = {i:j for j,i in enumerate(unq)}
	vertex_labels = np.vectorize(map.get)(arr)
	com_labels = (unq / (n_vertices / N_COMS)).astype(np.int_)
	return com_labels, vertex_labels

def generate_benchmark_deg_dist(p_,DIST,exp_deg,n_vertices,N_COMS):
	""" Generate a benchmark bipartite network with a given degree distribution, number of vertices and number of communities. """
	
	if DIST == 'z':
		# finite m (preferential attachment)
		m = D
		F = lambda k: (m + exp_deg) * scipy.special.poch(m, m / exp_deg + 2) / (m * (1. + exp_deg) + 1.) / scipy.special.poch(m + k, m / exp_deg + 2)
	elif DIST == 'e':
		# infinite m (callaway et al.)
		F = lambda k: (exp_deg ** k) / ((exp_deg + 1.) ** (k + 1.))
	elif DIST == 'b':
		# binomial distribution
		F = lambda k: scipy.stats.binom.pmf(k, 1000, 1. * exp_deg / 1000)
	else: raise Exception('invalid distribution')
	
	## Calculate the degree probabilities.
	pvec = F(np.arange(n_vertices))
	pvec[np.isnan(pvec)] = 0.
	pvec /= pvec.sum()
	
	## Sample the left and right degree sequences.
	degree_vec_L = np.random.choice(np.arange(n_vertices), size = n_vertices, p = pvec)
	degree_vec_R = np.random.choice(np.arange(n_vertices), size = n_vertices, p = pvec)
	
	## Make the edge stubs from the degree sequence.
	edge_vec_L = np.array([i for i, j in enumerate(degree_vec_L) for _ in range(j)])
	edge_vec_R = np.array([i for i, j in enumerate(degree_vec_R) for _ in range(j)])
	
	## Split the edge stubs into equally sized communities.
	partition_edges = lambda arr: np.split(arr, np.where(np.diff(np.mod(arr, n_vertices / N_COMS))<0)[0] + 1)
	com_edge_vec_L = partition_edges(edge_vec_L)
	com_edge_vec_R = partition_edges(edge_vec_R)
	
	## We can check that the communities have roughly the same number of edges to assign.
	[len(arr) for arr in com_edge_vec_L]
	[len(arr) for arr in com_edge_vec_R]
	
	## Shuffle all these community lists.
	[np.random.shuffle(arr) for arr in com_edge_vec_L]
	[np.random.shuffle(arr) for arr in com_edge_vec_R]
	
	## Match everything up as best we can.
	## Take fraction p_ of the community edge vectors to be within community.
	edges = []
	for evL, evR in zip(com_edge_vec_L, com_edge_vec_R):
		n = min(evL.shape[0], evR.shape[0])
		m = np.random.binomial(n, p_)
		edges.append(np.vstack((evL[:m],evR[:m])).transpose())
		evL[:m] = -1 
		evR[:m] = -1
	
	## Cut down the edge vecs to remove -1s and leave only the stubs left to assign.
	edge_vec_L = edge_vec_L[np.where(edge_vec_L > 0)]
	edge_vec_R = edge_vec_R[np.where(edge_vec_R > 0)]
	
	## Shuffle the remaining stubs.
	np.random.shuffle(edge_vec_L)
	np.random.shuffle(edge_vec_R)
	
	## Zip them together and append to edge list to create the out of community edges.
	n = min(edge_vec_L.shape[0], edge_vec_R.shape[0])
	edges.append(np.vstack((edge_vec_L[:n], edge_vec_R[:n])).transpose())
	edges = np.vstack(edges)
	
	## Clean up the edge list to contain only sequential integers
	L_node_com_label, edges[:,0] = cleanup_w_communities(edges[:,0],n_vertices,N_COMS)
	R_node_com_label, edges[:,1] = cleanup_w_communities(edges[:,1],n_vertices,N_COMS)
	
	## Return the graph data.
	return L_node_com_label, R_node_com_label, scipy.sparse.coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), dtype = np.int_).astype(np.float_).tocsc()

def generate_benchmark_edge_no(community_sizes,community_prefs,n_edges,allow_multiple_edges=False,file_id=''):

	## Parameters:
	## community sizes: A list of two lists giving the number of nodes in each community in each mode.
	## community_prefs: A list of two lists of lists for each community giving the preference of connecting to each community in the other mode. A list of arrays will also work.
	## n_edges: Integer, the number of edges required in the final graph.
	## allow_multiple_edges: Boolean, by default we do not allow for multiple edges.
	## file_id: string, defaults to empty string. Used to label the files generate by this instance of the generator.
	
	## Check the parameters are valid.
	assert len(community_sizes) == 2, 'You need to provide exactly two community size lists'
	assert len(community_prefs) == 2, 'You need to provide exactly two community preference lists'
	if not allow_multiple_edges:
		assert sum(community_sizes[0])*sum(community_sizes[1]) >= n_edges, 'n_edges is too high to prevent multiple edges'
	
	## Turn the community preferences into a cumulative sum.
	community_prefs = [np.array([np.cumsum(c) for c in m]) for m in community_prefs]
	
	for m in community_prefs:
		for c in m:
			assert c[-1] == 1., 'One of the community preferences does not not sum to 1'
			
	## Make some useful calculations to save time.
	n_top = sum(community_sizes[0])
	n_bottom = sum(community_sizes[1])
	n_top_comms = len(community_sizes[0])
	n_bottom_comms = len(community_sizes[1])
	
	i = 0
	j = 0
	comm_labels = {}  ## This dict will give node id: community label with numbering sequentially across both modes, starting from 0.
	## That is, the first node in the bottom set has id n_top and the first bottom community has id len(community_prefs[0]).
	for m in community_sizes:
		for c in m:
			for n in range(c):
				comm_labels[i] = j
				i += 1
			j += 1
	
	## Reverse this dict to get a list of node ids within each community label.
	comm_nodes = {}
	for k in comm_labels:
		if comm_nodes.get(comm_labels[k]):
			comm_nodes[comm_labels[k]].append(k)
		else:
			comm_nodes[comm_labels[k]] = [k]
	
	## Sample the top nodes for each edge.
	source_nodes = np.random.choice(n_top,size=n_edges)
	
	## Calculate the probability to determine the community of the bottom node for each edge.
	comm_probs = np.random.rand(n_edges)
	
	## Turn the probabilities into communities to target.
	target_comms = [np.argwhere(community_prefs[0][comm_labels[source_nodes[i]]] - p > 0)[0][0] + n_top_comms for i,p in enumerate(comm_probs)]
	
	## Sample the target nodes from the communities.
	target_nodes = [np.random.choice(comm_nodes[c]) for c in target_comms]
	
	edges = list(zip(source_nodes,target_nodes))
	
	## Check for multiple edges if required.
	if not allow_multiple_edges:
		while len(edges) != len(set(edges)):
			seen_pairs = set()
			duplicates = 0
			for i,e in enumerate(edges):
				if e in seen_pairs:
					duplicates += 1
					## We have a duplicate edge, resample it.
					e = (e[0],np.random.choice(comm_nodes[target_comms[i]]))
					edges[i] = e
				
				seen_pairs.add(e)
			print('Updated %d duplicate edges' % duplicates)

	## Write the generated graph to file.
	with open('edge_list_%s.txt' % file_id,'w') as f:
		for e in edges:
			f.write(' '.join(map(str,e)) + '\n')  ## Each line in the edge file is source, target.
		
	with open('comm_labels_%s.txt' % file_id,'w') as f:
		for k in comm_labels:
			f.write(str(k) + ',' + str(comm_labels[k]) + '\n')  ## Each line in the label file is node id, community label.

## Note here for possible extension to include degree distributions in the edge number version - use this to bias the np.random.choice when selecting nodes for source or target.

if __name__ == '__main__':
	#l_labels, r_labels, sparse_array = generate_benchmark_deg_dist(0.4, 'b', 4, 100, 5)
	sizes = [[50,50],  ## Top community sizes
				[50,50]]  ## Bottom community sizes
	prefs = [[[0.9,0.1],[0.1,0.9]],  ## Preferences for top onto bottom, n_top_comms rows, n_bottom comms columns.
				[[0.9,0.1],[0.1,0.9]]]  ## Preferences for bottom onto top, shape opposite to above.
	n_edges = 200
	
	generate_benchmark_edge_no(sizes,prefs,n_edges,allow_multiple_edges = False)
	
	import networkx as nx
	import matplotlib.pyplot as plt
	
	G = nx.read_edgelist('edge_list_.txt')
	nx.draw(G)
	plt.show()
	
	#code.interact(local=locals())