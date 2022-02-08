import numpy as np
import scipy.special
import scipy.stats
import code

def cleanup_w_communities(arr,n_vertices,N_COMS):
	""" Given an array on n_vertices with split into N_COMS, tidy up the labelling of nodes and communities to be sequential. """
	unq = numpy.unique(arr)
	map = {i:j for j,i in enumerate(unq)}
	vertex_labels = numpy.vectorize(map.get)(arr)
	com_labels = unq / (n_vertices / N_COMS)
	return com_labels, vertex_labels

def generate_benchmark_deg_dist(p_,DIST,n_vertices,N_COMS):
	""" Generate a benchmark bipartite network with a given degree distribution, number of vertices and number of communities. """
	
	if DIST == 'z':
		# finite m (preferential attachment)
		m = D
		F = lambda k: (m + D) * scipy.special.poch(m, m / D + 2) / (m * (1. + D) + 1.) / scipy.special.poch(m + k, m / D + 2)
	elif DIST == 'e':
		# infinite m (callaway et al.)
		F = lambda k: (D ** k) / ((D + 1.) ** (k + 1.))
	elif DIST == 'b':
		# binomial distribution
		F = lambda k: scipy.stats.binom.pmf(k, 1000, 1. * D / 1000)
	else: raise Exception('invalid distribution')
	
	## Calculate the degree probabilities.
	pvec = F(np.arange(n_vertices))
	pvec[np.isnan(pvec)] = 0.
	pvec /= pvec.sum()
	
	## Sample the left and right degree sequences.
	degree_vec_L = numpy.random.choice(numpy.arange(n_vertices), size = n_vertices, p = pvec)
	degree_vec_R = numpy.random.choice(numpy.arange(n_vertices), size = n_vertices, p = pvec)
	
	## Make the edge stubs from the degree sequence.
	edge_vec_L = numpy.array([i for i, j in enumerate(degree_vec_L) for _ in range(j)])
	edge_vec_R = numpy.array([i for i, j in enumerate(degree_vec_R) for _ in range(j)])
	
	## Split the edge stubs into equally sized communities.
	partition_edges = lambda arr: numpy.split(arr, numpy.where(numpy.diff(numpy.mod(arr, n_vertices / N_COMS))<0)[0] + 1)
	com_edge_vec_L = partition_edges(edge_vec_L)
	com_edge_vec_R = partition_edges(edge_vec_R)
	
	## We can check that the communities have roughly the same number of edges to assign.
	[len(arr) for arr in com_edge_vec_L]
	[len(arr) for arr in com_edge_vec_R]
	
	## Shuffle all these community lists.
	[numpy.random.shuffle(arr) for arr in com_edge_vec_L]
	[numpy.random.shuffle(arr) for arr in com_edge_vec_R]
	
	## Match everything up as best we can.
	## Take fraction p_ of the community edge vectors to be within community.
	edges = []
	for evL, evR in zip(com_edge_vec_L, com_edge_vec_R):
		n = min(evL.shape[0], evR.shape[0])
		m = numpy.random.binomial(n, p_)
		edges.append(numpy.vstack((evL[:m],evR[:m])).transpose())
		evL[:m] = -1 
		evR[:m] = -1
	
	## Cut down the edge vecs to remove -1s and leave only the stubs left to assign.
	edge_vec_L = edge_vec_L[numpy.where(edge_vec_L > 0)]
	edge_vec_R = edge_vec_R[numpy.where(edge_vec_R > 0)]
	
	## Shuffle the remaining stubs.
	numpy.random.shuffle(edge_vec_L)
	numpy.random.shuffle(edge_vec_R)
	
	## Zip them together and append to edge list to create the out of community edges.
	n = min(edge_vec_L.shape[0], edge_vec_R.shape[0])
	edges.append(numpy.vstack((edge_vec_L[:n], edge_vec_R[:n])).transpose())
	
	edges = numpy.vstack(edges)
	
	## Clean up the edge list to contain only sequential integers
	L_node_com_label, edges[:,0] = cleanup_w_communities(edges[:,0],n_vertices,N_COMS)
	R_node_com_label, edges[:,1] = cleanup_w_communities(edges[:,1],n_vertices,N_COMS)
	
	## Return the graph data.
	return L_node_com_label, R_node_com_label, scipy.sparse.coo_matrix((numpy.ones(edges.shape[0]), (edges[:,0], edges[:,1])), dtype = numpy.int_).astype(numpy.float_).tocsc()


if __name__ == 'main':
	l_labels, r_labels, sparse_array = generate_benchmark_deg_dist(0.4, 'b', 100, 5)
	
	code.interact(local=locals())