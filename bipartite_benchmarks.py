import numpy as np

def generate_badj(p_):
	
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
	
	## Set the seed for numpy random.
	pvec = F(numpy.arange(N_STEPS))
	pvec[numpy.isnan(pvec)] = 0.
	pvec /= pvec.sum()
	degree_vec_L = numpy.random.choice(numpy.arange(N_STEPS), size = N_STEPS, p = pvec)
	degree_vec_R = numpy.random.choice(numpy.arange(N_STEPS), size = N_STEPS, p = pvec)
	
	edge_vec_L = numpy.array([i for i, j in enumerate(degree_vec_L) for _ in range(j)])
	edge_vec_R = numpy.array([i for i, j in enumerate(degree_vec_R) for _ in range(j)])
	
	partition_edges = lambda arr: numpy.split(arr, numpy.where(numpy.diff(numpy.mod(arr, N_STEPS / N_COMS))<0)[0] + 1)  ## Change to N_STEPS here fixes community bug.
	com_edge_vec_L = partition_edges(edge_vec_L)
	com_edge_vec_R = partition_edges(edge_vec_R)
	
	# we can check that the communities have roughly the same number of edges to assign
	[len(arr) for arr in com_edge_vec_L] # [203225, 200546, 201171, 198982, 197272]
	[len(arr) for arr in com_edge_vec_R] # [207152, 197663, 197822, 196501, 196249]
	
	# shuffle all these sublists
	[numpy.random.shuffle(arr) for arr in com_edge_vec_L]
	[numpy.random.shuffle(arr) for arr in com_edge_vec_R]
	
	# now all that's left to do is match everything up as best we can
	# take fraction P of the community edge vectors
	edges = []
	for evL, evR in zip(com_edge_vec_L, com_edge_vec_R):
		n = min(evL.shape[0], evR.shape[0])
		m = numpy.random.binomial(n, p_)
		edges.append(numpy.vstack((evL[:m],evR[:m])).transpose())
		evL[:m] = -1 # since we never made a deep copy of edge_vec_L, this makes replacements in the original vec
		evR[:m] = -1
	
	# cut down the edge vecs to remove -1s
	edge_vec_L = edge_vec_L[numpy.where(edge_vec_L > 0)]
	edge_vec_R = edge_vec_R[numpy.where(edge_vec_R > 0)]
	
	# shuffle
	numpy.random.shuffle(edge_vec_L)
	numpy.random.shuffle(edge_vec_R)
	
	# zip them together and append to edge list
	n = min(edge_vec_L.shape[0], edge_vec_R.shape[0])
	edges.append(numpy.vstack((edge_vec_L[:n], edge_vec_R[:n])).transpose())
	
	edges = numpy.vstack(edges)
	
	# clean up the edge list to contain only sequential integers
	L_node_com_label, edges[:,0] = cleanup_w_communities(edges[:,0])
	R_node_com_label, edges[:,1] = cleanup_w_communities(edges[:,1])
	
	# load into a sparse matrix
	return L_node_com_label, R_node_com_label, scipy.sparse.coo_matrix((numpy.ones(edges.shape[0]), (edges[:,0], edges[:,1])), dtype = numpy.int_).astype(numpy.float_).tocsc()
