def BFS(f, root):
	Q = [root]
	while Q:
		n = Q.pop(0)
		f(n)
		for child in n:
			if not n.discovered:
				n.discovered = True
				Q.append(n)
        
