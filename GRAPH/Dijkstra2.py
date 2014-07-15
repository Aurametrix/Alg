def find_all_paths(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        if start not in graph:
            return []
        paths = []
        for node in graph[start]:
            if node not in path:
                newpaths = find_all_paths(graph, node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths       

def min_path(graph, start, end):
    paths=find_all_paths(graph,start,end)
    mt=10**99
    mpath=[]
    print '\tAll paths:',paths
    for path in paths:
        t=sum(graph[i][j] for i,j in zip(path,path[1::]))
        print '\t\tevaluating:',path, t
        if t<mt: 
            mt=t
            mpath=path

    e1=' '.join('{}->{}:{}'.format(i,j,graph[i][j]) for i,j in zip(mpath,mpath[1::]))
    e2=str(sum(graph[i][j] for i,j in zip(mpath,mpath[1::])))
    print 'Best path: '+e1+'   Total: '+e2+'\n'  

if __name__ == "__main__":
    graph = {'D1': {'D2':1, 'C1':1},
             'D2': {'C2':1, 'D1':1},
             'C1': {'C2':1, 'B1':1, 'D1':1},
             'C2': {'D2':1, 'C1':1, 'B2':1},
             'B1': {'C1':1, 'B2':1},
             'B2': {'B1':1, 'A2':1, 'C2':1},
             'A2': {'B2':1, 'A1':1},
             'A1': {'A2':1}}
    min_path(graph,'D1','A1')
