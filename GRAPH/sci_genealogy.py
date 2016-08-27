from tempfile import gettempdir
from pydot import graph_from_dot_file
from networkx import drawing, nx_agraph, compose,MultiDiGraph#,nx_pydot,to_networkx_graph
from subprocess import call

TMP = gettempdir() +'/'

def generate_dot(sciid):
    print("gathering data for SciID: {}".format(sciid))
    fn = TMP+'{}.dot'.format(sciid)
    call(['ggrapher','-f',fn,'-a',sciid])  # Geneagrapher

def make_graph(sciid,outfmt='pdf'):
    gc = graph_from_dot_file(TMP+'{}.dot'.format(sciid))
    gc.set_overlap(0)

    if 'png' in outfmt:
        gc.write_png('{}.png'.format(sciid), prog='dot')

    if 'pdf' in outfmt:
        gc.write_pdf('{}.pdf'.format(sciid), prog='dot') #fdp, dot,


def combine_dots(dotlist,sciid):
    g1=drawing.nx_agraph.read_dot(TMP+dotlist[-1])
    g = MultiDiGraph()
    for d in dotlist:
        g2=drawing.nx_agraph.read_dot(TMP + d)
        g1=compose(g1,g2)
    g.add_nodes_from(g1.nodes(data=True))
    g.add_edges_from(g1.edges(data=True))
    g.to_directed()
    g = nx_agraph.to_agraph(g)
    g.write(TMP+'-'.join(sciid)+'.dot')

def graph_genealogy(sci_id,outformat='pdf'):
    try:
        make_graph(sci_id,outformat)
    except Exception: #file wasn't found or is corrupt
        generate_dot(sci_id)
        make_graph(sci_id,outformat)

def graph_combined_genealogy(sciid,outfmt='pdf'):
    for i in sciid:
        graph_genealogy(i,outfmt)
#%% combined plot if more than one sciid specified
    if len(sciid)>1:
        names = [i+".dot" for i in sciid]
        combine_dots(names,sciid)
        make_graph('-'.join(sciid),outfmt)

if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser(description='easy interface to Scientific Genealogy Project Plotter')
    p.add_argument('sciid',help='Scientific ID of person(s)',type=str,nargs='+')
    p.add_argument('-o','--output',help='output format [pdf png]',default=['pdf'],nargs='+')
    p = p.parse_args()

    graph_combined_genealogy(p.sciid,p.output)
