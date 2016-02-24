from pyGeno.Genome import *

g = Genome(name = "GRCh37.75")
prot = g.get(Protein, id = 'ENSP00000438917')[0]
#print the protein sequence
print prot.sequence
#print the protein's gene biotype
print prot.gene.biotype
#print protein's transcript sequence
print prot.transcript.sequence

#fancy queries
for exon in g.get(Exon, {"CDS_start >": x1, "CDS_end <=" : x2, "chromosome.number" : "22"}) :
        #print the exon's coding sequence
        print exon.CDS
        #print the exon's transcript sequence
        print exon.transcript.sequence

#You can do the same for your subject specific genomes
#by combining a reference genome with polymorphisms
g = Genome(name = "GRCh37.75", SNPs = ["STY21_RNA"], SNPFilter = MyFilter())
