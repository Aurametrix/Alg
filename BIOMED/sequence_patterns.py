#returns the position of the first occurence
transcript.find("AT/GCCG")
#returns the positions of all occurences
transcript.findAll("AT/GCCG")

#similarly, you can also do
transcript.findIncDNA("AT/GCCG")
transcript.findAllIncDNA("AT/GCCG")
transcript.findInUTR3("AT/GCCG")
transcript.findAllInUTR3("AT/GCCG")
transcript.findInUTR5("AT/GCCG")
transcript.findAllInUTR5("AT/GCCG")

#same for proteins
protein.find("DEV/RDEM")
protein.findAll("DEV/RDEM")

#and for exons
exon.find("AT/GCCG")
exon.findAll("AT/GCCG")
exon.findInCDS("AT/GCCG")
exon.findAllInCDS("AT/GCCG")
#...
