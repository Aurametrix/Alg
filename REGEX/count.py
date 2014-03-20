import sys

def count_letters(word, char):
  return sum(char == a for a in word)

#print 'You entered', len(sys.argv), 'inputs.'
#print 'Here they are:', str(sys.argv)

if len(sys.argv) >= 2:
    print 'Let me count how many times the symbols \'a\', \'c\', \'g\', and \'t\' occur in', sys.argv[1]

    print count_letters(sys.argv[1] ,'a')
    print count_letters(sys.argv[1] ,'c')
    print count_letters(sys.argv[1] ,'g')
    print count_letters(sys.argv[1] ,'t')


#in a given string
string = 'ATCTAGCTACTCATCCTGCCCCATGTATATGATTTAGCGACGACCAGGACAAGTCAGCTTCTGAATGCTATCAACTCAGACGTCCAGGGTAGTGTACCACCGTGGAGGAAGAAGCGGACGCCGAAGTAGGTATAAGCCGACACACGTCCCCAACAGGTCCCGCACAGATGAGTGTCAACTGCATCTATCAAGGCCCGTCTGCTAAATCGGTGGGGGTGTACCTCTTGGCCTCTTGTTCCATAATTGCCGCATTATGCGGACTCTGCCACCTTGAAACGCCTTACGTGGTTATGGGACCAATCCCTTCAGGCCTCCGTCCCGCTCCTCTACAGAACAACAGACTTGATGGGGGAATTAATCCGTCAACGAATAAGAAGACACGCGCACTAGTACAAGATCATAGGAAACCCGTCATCAAGGACTACCTGGTGTCATATATCCGAAATATGGAGACGTTCTCTACATCAAGCAATTGTGGAACGGTAAGAGTATTATGTTTTCATAATCGATCAGTCTTGGTCACAGACTGTGTGTGATCAATTGCTCTGACGTGTTTACAGAGATTGGGGTAAGGCATGGTTTCACTCTCCGCTCCCACGCTCGATCCCGGACATGTTCTAAGGGATGATTGGCACTAATCACGCAGTATATACAAGCGCTTGCACTTATGACGCCGCGACAATGTTGGCAGCTTTCGCATCCCGACAGCCTGAGAGTCGACGAATACAAGTCGCAAGGCCTCTATCCAAGGAGAATAGGCCAACTGCAGGACCACGGAAGTAGGAACCAGAAATCGAAGTGAAGCACCAAGTCTGTGGATAAGACCGAGGTCGCCCTTATATACTATGTGGAACCGCACGACGAAGGCGGTGTACGGATCGTCC'

print 'Counting how many times the symbols \'A\', \'C\', \'G\', and \'T\' occur in', string

print count_letters(string ,'A')
print count_letters(string ,'C')
print count_letters(string ,'G')
print count_letters(string ,'T')

#read from a file

with open ("rosalind_dna.txt", "r") as myfile:
    string=myfile.read().replace('\n', '')

print 'Counting how many times the symbols \'A\', \'C\', \'G\', and \'T\' occur in rosalind_dna.txt'

print count_letters(string ,'A')
print count_letters(string ,'C')
print count_letters(string ,'G')
print count_letters(string ,'T')

print 'RNA will look like this: \n', string.replace('T', 'U')
