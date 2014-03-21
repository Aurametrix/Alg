def reversecomplement(sequence):
    """Return the reverse complement of the dna string.""" 

    complement = {"A":"T", "T":"A", "C":"G", "G":"C", "N":"N"}
    
    reverse_complement_sequence = ""

    sequence_list = list(sequence)
    sequence_list.reverse()

    for letter in sequence_list:
        reverse_complement_sequence += complement[letter.upper()]
    
    return reverse_complement_sequence

with open ("rosalind_dna.txt", "r") as myfile:
    string=myfile.read().replace('\n', '')

print reversecomplement(string)