 cssscl build_dbs -btax -c -blast -nt 2 PATH_TO/test_data/TRAIN.fa PATH_TO/taxon/
 
 
#  use cssscl to classify sequences in a test set  TEST.fa
cssscl classify -c -blast blastn -tax genus -nt 2 PATH_TO/test_data/test/TEST.fa PATH_TO/test_data/
