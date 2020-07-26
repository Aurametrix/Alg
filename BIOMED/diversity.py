mport numpy as np
counts = [1, 0, 0, 4, 1, 2, 3, 0]
observed_otus(counts)
observed_otus((1, 0, 0, 4, 1, 2, 3, 0)) # tuple
observed_otus(np.array([1, 0, 0, 4, 1, 2, 3, 0])) # numpy array
singles(counts)
doubles(counts)

# ee also betea diversity
""
Author : Shivankur Kapoor
Contact : kapoors@usc.edu
"""
import glob
import os
import pandas as pd
import pickle
import sys
from Bio import SeqIO
from Bio.Alphabet import DNAAlphabet
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from argparse import ArgumentParser

from diversity_boxplot import *

PYTHON_ENV = '/home/leelab/anaconda2/bin/'
SCRIPT = '/home/leelab/PycharmProjects/BioInfoPipeLine/BetaDiversity/betadiversity.py'


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def combine_fasta(**kwargs):
    output = kwargs.pop('output')
    rec_list = []
    count_dict = {}
    for group in kwargs:
        for sample, file in kwargs[group].iteritems():
            records = SeqIO.parse(open(file), 'fasta')
            for record in records:
                seqid, seq = record.id, str(record.seq)
                new_seqid = group + '.' + sample + '_' + seqid
                count_dict[new_seqid] = int(seqid.split(':')[-3])
                rec_list.append(SeqRecord(Seq(seq, DNAAlphabet), id=new_seqid, description=''))
    output_file = os.path.join(output, 'combined.fasta')
    SeqIO.write(rec_list, output_file, "fasta")
    return count_dict, output_file


def rep_fasta(outs_dict, fasta):
    seq_to_otu_dict = {}
    for otu, sequences in otus_dict.iteritems():
        for sequence in sequences:
            first, second = sequence.split('_', 1)
            id = first + '_' + second.replace('_', ':')
            seq_to_otu_dict[id] = otu

    fasta_sequences = SeqIO.parse(open(fasta), 'fasta')
    rec_list = []
    inserted_otu = set()
    for rec in fasta_sequences:
        seqid, seq = rec.id, str(rec.seq)
        otu = seq_to_otu_dict[seqid]
        if otu not in inserted_otu:
            inserted_otu.add(otu)
            new_seqid = otu + ' ' + seqid
            rec_new = SeqRecord(Seq(seq, DNAAlphabet), id=new_seqid, description='')
            rec_list.append(rec_new)
    modified_fasta = fasta.rsplit('.', 1)[0] + '_modified.fasta'
    SeqIO.write(rec_list, modified_fasta, "fasta")
    return modified_fasta


def gen_mapping(file_dict, file_name):
    output = os.path.join(file_dict.pop('output'), file_name)
    recs = []
    recs.append('\t'.join(['#SampleID', 'Group']))
    for group in file_dict:
        for sample in file_dict[group]:
            recs.append(group + '.' + sample + '\t' + group)
    with open(output, 'w') as f:
        for rec in recs:
            f.write(rec + '\n')
    return output


def get_command(*args, **kwags):
    qiime = args[0]
    cmd = args[1]
    script = args[2]
    command = os.path.join(qiime, cmd)
    command += ' ' + script
    for key, value in kwags.items():
        command += ' ' + key + ' ' + str(value)
    return command


def gen_bash(*args, **kwargs):
    script_path = args[0]
    script_name = args[1]
    command = kwargs.pop('command')
    extra = kwargs.pop('extra', None)
    source = 'source  activate qiime1'
    script = '.'.join([os.path.join(script_path, script_name), 'sh'])
    try:
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        with open(script, 'w') as f:
            f.write('chmod 775 ' + script + '\n')
            f.write(source + '\n')
            if command:
                f.write(command + '\n')
            if extra:
                f.write(extra)
        return script
    except IOError as e:
        print 'Error in generating bash script', e, script_name


if __name__ == '__main__':

    file_dict = {}
    file_dict = {}
    root = '/home/leelab/PycharmProjects/BioInfoPipeLine/BetaDiversity/Root'
    for _, groups, _ in os.walk(root):
        for group in groups:
            for _, samples, _ in os.walk(os.path.join(root, group)):
                for sample in samples:
                    if group not in file_dict:
                        file_dict[group] = {}
                    fasta_file = glob.glob(os.path.join(root, group, sample) + '/*fasta')
                    file_dict[group][sample] = os.path.join(fasta_file[0])

    qiime = '~/anaconda2/envs/qiime1/bin'

    parser = ArgumentParser(description="Beta Diversity")
    '''
    Defining arguments
    '''
    parser.add_argument("--combined_file", dest="combined_file", default="")
    parser.add_argument("--count_dict", dest="count_dict", default="")
    parser.add_argument("--modified_otus", dest="modified_otus", default="")
    parser.add_argument("--aligned_file", dest="aligned_file", default="")
    parser.add_argument("--modified_fasta", dest="modified_fasta", default="")
    parser.add_argument("--biom", dest="biom", default="")
    parser.add_argument("--mapping", dest="mapping", default="")
    parser.add_argument("--output", dest="output", default="")
    parser.add_argument("--task", dest="task", default="")
    parser.add_argument("--input", dest="input", default="")
    parser.add_argument("--script_path", dest="script_path", default="")
    parser.add_argument("--phylo_tree", dest="phylo_tree", default="")
    parser.add_argument("--otus_dict", dest="otus_dict", default="")
    args = parser.parse_args()

    try:
        assert args.task != ""
    except AssertionError as e:
        print 'Missing task', e
        sys.exit(1)

    if args.task == 'combine':
        try:
            assert args.output != ""
            assert args.script_path != ""
        except AssertionError as e:
            print 'Wrong input', e
            sys.exit(1)
        file_dict.update({'output': args.output})
        count_dict, combined_file = combine_fasta(**file_dict)
        name = os.path.join(args.output, 'count_dict.pkl')
        save_obj(count_dict, name)

        '''
        Step:1 Alignment using muscle
        '''
        try:
            params = {'-i': combined_file, '-o': args.output, '-m': 'muscle'}
            command = get_command(qiime, 'python', os.path.join(qiime, 'align_seqs.py'), **params)
            script_name = 'Align'
            extra = PYTHON_ENV + 'python ' + SCRIPT + ' --task=pick_otus --combined_file=' + combined_file + ' --output=' + args.output + ' --script_path=' + args.script_path
            script = gen_bash(args.script_path, script_name, command=command, extra=extra)
        except Exception as e:
            print 'Error in QIIME align_seqs.py', e
            sys.exit(1)

    elif args.task == 'pick_otus':
        '''
        Step:2 Pick OTUs
        '''
        try:
            assert args.output != ""
            assert args.script_path != ""
            assert args.combined_file != ""
        except AssertionError as e:
            print 'Wrong input', e
            sys.exit(1)
        file_dict.update({'output': args.output})
        aligned_file = args.combined_file.rsplit('.', 1)[0] + '_aligned.fasta'
        try:
            params = {'-i': aligned_file, '-o': args.output, '-m': 'mothur'}
            command = get_command(qiime, 'python', os.path.join(qiime, 'pick_otus.py'), **params)
            script_name = 'Pick_OTUs'
            extra = PYTHON_ENV + 'python ' + SCRIPT + ' --task=modify_otus --aligned_file=' + aligned_file + ' --output=' + args.output + ' --script_path=' + args.script_path
            script = gen_bash(args.script_path, script_name, command=command, extra=extra)
        except Exception as e:
            print 'Error in QIIME pick_otus.py', e
            sys.exit(1)

    elif args.task == 'modify_otus':
        '''
        Step:3 Putting frequency info to otu mapping
        '''
        try:
            assert args.output != ""
            assert args.script_path != ""
            assert args.aligned_file != ""
        except AssertionError as e:
            print 'Wrong input', e
            sys.exit(1)
        file_dict.update({'output': args.output})
        otus = args.aligned_file.rsplit('.', 1)[0] + '_otus.txt'
        modified_otus = otus.rsplit('.', 1)[0] + '_modified.txt'
        otus_dict = {}
        count_dict = load_obj(os.path.join(args.output, 'count_dict.pkl'))
        try:
            with open(otus, 'r') as f:
                for line in f.readlines():
                    otu, sequences = line.split('\t', 1)
                    otus_dict[otu] = map(lambda x: x.strip(), sequences.split('\t'))

            mod_otus_dict = {}
            for otu, sequences in otus_dict.iteritems():
                seq_list = []
                for sequence in sequences:
                    first, second = sequence.split('_', 1)
                    id = first + '_' + second.replace('_', ':')
                    for i in range(count_dict[id]):
                        seq_list.append(sequence + '.' + str(i))
                mod_otus_dict[otu] = seq_list

            with open(modified_otus, 'w') as f:
                for otu, seq_list in mod_otus_dict.items():
                    line = '\t'.join([str(otu)] + seq_list) + '\n'
                    f.write(line)

            name = os.path.join(args.output, 'otus_dict.pkl')
            save_obj(otus_dict, name)


        except Exception as e:
            print 'Error in generating modified_otus.txt'
            sys.exit(1)

        '''
        Step:4 Creating OTU table
        '''
        biom = modified_otus.rsplit('/', 1)[0] + '/' + 'otu_table.biom'
        try:
            params = {'-i': modified_otus, '-o': biom}
            command = get_command(qiime, 'python', os.path.join(qiime, 'make_otu_table.py'), **params)
            script_name = 'Create_OTU_Table'

            '''
            Step:5 Generating representative sequences for phylogeny tree
            '''
            otus_dict = load_obj(os.path.join(args.output, 'otus_dict.pkl'))
            modified_fasta = rep_fasta(otus_dict, args.aligned_file)
            extra = PYTHON_ENV + 'python ' + SCRIPT + ' --task=make_phylo_tree --modified_fasta=' + modified_fasta + ' --output=' + args.output + ' --script_path=' + args.script_path + ' --biom=' + biom
            script = gen_bash(args.script_path, script_name, command=command, extra=extra)
        except Exception as e:
            print 'Error in QIIME make_otu_table.py', e
            sys.exit(1)

    elif args.task == 'make_phylo_tree':
        try:
            assert args.output != ""
            assert args.script_path != ""
            assert args.modified_fasta != ""
            assert args.biom != ""
        except AssertionError as e:
            print 'Wrong input', e
            sys.exit(1)
        file_dict.update({'output': args.output})
        '''
        Step:6 Generating phylogeny tree
        '''
        phylo_tree = args.modified_fasta.rsplit('/', 1)[0] + '/' + 'phylo.tre'
        try:
            params = {'-i': args.modified_fasta, '-o': phylo_tree}
            command = get_command(qiime, 'python', os.path.join(qiime, 'make_phylogeny.py'), **params)
            script_name = 'Phylogeny_Tree'
            extra = PYTHON_ENV + 'python ' + SCRIPT + ' --task=beta_diversity --phylo_tree=' + phylo_tree + ' --output=' + args.output + ' --script_path=' + args.script_path + ' --biom=' + args.biom
            script = gen_bash(args.script_path, script_name, command=command, extra=extra)
        except Exception as e:
            print 'Error in QIIME make_phylogeny.py', e
            sys.exit(1)


    elif args.task == 'beta_diversity':
        try:
            assert args.output != ""
            assert args.script_path != ""
            assert args.biom != ""

        except AssertionError as e:
            print 'Wrong input', e
            sys.exit(1)

        file_dict.update({'output': args.output})
        '''
        Step:7 Generating Mapping file
        '''
        mapping = gen_mapping(file_dict, 'mapping.txt')

        '''
        Step:8 Performing Jackknified Beta Diversity
        '''
        try:
            params = {'-i': args.biom, '-o': args.output, '-e': 80, '-m': mapping, '-t': args.phylo_tree, '-f': ''}
            command = get_command(qiime, 'python', os.path.join(qiime, 'jackknifed_beta_diversity.py'), **params)
            script_name = 'Beta_Diversity'
            extra = PYTHON_ENV + 'python ' + SCRIPT + ' --task=2d_plots --mapping=' + mapping + ' --output=' + args.output + ' --script_path=' + args.script_path + ' --biom=' + args.biom
            script = gen_bash(args.script_path, script_name, command=command, extra=extra)
        except Exception as e:
            print 'Error in QIIME jackknifed_beta_diversity.py', e

    elif args.task == '2d_plots':
        try:
            assert args.output != ""
            assert args.script_path != ""
            assert args.mapping != ""
            assert args.biom != ""

        except AssertionError as e:
            print 'Wrong input', e
            sys.exit(1)

        try:
            os.mkdir(os.path.join(args.output, '2d_plots'))
            unweighted_pcoa = os.path.join(args.output, 'unweighted_unifrac', 'pcoa')
            unweighted_output = os.path.join(args.output, '2d_plots', 'unweighted')
            params1 = {'-i': unweighted_pcoa, '-o': unweighted_output, '-m': args.mapping}
            command_unweighted = get_command(qiime, 'python', os.path.join(qiime, 'make_2d_plots.py'), **params1)

            weighted_pcoa = os.path.join(args.output, 'weighted_unifrac', 'pcoa')
            weighted_output = os.path.join(args.output, '2d_plots', 'weighted')
            params2 = {'-i': weighted_pcoa, '-o': weighted_output, '-m': args.mapping, '-b' : 'Group'}
            command_weighted = get_command(qiime, 'python', os.path.join(qiime, 'make_2d_plots.py'), **params2)
            command = '\n'.join([command_unweighted, command_weighted])
            script_name = '2d_Plots'
            extra = PYTHON_ENV + 'python ' + SCRIPT + ' --task=alpha_diversity' + ' --output=' + args.output + ' --script_path=' + args.script_path + ' --biom=' + args.biom + ' --mapping=' + args.mapping
            script = gen_bash(args.script_path, script_name, command=command, extra=extra)
            if not os.path.exists(unweighted_output):
                os.mkdir(unweighted_output)
            if not os.path.exists(weighted_output):
                os.mkdir(weighted_output)
        except Exception as e:
            print 'Error in QIIME 2d plots', e

    elif args.task == 'alpha_diversity':
        try:
            assert args.output != ""
            assert args.script_path != ""
            assert args.biom != ""
            assert args.mapping != ""
        except AssertionError as e:
            print 'Wrong input', e
            sys.exit(1)
        try:

            # # Performing multiple rarefactions
            # params_multirarefaction = {'-i': args.biom, '-o': os.path.join(args.output, 'multirarefaction'), '-m': '20',
            #                            '-x': '100', '-s': '10', '-n': '10'}
            # command_multirarefaction = get_command(qiime, 'python', os.path.join(qiime, 'multiple_rarefactions.py'),
            #                                        **params_multirarefaction)

            # Performing Alpha Diversity
            params_alpha = {'-i': args.biom, '-m': 'simpson_reciprocal,shannon,simpson',
                            '-o': os.path.join(args.output, 'result.txt')}
            command_alpha = get_command(qiime, 'python', os.path.join(qiime, 'alpha_diversity.py'),
                                        **params_alpha)

            # # Collating results
            # params_collate = {'-i': os.path.join(args.output, 'result.txt'), '-o': os.path.join(args.output, 'collate')}
            # command_collate = get_command(qiime, 'python', os.path.join(qiime, 'collate_alpha.py'),
            #                               **params_collate)

            # # Rarefaction Plots
            # params_rareplots = {'-i': os.path.join(args.output, 'collate'),
            #                     '-o': os.path.join(args.output, 'alpha_plots'), '-m': args.mapping}
            # command_rareplots = get_command(qiime, 'python', os.path.join(qiime, 'make_rarefaction_plots.py'),
            #                                 **params_rareplots)
            script_name = 'Alpha_Diversity'
            command = '\n'.join([command_alpha])
            extra = PYTHON_ENV + 'python ' + SCRIPT + ' --task=alpha_plots' + ' --output=' + args.output
            script = gen_bash(args.script_path, script_name, command=command, extra=extra)
        except Exception as e:
            print'Error in QIIME group significance', e
            sys.exit(1)


    elif args.task == 'alpha_plots':
        try:
            assert args.output != ""
        except AssertionError as e:
            print 'Wrong input', e
            sys.exit(1)

        try:
            result = os.path.join(args.output, 'result.txt')
            measures = []
            alpha_data = []
            with open(result, 'r') as f:
                for line in f.readlines():
                    tokens = line.split('\t')
                    tokens = map(lambda x: x.strip(), tokens)
                    if not measures:
                        measures = tokens[1:]
                        # print measures
                    else:
                        data = {}
                        group, sample = map(lambda x: x.strip(), tokens[0].split('.'))
                        values = tokens[1:]
                        data.update({'group': group,
                                     'sample': sample,
                                     })
                        for i, measure in enumerate(measures):
                            data.update({measure: values[i]})
                        alpha_data.append(data)
            df = pd.DataFrame(alpha_data)

            # Generating box plots
            output = os.path.join(args.output, 'alpha_plots')
            if not os.path.exists(output):
                os.makedirs(output)
            name_dict = {'simpson_reciprocal': 'Simpson Reciprocal',
                         'shannon': 'Shannon Diversity',
                         'simpson': 'Simpson Index'}
            groups = list(set(df['group']))
            for measure in measures:
                df_ = df[['group', 'sample', measure]]
                boxplot(df_, measure, output, name_dict[measure])

        except Exception as e:
            print 'Error in alpha diversity box plots', e
            sys.exit(1)
