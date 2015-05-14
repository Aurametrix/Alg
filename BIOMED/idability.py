#! /usr/bin/env python

"""
idability.py
============
Please type "./idability.py -h" for usage help

Authors:
  Eric A. Franzosa (franzosa@hsph.harvard.edu)
  Lauren McIver
  Curtis Huttenhower

Copyright (c) 2015 Harvard T. H. Chan School of Public Health

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from __future__ import print_function # PYTHON 2.7+ REQUIRED 
import os, sys, argparse, csv

# ---------------------------------------------------------------
# description
# ---------------------------------------------------------------

description = """
DESCRIPTION:

  This is a python program for generating and evaluating 
  hitting-set based codes. The program operates on tabular data 
  organized with samples  as columns and features as rows.
  (Also known as PCL format.)

BASIC OPERATION:

  When the program is given a table, it will attempt to construct 
  a unique set of features for each sample:

  $ ./idability.py demo1.pcl

  When given a table and a set of codes, the program will report 
  which samples are hit by which codes:

  $ ./idability.py demo2.pcl --codes demo1.codes.txt

  Without setting any additional arguments, the code construction
  process will be naive: only presence/absence information is
  considered and minimal codes are prioritized.

  Setting '--meta_mode [relab/rpkm]' will configure all settings
  to behave optimally for metagenomics features
  (measured in relative abundance [relab] or RPKM units)

  $ ./idability.py stool-markers-visit1.pcl --meta_mode rpkm

  is equivalent to:

  $ ./idability.py stool-markers-visit1.pcl -s 0.8 -m 7 -d 5 -n 0.05 -r abundance_gap

  Parameters can be fine-tuned for user-specific applications.

ARGUMENTS:
"""

# ---------------------------------------------------------------
# constants
# ---------------------------------------------------------------

c_na = "#N/A"
c_epsilon = 1e-20
c_codes_extension = "codes.txt"
c_hits_extension = "hits.txt"

# ---------------------------------------------------------------
# arguments
# ---------------------------------------------------------------

def get_args ():
    """ master argument parser """
    parser = argparse.ArgumentParser( 
        description=description, 
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument( 
        'table',
        type=str,
        help="""
        Tab-delimited table file to encode/decode.
        PCL format: rows are features, cols are samples (or subjects), both have headers
        """,
    )
    parser.add_argument( 
        "-c", '--codes',
        type=str,
        help="""
        Codes file produced in an earlier run.
        Specifying this option will compare codes to input table.
        """,
    )
    parser.add_argument( 
        "-j", '--jaccard_similarity_cutoff',  
        type=float,
        help="""
        If set, an encoded feature will 'knock out' similar features
        at the specified threshold (Jaccard score in [0-1]).
        """,
    )
    parser.add_argument( 
        "-m", '--min_code_size',
        type=int,
        default=1,
        help="""
        If set, codes will continue to be lengthened beyond
        the point of uniqueness. Limits spurious hits in time-varying data.
        """,
    )
    parser.add_argument( 
        "-d", '--abund_detect',
        type=float,
        default=c_epsilon,
        help="""
        Features with values above this are scored as confidently present.
        When running in encode mode, this restricts the features that can be added to a code.
        When running in decode mode, this restricts the features that can be hit by a code.
        """,
    )
    parser.add_argument( 
        "-n", '--abund_nondetect',        
        type=float,
        default=c_epsilon,
        help="""
        Features with values below this are scored as confidently absent.
        Only applied to encode mode. A sample with a feature below this threshold
        is considered to be 'missing' the feature for hitting set purposes.
        """,
    )
    parser.add_argument( 
        "-r", "--ranking",
        type=str,
        default="rarity",
        choices=["rarity", "abundance_gap"],
        help="""
        The method by which an individual's features should be prioritized when
        building codes. The default, rarity, prioritizess-less prevalent features.
        The alternative method, abundance_gap, prioritizes features with a large 
        abundance gap between the individual's value and the next highest value.
        """,
    )
    parser.add_argument( 
        "-o", "--output",
        type=str,
        help="""
        Name for the output file (codes or confusion matrix, depending on mode).
        If not supplied, a default will be constructed from the input file names.
        """,
    )
    parser.add_argument( 
        "-e", "--meta_mode",
        type=str,
        choices=["off", "relab", "rpkm"],
        default="off",
        help="""
        Automatically optimize all variables for working with metagenomic codes.
        If working with relative abundance data, select the "relab" mode.
        If working with reads per kilobase per million reads (RPKM) data, select the "rpkm" mode.
        """,
    )

    args = parser.parse_args()
    return args

# ---------------------------------------------------------------
# utilities and file i/o
# ---------------------------------------------------------------

def try_open( path, *args ):
    """ open and fail gracefully """
    fh = None
    try:
        fh = open( path, *args )
    except:
        sys.exit( "unable to open: %s, please check the path" % ( path ) )
    return fh

def map_path_name ( path ):
    """ extract file name from path name """
    return os.path.split( path )[1].split( "." )[0]

def load_sfv ( path, cutoff ):
    """ 
    loads a table file to a nested dict (sfv: sample->feature->value)
    values below cutoff are ignored to save space 
    """
    sfv = {}
    with try_open( path ) as fh:
        headers = None
        for row in csv.reader( fh, dialect="excel-tab" ):
            if headers is None:
                headers = row[1:]
                sfv = {header:{} for header in headers}
            else:
                feature, values = row[0], [float( k ) for k in row[1:]]
                assert len( values ) == len( headers ), \
                    "row length mismatch"
                for header, value in zip( headers, values ):
                    if value >= cutoff:
                        sfv[header][feature] = value
    return sfv

def reduce_sfv ( sfv, cutoff, greater=True ):
    """
    rebuild sfv with only entries > cutoff
    maintain all samples(s), even if no features in sample meet cutoff
    """
    temp = {sample:{} for sample in sfv}
    for sample, fdict in sfv.items():
        for feature, value in fdict.items():
            if ( greater and value >= cutoff ) or ( not greater and value < cutoff ):
                temp[sample][feature] = value
    return temp

def flip_sfv ( sfv ):
    """ make a fsv object, i.e. feature->sample->value map """
    fsv = {}
    for sample, fdict in sfv.items():
        for feature, value in fdict.items():
            fsv.setdefault( feature, {} )[sample] = value
    return fsv

def coerce_to_sets ( nested_dict ):
    """ reduces inner dict to key set when we're done with values """
    return {outer_key:set( inner_dict ) \
            for outer_key, inner_dict in nested_dict.items()}

def check_hits ( sample_hits ):
    """ produces confusion results by comparing keys to lists of hit samples """
    counts = {k:0 for k in "1|TP 3|FN+FP 2|TP+FP 4|FN 5|NA".split()}
    for sample, hits in sample_hits.items():
        if hits is None:
            counts["5|NA"] += 1
        else:
            tp_hit = True if sample in hits else False
            fp_hit = True if len([sample2 for sample2 in hits if sample2 != sample]) > 0 else False
            if tp_hit:
                key = "2|TP+FP" if fp_hit else "1|TP"
            else:
                key = "3|FN+FP" if fp_hit else "4|FN"
            counts[key] += 1
    return counts

def write_codes ( sample_codes, path ):
    """ write code sets to a text file """
    with try_open( path, "w" ) as fh:
        print( "#SAMPLE\tCODE", file=fh )
        for sample in sorted( sample_codes ):
            code = sample_codes[sample]
            items = [sample] 
            items += [c_na] if code is None else code
            print( "\t".join( items ), file=fh )
    print( "wrote codes to:", path, file=sys.stderr )

def read_codes ( path ):
    """ read back in the codes written by write_codes """
    sample_codes = {}
    with try_open( path ) as fh:
        fh.readline() # headers
        for line in fh:
            items = line.strip().split( "\t" )
            sample, code = items[0], items[1:]
            sample_codes[sample] = code if c_na not in code else None
    return sample_codes

def write_hits ( sample_hits, path ):
    """ write hit results and summary to a text file """
    # compute confusion line
    confusion = check_hits( sample_hits )
    with try_open( path, "w" ) as fh:
        for confusion_class in sorted( confusion ):
            count = confusion[confusion_class]
            print( "# %s: %d" % ( confusion_class, count ), file=fh )
        for sample in sorted( sample_hits ):
            hits = sample_hits[sample]
            items = [sample]
            if hits is None:
                items += ["no_code"]
            else:
                items += ["matches" if len( hits ) > 0 else "no_matches"]
                items += hits
            print( "\t".join( items ), file=fh )
    print( "wrote hits to:", path, file=sys.stderr )

# ---------------------------------------------------------------------------
# encode part
# ---------------------------------------------------------------------------

def jaccard ( set1, set2 ):
    """ jaccard similarity for two sets """
    count_union = len( set1.__or__( set2 ) )
    count_intersection = len( set1.__and__( set2 ) )
    return count_intersection / float( count_union )

def rank_by_abundgap( sfv, fsv, abund_nondetect ):
    """ abundance gap sorting sfv features """
    sorted_features = {}
    for sample, fdict in sfv.items():
        gaps = {}
        for feature, focal_value in fdict.items():
            lesser_values = [abund_nondetect]
            lesser_values += [value for sample2, value in fsv[feature].items() \
                             if value <= focal_value and sample2 != sample]
            gaps[feature] = focal_value - max( lesser_values )
        sorted_features[sample] = sorted( gaps, key=lambda feature: gaps[feature] )
    return sorted_features

def rank_by_rarity( sfv, fsv, abund_nondetect ):
    """ rarity sorting of sfv features """
    sorted_features = {}
    for sample, fdict in sfv.items():
        sorted_features[sample] = sorted( 
            fdict, 
            key=lambda feature: len( fsv[feature] ), 
            reverse=True, 
        )
    return sorted_features

def make_one_code ( sample, ranked_features, sfv_sets, fsv_sets, \
                      similarity_cutoff, min_code_size ):
    """ execute the idabilty algorithm for one sample """
    features = ranked_features[:]
    other_samples = {sample2 for sample2 in sfv_sets if sample2 != sample}
    code = []
    while len( features ) > 0 and \
          ( len( other_samples ) > 0 or len( code ) < min_code_size ):
        feature = features.pop()
        code.append( feature )
        # restrict other samples
        old_count = len( other_samples )
        other_samples = other_samples.__and__( fsv_sets[feature] )
        new_count = len( other_samples )
        # forget current feature if it doesn't knock out 1+ additions samples
        # *** unless we've already knocked everyone out and are just lengthening code ***
        if old_count == new_count and old_count != 0:
            code.pop()
        # restrict remaining features to avoid similarity to best feature
        if similarity_cutoff is not None:
            features = [feature2 for feature2 in features \
                        if jaccard( fsv_sets[feature], fsv_sets[feature2] ) \
                        < similarity_cutoff ]
    return code if len( other_samples ) == 0 else None

def encode_all ( sfv, abund_detect, abund_nondetect, similarity_cutoff, min_code_size, ranking="rarity" ):
    """ run idability algorithm on all samples """
    # flip sfv to fsv
    fsv = flip_sfv( sfv )
    # rebuild sfv with only features above abund threshold
    sfv = reduce_sfv( sfv, cutoff=abund_detect )
    # prioritize features
    print( "performing requested feature ranking:", ranking, file=sys.stderr )
    rank_function = {"rarity":rank_by_rarity, "abundance_gap":rank_by_abundgap}[ranking]
    sorted_features = rank_function( sfv, fsv, abund_nondetect )
    # simplify sfv and fsv to sets
    sfv_sets = coerce_to_sets( sfv )
    fsv_sets = coerce_to_sets( fsv )
    # make codes for each sample
    sample_codes = {}
    for i, sample in enumerate( sfv_sets ):
        sample_codes[sample] = make_one_code( 
            sample, 
            sorted_features[sample],
            sfv_sets,
            fsv_sets,
            similarity_cutoff, 
            min_code_size,
        )
    return sample_codes

# ---------------------------------------------------------------------------
# decode part
# ---------------------------------------------------------------------------

def check_one_code ( code, sfv_sets ):
    """ compare a single code to a population """
    code_set = set( code )
    hits = []
    for sample, features_set in sfv_sets.items():
        if code_set.issubset( features_set ):
            hits.append( sample )
    return hits

def decode_all ( sfv, sample_codes, abund_detect ):
    """ compare all codes to a population """
    sfv_sets = coerce_to_sets( reduce_sfv( sfv, abund_detect ) )
    sample_hits = {}
    for sample, code in sample_codes.items():
        sample_hits[sample] = None if code is None else check_one_code( code, sfv_sets )
    return sample_hits

# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------

def main ( ):
    
    """ main """

    # process arguments
    args = get_args()
    table_path = args.table
    codes_path = args.codes
    abund_detect = args.abund_detect
    abund_nondetect = args.abund_nondetect
    similarity_cutoff = args.jaccard_similarity_cutoff
    min_code_size = args.min_code_size
    ranking = args.ranking
    output_path = args.output

    # overrides
    if args.meta_mode != "off":
        choice = args.meta_mode
        abund_detect = 5.0 if choice == "rpkm" else 0.001
        abund_nondetect = abund_detect / 100.0
        # relax detection parameter in decoding step
        abund_detect = abund_detect / 10.0 if args.codes is not None else abund_detect
        similarity_cutoff = 0.8
        min_code_size = 7
        ranking = "abundance_gap"

    # determine output file name
    if output_path is None:
        items = [map_path_name( table_path )]
        if codes_path is None:
            items.append( c_codes_extension )
        else:
            items.append( map_path_name( codes_path ) )
            items.append( c_hits_extension )
        output_path = ".".join( items )

    # do this for either encoding/decoding
    print( "loading table file:", table_path, file=sys.stderr )
    sfv = load_sfv( table_path, abund_nondetect )

    # make codes mode
    if codes_path is None:
        print( "encoding the table", file=sys.stderr )
        sample_codes = encode_all( 
            sfv, 
            abund_detect=abund_detect, 
            abund_nondetect=abund_nondetect, 
            similarity_cutoff=similarity_cutoff,
            min_code_size=min_code_size,
            ranking=ranking,
        )
        write_codes( sample_codes, output_path )

    # compare codes to table mode
    else:
        print( "decoding the table", file=sys.stderr )
        sample_codes = read_codes( codes_path )
        sample_hits = decode_all( 
            sfv, 
            sample_codes,
            abund_detect=abund_detect,
        )
        write_hits( sample_hits, output_path )

if __name__ == "__main__":
    main()

