
"""

===============
PHageR PIPELINE
===============

The purpose of this program is to hunt for evidence of phage gene regulation by
transcription factors of the host.

Given a set of TF motif models and a set of MGE sequences (phage genomes), all
the MGE-TF pairs are searched for evidence of cross-regulation.

The program scans the sequences with the PSSMs. Than it analyses the
distribution of the obtained PSSM-scores, the positional distribution of
putative TF binding sites, and the proportion of putative sites that are
intergenic. Several statistics are computed. The obtained values are converted
into p-values by performing the analysis with background sets of
"pseudogenomes", to obtain a null distribution.

"""

import json
import os
from Bio import SeqIO
import warnings

from mge import MGE
from tf import TF
from mge_tf import MGE_TF


config_filename = 'PHageR_pipeline_config.json'


def read_config_file(filename):
    '''
    Returns the settings as a dictionary.
    '''
    with open(filename, 'r') as f:
        config = json.load(f)
    return config


def check_dir(dir_path):
    '''
    If the directory at the specified path doesn't exists, it's created.
    Any missing parent directory is also created. If the directory already
    exists, it is left un modified.
    '''
    os.makedirs(dir_path, exist_ok=True)


def check_pseudogenomes(mge_dirname, mge_config):
    '''
    Iterates over all the MGEs to be analyzed, and checks that the number of
    required pseudogenomes is already available to be read from the cache
    folder. If some required pseudogenomes are not present, they are generated
    and saved to the cache folder.
    '''
    
    mge_list = os.listdir('../datasets/' + mge_dirname)
    
    for mge_name in mge_list:
        mge_path = '../datasets/' + mge_dirname + "/" + mge_name
        my_mge = MGE(mge_path, 'gb', mge_config)
        
        for i in range(my_mge.n_pseudogenomes):
            pg_filename = my_mge.original.id + '_pseudo_' + str(i) + ".gb"
            if not os.path.exists("../cache/" + pg_filename):
                my_mge.generate_pseudogenome(i)


def save_parameters_used(config_dict, out_dirpath):
    '''
    Saves the paramters (from the config file) used to run this pipeline as
    a JSON file, for future reference.
    '''
    filepath = out_dirpath + "parameters.json"
    with open(filepath, 'w') as f:
        json.dump(config_dict, f)


def check_mge_files(config):
    '''
    This function goes over the MGE genome files and catches all the warnings
    and errors that would be raised when parsing them. It organizes them into
    a comprehensive warning message and a comprehensive error message.
    
    The comprehensive warning message is saved to a warnings.txt file into the
    output directory for the run. A warning message will inform the user about
    the warnings.txt file.
    
    The comprehensive error message will be raised, if errors were encountered.
    '''
    
    mge_dirpath = '../datasets/' + config['mge_dirname'] + "/"
    
    # Catch all warnings
    warnings_causes_dict = {}
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        for filename in os.listdir(mge_dirpath):
            filepath = mge_dirpath + filename
            try:
                SeqIO.read(filepath, "genbank")
            except Warning as w:
                if str(w) in warnings_causes_dict.keys():
                    warnings_causes_dict[str(w)].append(filename)
                else:
                    warnings_causes_dict[str(w)] = [filename]
    
    # Produce a comprehensive warning message
    full_warning_message = ""
    if len(warnings_causes_dict) > 0:
        for k, v in warnings_causes_dict.items():
            warning_message = 'The warning "{}" is raised by the following files: {}.\n'.format(k, v)
            full_warning_message += warning_message
        warnings_filepath = "../results/" + config['out_dirname'] + "/warnings.txt"
        f = open(warnings_filepath, "w")
        f.write(full_warning_message)
        f.close()
        warnings.warn("Some warnings were catched and saved to " + warnings_filepath + "\n")
    
    # Catch all errors
    errors_causes_dict = {}
    for filename in os.listdir(mge_dirpath):
        filepath = mge_dirpath + filename
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                SeqIO.read(filepath, "genbank")
            except Exception as e:
                if str(e) in errors_causes_dict.keys():
                    errors_causes_dict[str(e)].append(filename)
                else:
                    errors_causes_dict[str(e)] = [filename]
    
    # Produce a comprehensive error message
    full_error_message = ""
    if len(errors_causes_dict) > 0:
        for k, v in errors_causes_dict.items():
            error_message = 'The error "{}" is raised by the following files: {}.\n'.format(k, v)
            full_error_message += error_message
        raise ValueError(full_error_message)


def main():
    '''
    Main program.
    '''
    
    # Set up
    config = read_config_file(config_filename)
    
    mge_dirname = config['mge_dirname']
    tf_dirname = config['tf_dirname']
    out_dirname = config['out_dirname']
    mge_config = config["mge"]
    tf_config = config["tf"]
        
    # Output directory and subfolder
    out_dirpath = '../results/' + out_dirname + '/'
    mgetf_pairs_dir = out_dirpath + 'mge_tf_pairs_results/'
    check_dir(mgetf_pairs_dir)
    
    check_mge_files(config)
    
    # Cache folder for pseudogenomes
    if mge_config["save_pseudogenomes"]:
        check_dir("../cache")
    
    # Save settings used in this run
    save_parameters_used(config, out_dirpath)
    
    MGE_LIST = os.listdir(mge_dirname)
    TF_LIST = os.listdir(tf_dirname)
    check_pseudogenomes(mge_dirname, mge_config)
    
    # Run pipeline
    
    print("Generating TF objects ...\n")
    # Prepare TF objects to avoid regenerating them at every iteration of the
    # loop over MGE_LIST (the permutations would not be the same for every MGE)
    tf_obj_list = []
    for tf_name in TF_LIST:
        tf_path = '../datasets/' + tf_dirname + "/" + tf_name
        tf_obj_list.append(TF(tf_path, tf_config))
        tf_obj_list[-1].set_permuted_motifs()
    
    n_mges = len(MGE_LIST)
    # For loop over MGEs
    for i, mge_name in enumerate(MGE_LIST):
        
        print("\nMGE: {}\t({}/{})".format(mge_name, i+1, n_mges))
        
        # Generate MGE object
        mge_path = '../datasets/' + mge_dirname + "/" + mge_name
        my_mge = MGE(mge_path, 'gb', mge_config)
        my_mge.set_pseudogenomes()
        
        # For loop over TFs
        for my_tf in tf_obj_list:
            print("\tTF: " + my_tf.original.name)
            
            # Generate MGE-TF object
            my_mge_tf = MGE_TF(my_mge, my_tf)
            
            # Compute values
            my_mge_tf.compute_motif_specific_vals(outdir=mgetf_pairs_dir)
            # Analyze hits
            my_mge_tf.analyze_scores()
            my_mge_tf.analyze_positional_distribution()
            my_mge_tf.analyze_intergenicity()
            
            # Save results
            my_mge_tf.save_p_vals(mgetf_pairs_dir)
            my_mge_tf.save_motif_specific_vals(mgetf_pairs_dir)
    
    print('\nDone. Run "' + out_dirname + '" is complete.')


if __name__ == "__main__":
    
    # Execute pipeline
    main()









