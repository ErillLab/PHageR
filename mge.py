
"""

MGE class.

The MGE object represents a model of a mobile genetic element (MGE). Together
with the original DNA sequence ("genome"), it stores all the relevant
attributes and methods for the MGE-TF pipeline, included the "pseudogenomes" as
a control. The "genome" and the "pseudogenomes" store several statistics. The
corresponding p-values are stored as attributes in the MGE object.

It is paired with an TF object to form an MGE_TF object (see mge_tf.py).

"""

import numpy as np
import random
import os
import copy
from Bio.Seq import Seq
from genome import Genome


class MGE():
    
    def __init__(self, filepath, fileformat, mge_config_dict):
        
        self.original = Genome(filepath, fileformat, "original")
        self.pseudogenomes = []
        self.source = (filepath, fileformat)
        self.pseudo_g_counter = 0
        self.n_pseudogenomes = mge_config_dict["n_pseudogenomes"]
        self.kmer_len = mge_config_dict["kmer_len"]
        self.n_bins = mge_config_dict["n_positional_bins"]
        self.ripley_d = mge_config_dict["ripley_d_value"]
        self.neighbor_degree = mge_config_dict["neighbor_degree"]
        self.save_pseudogenomes = mge_config_dict["save_pseudogenomes"]
        self.save_all_pssm_scores_of_original_genome = (
            mge_config_dict["save_all_pssm_scores_of_original_genome"])  # XXX
        
        # p-values
        self.site_density = None
        self.avg_score = None
        self.avg_score_sample_size = None
        self.extremeness = None
        self.entropy = None
        self.norm_entropy = None
        self.gini = None
        self.norm_gini = None
        self.evenness = None
        self.new_evenness = None
        self.ripleyl = None
        self.intergenicity = None
    
    def get_pseudogenome_filepath(self, id_number):
        '''
        Returns the file path associated to each pseudogenome. It is the path
        used to save pseudogenomes to genbank files into the cache folder.
        
        id_number : specifies the n-th pseudogenome (where n = id_number)
        '''
        pseudogenome_name = self.original.id + "_pseudo_" + str(id_number)
        return "../cache/" + pseudogenome_name + ".gb"
    
    def get_pseudogenome(self):
        '''
        Returns the n-th pseudogenome, where n depends on how many times this
        function has been called already. If pseudogenomes up to the n-th one
        (or more) are already present in the cache folder, the n-th pseudogenome
        will be loaded from that folder. Otherwise, the n-th pseudogenome will
        be generated through the 'generate_pseudogenome' function.
        '''
        
        # The pseudogenome is assigned a unique ID (starting from 1)
        self.increase_pseudo_g_counter()
        pseudogenome_id = self.pseudo_g_counter
        
        # Load or generate pseudogenome
        pseudogenome_filepath = self.get_pseudogenome_filepath(pseudogenome_id)
        #pseudogenome_name = pseudogenome_filepath.split("/")[-1][:-3]
        if os.path.exists(pseudogenome_filepath):
            # Load pseudogenome from cache folder
            #print("Loading {} from cache".format(pseudogenome_name))
            pseudogenome = self.load_pseudogenome(pseudogenome_filepath)
        else:
            # Generate pseudogenome (and save it into the cache folder)
            #print("Generating {} for the first time".format(pseudogenome_name))
            pseudogenome = self.generate_pseudogenome(pseudogenome_id)
        
        return pseudogenome
    
    def load_pseudogenome(self, filepath):
        '''
        This function can be used to load a pseudogenome from the cache folder.
        
        filepath : path of the genbank file for the pseudogenome to be loaded
        and returned.
        '''
        # Read GenBank file into Genome object
        pseudogenome = Genome(filepath, 'gb', "pseudogenome")
        # Restore 'genomic units' attribute from original genome
        pseudogenome.genomic_units = self.original.genomic_units
        return pseudogenome
    
    def generate_pseudogenome(self, id_number):
        '''
        It generates a 'pseudogenome'. For each genomic unit in the original
        genome sequence, a k-sampled sequence is generated. The pseudogenome is
        composed of these pseudo-units (k-sampled sequences) joined in the same
        order as their corresponding units appear on the original genome, to
        preserve genomic structure. In other words, each genomic unit is
        independently 'k-sampled' (using the 'get_k_sampled_sequence' method).
        '''
        pseudogenome = copy.deepcopy(self.original)
        self.clear_stats(pseudogenome)
        pseudogenome.type = 'pseudogenome'
        # Generate the sequence
        pseudogenome.seq = Seq("")
        units_bounds = pseudogenome.genomic_units['bounds']
        for i in range(len(units_bounds)-1):
            unit = self.original.seq[units_bounds[i]: units_bounds[i+1]]
            pseudogenome.seq += self.get_k_sampled_sequence(unit)
        
        # Make ID, Name, Description specific for this pseudogenome
        pseudogenome.id = str(pseudogenome.id) + '_' + str(id_number)
        pseudogenome.name = str(pseudogenome.name) + '_pseudo_' + str(id_number)
        pseudogenome.description = 'Pseudogenome {} as control for {}'.format(
            id_number, pseudogenome.description)
        
        # Save pseudogenome into cache folder if requested
        if self.save_pseudogenomes:
            pseudogenome_filepath = self.get_pseudogenome_filepath(id_number)
            print("Writing {} into cache".format(pseudogenome.id))
            pseudogenome.save_as_genbank(pseudogenome_filepath)
        
        return pseudogenome
    
    def set_pseudogenomes(self):
        '''
        Sets the  pseudogenomes  attribute: a list of pseudogenomes.
        It also keeps track of how many pseudogenomes were generated on the fly
        and how many were loaded from an already existing file.
        '''
        count_loaded = 0
        count_generated = 0
        for i in range(self.n_pseudogenomes):
            pseudogenome = self.get_pseudogenome()
            self.pseudogenomes.append(pseudogenome)
            # Check if it was loaded from a pseudogenome file or generated from
            # a genome file
            if 'pseudo' in pseudogenome.source[0]:
                count_loaded += 1
            else:
                count_generated += 1
        print("{} pseudogenomes:\t{} generated, {} loaded from cache.".format(
            self.original.name, count_generated, count_loaded))
    
    def increase_pseudo_g_counter(self):
        ''' Updates the counter used by the 'get_pseudogenome' function.'''
        self.pseudo_g_counter += 1
    
    def get_k_sampled_sequence(self, sequence):
        '''
        All kmers are stored. Than sampled without replacement.
        Example with k = 3:
        ATCAAAGTCCCCGTACG
        for which 3-mers are
        ATC, TCA, CAA, AAA, AAG, ...
        A new sequence is generated by sampling (without replacement) from that
        complete set of k-mers.
        The nucleotide content (1-mers content) may not be perfectly identical
        because of overlap between k-mers that are then randomly sampled.
        The length of the sequence is preserved.
        '''
        
        if self.kmer_len > 1:
            n_kmers = len(sequence) // self.kmer_len
            n_nuclotides_rem = len(sequence) % self.kmer_len
            
            all_kmers = self.get_all_kmers(sequence)
            sampled_seq_list = random.sample(all_kmers, n_kmers)
            n_nucleotides = random.sample(str(sequence), n_nuclotides_rem)
            sampled_seq_list += n_nucleotides
        
        else:
            sampled_seq_list = random.sample(str(sequence), len(sequence))
        
        sampled_seq = Seq("".join(sampled_seq_list))
        return sampled_seq
    
    def get_all_kmers(self, seq):
        '''
        Returns the list of all the k-mers of length k in sequence seq.
        '''
        return [str(seq)[i:i+self.kmer_len] for i in range(len(seq)-self.kmer_len+1)]
    
    def clear_stats(self, genome):
        ''' Ensures all the statistics in the 'stats' list are set to None. '''
        stats = ['n_sites', 'site_density', 'avg_score', 'extremeness',
                 'counts', 'entropy', 'norm_entropy', 'gini', 'norm_gini',
                 'evenness', 'new_evenness', 'ripleyl', 'intergenicity']
        for stat in stats:
            vars(genome)[stat] = None
    
    def scan(self, motif, pseudocount, threshold=None):
        '''
        Scans the original genome and all the pseudogenomes with the PSSM of a
        given motif.
        '''
        self.original.scan(motif, pseudocount, threshold=threshold)
        for pg in self.pseudogenomes:
            pg.scan(motif, pseudocount, threshold=threshold)
    
    def analyze_scores(self):
        ''' Sets the p-value for the statistics related to the PSSM-scores. '''
        genomes = [self.original] + self.pseudogenomes
        for g in genomes:
            g.analyze_scores()
        # Set p-values
        self.set_pvalue('avg_score', 'greater', sample_size_report_attr=True)
        self.set_pvalue('extremeness', 'greater')
    
    def analyze_positional_distribution(self):
        ''' Sets the p-value for the statistics related to the positional
        distribution. '''
        # Analyze original genome
        self.original.analyze_positional_distribution(self.n_bins, self.ripley_d)
        # Analyze pseudogenomes considering the same number of hits
        n_matches = self.original.n_sites
        for pg in self.pseudogenomes:
            pg.analyze_positional_distribution(self.n_bins, self.ripley_d,
                                               n_top_scores=n_matches)
        # Set p-value(s)
        self.set_pvalue('entropy', 'smaller')
        self.set_pvalue('norm_entropy', 'smaller')
        self.set_pvalue('gini', 'greater')
        self.set_pvalue('norm_gini', 'greater')
        self.set_pvalue('evenness', 'greater')
        self.set_pvalue('new_evenness', 'smaller')
        self.set_pvalue('ripleyl', 'greater')
    
    def analyze_intergenicity(self):
        ''' Sets the p-value for the statistics related to the intergenicity. '''
        # Analyze original genome
        self.original.analyze_intergenicity()
        # Analyze pseudogenomes considering the same number of hits
        n_matches = self.original.n_sites
        for g in self.pseudogenomes:
            g.analyze_intergenicity(n_matches)
        # Set p-value
        self.set_pvalue('intergenicity', 'greater')
    
    def estimate_p_value(self, b, m):
        '''
        Estimates E[p] where p is the true p-value.
        Parameters:
            
            m is the number of values composing the control set
            
            b is the number of values composing the control set that are as
            extreme as the original observation, or more extreme.
        
        b/m is the frequency, which would be a problematic estimator of the
        p-value (especially when b=0).
        
        This function returns E[p] = (b + 1) / (m + 2), which equivalent to
        Laplace's rule of succession.
        '''
        return (b + 1) / (m + 2)
    
    def set_pvalue(self, metric, alternative, sample_size_report_attr=False):
        '''
        Sets the p-value for the specified metric. The estimate of the p-value
        is based on the frequency of pseudogenomes that can reproduce a metric
        score as extreme as the one observed on the original genome, or more
        estreme.

        Parameters
        ----------
        metric : str
            Name of the metric for which the p-value has to be estimated.
        alternative : str
            The type of alternative hypothesis: either "greater" or "smaller".
        sample_size_report_attr : bool, optional
            If set to True, an additional attribute will be defined, storing
            the number of valid values used as a control (the effective size of
            the sample used to estimate the p-value). The name of the attribute
            will be the name of the metric concatenated with "_sample_size".
            The default is False.
        '''
        control_values = []
        for genome in self.pseudogenomes:
            control_values.append(vars(genome)[metric])
        
        if None in control_values:
            raise ValueError('The value of ' + str(metric) +
                             ' is not set for all pseudogenomes.')
        
        valid_values = [x for x in control_values if not isinstance(x, str)]
        
        if sample_size_report_attr:
            vars(self)[metric + "_sample_size"] = len(valid_values)
                
        # Valid values from pseudogenomes that can be used as a control
        control = np.array(valid_values)
        # Observed value on the original genome
        obs = vars(self.original)[metric]
        
        if isinstance(obs, (str, type(None))):
            p_val = 'no_obs'
        
        elif len(control)==0:
            p_val = 'no_control_vals'
        
        else:
            if alternative == 'greater':
                b = (control >= obs).sum()
            elif alternative == 'smaller':
                b = (control <= obs).sum()
            else:
                raise ValueError('alternative should be "greater" or "smaller".')
            p_val = self.estimate_p_value(b, len(control))
        
        # Set p_value
        vars(self)[metric] = p_val














