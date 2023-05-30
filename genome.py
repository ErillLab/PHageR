
"""

Genome class.

The Genome object represents a model of a DNA sequence to be analyzed with the
MGE-TF pipeline. It stores all the relevant attributes and methods for the
MGE-TF pipeline, relevant to the PSSM scan, and the statistics dependent on the
location, positional distribution and score distribution of the hits.

It can be an "original" genome or a "pseudogenome".

"""

import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import pandas as pd
import os
import json
import copy
from Bio import motifs
import warnings


class Genome():
    
    def __init__(self, filepath, fileformat, seq_type):
        '''
        Initialize Genome object, reading from a GenBank file.
        
        filepath : path of the input GenBank file
        fileformat : format of the input GenBank file (e.g.: "gb")
        seq_type : can be "original" or "pseudogenome", depending on what type
                   of genome object is being generated.
        '''
        
        # Ignore warnings when reading the GenBank file (they have already been
        # catched at the beginning of the run by the check_mge_files function)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            SR = SeqIO.read(filepath, "genbank")
        
        # For traceability
        self.source = (filepath, fileformat)
        self.type = seq_type
        
        # SeqRecord
        self.seq = SR.seq
        self.id = SR.id
        self.name = SR.name
        self.description = SR.description
        self.dbxrefs = SR.dbxrefs
        self.features = SR.features
        self.annotations = SR.annotations
        self.letter_annotations = SR.letter_annotations
        self.format = SR.format  # Remove if never used (it's not what it should be)
        
        # Additional attributes
        self.length = len(SR.seq)
        self.genomic_units = {'bounds': None, 'coding': None}
        self.pssm_scores = None
        self.hits = {'scores': None,
                     'positions': None,
                     'threshold': None,
                     'motif_length': None,
                     'intergenic': None,
                     'min_distance': None,
                     'sequences': None,
                     'avg_hamm_dist_pval': None,
                     'IC_pval': None,
                     'closest_genes': None}
        self.n_genes = None
        self.n_sites = None
        self.site_density = None
        self.avg_score = None
        self.extremeness = None
        self.counts = None
        self.entropy = None
        self.norm_entropy = None
        self.gini = None
        self.norm_gini = None
        self.evenness = None
        self.new_evenness = None
        self.ripleyl = None
        self.intergenicity = None
        
        # Features considered as 'coding' => When a site falls within one of
        # those features it is considered 'intergenic'.
        self.coding_feat = ['CDS','rRNA','tRNA','ncRNA','preRNA','tmRNA','misc']
        
        # Set genomic units, used to generate pseudogenomes and to define
        # intergenic sites
        self.set_genomic_units()
        
        # Set the number of annotated genes (coding features) found
        self.set_n_genes()
    
    def set_genomic_units(self):
        '''
        Sets the  genomic_units  attribute.
        genomic_units  has two keys:
            "bounds": list of the bounds that can be used to split the genome
                      into units. The first and last bounds are the start and
                      the end of the genome.
            "coding": array of booleans of length n, where n is the number of
                      units. The i-th element is True when the i-th unit is
                      a coding unit, False otherwise.
        '''
        
        # Define bounds of units
        units_bounds = [0, self.length]
        coding_regions = []
        for feat in self.features:
            start, end = int(feat.location.start), int(feat.location.end)
            units_bounds.append(start)
            units_bounds.append(end)
            if feat.type in self.coding_feat:
                coding_regions.append([start, end])
        units_bounds = list(set(units_bounds))
        units_bounds.sort()
        
        # Check what units are 'coding'
        n_units = len(units_bounds) - 1
        coding_units = np.array([False] * n_units)
        for cod_reg in coding_regions:
            start, end = cod_reg
            unit_idx_start = units_bounds.index(start)
            unit_idx_end = units_bounds.index(end)
            coding_units[unit_idx_start:unit_idx_end] = True
        coding_units = list(coding_units)
        
        self.genomic_units['bounds'] = units_bounds
        self.genomic_units['coding'] = coding_units
    
    def set_n_genes(self):
        '''
        Sets the  n_genes  attribute to the number of coding features present.
        Here a feature is considered as 'coding' if its type is one of those
        defined in the  coding_feat  attribute.
        '''
        self.n_genes = 0
        for feat in self.features:
            if feat.type in self.coding_feat:
                self.n_genes += 1
    
    def choose_tfbs_orientation(self, seq, motif):
        '''
        Compares the sequence with its reverse complement, and returns the
        one that best matches the (forward) motif.
        '''
        if motif.pssm.calculate(seq) >= motif.pssm.calculate(seq.reverse_complement()):
            return seq
        else:
            return seq.reverse_complement()
    
    def get_oriented_tfbs_set(self, seq_list, motif):
        '''
        Achieves "same orientation" among the sequeces in seq_list, by
        choosing between every sequence and its reverse complement. The chosen
        orientation for each sequence is the one that maximizes the (forward)
        PSSM score for the given motif.
        '''
        oriented_set = []
        for seq in seq_list:
            oriented_set.append(self.choose_tfbs_orientation(seq, motif))
        return oriented_set
    
    def get_hamm_dist(self, seq1, seq2):
        ''' Returns the Hamming distance between the two input sequences. '''
        d = 0
        for i in range(len(seq1)):
            d += not seq1[i]==seq2[i]
        return d
    
    def get_avg_hamm_dist(self, sequences):
        ''' Computes all the pair-wise Hamming distances between the given sequences.
        Returns the average Hamming distance. '''
        distances = []
        for i in range(len(sequences)-1):
            for j in range(i+1, len(sequences)):
                distances.append(self.get_hamm_dist(sequences[i], sequences[j]))
        return sum(distances)/len(distances)
    
    def get_avg_hamm_dist_pval(self, hit_sequences, control_sets, motif):
        '''
        Returns the p-value for the Average Hamming Distance (AHD) of the aligned
        'hit_sequences'. The p-value is estimated by calculating AHD on n control
        sets of m sequences each, where m is the number of sequences in 'hit_sequences'.
        The p-value is computed by estimating P(ADH < obs_AHD), where obs_AHD is
        the AHD observed in the original set 'hit_sequences'.
        '''
        
        # Observed value of AHD
        obs_AHD = self.get_avg_hamm_dist(hit_sequences)
        
        # Control values of AHD
        control_vals = []
        for control_set in control_sets:
            control_vals.append(self.get_avg_hamm_dist(control_set))
        control_vals = np.array(control_vals)
        
        # Return p-value
        b = (control_vals <= obs_AHD).sum()
        return (b + 1) / (len(control_vals) + 2)
    
    def get_IC(self, sequences):
        '''
        Returns the Information Content (IC) for the aligned sequences.
        The maximum IC is 2 * L, where L is the legth of the motif, and it's
        achieved when the entropy (H) is 0. H is 0 when every column in the
        matrix obtained from aligning the sequences is a on-hot vector.
        '''
        seq_m = motifs.create(sequences)
        pwm_mat = np.array(list(seq_m.pwm.values()))
        H = 0
        for j in range(seq_m.length):
            for i in range(4):
                element = pwm_mat[i,j]
                if element != 0:
                    H -= element * np.log2(element)
        return 2 * seq_m.length - H
    
    def get_IC_pvalue(self, hit_sequences, control_sets, motif):
        '''
        Returns the p-value for the Information Content (IC) of the aligned
        'hit_sequences'. The p-value is estimated by calculating IC on n control
        sets of m sequences each, where m is the number of sequences in 'hit_sequences'.
        The p-value is computed by estimating P(IC > obs_IC), where obs_IC is
        the IC observed in the original set 'hit_sequences'.
        '''
        
        # Observed value of IC
        obs_IC = self.get_IC(hit_sequences)
        
        # Control values of IC
        control_vals = []
        for control_set in control_sets:
            control_vals.append(self.get_IC(control_set))
        control_vals = np.array(control_vals)
        
        # Return p-value
        b = (control_vals >= obs_IC).sum()
        return (b + 1) / (len(control_vals) + 2)
    
    def scan(self, motif, pseudocount, threshold=None, top_n=None):
        '''
        Scans the genome sequence with the PSSM generated from the input motif.
        For each position, an effective PSSM score is calculated by combining
        the scores on the two strands. The scores are stored in the pssm_scores
        attribute.
        
        If a threshold is specified, it will be used to define hits (the
        positions with a score above the threshold). The results will be stored
        in the hits attribute, and the number of hits will be stored in the
        n_sites attribute.
        '''
        
        pwm = motif.counts.normalize(pseudocounts=pseudocount)
        rpwm = pwm.reverse_complement()
        # Generate PSSM (and reverse complement)
        pssm = pwm.log_odds()
        rpssm = rpwm.log_odds()
        f_scores = pssm.calculate(self.seq)  # Scan on forward strand
        r_scores = rpssm.calculate(self.seq)  # Scan on reverse strand
        effective_scores = self.combine_f_and_r_scores(f_scores, r_scores)
        
        self.pssm_scores = {'forward': f_scores,
                            'reverse': r_scores,
                            'combined': effective_scores}
        
        if threshold:
            
            # Define and study hits
            hits_scores = effective_scores[effective_scores > threshold]
            hits_positions = np.argwhere(effective_scores > threshold).flatten()
            hits_sequences = [self.seq[start:start+pssm.length] for start in hits_positions]
            
            self.n_sites = len(hits_scores)
            
            # Compile 'hits' attribute
            self.hits['scores'] = hits_scores
            self.hits['positions'] = hits_positions
            self.hits['threshold'] = threshold
            self.hits['motif_length'] = pssm.length
            self.hits['sequences'] = [str(s) for s in hits_sequences]
            
            # Two additional elements in 'hits' dictionary: AHD and IC
            if self.type == 'original':
                if self.n_sites > 1:
                    
                    # Observed sites-set
                    obs_set = self.get_oriented_tfbs_set(hits_sequences, motif)
                    
                    # Control sites-sets
                    rnd_inst = motif.get_random_instances(self.n_sites * 100)  # !!! Hard-coded control set size
                    rnd_inst = self.get_oriented_tfbs_set(rnd_inst, motif)
                    contr_sets = [rnd_inst[i:i+self.n_sites] for i in range(0, len(rnd_inst), self.n_sites)]
                    
                    # Compute p-value for Average Hamming Distance and Information Content
                    ahd_pval = self.get_avg_hamm_dist_pval(obs_set, contr_sets, motif)
                    ic_pval = self.get_IC_pvalue(obs_set, contr_sets, motif)
                    
                else:
                    ahd_pval = 'not_enough_hits'
                    ic_pval = 'not_enough_hits'
                    
                # Compile PSFM-based p-val for Average Hamming distance and IC
                self.hits['avg_hamm_dist_pval'] = ahd_pval
                self.hits['IC_pval'] = ic_pval
    
    def combine_f_and_r_scores(self, f_scores, r_scores):
        '''
        Combines the PSSM scores on the forward and reverse strand into
        'effective scores', according to the
        method developed in:
        
        Hobbs ET, Pereira T, O'Neill PK, Erill I. A Bayesian inference method for
        the analysis of transcriptional regulatory networks in metagenomic data.
        Algorithms Mol Biol. 2016 Jul 8;11:19. doi: 10.1186/s13015-016-0082-8.
        PMID: 27398089; PMCID: PMC4938975.
        '''
        effective_scores = np.log2(2**f_scores + 2**r_scores)
        return effective_scores
    
    def get_entropy(self, counts):
        ''' Returns the Shannon entropy of the counts vector. '''
        counts_vector = np.array(counts)
        frequencies = counts_vector / counts_vector.sum()
        H = 0
        for p in frequencies:
            if p != 0:
                H -= p * np.log(p)
        return H
    
    def get_norm_entropy(self, counts):
        '''
        Entropy divided by the maximum entropy possible with that number of counts
        and that number of bins.
        
        Parameters
        ----------
        counts : array-like object
            Counts associated to each class.

        Returns
        -------
        rel_possible_ent : float
            Ranges from 0, when entropy is 0, to 1, when entropy is the maximum
            possible entropy. The maximum possible entropy depends on the number of
            counts and bins, and it's achieved when the counts are distributed as
            evenly as possible among the bins. Example: with 10 bins and 12 counts,
            maximum possible entropy is the entropy of the distribution where 2
            bins contain 2 counts, and 8 bins contain 1 count.
        '''
        
        counts_vector = np.array(counts)
        n_obs = counts_vector.sum()
        n_bins = len(counts_vector)
        if n_obs == 1:
            rel_possible_ent = 1
        else:
            # Compute max entropy possible with that number of obs and bins
            quotient = n_obs // n_bins
            remainder = n_obs % n_bins
            chunk_1 = np.repeat(quotient, n_bins - remainder)
            chunk_2 = np.repeat(quotient + 1, remainder)
            values = np.hstack((chunk_1, chunk_2))  # values distr as evenly as possible
            max_possible_entropy = self.get_entropy(values)
            # Compute relative entropy
            rel_possible_ent = self.get_entropy(counts) / max_possible_entropy
        return rel_possible_ent
    
    def get_gini_coeff(self, counts):
        '''
        Gini coefficient measures distribution inequality.
    
        Parameters
        ----------
        counts : array-like object
            Values associated to each class.
            They don't need to be already sorted and/or normalized.
    
        Returns
        -------
        gini_coeff : float
            Ranges from 0 (perfect equality) to 1 (maximal inequality).
        '''
        
        values = np.array(counts)
        norm_values = values / values.sum()  # normalize
        
        # Generate Lorenz curve
        norm_values.sort()
        cum_distr = np.cumsum(norm_values)
        cum_distr = list(cum_distr)
        cum_distr.insert(0, 0)
        
        # Get area under Lorenz curve
        n_classes = len(cum_distr)-1
        under_lorenz = np.trapz(y = cum_distr, dx = 1/n_classes)
        
        # Area under Perfect Equality curve
        # It's the area of a triangle with base = 1 and height = 1
        under_PE = 0.5
        
        # Compute Gini coefficient
        gini_coeff = (under_PE - under_lorenz) / under_PE
        
        return gini_coeff
    
    def get_norm_gini_coeff(self, counts):
        '''
        Normalized Gini coefficient.
        The minimum and maximum possible Gini coefficient with that number of
        bins and observations are computed. Then, norm_Gini_coefficient is
        defined as
        norm_Gini_coefficient := (Gini - min_Gini) / (max_Gini - min_Gini)
    
        Parameters
        ----------
        counts : array-like object
            Values associated to each class.
            They don't need to be already sorted and/or normalized.
    
        Returns
        -------
        norm_gini_coeff : float
            Ranges from 0 (minimal inequality possible) to 1 (maximal
            inequality possible).
        '''
    
        # Compute Gini coefficient
        nuber_of_bins = len(counts)
        number_of_obs = np.array(counts).sum()
        gini = self.get_gini_coeff(counts)
        
        # Compute minimum possible Gini coefficient
        quotient = number_of_obs // nuber_of_bins
        remainder = number_of_obs % nuber_of_bins
        chunk_1 = np.repeat(quotient, nuber_of_bins - remainder)
        chunk_2 = np.repeat(quotient + 1, remainder)
        vect = np.hstack((chunk_1, chunk_2))  # values distr as evenly as possible
        min_gini = self.get_gini_coeff(vect)
        
        # Compute maximum possible Gini coefficient
        chunk_1 = np.repeat(0, nuber_of_bins - 1)
        chunk_2 = np.repeat(number_of_obs, 1)
        vect = np.hstack((chunk_1, chunk_2))  # values distr as unevenly as possible
        vect = [int(v) for v in vect]
        max_gini = self.get_gini_coeff(vect)
        
        # Compute normalized Gini coefficient
        if max_gini - min_gini == 0:
            norm_gini = 0
        else:
            norm_gini = (gini - min_gini) / (max_gini - min_gini)
        
        return norm_gini
    
    def get_hits_distances(self, positions):
        '''
        Returns the distance (in bp) between consecutive hits on the genome.
        '''
        distances = []
        for i in range(len(positions)):
            if i == 0:
                distance = self.length - positions[-1] + positions[i]
            else:
                distance = positions[i] - positions[i-1]
            distances.append(distance)
        return distances

    def get_original_evenness(self, positions):
        '''
        Evenness as defined in Philip and Freeland (2011).
        It's the variance of the distances between consecutive (sorted)
        datapoints.
        '''
        
        intervals = self.get_hits_distances(positions)
        return np.var(intervals)

    def get_norm_evenness(self, positions):
        '''
        Normalized evenness.
        Norm_Evenness = Evenness / Max_Evenness
        '''
        
        intervals = self.get_hits_distances(positions)
        var = np.var(intervals)
        
        n_intervals = len(intervals)
        mean = self.length/n_intervals
        max_var = ((n_intervals - 1) * mean**2 + (self.length - mean)**2)/n_intervals
        norm_var = var / max_var
        return norm_var

    def get_new_evenness(self, positions):
        '''
        A transformation is applied so that large evenness values imply a very
        even distribution (it's the opposite in the original definition of
        evenness by Philip and Freeland).
        '''
        
        norm_var = self.get_norm_evenness(positions)
        new_evenness = 1 - norm_var
        return new_evenness
    
    def get_ecdf(self, values):
        ''' Returns the empyrical cumulative distribution function for the
        values vector. '''
        x = np.sort(values)
        y = np.arange(1, len(values)+1) / len(values)
        return x, y
    
    def get_ripleyk_function(self, positions):
        ''' Returns the Ripley's K function as a pair of vectors:
        the x values (distances) and their associated k values (cumulative
        frequencies). '''
        # Get all unique pairwise distances (self-distances are not considered)
        distances = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                distances.append(abs(positions[j] - positions[i]))
        distances.sort()
        # return estimated cumulative distribution of distances
        x, y = self.get_ecdf(distances)
        # Add a 0 at the beginning of each array to get the f:x->k function
        x = np.insert(x, 0, 0)
        k = np.insert(y, 0, 0)
        return x, k
    
    def get_expected_k(self, d):
        '''
        Returns the expected k value (from the 1D Ripley's K function) given a
        random (uniform) distribution of positions over the genome.
        '''
        d = int(d)
        # Number of distance values >= d
        n_great_distances = (self.length - d) * (self.length - d + 1)
        # If N is the number of possible positions, the total number of
        # considered distances is not N^2, but instead it's N^2 - N, because we
        # don't consider "self-distances", i.e., the 0-valued cells on the
        # diagonal of the distance matrix (so we subtract N).
        tot_n_distances = self.length ** 2 - self.length
        # Number of distance values < d
        n_small_distances = tot_n_distances - n_great_distances
        # Frequency of distance values smaller < d
        return n_small_distances / tot_n_distances
    
    def get_ripleyl(self, positions, d):
        '''
        Applies the Ripley's L function and returns the l value for a given
        distance d. The Ripley's L function is applied to the observed position
        of hits along the genome (it uses the 1D version of Ripley's function).
        The l value is the difference between the observed k value (from the
        Ripley's K function) and the expected k value.
        '''
        # Ripley's K function
        x, k = self.get_ripleyk_function(positions)
        # Observed k value for distance d
        idx = x.searchsorted(d, 'right') - 1
        obs_k = k[idx]
        # Expected k value for distance d
        exp_k = self.get_expected_k(d)
        # return the L value (difference between observed K and expected K)
        return obs_k - exp_k
    
    def analyze_scores(self):
        '''
        Computes the metrics related to the PSSM scores of the hits:
            - average PSSM score
            - extremeness
        '''
        
        if not self.hits:
            raise TypeError(
                "The 'hits' attribute is 'NoneType'. Make sure you call the\
                'scan' method specifying a threshold to get PSSM-hits before\
                calling 'analyze_positional_distribution'.")
        
        if self.n_sites == 0:
            self.avg_score = 'no_sites'
            self.extremeness = 0
        else:
            self.avg_score = self.hits['scores'].mean()
            self.extremeness = (self.hits['scores'] - self.hits['threshold']).sum()
    
    def set_counts(self, positions, n_bins, use_double_binning):
        '''
        Sets the 'counts' attribute.
        The function counts the number of hits per bin. If use_double_binning
        is True, two binning procedures are performed (the second one is
        shifted by half the bin size, compared to the first one).
        '''
        # Counts in each bin (for Entropy and Gini)
        counts, bins = np.histogram(
            positions, bins=n_bins, range=(0, self.length))
        counts_shifted = None
        
        if use_double_binning:
            # The coordinate system will be shifted by half the bin size
            half_bin_size = int((bins[1] - bins[0])/2)
            # Change coordinates (start point moved from 0 to half_bin_size)
            shifted_matches_positions = []
            for m_pos in positions:
                shifted_m_pos = m_pos - half_bin_size
                if shifted_m_pos < 0:
                    shifted_m_pos += self.length
                shifted_matches_positions.append(shifted_m_pos)
            shifted_matches_positions.sort()   
            # Counts in each shifted bin (for Entropy and Gini)
            counts_shifted, bins_shifted = np.histogram(
                shifted_matches_positions, bins=n_bins, range=(0, self.length))
        
        self.counts = {'regular_binning': counts,
                       'shifted_binning': counts_shifted}
    
    def analyze_positional_distribution(self, n_bins, ripley_d,
                                        use_double_binning=True,
                                        n_top_scores=None):
        '''
        Computes the metrics related to the positional distribution of the hits:
            - entropy (and norm_entropy)
            - gini (and norm_gini)
            - evenness (and norm_evenness)
            - Ripley's L function
        '''
        
        if not self.hits:
            raise TypeError(
                "The 'hits' attribute is 'NoneType'. Make sure you call the\
                'scan' method specifying a threshold to get PSSM-hits before\
                calling 'analyze_positional_distribution'.")
        
        # Site density (sites per thousand bp)
        self.site_density = 1000 * self.n_sites / self.length
        
        # Compute intergenicity
        
        # First define the positions of the matches to be analyzed
        
        if self.type == 'original':
            if self.n_sites >= 2:
                # On the original genome, use the hits (defined by the Patser threshold)
                n_top_scores = self.n_sites
            else:
                # But if there are < 2 hits, consider the best 2 matches
                n_top_scores = 2
        
        elif self.type == 'pseudogenome':
            # Consider the best n matches as specified by n_top_scores.
            # If not specified, use the hits of the pseudogenome (which may be
            # less or more numerous than the hits on the original genome!)
            if n_top_scores == None:
                n_top_scores = self.n_sites
            # If specified but lower than 2, consider the best 2 matches
            elif n_top_scores < 2:
                n_top_scores = 2
        
        if n_top_scores == self.n_sites:
            # the positions were already computed and stored in self.hits
            positions = self.hits['positions']
        else:
            # find the positions of the n top matches
            positions = np.argpartition(
                self.pssm_scores['combined'], -n_top_scores)[-n_top_scores:]
        
        # Now analyze positional distribution of the matches at those positions.
        
        # Set counts (regular binning and shifted binning)
        self.set_counts(positions, n_bins, use_double_binning)
        counts_regular, counts_shifted = self.counts.values()
        
        # Entropy, Normalized entropy, Gini, Normalized Gini (regular frame)
        entr = self.get_entropy(counts_regular)
        norm_entr = self.get_norm_entropy(counts_regular)
        gini = self.get_gini_coeff(counts_regular)
        norm_gini = self.get_norm_gini_coeff(counts_regular)
        
        if use_double_binning:
            # Entropy, Normalized entropy, Gini, Normalized Gini (shifted frame)
            entr_sh = self.get_entropy(counts_shifted)
            norm_entr_sh = self.get_norm_entropy(counts_shifted)
            gini_sh = self.get_gini_coeff(counts_shifted)
            norm_gini_sh = self.get_norm_gini_coeff(counts_shifted)
            
            # Chose frame that detects clusters the most
            entr = min(entr, entr_sh)
            norm_entr = min(norm_entr, norm_entr_sh)
            gini = max(gini, gini_sh)
            norm_gini = max(norm_gini, norm_gini_sh)
        
        # Set entropy, normalized entropy, Gini and normalized Gini
        self.entropy = entr
        self.norm_entropy = norm_entr
        self.gini = gini
        self.norm_gini = norm_gini
        
        # Set original evenness and new evenness
        self.evenness = self.get_original_evenness(positions)
        self.new_evenness = self.get_new_evenness(positions)
        
        # Set Ripley's l value
        self.ripleyl = self.get_ripleyl(positions, ripley_d)
    
    def overlaps_with_feature(self, site_pos, feat):
        '''
        Given a genome feature and a TF binding site position, it returns True
        if there is overlap, False otherwise.
        '''
        
        site_start = site_pos
        site_end = site_start + self.hits['motif_length']
        
        feat_start = int(feat.location.start)
        feat_end = int(feat.location.end)
        
        if site_start < feat_end and feat_start < site_end:
            return True
        else:
            return False
    
    def gene_to_site_distance(self, feat, site_pos, circular_genome=False):
        '''
        Returns : int
            The function returns a 'distance' as an integer.
            Its absolute value is the distance between the gene start position
            and the closest edge of the TFBS.
            - It's 0 if the gene start is contained into the TFBS
            - It's negative for TFBS that are to the genomic 'left' of the gene start
            - It's positive for TFBS that are to the genomic 'right' of the gene start           
            
            EXAMPLE 1:
                A 23 bp TFBS located at position 1000, will be reported to be +3 bp
                from a gene start at position 997, -5 bp from a gene start at
                position 1028, and 0 bp from a gene start at position 1016 (because
                the TFBS would contain the gene start).
            
            EXAMPLE 2:
                In a circular genome of 1,000,000 bp a TFBS located at position
                1000 would be reported to be at +1030 bp from a gene start located
                at position 999,970.
        '''
        
        # Define site center "position" (it can be non-integer)
        edge_to_center = (self.hits['motif_length'] - 1)/2
        site_center = site_pos + edge_to_center
        
        # Feature coordinates
        coord = np.array([int(feat.location.start), int(feat.location.end)])
        
        # Three pairs of coordinates for circular genomes. One pair otherwise.
        coordinates = [coord]  # Initialize list with first pair
        if circular_genome == True:
            coordinates.append(coord + self.length)  # second pair
            coordinates.append(coord - self.length)  # third pair
        
        # In this list, a single distance will be appended for non circular
        # genomes. For circular genomes three distances will be recorded (for
        # the three coordinate systems)
        tmp_distances = []
        
        for coord in coordinates:
            # Identify gene start position and compute distance from site_center
            
            # If gene is on forward strand
            if feat.location.strand in [1, '1', '+']:
                gene_start = coord[0]
                tmp_distances.append(site_center - gene_start)
            
            # If gene is on reverse strand
            elif feat.location.strand in [-1, '-1', '-']:
                gene_start = coord[1]
                tmp_distances.append(gene_start - site_center)
            
            else:
                raise ValueError("Unknown 'location.strand' value: " +
                                 str(feat.location.strand))
        
        # Choose the distance with the lowest absolute value.
        tmp_absolute_distances = [abs(x) for x in tmp_distances]
        gene_to_site_center = tmp_distances[np.argmin(tmp_absolute_distances)]
        
        # Define distance
        if abs(gene_to_site_center) < edge_to_center:
            # Overlapping
            distance = 0
        else:
            # Reduce the absoulte value of the distance by  edge_to_center
            if gene_to_site_center > 0:
                # TFBS is to the left
                distance = round(gene_to_site_center - edge_to_center)
            else:
                # TFBS is to the right
                distance = round(gene_to_site_center + edge_to_center)
        
        return distance
    
    def get_closest_genes(self, site_pos, neighbor_degree):
        '''
        Finds the genes that are closest to a given hit position.
        
        Parameters
        ----------
        site_pos : int
            Position of hit.
        neighbor_degree : int
            Defines how many genes close to site_pos are going to be considered.
            For example, if neighbor_degree = 2, we are going to consider 5 genes
            in total: the closest to the hit, as well as the 2 next genes on the
            left and the 2 next genes on the right.
            
            Visual scheme of how genes are assigned a neighbor-degree:
            
            -------gene------gene------hit--gene-------gene--gene--------------
            _______ND=2______ND=1___________ND=0_______ND=1__ND=2______________
            
            where ND stands for neighbor-degree.
            The closest gene to the hit has ND=0.

        Returns
        -------
        If genes are annotated, three lists are returned:
            - closest_genes [containing genome record feature objects]
            - closest_genes_indexes [containing their index in the
              record 'features' attribute]
            - closest_genes_tags [containing their ND (neighbor-degree) values]
        If genes are not annotated, the string "no_genes" is rerturned.
        '''
        
        genes = []
        indexes = []
        distances = []
        
        for (index, feat) in enumerate(self.features):
            
            # Ignore the feature if it's not a coding sequence.
            if feat.type.upper() != 'CDS':
                continue
            
            # Distance (from gene start to site)
            distance = self.gene_to_site_distance(feat, site_pos, circular_genome=True)
            
            # Store the feature, its index, its distance from TFBS
            genes.append(feat)
            indexes.append(index)
            distances.append(distance)
        
        if len(genes) > 0:
            # Get closest gene record
            abs_distances = [abs(x) for x in distances]
            j = np.argmin(abs_distances)  # j-th gene was the closest one
            min_distance = distances[j]  # j-th gene to site distance
            
            first_idx = j - neighbor_degree
            stop_idx  = j + neighbor_degree + 1
            # The following code assumes genome circularity
            if first_idx < 0:
                closest_genes = genes[first_idx % len(genes):] + genes[:stop_idx]
                closest_genes_indexes = indexes[first_idx % len(genes):] + indexes[:stop_idx]
            elif stop_idx > len(genes):
                closest_genes = genes[first_idx:] + genes[:stop_idx % len(genes)]
                closest_genes_indexes = indexes[first_idx:] + indexes[:stop_idx % len(genes)]
            else:
                closest_genes = genes[first_idx : stop_idx]
                closest_genes_indexes = indexes[first_idx : stop_idx]
            closest_genes_tags = list(range(-neighbor_degree, neighbor_degree+1))
            
            closest_genes = {'features': closest_genes,
                             'features_indexes': closest_genes_indexes,
                             'neighbor_tags': closest_genes_tags}
            return closest_genes, min_distance

        else:
            return 'no_genes'
    
    def set_hits_closest_genes(self, neighbor_degree):
        '''
        Sets the value of hits['closest_genes'].
        
        The hits attribute is a dictionary, and this function sets the value to
        be associated with the key 'closest_genes'.
        If there are annotated genes:
            The value will be a dictionary with the following content:
                - 'features': [list containing genome record feature objects]
                - 'features_indexes': [list containing their index in the
                                       record 'features' attribute]
                - 'neighbor_tags': [list containing their neighbor-degree values]
        If there aren't any annotated genes:
            The value will be the string "no_genes".
        '''
        
        if not self.hits:
            raise TypeError(
                "The 'hits' attribute is 'NoneType'. Make sure you call the\
                'scan' method specifying a threshold to get PSSM-hits before\
                calling 'get_intergenicity'.")
        
        if sum(self.genomic_units['coding']) == 0:
            self.hits['closest_genes'] = 'no_genes'
        
        else:
            self.hits['closest_genes'] = []
            self.hits['min_distance'] = []
            for hit_pos in self.hits['positions']:
                closest_genes, min_dist = self.get_closest_genes(hit_pos, neighbor_degree)
                self.hits['closest_genes'].append(closest_genes)
                self.hits['min_distance'].append(min_dist)
    
    def is_intergenic(self, hit_pos):
        ''' Returns True if hit_pos (the position of a given hit) lies into an
        intergenic region. Returns False otherwise. '''
        idx_right_bound = np.searchsorted(self.genomic_units['bounds'], hit_pos)
        idx_unit = idx_right_bound - 1
        return not self.genomic_units['coding'][idx_unit]
    
    def set_hits_intergenic(self):
        '''
        Sets the value of hits['intergenic'] as a list of booleans.
        If the n-th hit is intergenic, the n-th element of that list will be
        True, otherwise it will be False.
        The hits attribute is a dictionary, so the list of booleans will serve
        as a value to be associated with the key 'intergenic'.
        '''
        
        if not self.hits:
            raise TypeError(
                "The 'hits' attribute is 'NoneType'. Make sure you call the\
                'scan' method specifying a threshold to get PSSM-hits before\
                calling 'get_intergenicity'.")
        
        if sum(self.genomic_units['coding']) == 0:
            self.hits['intergenic'] = 'no_genes'
        
        else:
            self.hits['intergenic'] = []
            for hit_pos in self.hits['positions']:
                self.hits['intergenic'].append(self.is_intergenic(hit_pos))
    
    def analyze_intergenicity(self, n_top_scores=None):
        '''
        It computes the proportion of hits that are located into intergenic
        regions, and sets the 'intergenicity' attribute to that value.
        '''
        
        if not self.hits:
            raise TypeError(
                "The 'hits' attribute is 'NoneType'. Make sure you call the\
                'scan' method specifying a threshold to get PSSM-hits before\
                calling 'get_intergenicity'.")
        
        # Identify genetic context for each hit
        self.set_hits_intergenic()
        
        # Compute intergenicity
        
        # First define the positions of the matches to be analyzed
        
        if self.type == 'original':
            if self.n_sites >= 2:
                # On the original genome, use the hits (defined by the Patser threshold)
                n_top_scores = self.n_sites
            else:
                # But if there are < 2 hits, consider the best 2 matches
                n_top_scores = 2

        elif self.type == 'pseudogenome':
            # Consider the best n matches as specified by n_top_scores.
            # If not specified, use the hits of the pseudogenome (which may be
            # less or more numerous than the hits on the original genome!)
            if n_top_scores == None:
                n_top_scores = self.n_sites
            # If specified but lower than 2, consider the best 2 matches
            elif n_top_scores < 2:
                n_top_scores = 2
        
        if n_top_scores == self.n_sites:
            # the positions were already computed and stored in self.hits
            positions = self.hits['positions']
        else:
            # find the positions of the n top matches
            positions = np.argpartition(
                self.pssm_scores['combined'], -n_top_scores)[-n_top_scores:]
        
        # Now compute intergenicity of the matches at those positions.
        
        if len(positions) == 0:
            # This could happen for pseudogenomes if the n_top_scores parameter
            # is not used
            self.intergenicity = 'no_sites'
        
        elif sum(self.genomic_units['coding']) == 0:
            # This could happen in the case of non-annotated MGEs
            self.intergenicity = 'no_genes'
        
        else:
            # There are both matches and genes, so we can compute intergenicity
            intergenicity_list = []
            for pos in positions:
                intergenicity_list.append(self.is_intergenic(pos))
            
            # Count number of intergenic sites
            n_intergenic = sum(intergenicity_list)
            # Intergenic frequency
            intergenic_freq = n_intergenic / len(positions)
            # Set intergenicity attribute
            self.intergenicity = intergenic_freq
    
    def save_hits_table(self, filename_tag, outdir):
        '''
        Returns a table, saved as a CSV file.
        For each hit (putative TF binding site), all relevant info are
        stored in different columns (PSSM score, position, etc ...)
        '''
        
        hits_table = pd.DataFrame(self.hits).drop(['closest_genes'], axis=1)
        
        # If genome has annotated features, add table extension with info
        # about "closest genes"
        if not self.hits['closest_genes'] in ["no_genes", None]:
            
            # List of the rows in the table extension
            list_of_rows = []
            for closest_genes_dict in self.hits['closest_genes']:
                
                # Initialize a new row of the extension table
                row = []
                
                # Loop over the n closest genes reported
                for i in range(len(closest_genes_dict['features'])):
                    feat  = closest_genes_dict['features'][i]
                    f_idx = closest_genes_dict['features_indexes'][i]
                    
                    f_start  = int(feat.location.start)
                    f_end    = int(feat.location.end)
                    f_strand = feat.location.strand
                    
                    qualifiers = feat.qualifiers
                    
                    if 'locus_tag' in qualifiers.keys():
                        locus_tag = qualifiers['locus_tag'][0]
                    else:
                        locus_tag = None
                    
                    if 'protein_id' in qualifiers.keys():
                        protein_id = qualifiers['protein_id'][0]
                    else:
                        protein_id = None
                    
                    if 'product' in qualifiers.keys():
                        product = qualifiers['product'][0]
                    else:
                        product = None
                    
                    # Extend row (add columns for this "close gene")
                    row = row + [locus_tag, protein_id, product, f_start, f_end, f_strand, f_idx]
                
                # The row is complete and can be stored
                list_of_rows.append(row)
    
            # From a list of rows to a list of columns (transpose 2D list)
            list_of_cols = list(map(list, zip(*list_of_rows)))
    
            # 7 types of columns (less than the actual number of columns)
            column_types = ['locus_tag', 'protein_id', 'product', 'start', 'end', 'strand', 'feat_index']
    
            # Get neighbor_degree number
            if len(list_of_cols) % len(column_types) != 0:
                raise ValueError('Incorrect number of columns/column types')
            else:
                # Number of close genes reported
                n_close_genes = len(list_of_cols) // len(column_types)
                # Neighbor degree
                neighbor_degree = n_close_genes // 2
    
            # Neighbor tags: to be combined with column_types to produce unique column names
            neighbor_tags = []
            for i in range(-neighbor_degree, neighbor_degree+1):
                if i < 0:
                    neighbor_tag = 'L' + str(abs(i))
                elif i == 0:
                    neighbor_tag = '0'
                else:
                    neighbor_tag = 'R' + str(i)
                neighbor_tags.append(neighbor_tag)
    
            # Combine neighbor_tags with column_types to produce unique column names
            colnames = []
            for tag in neighbor_tags:
                for col_type in column_types:
                    colname = tag + "_" + col_type
                    colnames.append(colname)
    
            # Generate extension table
            extension_dict = dict(zip(colnames, list_of_cols))
            extension_table = pd.DataFrame(extension_dict)
            # Add extension table to the hits_table to produce an "extended table"
            hits_table = pd.concat([hits_table, extension_table], axis=1)
        
        # Add column reporting the source filename for the MGE sequence record
        mge_filepath = self.source[0]
        mge_filename = os.path.basename(mge_filepath)
        mge_filename_col = pd.DataFrame(
            {'MGE_filename': [mge_filename]*len(hits_table)})
        hits_table = pd.concat([mge_filename_col, hits_table], axis=1)
        
        # Save the 'hits_table' as a CSV file
        filename = filename_tag + '.hits_table.csv'
        if outdir != None:
            filename = outdir + filename
        hits_table.to_csv(filename)
    
    def save_stats_report(self, filename_tag, outdir):
        '''
        Returns a dictionary, saved as a JSON file.
        It stores the original values of the various metrics (the raw scores,
        not their corresponding p-values), as well as genome-specific info
        (seq length, position and identity of all "genomic units", etc...).
        '''
        stats = ['n_sites', 'site_density', 'avg_score', 'extremeness',
                 'entropy', 'norm_entropy', 'gini', 'norm_gini',
                 'evenness', 'new_evenness', 'ripleyl', 'intergenicity',
                 'length', 'genomic_units']
        stats_report = copy.deepcopy({k:vars(self)[k] for k in stats})
        for k in stats_report.keys():
            if isinstance(stats_report[k], np.float32):
                stats_report[k] = stats_report[k].item()
        str_list = [str(b) for b in stats_report['genomic_units']['coding']]
        stats_report['genomic_units']['coding'] = str_list
        stats_report['n_genes'] = self.n_genes
        stats_report['AHD_pval'] = self.hits['avg_hamm_dist_pval']
        stats_report['IC_pval'] = self.hits['IC_pval']
        filename = filename_tag + '.stats_report.json'
        if outdir != None:
            filename = outdir + filename
        with open(filename, 'w') as f:
            json.dump(stats_report, f)
    
    def save_pssm_scores(self, filename_tag, outdir):
        '''
        Returns a dictionary, saved as a JSON file.
        It stores ALL the raw PSSM scores on the '+' strand, on the '-'
        strand, as well as the combined ('+'&'-') PSSM scores, obtained
        using the method described in Hobbs et al. (2016)
        [doi: https://doi.org/10.1186/s13015-016-0082-8]
        '''
        d = {k : list(v.astype(np.float64)) for k, v in self.pssm_scores.items()}
        filename = filename_tag + '.pssm_scores.json'
        if outdir != None:
            filename = outdir + filename
        with open(filename, 'w') as f:
            json.dump(d, f)
    
    def save_report(self, filename_tag, save_all_pssm_scores, outdir=None):
        '''
        It calls the functions that save
            - the hit-specific results
            - the original values of the metrics (together with genomic info)
            - the raw PSSM scores
        '''
        
        # Save hits (CSV)
        self.save_hits_table(filename_tag, outdir)
        
        # Save stats report (JSON)
        self.save_stats_report(filename_tag, outdir)
        
        # Save PSSM scores (JSON)
        if save_all_pssm_scores:
            self.save_pssm_scores(filename_tag, outdir)  # XXX
    
    def save_as_genbank(self, out_filepath):
        '''
        Saves the Genome object as a Genbank file. The additional attributes
        present in Genome objects (and not in Genbank SeqRecords) will be lost.
        '''
        
        record = SeqRecord(self.seq, id=self.id, name=self.name,
                           description=self.description,
                           annotations={'molecule_type': 'DNA'})
        with open(out_filepath, 'w') as f:
            SeqIO.write(record, f, 'genbank')











