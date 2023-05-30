
"""

MGE_TF class.

The MGE_TF object represents an MGE-TF pair. It stores an MGE object and a TF
object.

The MGE object stores the p-values for several statistics.

"""

import numpy as np
import pandas as pd
import json


class MGE_TF():
    
    def __init__(self, mge, tf):
        
        self.mge = mge
        self.tf = tf
        self.motif_specific_vals = None
        
        # p-values
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
    
    def compute_motif_specific_vals(self, save_original_data=True, outdir=None):
        '''
        Each motif in the TF object (the original motif and the permuted motifs)
        is used to analyze the MGE object. The results for each motif are
        stored in the motif_specific_vals attribute.
        
        motif_specific_vals is a dictionary where the keys are the different
        metrics, and the values are lists with the results from all the motifs
        (where the first element of each is list is the result for the original
        motif, followed by the results for the permuted motifs).
        '''
        
        stats = ['avg_score', 'avg_score_sample_size', 'extremeness',
                 'entropy', 'norm_entropy', 'gini', 'norm_gini',
                 'evenness', 'new_evenness', 'ripleyl', 'intergenicity']
        
        self.init_motif_specific_vals(stats)
        all_motifs = [self.tf.original] + self.tf.permuted
        for m in all_motifs:
            self.mge.scan(m, self.tf.pseudocount, self.tf.patser_threshold)
            self.mge.analyze_scores()
            self.mge.analyze_positional_distribution()
            self.mge.analyze_intergenicity()
            self.compile_motif_specific_vals(stats)
            
            # Ensure that only the report about the analysis with the original
            # motif is saved
            if m.type == 'original':
                
                if save_original_data:
                    
                    # Run analysis of close genes
                    neghbor_degree = self.mge.neighbor_degree
                    self.mge.original.set_hits_closest_genes(neghbor_degree)
                    
                    # Save files
                    filename_tag = self.get_filename_tag()
                    save_all_pssm_scores = self.mge.save_all_pssm_scores_of_original_genome  # XXX
                    self.mge.original.save_report(filename_tag, save_all_pssm_scores, outdir)
    
    def init_motif_specific_vals(self, stats):
        ''' Initialize motif_specific_vals attribute as a dictionary where the
        keys are the stats and the values are empty lists. '''
        self.motif_specific_vals = dict(zip(stats, [[] for x in range(len(stats))]))
    
    def compile_motif_specific_vals(self, stats):
        ''' Compile motif_specific_vals attribute. The current statistic values
        from the MGE object are appended to the corresponding values of the
        motif_specific_vals dictionary. '''
        for stat in stats:
            val = vars(self.mge)[stat]
            self.motif_specific_vals[stat].append(val)
    
    def analyze_scores(self):
        ''' Sets the p-value for the statistics related to the PSSM-scores.
        The alternative Hp is always 'smaller' because the values used for this
        test are already p-values. '''
        self.set_pvalue('avg_score', 'smaller',
                        sample_size_report_attr=True)
        self.set_pvalue('extremeness', 'smaller')
    
    def analyze_positional_distribution(self):
        ''' Sets the p-value for the statistics related to the positional distribution.
        The alternative Hp is always 'smaller' because the values used for this
        test are already p-values. '''
        self.set_pvalue('entropy', 'smaller')
        self.set_pvalue('norm_entropy', 'smaller')
        self.set_pvalue('gini', 'smaller')
        self.set_pvalue('norm_gini', 'smaller')
        self.set_pvalue('evenness', 'smaller')
        self.set_pvalue('new_evenness', 'smaller')
        self.set_pvalue('ripleyl', 'greater')
    
    def analyze_intergenicity(self):
        ''' Sets the p-value for the statistics related to the intergenicity.
        The alternative Hp is always 'smaller' because the values used for this
        test are already p-values. '''
        self.set_pvalue('intergenicity', 'smaller')
    
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
        is based on the frequency of control values for the given metric that
        are as extreme as the one observed on the original MGE-TF pair, or more
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
        
        # Observed value on the original MGE-TF pair
        obs = self.motif_specific_vals[metric][0]
        control_values = self.motif_specific_vals[metric][1:]
        
        valid_values = [x for x in control_values if not isinstance(x, str)]

        if sample_size_report_attr:
            vars(self)[metric + "_sample_size"] = len(valid_values)
        
        # Valid values from MGEs that can be used as a control
        control = np.array(valid_values)
        
        if isinstance(obs, (str, type(None))):
            p_val = 'no_valid_obs'
        
        elif len(control) == 0:
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
    
    def save_p_vals(self, outdir=None):
        '''
        Saves the p-values (and corrected p-values) as a CSV table:
            *.pvals_and_corr_pvals.csv
        '''
        
        # Save p-values to CSV file
        stats = ['avg_score', 'avg_score_sample_size', 'extremeness',
                 'entropy', 'norm_entropy', 'gini', 'norm_gini',
                 'evenness', 'new_evenness', 'ripleyl', 'intergenicity']
        first_pval = []
        second_pval = []
        for stat in stats:
            first_pval.append(self.motif_specific_vals[stat][0])
            second_pval.append(vars(self)[stat])

        res = pd.DataFrame({'stat_name': stats,
                            'pval': first_pval,
                            'corrected_pval': second_pval})
        filename = self.get_filename_tag() + '.pvals_and_corr_pvals.csv'
        if outdir != None:
            filename = outdir + filename
        res.to_csv(filename, index=False)
    
    def save_motif_specific_vals(self, outdir=None):
        '''
        Saves the 'motif_specific_vals' attribute to JSON file.
        
        'motif_specific_vals' is a dictionary where the keys are the different
        metrics, and the values are lists with the results from all the motifs
        (where the first element of each is list is the result for the original
        motif, followed by the results for the permuted motifs).
        It is compiled by the 'compute_motif_specific_vals' function.
        '''
        
        # Save motif-specific results to JSON file
        filename = self.get_filename_tag() + '.motif_specific_values.json'
        if outdir != None:
            filename = outdir + filename
        with open(filename, 'w') as f:
            json.dump(self.motif_specific_vals, f)
    
    def get_filename_tag(self):
        '''
        Returns a "file-name tag" for output files as a string. All the output
        files for this MGE-TF pair should begin with this "file-name tag".
        '''
        return self.tf.original.name + "." + self.mge.original.name











