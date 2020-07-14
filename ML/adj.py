def fdr(p_vals):

    from scipy.stats import rankdata
    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr

# Given a set of p-values, returns p-values adjusted using one of several methods

def pval_corrected(self, method=None):
    '''p-values corrected for multiple testing problem
 
    This uses the default p-value correction of the instance stored in
    ``self.multitest_method`` if method is None.
 
    '''
    import statsmodels.stats.multitest as smt
    if method is None:
        method = self.multitest_method
    #TODO: breaks with method=None
    return smt.multipletests(self.pvals_raw, method=method)


def correct_pvalues_for_multiple_testing(pvalues, correction_type = "Benjamini-Hochberg"):                
    """                                                                                                   
    consistent with R - print correct_pvalues_for_multiple_testing([0.0, 0.01, 0.029, 0.03, 0.031, 0.05, 0.069, 0.07, 0.071, 0.09, 0.1]) 
    """
    from numpy import array, empty                                                                        
    pvalues = array(pvalues) 
    n = float(pvalues.shape[0])                                                                           
    new_pvalues = empty(n)
    if correction_type == "Bonferroni":                                                                   
        new_pvalues = n * pvalues
    elif correction_type == "Bonferroni-Holm":                                                            
        values = [ (pvalue, i) for i, pvalue in enumerate(pvalues) ]                                      
        values.sort()
        for rank, vals in enumerate(values):                                                              
            pvalue, i = vals
            new_pvalues[i] = (n-rank) * pvalue                                                            
    elif correction_type == "Benjamini-Hochberg":                                                         
        values = [ (pvalue, i) for i, pvalue in enumerate(pvalues) ]                                      
        values.sort()
        values.reverse()                                                                                  
        new_values = []
        for i, vals in enumerate(values):                                                                 
            rank = n - i
            pvalue, index = vals                                                                          
            new_values.append((n/rank) * pvalue)                                                          
        for i in xrange(0, int(n)-1):  
            if new_values[i] < new_values[i+1]:                                                           
                new_values[i+1] = new_values[i]                                                           
        for i, vals in enumerate(values):
            pvalue, index = vals
            new_pvalues[index] = new_values[i]                                                                                                                  
    return new_pvalues



# Benjamini and Hochberg FDR at alpha=0.01
from statsmodels.stats.multitest import multipletests

AdP = multipletests(pvals=Pval, alpha=0.01, method="fdr_bh", is_sorted=False, returnsorted=False)

#print(AdP[1])
#print(len(AdP[1][np.where(AdP[1]<0.01)]))  # AdP[1] returns corrected P-vals (array)
