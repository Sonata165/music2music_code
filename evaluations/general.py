import numpy as np
from typing import List


def calculate_wer(out_seq:List[str], tgt_seq:List[str]) -> float:
    """
    Calculate word error rate

    res = (number of tokens in output that overlap with tgt) / min(|out|, |tgt|)
    """
    # Initialize a matrix with size |ref_words|+1 x |hyp_words|+1
    # The extra row and column are for the case when one of the strings is empty
    d = np.zeros((len(tgt_seq) + 1, len(out_seq) + 1))
    # The number of operations for an empty hypothesis to become the reference
    # is just the number of words in the reference (i.e., deleting all words)
    for i in range(len(tgt_seq) + 1):
        d[i, 0] = i
    # The number of operations for an empty reference to become the hypothesis
    # is just the number of words in the hypothesis (i.e., inserting all words)
    for j in range(len(out_seq) + 1):
        d[0, j] = j
    # Iterate over the words in the reference and hypothesis
    for i in range(1, len(tgt_seq) + 1):
        for j in range(1, len(out_seq) + 1):
            # If the current words are the same, no operation is needed
            # So we just take the previous minimum number of operations
            if tgt_seq[i - 1] == out_seq[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                # If the words are different, we consider three operations:
                # substitution, insertion, and deletion
                # And we take the minimum of these three possibilities
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)
    
    # The minimum number of operations to transform the hypothesis into the reference
    # is in the bottom-right cell of the matrix
    # We divide this by the number of words in the reference to get the WER
    if len(tgt_seq) == 0: # if target empty
        if len(out_seq) != 0: # out not empty
            wer = 1 # all insertion error
        else: # out also empty
            wer = 0 # no error
    else: # target not empty
        wer = d[len(tgt_seq), len(out_seq)] / len(tgt_seq)
    
    return wer