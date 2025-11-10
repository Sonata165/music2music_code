import math

def calculate_perplexity(loss):
    """
    Calculate the perplexity of a language model based on the validation loss.
    
    Parameters:
        loss (float): The validation loss (cross-entropy loss) of the model.
    
    Returns:
        float: The perplexity of the language model.
    """
    return math.exp(loss)

print(calculate_perplexity(0.6533))