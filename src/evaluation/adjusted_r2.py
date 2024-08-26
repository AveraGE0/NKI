"""Module providing the adjusted r2 score. This might not be interpreted as direct as the
r2 score, but it is adjusted towards the number of independent variables used."""


def adjusted_r2_score(n: int, p: int, r2_score: float) -> float:
    """Function that calculates the adjusted r2 score from the number of
    samples, the independent variables and the r2 score. Source:
    https://stackoverflow.com/questions/49381661/how-do-i-calculate-the-adjusted-r-squared-score-using-scikit-learn.

    Args:
        n (int): number of samples used to train the model
        p (int): number of independent variables
        r2_score (float): the original r2 score of the model

    Returns:
        float: _description_
    """
    return 1 - (1-r2_score)*((n-1)/(n-p-1))