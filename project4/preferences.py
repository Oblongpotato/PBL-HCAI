"""Task 2: extending Bradley-Terry from a pairwise choice to a full ranking.

Bradley-Terry says one item beats another with probability proportional to its utility on the
exponential scale. It only speaks about pairs, and one of the two interfaces in this study asks
for a ranking of ten films, so the model has to be extended.

The extension is Plackett-Luce, and it follows from reading a ranking as a sequence of choices.
Given a slate, the participant picks their favourite from the whole slate by the Luce rule; then
picks their favourite from what remains; and so on. Multiplying those choice probabilities:

    P(i1 > i2 > ... > in) = prod_t  exp(w.x_it) / sum_{s>=t} exp(w.x_is)

Setting n = 2 leaves a single factor, exp(w.x_i1) / (exp(w.x_i1) + exp(w.x_i2)), which is exactly
Bradley-Terry. That identity is what makes the whole user study coherent: a pairwise choice and a
ten-item ranking are the same likelihood evaluated at different n, so both interfaces produce
evidence about the same w and the two conditions can be compared directly rather than through
some conversion factor.

Estimation is maximum a posteriori with a Gaussian prior. A participant supplies a few dozen
comparisons for a vector of ~45 weights, so the unpenalised likelihood is under-determined and
would happily send weights to infinity on any feature that was never contradicted; the prior is
what keeps the problem well posed.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

PRIOR_VARIANCE = 1.0


def log_likelihood(w, rankings, X):
    """Log P(rankings | w) under Plackett-Luce.

    `rankings` is a sequence of index lists, each ordered best first.
    """
    total = 0.0
    for ranking in rankings:
        utilities = X[list(ranking)] @ w
        # The last item has no one left to be preferred over, so it contributes nothing.
        for position in range(len(utilities) - 1):
            total += utilities[position] - logsumexp(utilities[position:])
    return total


def gradient(w, rankings, X):
    """d/dw of log_likelihood: the chosen item minus what the model expected to be chosen."""
    total = np.zeros_like(w)
    for ranking in rankings:
        rows = X[list(ranking)]
        utilities = rows @ w
        for position in range(len(utilities) - 1):
            remaining = utilities[position:]
            weights = np.exp(remaining - logsumexp(remaining))
            total += rows[position] - weights @ rows[position:]
    return total


def _objective(w, rankings, X, prior_variance):
    """Negative log posterior, with its gradient, for the minimiser."""
    penalty = w @ w / (2 * prior_variance)
    value = -(log_likelihood(w, rankings, X) - penalty)
    grad = -(gradient(w, rankings, X) - w / prior_variance)
    return value, grad


def fit(rankings, X, prior_variance=PRIOR_VARIANCE):
    """MAP estimate of the preference vector from a participant's rankings."""
    start = np.zeros(X.shape[1])
    if not rankings:
        return start

    result = minimize(
        _objective,
        start,
        args=(rankings, X, prior_variance),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500},
    )
    return result.x


def pairwise_probability(w, winner, loser, X):
    """Bradley-Terry, written out directly, for checking the n = 2 case against the general one."""
    difference = (X[winner] - X[loser]) @ w
    return float(1.0 / (1.0 + np.exp(-difference)))


def recommend(w, X, k=10, exclude=()):
    """The k films the fitted preferences rank highest."""
    scores = X @ w
    if len(exclude):
        scores = scores.copy()
        scores[list(exclude)] = -np.inf
    return np.argsort(scores)[::-1][:k].tolist()


def held_out_log_likelihood(w, rankings, X):
    """Per-comparison fit on rankings the model was not trained on.

    This is the study's primary measure: how much a block of elicitation actually taught us
    about someone, normalised so a block of ten-item rankings and a block of pairs are on the
    same footing.
    """
    if not rankings:
        return 0.0
    return log_likelihood(w, rankings, X) / sum(len(r) - 1 for r in rankings)
