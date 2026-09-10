"""Task 3: deciding when to answer and when to hand over.

Two strategies, and the comparison between them is the point.

Confidence-based rejection is the naive one from the start of lecture 5: defer whenever the
classifier's own certainty falls below a threshold. It never looks at the expert, so it
cannot know whether handing over actually helps.

The second is the lecture's deferral formulation. Instead of modelling the decision directly
we learn K+1 scorers, where class K+1 means "ask the expert", and train them on the
cost-sensitive softmax cross-entropy surrogate:

    L = - sum_k [ max_j c(j, x, y) - c(k, x, y) ] * log softmax_k(g(x))

with c(k, x, y) = 1[y != k] for a real class, and c(K+1, x, y) = 1[expert(x) != y] + kappa
for the deferral option. The bracket is zero for every option that is not among the best, so
in practice the true class always carries weight, and deferral carries weight only when the
expert would have been right. kappa prices the expert's time.

The gradient of that loss with respect to the logits is p * sum(w) - w, which is short
enough to write directly, so the model is plain numpy over the SVD features.
"""

import numpy as np

from . import classifier, data, experts

# When the expert is right the deferral option carries weight 1 - kappa, so at kappa >= 1
# deferral is never encouraged and the option is effectively switched off. The grid therefore
# brackets the optimum by construction rather than stopping at an arbitrary value.
KAPPAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
VALIDATION_FRACTION = 0.25
EPOCHS = 120
BATCH = 256
LEARNING_RATE = 0.05
L2 = 1e-4
SEED = 0


def _standardise(Z_train, Z_test):
    mean, scale = Z_train.mean(axis=0), Z_train.std(axis=0) + 1e-9
    return (Z_train - mean) / scale, (Z_test - mean) / scale


def deferral_features():
    """SVD text features plus what the deployed classifier believes.

    Without the second part the scorers would have to relearn topic classification from
    scratch on 300 dense dimensions, and would do it worse than the classifier already in
    production. The deferral decision would then be made against a straw man. Training uses
    out-of-fold probabilities so the confidences are not inflated by having been fitted on
    the same rows.
    """
    parts = data.features()
    Z_train, Z_test = _standardise(parts["Z_train"], parts["Z_test"])
    log_train = np.log(classifier.out_of_fold_probabilities() + 1e-9)
    log_test = np.log(classifier.baseline()["probabilities"]["test"] + 1e-9)
    return np.hstack([Z_train, log_train]), np.hstack([Z_test, log_test])


def _softmax(scores):
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def css_weights(y, expert_correct, n_classes, kappa=0.0):
    """The bracket [max_j c_j - c_k] for every option, per example."""
    n = len(y)
    costs = np.ones((n, n_classes + 1))
    costs[np.arange(n), y] = 0.0
    costs[:, n_classes] = (1 - expert_correct) + kappa
    return costs.max(axis=1, keepdims=True) - costs


def css_loss(scores, weights):
    """Mean CSS loss and its gradient with respect to the scores."""
    probabilities = _softmax(scores)
    loss = -(weights * np.log(probabilities + 1e-12)).sum(axis=1).mean()
    gradient = (probabilities * weights.sum(axis=1, keepdims=True) - weights) / len(scores)
    return loss, gradient


def train_css(Z, y, expert_correct, n_classes, kappa=0.0, epochs=EPOCHS, seed=SEED):
    """Fit the K+1 linear scorers by minibatch gradient descent (Adam)."""
    rng = np.random.default_rng(seed)
    weights = css_weights(y, expert_correct, n_classes, kappa)

    theta = np.zeros((Z.shape[1], n_classes + 1))
    bias = np.zeros(n_classes + 1)
    m_t, v_t = np.zeros_like(theta), np.zeros_like(theta)
    m_b, v_b = np.zeros_like(bias), np.zeros_like(bias)
    step = 0
    history = []

    for _ in range(epochs):
        order = rng.permutation(len(Z))
        for start in range(0, len(Z), BATCH):
            rows = order[start : start + BATCH]
            batch, batch_weights = Z[rows], weights[rows]

            loss, gradient = css_loss(batch @ theta + bias, batch_weights)
            grad_theta = batch.T @ gradient + L2 * theta
            grad_bias = gradient.sum(axis=0)

            step += 1
            for param, grad, m, v in (
                (theta, grad_theta, m_t, v_t),
                (bias, grad_bias, m_b, v_b),
            ):
                m *= 0.9
                m += 0.1 * grad
                v *= 0.999
                v += 0.001 * grad**2
                param -= LEARNING_RATE * (m / (1 - 0.9**step)) / (
                    np.sqrt(v / (1 - 0.999**step)) + 1e-8
                )

        history.append(float(css_loss(Z @ theta + bias, weights)[0]))

    return {"theta": theta, "bias": bias, "history": history}


def evaluate(defer_mask, class_prediction, expert_prediction, truth):
    """How good the decisions were, not just how accurate the answers were.

    The brief asks for the quality of the deferral decisions, which is not the same thing as
    the accuracy of the answers. On most articles both parties agree and the routing cannot
    matter; scoring the router over all of them just reproduces system accuracy. So routing is
    scored only where exactly one party was right, which is where the decision had a cost.
    """
    system = np.where(defer_mask, expert_prediction, class_prediction)
    classifier_right = class_prediction == truth
    expert_right = expert_prediction == truth
    ideal = np.maximum(classifier_right, expert_right)

    # Deferring was the right call when the expert was right and the classifier was not, and
    # keeping was right in the mirror case. Elsewhere either choice gives the same answer.
    decisive = classifier_right != expert_right
    routed_well = defer_mask == expert_right

    kept = ~defer_mask
    return {
        "system_accuracy": round(float((system == truth).mean()), 4),
        "deferral_rate": round(float(defer_mask.mean()), 4),
        "classifier_accuracy_on_kept": round(float(classifier_right[kept].mean()), 4)
        if kept.any()
        else None,
        "expert_accuracy_on_deferred": round(float(expert_right[defer_mask].mean()), 4)
        if defer_mask.any()
        else None,
        "routing_accuracy": round(float(routed_well[decisive].mean()), 4)
        if decisive.any()
        else None,
        "decisive_share": round(float(decisive.mean()), 4),
        "oracle_ceiling": round(float(ideal.mean()), 4),
    }


def confidence_curve(expert_name, thresholds=None):
    """Baseline: defer the least confident predictions, ignoring who the expert is."""
    parts = data.features()
    truth = parts["y_test"]
    class_prediction = classifier.baseline()["predictions"]["test"]
    expert_prediction = experts.predictions(expert_name, "test")
    confidence = classifier.confidence("test")

    if thresholds is None:
        thresholds = np.quantile(confidence, np.linspace(0, 0.95, 20))

    points = []
    for threshold in thresholds:
        result = evaluate(confidence < threshold, class_prediction, expert_prediction, truth)
        result["threshold"] = round(float(threshold), 4)
        points.append(result)
    return points


def css_curve(model, Z_test, expert_name, class_prediction, offsets=None):
    """Trace the CSS model's coverage by shifting the deferral score."""
    parts = data.features()
    truth = parts["y_test"]
    expert_prediction = experts.predictions(expert_name, "test")

    scores = Z_test @ model["theta"] + model["bias"]
    n_classes = len(data.TOPICS)
    margin = scores[:, n_classes] - scores[:, :n_classes].max(axis=1)

    if offsets is None:
        offsets = np.quantile(-margin, np.linspace(0.02, 0.98, 20))

    points = []
    for offset in offsets:
        result = evaluate(margin + offset > 0, class_prediction, expert_prediction, truth)
        result["offset"] = round(float(offset), 4)
        points.append(result)
    return points


def _fit_and_apply(Z_fit, y_fit, expert_correct_fit, Z_eval, kappa, n_classes, seed=SEED):
    model = train_css(Z_fit, y_fit, expert_correct_fit, n_classes, kappa=kappa, seed=seed)
    scores = Z_eval @ model["theta"] + model["bias"]
    return model, scores.argmax(axis=1) == n_classes


def choose_kappa(expert_name, n_classes, seed=SEED):
    """Pick the query cost on held-out training data, never on the test set.

    kappa prices the expert's time, and the surrogate is loose enough that the value matters.
    Choosing it on the test set would report a score that the search had already seen, which
    is the mistake project 1 had to fix.
    """
    parts = data.features()
    Z_train, _ = deferral_features()
    y = parts["y_train"]
    expert_correct = experts.correctness(expert_name, "train")
    expert_prediction = experts.predictions(expert_name, "train")
    class_prediction = classifier.out_of_fold_probabilities().argmax(axis=1)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    cut = int(len(y) * (1 - VALIDATION_FRACTION))
    fit, validation = order[:cut], order[cut:]

    trace = []
    for kappa in KAPPAS:
        _, defer_mask = _fit_and_apply(
            Z_train[fit], y[fit], expert_correct[fit], Z_train[validation], kappa, n_classes
        )
        result = evaluate(
            defer_mask,
            class_prediction[validation],
            expert_prediction[validation],
            y[validation],
        )
        trace.append({"kappa": kappa, **result})

    best = max(trace, key=lambda row: row["system_accuracy"])
    return best["kappa"], trace


def run(expert_name):
    """Train the deferral model for one expert and measure both strategies.

    The class prediction always comes from the classifier built in task 1. The deferral model
    decides *whether* to answer, not what the answer is; letting it relearn topic
    classification from 300 dense dimensions produced a much weaker classifier and made
    deferral look better than it was.
    """
    parts = data.features()
    Z_train, Z_test = deferral_features()
    truth = parts["y_test"]
    n_classes = len(data.TOPICS)

    class_prediction = classifier.baseline()["predictions"]["test"]
    expert_prediction = experts.predictions(expert_name, "test")
    confidence = classifier.confidence("test")

    kappa, kappa_trace = choose_kappa(expert_name, n_classes)
    model, defer_mask = _fit_and_apply(
        Z_train,
        parts["y_train"],
        experts.correctness(expert_name, "train"),
        Z_test,
        kappa,
        n_classes,
    )
    css = evaluate(defer_mask, class_prediction, expert_prediction, truth)

    # Confidence rejection at the same workload: the only fair comparison, since any
    # strategy looks better if it is allowed to ask the human more often. When the learned
    # rule defers nothing there is no budget to match, and forcing a quantile through anyway
    # hands the baseline a single article and reports statistics computed from it.
    if css["deferral_rate"] > 0:
        threshold = float(np.quantile(confidence, css["deferral_rate"]))
        confidence_mask = confidence < threshold
        matched = evaluate(confidence_mask, class_prediction, expert_prediction, truth)
    else:
        confidence_mask = np.zeros_like(defer_mask)
        matched = None

    # Where each strategy spends its budget. This is the clearest evidence for the lecture's
    # objection: confidence rejection cannot aim, because it never looks at the expert.
    by_topic = {
        topic: {
            "css": round(float(defer_mask[truth == code].mean()), 3),
            "confidence": round(float(confidence_mask[truth == code].mean()), 3),
            "expert_better": bool(
                (expert_prediction[truth == code] == code).mean()
                > (class_prediction[truth == code] == code).mean()
            ),
        }
        for topic, code in data.CODES.items()
    }

    return {
        "expert": expert_name,
        "kappa": kappa,
        "kappa_trace": kappa_trace,
        "css": css,
        "confidence_matched": matched,
        "deferral_by_topic": by_topic,
        "css_curve": css_curve(model, Z_test, expert_name, class_prediction),
        "confidence_curve": confidence_curve(expert_name),
        "baseline_accuracy": round(float((class_prediction == truth).mean()), 4),
    }
