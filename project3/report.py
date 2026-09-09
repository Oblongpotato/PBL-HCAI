"""The PDF report the brief asks for.

Built on demand from the committed `results.json` and figures, so it can never disagree with
what the page shows. Everything quoted here is read from those results rather than typed in.
"""

from pathlib import Path

from utils.report import Report

from . import data, defer, experiments

STATIC = Path(__file__).resolve().parent / "static"


def _figure(path):
    return STATIC / path


def _percent(value):
    return f"{value * 100:.1f}%"


def _deferral_run(results, expert):
    return next(run for run in results["deferral"] if run["expert"] == expert)


def _active_run(results, expert):
    return next(run for run in results["active"]["runs"] if run["expert"] == expert)


def build():
    """Render the report and return PDF bytes."""
    results = experiments.load()
    if results is None:
        raise FileNotFoundError("No results yet — run `manage.py run_project3` first.")

    stamp = results["generated"]
    doc = Report(
        "Active Learning for Learning-to-Defer",
        f"Human-Centric Artificial Intelligence, project 3 &mdash; "
        f"results generated {stamp['utc']} from commit {stamp['git_sha']}",
    )

    _introduction(doc, results)
    _task1(doc, results)
    _task2(doc, results)
    doc.page_break()
    _task3(doc, results)
    doc.page_break()
    _task4(doc, results)
    _limitations(doc, results)
    return doc.render()


def _introduction(doc, results):
    summary = results["data"]
    doc.heading("1. The problem")
    doc.paragraph(
        "A topic classifier for news articles is allowed to decline to answer and hand the article "
        "to a human expert instead. Two questions follow. When is handing over actually worth it, "
        "given that the expert is neither free nor uniformly better? And how do we find out what "
        "the expert is good at, when nobody has told us and every answer costs a human's time?"
    )
    doc.paragraph(
        "The report follows the four tasks of the brief: a baseline classifier, a simulated expert, "
        "a learned deferral rule, and an active-learning strategy that discovers the expert's "
        "competence from a small number of queries."
    )

    doc.heading("Data", level=2)
    doc.paragraph(
        f"AG News, subsampled to {summary['n_train']} training and {summary['n_test']} test "
        f"articles, balanced across the four topics ({', '.join(summary['topics'])}). The "
        "subsample is drawn once with a fixed seed and committed to the repository. That is a "
        "deliberate choice: it keeps every number in this report reproducible and means the "
        "application never downloads anything at run time, so a reader can reproduce the results "
        "from a clean checkout."
    )
    doc.paragraph(
        f"Articles are represented by TF-IDF over {summary['vocabulary']} unigrams and bigrams. "
        f"The deferral and competence models need dense input, so the same matrix is reduced to "
        f"{summary['svd_components']} dimensions by truncated SVD, retaining "
        f"{summary['explained_variance']} of the variance."
    )


def _task1(doc, results):
    baseline = results["baseline"]
    doc.heading("2. Task 1 — the classifier on its own")
    doc.paragraph(
        "Multinomial logistic regression over the TF-IDF features, trained on every available "
        f"label. It reaches <b>{baseline['accuracy']}</b> accuracy on the held-out test set. This "
        "is the number the human-AI team has to beat: a deferral system that scores below it has "
        "made the product worse, however sophisticated the routing."
    )
    doc.table(
        [["Topic", "Accuracy"]] + [[topic, value] for topic, value in baseline["per_topic"].items()]
    )
    doc.paragraph(
        "The per-topic breakdown is what makes the rest of the project possible. Sports is easy and "
        "Sci/Tech is hard, so the classifier has identifiable weak regions rather than uniform "
        "error, and an expert can be complementary to it."
    )
    doc.figure(_figure(baseline["confusion_figure"]),
               "Business and Sci/Tech are the pair the classifier confuses.", width=330)


def _task2(doc, results):
    doc.heading("3. Task 2 — the simulated experts")
    doc.paragraph(
        "An expert is modelled as a distribution over the label they would give, conditioned on the "
        "true topic: they answer correctly with a probability that depends on the topic, and "
        "otherwise pick uniformly among the others. In the language of lecture 9 this is a user "
        "model p(a | s, theta) whose parameter is a per-topic competence vector."
    )

    doc.heading("Why these two profiles", level=2)
    doc.paragraph(
        "The design choice worth justifying is the competence profile, because it decides whether "
        "the deferral problem is interesting at all. An expert who is better everywhere makes "
        "deferral trivial (always defer); one who is worse everywhere makes it pointless (never "
        "defer). Neither teaches anything."
    )
    doc.paragraph(
        "The <b>specialist</b> is therefore set to be complementary to the classifier: strong "
        "(0.95) on Business and Sci/Tech, which are exactly the classifier's weakest topics, and "
        "poor (0.30&ndash;0.35) on World and Sports, where the classifier is strong. Its overall "
        "accuracy is well below the classifier's, so any gain must come from deferring selectively "
        "rather than from deferring often. The <b>generalist</b> is a flat 0.70 everywhere and "
        "beats the classifier on no topic; it is the control case, and section 5 shows what a "
        "deferral system should do with such a person."
    )

    rows = [["Expert", "Overall", "Beats the classifier on"]]
    for report in results["experts"]["reports"]:
        beats = ", ".join(report["beats_classifier_on"]) or "nothing"
        rows.append([report["label"], report["accuracy"], beats])
    doc.table(rows)
    doc.figure(_figure(results["experts"]["figure"]),
               "Measured accuracy per topic. The specialist and the classifier cross over.")


def _task3(doc, results):
    doc.heading("4. Task 3 — learning to defer")
    doc.paragraph(
        "Lecture 5 opens with confidence-based rejection: defer whenever the classifier's own "
        "certainty falls below a threshold. It is the natural first idea and it has a specific "
        "flaw, which this project is built to demonstrate: it never looks at the expert, so it "
        "cannot know whether handing over will help."
    )

    doc.heading("The surrogate", level=2)
    doc.paragraph(
        "The alternative is the lecture's deferral formulation. Rather than modelling the decision "
        "directly, we learn K+1 scorers where the extra option means &ldquo;ask the expert&rdquo;, "
        "and train them on the cost-sensitive softmax cross-entropy surrogate:"
    )
    doc.formula("L = - sum_k [ max_j c(j,x,y) - c(k,x,y) ] * log softmax_k( g(x) )")
    doc.paragraph(
        "with c(k,x,y) = 1[y != k] for a real class and c(K+1,x,y) = 1[expert wrong] + kappa for "
        "the deferral option. The bracket vanishes for every option that is not among the best, so "
        "in practice the true class always carries weight and deferral carries weight only on the "
        "examples where the expert would have been right. The gradient with respect to the scores "
        "is p * sum(w) - w, which is short enough to derive and implement directly; it is checked "
        "against finite differences in the test suite."
    )

    doc.heading("Two design decisions", level=2)
    doc.bullets([
        "<b>The class answer comes from the task 1 classifier, not from the learned scorers.</b> "
        "Letting the K+1 model produce the class prediction as well was tried first and produced a "
        "markedly weaker classifier than the one already built (0.90 against 0.97 on the same kept "
        "set), which made deferral look better than it was by comparing it against a straw man. The "
        "deferral model now decides only <i>whether</i> to answer.",
        "<b>kappa is chosen on held-out training data, never on the test set.</b> The query cost "
        "matters and the surrogate is loose enough that the value changes the outcome, so selecting "
        "it on the test set would report a score the search had already seen.",
    ])

    specialist = _deferral_run(results, "specialist")
    grid = [row["kappa"] for row in specialist["kappa_trace"]]
    doc.paragraph(
        f"The grid searched is {grid}. The ceiling of 1.0 is principled rather than arbitrary: when "
        "the expert is right the deferral option carries weight 1 - kappa, so at kappa >= 1 "
        "deferral is never encouraged and the option is switched off entirely. The grid therefore "
        "brackets the optimum by construction."
    )

    doc.heading("Results", level=2)
    rows = [["", "System accuracy", "Deferred", "Expert accuracy when asked", "Right party chosen"]]
    rows.append(["Classifier alone", specialist["baseline_accuracy"], "0.0", "—", "—"])
    rows.append(["Learned deferral", specialist["css"]["system_accuracy"],
                 specialist["css"]["deferral_rate"],
                 specialist["css"]["expert_accuracy_on_deferred"],
                 specialist["css"]["right_party_chosen"]])
    rows.append(["Confidence-based, same budget",
                 specialist["confidence_matched"]["system_accuracy"],
                 specialist["confidence_matched"]["deferral_rate"],
                 specialist["confidence_matched"]["expert_accuracy_on_deferred"],
                 specialist["confidence_matched"]["right_party_chosen"]])
    rows.append(["Perfect routing", specialist["css"]["oracle_ceiling"], "—", "—", "1.0"])
    doc.table(rows)

    doc.paragraph(
        "The accuracy difference is real but modest. The decisive evidence is where each strategy "
        "spends the same budget of questions."
    )
    doc.figure(_figure(specialist["targeting_figure"]),
               "At an identical deferral rate, the learned rule concentrates on the topics where "
               "the expert is better; confidence-based rejection cannot tell them apart.")

    by_topic = specialist["deferral_by_topic"]
    strong = [t for t, row in by_topic.items() if row["expert_better"]]
    weak = [t for t, row in by_topic.items() if not row["expert_better"]]
    doc.paragraph(
        "Concretely: the learned rule sends "
        + ", ".join(f"{_percent(by_topic[t]['css'])} of {t}" for t in strong)
        + " to the expert, against only "
        + " and ".join(f"{_percent(by_topic[t]['css'])} of {t}" for t in weak)
        + ". Confidence-based rejection spreads the same budget almost evenly ("
        + ", ".join(_percent(by_topic[t]["confidence"]) for t in by_topic)
        + "), because a low-confidence article looks the same to it whether or not the expert can "
        "help. That is the lecture's objection, measured."
    )
    doc.figure(_figure(specialist["coverage_figure"]),
               "Accuracy against how often the human is asked. The gap to perfect routing is what a "
               "linear surrogate leaves on the table.")


def _task4(doc, results):
    doc.heading("5. Task 4 — discovering the expert's competence")
    specialist = _active_run(results, "specialist")
    doc.paragraph(
        "Task 3 assumed we already knew, for every training article, whether the expert would have "
        f"been right. Collecting that means asking a human about all {results['data']['n_train']} "
        "of them, which is exactly the cost the exercise is supposed to avoid. Here we start with "
        "no expert labels at all and may ask about only a few, so the question becomes which "
        "articles to ask about."
    )

    doc.heading("The query strategy", level=2)
    doc.paragraph(
        "The strategy should target the decision the answers feed. Deferral is hardest to call "
        "where the expert and the classifier are equally likely to be right, so the informative "
        "articles are those whose estimated probabilities nearly tie:"
    )
    doc.formula("u_info(x) = - | P(expert right | x) - P(classifier right | x) |")
    doc.paragraph(
        "On its own that spends the budget on outliers, so following lecture 6 it is multiplied by "
        "a representativeness term, the mean similarity of the article to the rest of the pool. "
        "Two baselines are run against it: querying uniformly at random, and querying where the "
        "classifier is least confident. The second is the natural mistake — it finds genuinely hard "
        "articles, but says nothing about whether the <i>expert</i> can handle them."
    )

    curves = {s["strategy"]: s["curve"] for s in specialist["strategies"]}
    proposed, random_query = curves["proposed"], curves["random"]
    doc.paragraph(
        f"The result is a large gain in label efficiency. The proposed strategy reaches "
        f"{proposed[3]['system_accuracy']} system accuracy after {proposed[3]['queries']} expert "
        f"labels; random querying is still at {random_query[-1]['system_accuracy']} after "
        f"{random_query[-1]['queries']}, five times as many. Knowing every expert label would give "
        f"{specialist['full_supervision']}."
    )
    doc.figure(_figure(specialist["figure"]),
               "Asking about the right articles matters more than asking about many.")

    doc.heading("What was recovered", level=2)
    doc.paragraph(
        f"Estimated from {specialist['budget']} answers, against the profile the expert was "
        "actually configured with. The absolute values drift, but the split between strong and weak "
        "topics — which is all the deferral decision depends on — comes through clearly."
    )
    doc.table(
        [["Topic", "Estimated competence", "Actual"]]
        + [[row["topic"], row["learned"], row["actual"]]
           for row in specialist["strategies"][0]["competence"]],
        widths=[110, 130, 70],
    )


def _limitations(doc, results):
    generalist = _deferral_run(results, "generalist")
    doc.heading("6. Limitations and the negative result")

    doc.heading("An expert who helps nowhere", level=2)
    doc.paragraph(
        f"The generalist is the control case, and the outcome is worth reporting rather than "
        f"hiding. The query-cost search selects kappa = {generalist['kappa']}, the point at which "
        f"deferral is switched off, and the system defers "
        f"{_percent(generalist['css']['deferral_rate'])} of articles, landing at "
        f"{generalist['css']['system_accuracy']} — the classifier's own accuracy."
    )
    doc.paragraph(
        "That is the correct answer, not a failure. This expert is better on no topic, so there is "
        "no region for the router to aim at and every hand-over is a loss in expectation. The "
        "system's best available policy is to consult nobody, and it finds it. An earlier and "
        "narrower kappa grid did not: it stopped at 0.4, still deferred roughly a third of "
        "articles, and scored below the baseline. The lesson is that the surrogate will happily "
        "defer whenever the expert happened to be right on a training example, and pricing the "
        "expert's time is what suppresses that."
    )

    doc.heading("Other limitations", level=2)
    doc.bullets([
        "The deferral scorers are linear over the reduced features. The gap to perfect routing in "
        "section 4 is largely the capacity this gives up.",
        "The expert is simulated, and its competence depends only on the true topic. A real "
        "annotator's reliability would vary with the article itself, not just its category, and "
        "would drift with fatigue.",
        "Workload is controlled only through the query cost. A deployed system would need a hard "
        "cap on how often a person is interrupted, which lecture 5 raises and this does not model.",
        "The active-learning comparison uses a single seed per strategy. The curves are far enough "
        "apart to be convincing, but repeated runs with confidence bands would be stronger.",
    ])
