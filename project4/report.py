"""The PDF the brief asks for: the method (tasks 1 and 2) and the study design (task 3).

Rendered on demand so the catalogue figures quoted in it are always the ones the application is
actually using.
"""

from utils.report import Report

from . import data
from .study import SLATE_SIZE, TASKS_PER_BLOCK
from .models import PAIRWISE, RANKING


def build():
    summary = data.summary()
    doc = Report(
        "Comparing two ways of asking someone what they like",
        "Human-Centric Artificial Intelligence, group 39, project 4: method and study design",
    )
    _introduction(doc, summary)
    _task1(doc, summary)
    doc.page_break()
    _task2(doc, summary)
    doc.page_break()
    _task3(doc)
    _ethics(doc)
    _validity(doc)
    return doc.render()


def _introduction(doc, summary):
    doc.heading("1. What this document is")
    doc.paragraph(
        "A recommender that has never met you has to learn your taste from a handful of "
        "interactions. How it asks matters: you could be shown two films and asked which you "
        "prefer, or shown ten and asked to put them in order. The first is quick but says little "
        "each time; the second is slower but says a great deal. Which is the better use of a "
        "person's attention is an empirical question, and this document designs the experiment "
        "that would answer it."
    )
    doc.paragraph(
        "Sections 2 and 3 describe the method: how a film is represented, and how a preference "
        "over films is modelled and estimated. Section 4 is the study design proper. <b>The study "
        "has not been run.</b> What accompanies this document is the working participant "
        "interface, so the protocol described here could be executed as written."
    )
    doc.paragraph(
        f"The catalogue is the IMDB 5000 dataset: {summary['n_raw']} rows, of which "
        f"{summary['n_films']} films survive cleaning ({summary['n_dropped']} dropped for missing "
        f"year, genre, score or running time, or as duplicate titles), spanning "
        f"{summary['year_range'][0]} to {summary['year_range'][1]}. It carries no user ratings at "
        "all, which is the point: every preference in this study comes from the elicitation "
        "interface rather than from data someone else collected."
    )


def _task1(doc, summary):
    doc.heading("2. Task 1: representing a film")
    doc.paragraph(
        "The preference model is a linear utility, U(x) = w &middot; x, so the feature vector "
        "decides what a preference can be about at all. Two constraints shaped it."
    )
    doc.bullets([
        "<b>Every feature should be something a person can have a taste about.</b> Genre, era, "
        "running time, how celebrated a film is, how widely seen it is. A feature nobody holds an "
        "opinion on produces a weight nobody can interpret, and interpretability is most of the "
        "value of a linear model here.",
        "<b>The vector has to be short.</b> A participant will give perhaps thirty to fifty "
        "comparisons in a sitting. A representation with hundreds of dimensions cannot be "
        "estimated from that, so eras are decades rather than years and only the common "
        "certificates get their own column.",
    ])
    doc.paragraph(f"The result is {summary['n_features']} features:")
    doc.table([
        ["Group", "Source", "Encoding"],
        ["Genre", "genres", f"multi-hot over {summary['n_genres']} genres"],
        ["Era", "title_year", "one-hot by decade, everything before 1980 pooled"],
        ["Certificate", "content_rating", "one-hot over the eight common ratings, rest pooled"],
        ["Length", "duration", "standardised"],
        ["Acclaim", "imdb_score", "standardised"],
        ["Fame", "director / cast facebook likes", "log1p, then standardised"],
        ["Reach", "num_voted_users", "log1p, then standardised"],
        ["Origin", "language, country", "English-language, US-production indicators"],
    ], widths=[80, 170, 210])
    doc.paragraph(
        "Continuous features are standardised so the learned weights sit on a common scale and "
        "can be compared with each other; without it a weight on minutes and a weight on an IMDb "
        "score would not be commensurable. Counts of likes and votes are log-transformed first "
        "because both span several orders of magnitude, and the difference between a thousand and "
        "ten thousand votes means far more than the difference between a million and 1.01 million."
    )
    doc.paragraph(
        "<b>budget and gross are deliberately excluded.</b> Both are heavily missing, and the "
        "amounts are recorded in whatever currency the production reported with no unit column, so "
        "a model given them would learn a data-collection artefact and present it as taste. "
        "Dropping them costs little: acclaim and reach already carry most of what a viewer would "
        "notice about a film's scale."
    )


def _task2(doc, summary):
    doc.heading("3. Task 2: from pairwise choice to a ranking")
    doc.paragraph(
        "Bradley-Terry models a single comparison: item i beats item j with probability "
        "exp(U_i) / (exp(U_i) + exp(U_j)). One of the two interfaces asks for a ranking of ten "
        "films, so the model has to be extended to an ordering of n items."
    )

    doc.heading("The extension", level=2)
    doc.paragraph(
        "Read a ranking as a sequence of choices. The participant picks their favourite out of the "
        "whole slate; then their favourite of what remains; and so on until one film is left. Each "
        "pick is a Luce choice over the items still available, and multiplying the pick "
        "probabilities gives the probability of the whole ordering:"
    )
    doc.formula("P(i1 > i2 > ... > in) = prod_t  exp(w.x_it) / sum_{s>=t} exp(w.x_is)")
    doc.paragraph(
        "This is the Plackett-Luce model. Setting n = 2 leaves a single factor, "
        "exp(w.x_i1) / (exp(w.x_i1) + exp(w.x_i2)), which is Bradley-Terry exactly. That identity "
        "is not a curiosity, it is what makes the study coherent: <b>a pairwise choice and a "
        "ten-item ranking are the same likelihood evaluated at different n</b>. Both interfaces "
        "therefore produce evidence about the same w, on the same scale, and can be compared "
        "without any conversion factor between them. A ranking of ten contributes nine factors, a "
        "pair contributes one, which is also the natural way to count how much each interaction "
        "asked of the participant."
    )
    doc.paragraph(
        "The implementation asserts this rather than assuming it: a test evaluates the "
        "Plackett-Luce likelihood at n = 2 against the Bradley-Terry formula written out "
        "separately, and requires them to agree."
    )

    doc.heading("Estimating w", level=2)
    doc.paragraph(
        "The log-likelihood is sum_t [ w.x_it - logsumexp_{s>=t} w.x_is ], and its gradient is "
        "sum_t [ x_it - sum_{s>=t} p_s x_is ], where p is the softmax over the items still "
        "available at step t. The gradient has a readable form: the film that was chosen, minus "
        "the film the model expected to be chosen. Where those agree the weights do not move."
    )
    doc.paragraph(
        "Estimation is maximum a posteriori with a zero-mean Gaussian prior. This matters more "
        "than it might seem. A participant gives a few dozen comparisons for a vector of "
        f"{summary['n_features']} weights, so the unpenalised likelihood is under-determined: any feature that never "
        "appeared on the losing side would have its weight driven to infinity. The prior keeps the "
        "problem well posed and encodes the reasonable belief that most features matter little to "
        "most people. Optimisation is L-BFGS on the exact gradient, which is checked against "
        "finite differences in the tests."
    )


def _task3(doc):
    pairwise_tasks, ranking_tasks = TASKS_PER_BLOCK[PAIRWISE], TASKS_PER_BLOCK[RANKING]
    comparisons = pairwise_tasks * (SLATE_SIZE[PAIRWISE] - 1), ranking_tasks * (SLATE_SIZE[RANKING] - 1)

    doc.heading("4. Task 3: the study")
    doc.paragraph(
        "The study compares the two elicitation interfaces. It is designed here; it has not been "
        "conducted."
    )

    doc.heading("4.1 Question and hypotheses", level=2)
    doc.paragraph(
        "The question is not which interface people prefer, but which is the better use of their "
        "time. An interface that extracts twice as much information but takes three times as long "
        "is the worse instrument, and either measure alone would miss that."
    )
    doc.bullets([
        "<b>H1.</b> Ranking ten films yields a higher held-out preference log-likelihood per "
        "minute than repeated pairwise choice. Null: no difference.",
        "<b>H2.</b> Pairwise choice imposes lower subjective workload. Null: no difference.",
        "H1 is the primary hypothesis; the study is powered for it, and H2 is reported alongside "
        "because the two together decide which interface to ship.",
    ])

    doc.heading("4.2 Variables", level=2)
    doc.table([
        ["Role", "Variable", "Measurement"],
        ["Independent", "Elicitation interface", "two levels: pairwise, ranking-of-ten"],
        ["Dependent (primary)", "Information per minute",
         "held-out Plackett-Luce log-likelihood per comparison, divided by minutes on task"],
        ["Dependent", "Time on task", "server-side, presentation to submission, summed per block"],
        ["Dependent", "Subjective workload", "NASA-TLX after each block"],
        ["Dependent", "Recommendation satisfaction", "7-point rating of the final list"],
        ["Controlled", "Film slates", "uniform random, disjoint between blocks, seeded per participant"],
    ], widths=[95, 130, 235])
    doc.paragraph(
        f"The primary measure needs one more piece: a held-out set. Within each block the last "
        f"third of the tasks is withheld from fitting and used to score the w estimated from the "
        f"rest, so the measure is predictive accuracy on that participant's own later answers "
        f"rather than fit to the data used for training. A third rather than a fixed number of "
        f"responses, because the blocks hold different numbers of tasks: withholding two "
        f"responses would leave a single ranking to fit {ranking_tasks} ranking tasks worth of "
        f"weights from."
    )

    doc.heading("4.3 Design", level=2)
    doc.paragraph(
        "<b>Within-subject, with counterbalanced order.</b> Every participant uses both interfaces. "
        "The reason is variance: differences in film taste between people are enormous compared "
        "with any plausible difference between two interfaces, and a between-subject design would "
        "have to be several times larger to see past that noise. Each participant acts as their "
        "own control."
    )
    doc.paragraph(
        "The cost of the within-subject choice is order effects, so the order is counterbalanced: "
        "participants are assigned alternately at enrolment, so half start with pairwise and half "
        "with ranking. Alternating rather than randomising removes the chance of an accidentally "
        "lopsided split. It balances <i>starters</i>, though, not completers: someone who consents "
        "and then abandons still consumes their position in the sequence, so dropout can leave the "
        "analysed sample slightly uneven. The realised split should therefore be reported, and "
        "re-balanced by recruiting a few extra participants if it drifts."
    )
    doc.paragraph(
        f"Each block holds {pairwise_tasks} pairwise tasks or {ranking_tasks} ranking tasks, which "
        f"is {comparisons[0]} and {comparisons[1]} elementary comparisons respectively. The blocks "
        "are matched on neither wall-clock time nor comparison count, and deliberately so: time is "
        "half of what is being measured, and holding the evidence equal would fix the numerator "
        "of the other half. Each block is instead sized to be long enough to fit a preference "
        "vector and still hold answers back, and short enough that fatigue does not set in. The "
        "primary measure is a rate, log-likelihood per comparison per minute, so the two blocks "
        "supplying different amounts of evidence does not bias the comparison. "
        "Slates are drawn from one disjoint pool per participant, so no film is ever seen twice "
        "and neither condition can be handed easier material."
    )

    doc.heading("4.4 Participants", level=2)
    doc.paragraph(
        "<b>Sample size.</b> The primary comparison is a paired t-test. At alpha = 0.05, power "
        "0.8, and an assumed medium effect (Cohen's d = 0.5), the required sample is <b>N = 34</b>. "
        "Allowing for around 15% attrition and exclusions, <b>40 participants</b> would be "
        "recruited. The d = 0.5 assumption is the weakest point in the plan and should be revisited "
        "against the pilot; if the pilot suggests a smaller effect, the honest response is to "
        "recruit more rather than to report an underpowered result."
    )
    doc.bullets([
        "<b>Inclusion:</b> 18 or over, watches films often enough to hold opinions about them, "
        "sufficient English to read the interface.",
        "<b>Recruitment:</b> university mailing lists and a participant pool, which is a "
        "convenience sample and should be described as one. It skews young and educated, and that "
        "limits how far the result generalises.",
        "<b>Compensation:</b> a flat payment, not contingent on finishing, so that withdrawing "
        "carries no penalty.",
    ])

    doc.heading("4.5 Procedure", level=2)
    doc.bullets([
        "Information sheet and consent.",
        "Short demographics: age band, how often they watch films. Nothing identifying.",
        "Instructions and two practice tasks in the first interface, discarded.",
        "<b>Block A</b> in the assigned first interface, followed by NASA-TLX.",
        "A short washout: a filler task of about a minute, to blunt carry-over between blocks.",
        "<b>Block B</b> in the second interface, followed by NASA-TLX.",
        "The recommendations produced from their own answers, and a satisfaction rating.",
        "Debrief explaining what the two interfaces were and what was being compared.",
    ])
    doc.paragraph(
        "<b>Piloting.</b> Five to ten participants first, ideally observed in person, to catch "
        "unclear wording, mis-set task counts and anything that makes the ranking interface "
        "frustrating. Pilot data is <b>excluded from the analysis</b>; if the pilot causes a change "
        "to the materials, the pilot cannot also be evidence about them."
    )

    doc.heading("4.6 Analysis", level=2)
    doc.bullets([
        "Check the paired differences for normality (Shapiro-Wilk plus a look at the plot). Paired "
        "t-test if that holds, Wilcoxon signed-rank if it does not. The choice is stated in advance "
        "so it is not made after seeing which gives the nicer p-value.",
        "Report effect sizes with confidence intervals, not just significance. With N = 34 the "
        "interval is the more informative quantity.",
        "Correct across the multiple dependent measures (Holm), with H1 designated primary in "
        "advance.",
        "Test for an order effect by adding block order as a factor; if it is substantial, report "
        "it rather than averaging over it.",
        "Report the number of exclusions and why, including anyone who abandoned mid-study.",
    ])


def _ethics(doc):
    doc.heading("5. Ethics and data protection")
    doc.bullets([
        "<b>Informed consent</b> before anything is recorded, as an explicit action rather than a "
        "pre-ticked box, covering what is collected, what it is used for, and how long it is kept.",
        "<b>Right to withdraw</b> at any point without giving a reason and without losing "
        "compensation; withdrawn sessions are deleted rather than retained anonymously.",
        "<b>Data minimisation.</b> A participant is a random token. No name, no email, no IP "
        "address. What is stored is the film orderings they gave, how long each took, and the "
        "questionnaire answers. Nothing that identifies a person, and nothing not needed to "
        "answer the research question.",
        "<b>Retention.</b> Responses kept for the analysis and a defined period after publication, "
        "then deleted; the schedule stated on the information sheet.",
        "<b>Storage.</b> On institutional infrastructure, not a personal machine or a third-party "
        "service.",
        "<b>Approval</b> from the institutional ethics committee before any recruitment.",
    ])
    doc.paragraph(
        "The interface implements the parts of this that are code: consent is a required explicit "
        "action, and the data model has no column that could hold a personal detail."
    )


def _validity(doc):
    doc.heading("6. Threats to validity")
    doc.bullets([
        "<b>Order and learning effects.</b> Whichever interface comes second benefits from a "
        "participant who now understands the task. Counterbalancing spreads this across conditions "
        "but does not remove it, and it is tested for explicitly.",
        "<b>Fatigue.</b> Ranking ten films is tiring, and the second block is done by a more tired "
        "person. The washout helps a little; a shorter session would help more, at the cost of "
        "fewer comparisons each.",
        "<b>The confound in the primary measure.</b> Information per minute rewards an interface "
        "that is quick, but ranking is cognitively harder per second than choosing. Two interfaces "
        "could score alike while demanding very different amounts of effort, which is exactly why "
        "the workload measure is reported next to it rather than as an afterthought.",
        "<b>Idle time counts as time on task.</b> Duration is measured server-side from when a "
        "slate is first presented to when it is submitted, so a participant who opens a question, "
        "is interrupted, and returns has the interruption counted against the interface. Time is "
        "the denominator of the primary measure, so this inflates it. Implausibly long responses "
        "should be trimmed or winsorised before analysis, and a client-side timer that pauses on "
        "blur would measure it properly.",
        "<b>Satisficing on the ranking task.</b> A participant who finds ten items too hard may "
        "order the top few and leave the rest arbitrary. That looks like information but is not. "
        "Unusually fast rankings should be flagged, and a follow-up could compare the model's fit "
        "on early versus late positions to detect it.",
        "<b>Random slates.</b> Uniform sampling was chosen so neither interface is advantaged, but "
        "it means most pairs are easy and uninformative. An adaptive selector would extract more "
        "per question; it would also help the two interfaces by different amounts, which would "
        "confound the very comparison being made. It belongs in a follow-up study, not this one.",
        "<b>Generalisation.</b> A convenience sample of students, a Western film catalogue, and a "
        "single sitting. The result would speak about this population and this domain.",
    ])
