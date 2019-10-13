import numpy as np
import itertools
import statistics
import matplotlib.pyplot as plt

def plot_ellsworth_figs(data):

    emotions = ["sadness", "joy", "fear", "anger", "challenge", "boredom", "frustration"]
    vars = ["pleasantness", "attention", "control",
                          "certainty", "anticipated_effort", "responsibility"]

    em_plot = {
        "pleasantness":{x:[] for x in emotions},
        "attention":{x:[] for x in emotions},
        "control":{x:[] for x in emotions},
        "certainty":{x:[] for x in emotions},
        "anticipated_effort":{x:[] for x in emotions},
        "responsibility":{x:[] for x in emotions}
    }

    for x in data:
        for turn in ["turn1", "turn2", "turn3"]:
            emotion = x[turn]["emotion"]
            for var in vars:
                em_plot[var][emotion].append(x[turn]["appraisals"][var])
    
    for var in em_plot:
        for em in em_plot[var]:
            em_plot[var][em] = statistics.median(em_plot[var][em])

    fig, ax = plt.subplots()
    ax.scatter(list(em_plot["pleasantness"].values()), list(em_plot["anticipated_effort"].values()))
    ax.set_xlabel("Pleasantness")
    ax.set_ylabel("Effort")
    for i, txt in enumerate(emotions):
        ax.annotate(txt, ( list(em_plot["pleasantness"].values())[i], list(em_plot["anticipated_effort"].values())[i]))
    plt.savefig("results/figs/effort_vs_pleasantness.png")

    fig, ax = plt.subplots()
    ax.scatter(list(em_plot["responsibility"].values()), list(em_plot["control"].values()))
    ax.set_xlabel("Responsibility")
    ax.set_ylabel("Control")
    for i, txt in enumerate(emotions):
        ax.annotate(txt, ( list(em_plot["responsibility"].values())[i], list(em_plot["control"].values())[i]))
    plt.savefig("results/figs/control_vs_res.png")

    fig, ax = plt.subplots()
    ax.scatter(list(em_plot["attention"].values()), list(em_plot["certainty"].values()))
    ax.set_xlabel("Attention")
    ax.set_ylabel("Certainty")
    for i, txt in enumerate(emotions):
        ax.annotate(txt, ( list(em_plot["attention"].values())[i], list(em_plot["certainty"].values())[i]))
    plt.savefig("results/figs/certainty_vs_attention.png")


def plot_confusion_matrix(cm, classes, title, normalize=False, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Courtesy of scikit-learn
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig("results/confusion_matrices/" + title + ".png")
    plt.close()