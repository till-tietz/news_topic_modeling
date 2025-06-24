import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from matplotlib.cm import get_cmap
from typing import Dict

def evaluate_multiclass_model(
        y_true: np.array,
        y_score: np.ndarray,
        label_mapping: Dict,
        output_dir: str = '/app/output'
) -> None:
    """ Evaluates a multiclass classifier with classification report, confusion matrix,
    and one-vs-rest ROC AUC curves.

    Arguments
    ---------
        y_true: True class labels (numeric).
        y_score: Predicted probabilities, shape (n_samples, n_classes).
        label_mapping: Dictionary mapping numeric labels to string labels.
    """

    n_classes = y_score.shape[1]
    class_names = [label_mapping[i] for i in range(n_classes)]
    y_pred = np.argmax(y_score, axis=1)

    # Prepare classification report as a string
    report = classification_report(y_true, y_pred, target_names=class_names)

    # Prepare binarized labels for ROC
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    fpr, tpr, roc_auc = {}, {}, {}
    cmap = get_cmap('viridis', n_classes)

    # Create figure with 3 equally sized panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.tight_layout(pad=5)

    # --- Panel 1: Text report ---
    axes[0].axis('off')
    axes[0].set_title('Classification Report', fontsize=12, fontweight='bold')
    axes[0].text(0, 1, report, fontsize=10, fontfamily='monospace', verticalalignment='top')

    # --- Panel 2: Confusion Matrix ---
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names,
        xticks_rotation=45, ax=axes[1], colorbar=False
    )
    axes[1].set_title("Confusion Matrix")

    # --- Panel 3: ROC Curves ---
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        axes[2].plot(
            fpr[i], tpr[i],
            color=cmap(i), lw=2,
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})"
        )

    axes[2].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[2].set_xlabel('False Positive Rate')
    axes[2].set_ylabel('True Positive Rate')
    axes[2].set_title('One-vs-Rest ROC Curves')
    axes[2].legend(loc='lower right')
    axes[2].grid(True)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'model_performance.png'))