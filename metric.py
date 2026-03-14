# Metric utilities for tracking training progress
def reset_metrics(metrics):
    """ Resets all metrics in the provided dictionary. """
    for metric in metrics.values():
        metric.reset()

def update_metrics(metrics, values_dict):
    """ Updates all metrics in the provided dictionary with new predictions and targets. """
    for k, v in values_dict.items():
        if k in metrics:
            metrics[k](v)

def compute_metrics(metrics):
    """ Computes and returns the current value of all metrics in the provided dictionary. """
    return {k: metric.compute().item() for k, metric in metrics.items()}