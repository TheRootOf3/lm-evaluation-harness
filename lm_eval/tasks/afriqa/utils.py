import transformers.data.metrics.squad_metrics as squad_metrics


def compute_scores(ref, pred):
    em_sum = squad_metrics.compute_exact(ref, pred)
    f1_sum = squad_metrics.compute_f1(ref, pred)

    return {
        "em": em_sum,
        "f1": f1_sum,
    }


def process_results(doc, results):
    ref = doc["answer_pivot"]
    pred = results[0].strip().split("\n")[0]

    scores = compute_scores(ref, pred)
    return scores
