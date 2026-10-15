"""Aggregate segment similarities without loading the audio or Siamese models."""


def aggregate_decision_matrix(
    similarity_matrix,
    proportion_thres: float = 0.2,
    decision_thres: float = 0.5,
    min_hits: int = 1,
):
    """Require sufficient matched-window coverage in each piece independently.

    Rows represent windows in piece 1; columns represent windows in piece 2.
    Preserve the public API's strict score/proportion thresholds and min_hits.
    """
    decision_matrix = similarity_matrix > decision_thres
    assert len(decision_matrix.shape) == 2
    num_rows, num_columns = decision_matrix.shape
    matched_rows = decision_matrix.sum(dim=1) >= min_hits
    matched_columns = decision_matrix.sum(dim=0) >= min_hits
    if (matched_rows.sum() > proportion_thres * num_rows
            and matched_columns.sum() > proportion_thres * num_columns):
        return 1, decision_matrix
    return 0, decision_matrix
