import pulp

def compute_objective_score(data, x):
    """Recomputes objective score for either PuLP or heuristic schedules."""
    staff = list(data["staff"].keys())
    days = list(range(data["horizon"]))
    shifts = list(data["shifts"].keys())

    score = 0

    # Shift ON request penalties
    for emp, d, s, w in data["req_on"]:
        if pulp.value(x[emp][d][s]) < 0.5:
            score += w

    # Shift OFF request penalties
    for emp, d, s, w in data["req_off"]:
        if pulp.value(x[emp][d][s]) > 0.5:
            score += w

    # Coverage penalties
    for (d, s), cov in data["cover"].items():
        required = cov["required"]
        under_w = cov["under"]
        over_w  = cov["over"]

        assigned = sum(pulp.value(x[e][d][s]) for e in staff)

        if assigned < required:
            score += (required - assigned) * under_w
        elif assigned > required:
            score += (assigned - required) * over_w

    return score