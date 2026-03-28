import pulp

def validate_solution(data, x, model):
    staff = list(data["staff"].keys())
    days = list(range(data["horizon"]))
    shifts = list(data["shifts"].keys())

    errors = []
    warnings = []

    print("\n=== VALIDATING SOLUTION ===")

    # 1. One shift per day per nurse
    for e in staff:
        for d in days:
            assigned = sum(1 for s in shifts if pulp.value(x[e][d][s]) > 0.5)
            if assigned > 1:
                errors.append(f"Employee {e} works multiple shifts on day {d}")

    # 2. Days-off respected
    if "days_off" in data:
        for e, offdays in data["days_off"].items():
            for d in offdays:
                for s in shifts:
                    if pulp.value(x[e][d][s]) > 0.5:
                        errors.append(f"Employee {e} assigned on OFF day {d}")

    # 3. Coverage (warnings, not errors)
    for (d, s), cov in data["cover"].items():
        req = cov["required"]
        assigned = sum(pulp.value(x[e][d][s]) for e in staff)

        if assigned < req:
            warnings.append(
                f"UNDER-COVER: Day {d}, Shift {s}: assigned {assigned}, required {req}"
            )
        if assigned > req:
            warnings.append(
                f"OVER-COVER: Day {d}, Shift {s}: assigned {assigned}, required {req}"
            )

    # 4. Forbidden sequences
    for e in staff:
        for d in range(1, data["horizon"]):
            for s_prev in shifts:
                if pulp.value(x[e][d-1][s_prev]) > 0.5:
                    forbidden = data["shifts"][s_prev]["forbidden_follow"]
                    for s_next in forbidden:
                        if s_next in shifts and pulp.value(x[e][d][s_next]) > 0.5:
                            errors.append(
                                f"Forbidden sequence for {e}: {s_prev} on day {d-1} → {s_next} on day {d}"
                            )

    # 5. Recompute objective
    recomputed = 0

    for emp, d, s, w in data["req_on"]:
        if pulp.value(x[emp][d][s]) < 0.5:   # not assigned
            recomputed += w

    for emp, d, s, w in data["req_off"]:
        if pulp.value(x[emp][d][s]) > 0.5:   # assigned but they wanted off
            recomputed += w

    for (d, s), cov in data["cover"].items():
        req = cov["required"]
        under_w = cov["under"]
        over_w = cov["over"]

        assigned = sum(pulp.value(x[e][d][s]) for e in staff)

        if assigned < req:
            recomputed += (req - assigned) * under_w
        elif assigned > req:
            recomputed += (assigned - req) * over_w

    solver_obj = pulp.value(model.objective)

    print("\n--- OBJECTIVE CHECK ---")
    print(" Solver objective:     ", solver_obj)
    print(" Recomputed objective: ", recomputed)

    if abs(solver_obj - recomputed) > 1e-5:
        warnings.append("Objective mismatch detected!")

    # Summary
    print("\n--- FEASIBILITY SUMMARY ---")
    if errors:
        print("❌ Infeasible schedule:")
        for e in errors:
            print("  -", e)
    else:
        print("✅ All hard constraints satisfied.")

    print("\n--- WARNINGS ---")
    if warnings:
        for w in warnings:
            print("  •", w)
    else:
        print("✅ No warnings.")

    print("\nValidation complete.\n")