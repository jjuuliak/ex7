import random

def heuristic_solve(data):
    staff = list(data["staff"].keys())
    days = range(data["horizon"])
    shifts = list(data["shifts"].keys())

    # Create a simple 3‑level dict of integers (0 or 1)
    hx = {e: {d: {s: 0 for s in shifts} for d in days} for e in staff}

    # ----------------------------------------------------
    # Greedy heuristic: Fill required shifts day by day
    # ----------------------------------------------------
    for d in days:
        for s in shifts:
            required = data["cover"][(d, s)]["required"]
            assigned = 0

            candidates = staff.copy()
            random.shuffle(candidates)

            for e in candidates:
                # Can't assign on day off
                if e in data["days_off"] and d in data["days_off"][e]:
                    continue

                # Can't double assign
                if sum(hx[e][d].values()) > 0:
                    continue

                # Forbidden sequence check
                if d > 0:
                    for s_prev in shifts:
                        if hx[e][d-1][s_prev] == 1 and \
                           s in data["shifts"][s_prev]["forbidden_follow"]:
                            break
                    else:
                        # Previous shift OK → assign
                        hx[e][d][s] = 1
                        assigned += 1
                else:
                    hx[e][d][s] = 1
                    assigned += 1

                if assigned >= required:
                    break

    # ✅ Return the integer dictionary EXACTLY like the solver's x dict
    return hx