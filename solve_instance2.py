import pulp
import re
from validate_solution import validate_solution   # ✅ import validator
from heuristic_solver import heuristic_solve
from compute_objective_score import compute_objective_score  # ✅ import objective score calculator


# ============================================================
#  PARSER
# ============================================================
def parse_instance(filename):
    with open(filename, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("#")]

    i = 0
    data = {}

    while i < len(lines):
        line = lines[i]

        # SECTION: HORIZON
        if line == "SECTION_HORIZON":
            data["horizon"] = int(lines[i + 1])
            i += 2

        # SECTION: SHIFTS
        elif line == "SECTION_SHIFTS":
            shifts = {}
            i += 1
            while i < len(lines) and not lines[i].startswith("SECTION"):
                shift_id, minutes, forbidden = lines[i].split(",")
                shifts[shift_id] = {
                    "length": int(minutes),
                    "forbidden_follow": forbidden.split(";") if forbidden else []
                }
                i += 1
            data["shifts"] = shifts

        # SECTION: STAFF (robust parser)
        elif line == "SECTION_STAFF":
            i += 1
            staff = {}
            while i < len(lines) and not lines[i].startswith("SECTION"):
                line = lines[i].strip()
                emp, rest = line.split(",", 1)
                parts = rest.split(",")

                # numeric part begins where field is only digits
                split_index = None
                for idx, part in enumerate(parts):
                    if part.strip().isdigit():
                        split_index = idx
                        break
                if split_index is None:
                    raise ValueError("Bad STAFF line: " + line)

                shift_block = ",".join(parts[:split_index])
                numeric = parts[split_index:]

                # parse shift block
                maxshifts = {}
                for token in shift_block.split("\\"):
                    token = token.strip()
                    if "=" in token:
                        key, raw_val = token.split("=", 1)
                        m = re.search(r"(\d+)", raw_val)
                        if m:
                            maxshifts[key.strip()] = int(m.group(1))

                if len(numeric) != 6:
                    raise ValueError("Bad numeric STAFF section: " + line)

                max_minutes, min_minutes, max_consec, min_consec, min_days_off, max_weekends = map(int, numeric)

                staff[emp] = {
                    "max_per_shift": maxshifts,
                    "max_minutes": max_minutes,
                    "min_minutes": min_minutes,
                    "max_consec": max_consec,
                    "min_consec": min_consec,
                    "min_days_off": min_days_off,
                    "max_weekends": max_weekends
                }

                i += 1

            data["staff"] = staff

        # SECTION: DAYS_OFF
        elif line == "SECTION_DAYS_OFF":
            i += 1
            days_off = {}
            while i < len(lines) and not lines[i].startswith("SECTION"):
                emp, d = lines[i].split(",")
                days_off.setdefault(emp, []).append(int(d))
                i += 1
            data["days_off"] = days_off

        # SECTION: SHIFT_ON_REQUESTS
        elif line == "SECTION_SHIFT_ON_REQUESTS":
            i += 1
            req_on = []
            while i < len(lines) and not lines[i].startswith("SECTION"):
                emp, d, s, w = lines[i].split(",")
                req_on.append((emp, int(d), s, int(w)))
                i += 1
            data["req_on"] = req_on

        # SECTION: SHIFT_OFF_REQUESTS
        elif line == "SECTION_SHIFT_OFF_REQUESTS":
            i += 1
            req_off = []
            while i < len(lines) and not lines[i].startswith("SECTION"):
                emp, d, s, w = lines[i].split(",")
                req_off.append((emp, int(d), s, int(w)))
                i += 1
            data["req_off"] = req_off

        # SECTION: COVER
        elif line == "SECTION_COVER":
            i += 1
            cover = {}
            while i < len(lines):
                parts = lines[i].split(",")
                if len(parts) < 5:
                    break
                d, s, req, under, over = parts
                cover[(int(d), s)] = {
                    "required": int(req),
                    "under": int(under),
                    "over": int(over)
                }
                i += 1
            data["cover"] = cover

        else:
            i += 1

    return data


# ============================================================
#  MODEL
# ============================================================
def build_model(data):

    staff = list(data["staff"].keys())
    days = range(data["horizon"])
    shifts = list(data["shifts"].keys())

    x = pulp.LpVariable.dicts("x", (staff, days, shifts), 0, 1, pulp.LpBinary)

    model = pulp.LpProblem("NRP_Instance2", pulp.LpMinimize)
    obj_terms = []

    # Requests
    for emp, d, s, w in data["req_on"]:
        obj_terms.append(w * (1 - x[emp][d][s]))

    for emp, d, s, w in data["req_off"]:
        obj_terms.append(w * x[emp][d][s])

    # Coverage penalties
    for (d, s), cov in data["cover"].items():
        req = cov["required"]
        under_w = cov["under"]
        over_w = cov["over"]

        assigned = pulp.lpSum(x[e][d][s] for e in staff)

        u = pulp.LpVariable(f"u_{d}_{s}", 0)
        o = pulp.LpVariable(f"o_{d}_{s}", 0)

        model += assigned + u - o == req
        obj_terms.append(under_w * u + over_w * o)

    # HARD CONSTRAINTS -------------------------------

    # One shift per day
    for e in staff:
        for d in days:
            model += pulp.lpSum(x[e][d][s] for s in shifts) <= 1

    # Days off
    for e in staff:
        if e in data["days_off"]:
            for d in data["days_off"][e]:
                model += pulp.lpSum(x[e][d][s] for s in shifts) == 0

    # Max shifts of each type
    for e in staff:
        for s in shifts:
            if s in data["staff"][e]["max_per_shift"]:
                model += pulp.lpSum(x[e][d][s] for d in days) <= data["staff"][e]["max_per_shift"][s]

    # Minutes constraints
    for e in staff:
        total_minutes = pulp.lpSum(
            x[e][d][s] * data["shifts"][s]["length"]
            for d in days for s in shifts
        )
        model += total_minutes <= data["staff"][e]["max_minutes"]
        model += total_minutes >= data["staff"][e]["min_minutes"]

    # Max consecutive working days
    for e in staff:
        k = data["staff"][e]["max_consec"]
        for start in range(data["horizon"] - (k + 1)):
            model += pulp.lpSum(
                pulp.lpSum(x[e][d][s] for s in shifts)
                for d in range(start, start + k + 1)
            ) <= k

    # Forbidden shift sequences
    for e in staff:
        for d in range(1, data["horizon"]):
            for s_prev in shifts:
                for s_next in data["shifts"][s_prev]["forbidden_follow"]:
                    if s_next in shifts:
                        model += x[e][d - 1][s_prev] + x[e][d][s_next] <= 1

    # Max weekends
    weekend_pairs = [(5, 6), (12, 13)]
    for e in staff:
        max_w = data["staff"][e]["max_weekends"]
        wknd_vars = []
        for sat, sun in weekend_pairs:
            w = pulp.LpVariable(f"weekend_{e}_{sat}", 0, 1, pulp.LpBinary)
            model += w >= pulp.lpSum(x[e][sat][s] for s in shifts)
            model += w >= pulp.lpSum(x[e][sun][s] for s in shifts)
            wknd_vars.append(w)
        model += pulp.lpSum(wknd_vars) <= max_w

    # Objective
    model += pulp.lpSum(obj_terms)

    return model, x


# ============================================================
#  SOLVER
# ============================================================
def solve(data):
    model, x = build_model(data)
    model.solve(pulp.PULP_CBC_CMD(msg=1))
    return model, x


# ============================================================
#  OUTPUT
# ============================================================
def print_schedule(x, data):
    staff = list(data["staff"].keys())
    days = range(data["horizon"])
    shifts = list(data["shifts"].keys())

    print("\n=== FINAL SCHEDULE ===")
    print("Day | " + " | ".join(staff))
    print("-" * (5 + 4 * len(staff)))

    for d in days:
        row = f"{d:3d} | "
        for e in staff:
            assigned = [s for s in shifts if pulp.value(x[e][d][s]) > 0.5]
            row += (assigned[0] if assigned else "-") + " | "
        print(row)


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    data = parse_instance("Instance2.txt")

    model, x = solve(data)
    print_schedule(x, data)
    validate_solution(data, x, model)
    print("Exact objective:", compute_objective_score(data, x))

    print("\n=== HEURISTIC SOLUTION ===")
    hx = heuristic_solve(data)

    print_schedule(hx, data)
    validate_solution(data, hx, None)
    print("Heuristic objective:", compute_objective_score(data, hx))