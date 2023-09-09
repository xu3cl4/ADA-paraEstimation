def checkFeasibility(last2lines):
    if "SIMULATION_SUCCESSFUL" in last2lines[1]:
        return "feasible"
    else:
        if "SIMULATION_FAILED" in last2lines[1] and "timestep too small" in last2lines[0]:
            return "infeasible"
        else:
            return "other"
