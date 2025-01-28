from numpy import empty, append

def combine_coverage_structs(coverage_structs):
    coverage_struct_combined = {}
    coverage_struct_combined["data"] = empty((0, 8))
    coverage_struct_combined["filters"] = empty((0, 1))
    coverage_struct_combined["moc"] = []
    coverage_struct_combined["telescope"] = empty((0, 1))
    coverage_struct_combined["galaxies"] = []
    coverage_struct_combined["exposureused"] = []

    for coverage_struct in coverage_structs:
        coverage_struct_combined["data"] = append(
            coverage_struct_combined["data"], coverage_struct["data"], axis=0
        )
        coverage_struct_combined["filters"] = append(
            coverage_struct_combined["filters"], coverage_struct["filters"]
        )
        coverage_struct_combined["moc"] = (
            coverage_struct_combined["moc"] + coverage_struct["moc"]
        )
        coverage_struct_combined["telescope"] = append(
            coverage_struct_combined["telescope"], coverage_struct["telescope"]
        )
        coverage_struct_combined["exposureused"] += list(
            coverage_struct["exposureused"]
        )
        if "galaxies" in coverage_struct:
            coverage_struct_combined["galaxies"] = (
                coverage_struct_combined["galaxies"] + coverage_struct["galaxies"]
            )

    return coverage_struct_combined
