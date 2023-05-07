# not comprehensive
# load result and result2 separate ways and compare

def main():
    from GPFramework.analysis import analyze

    result = analyze.EmadeOutput(out_json="", xml_path="", out_path="", emade_path="", db=False)
    result2= analyze.EmadeOutput(out_json="", xml_path="", out_path="", emade_path="", db=False)

    assert result2.inds==result.inds
    assert result2.gens==result.gens
    assert result2.gens_complete==result.gens_complete
    assert result2.pareto==result.pareto
    assert result2.pareto_history==result.pareto_history
    assert result2.auc==result.auc
    assert result2.auc_history==result.auc_history
    assert result2.objectives==result.objectives
    assert result2.obj_ranges==result.obj_ranges

if __name__ == "__main__":
    main()