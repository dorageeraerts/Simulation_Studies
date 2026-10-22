import uproot

DEFAULT_DATA_DIR_TEMPLATE = "/pnfs/iihe/muraves/muraves_DATA/RECONSTRUCTED/BLU/v1/"
path = DEFAULT_DATA_DIR_TEMPLATE + "MURAVES_AnalyzedData_run9679.root"

with uproot.open(path) as f:
    print(f.keys())
    tree = f["AnalyzedData"]
    print(tree.num_entries)

branches = [
    "Phi_4p",
    "Theta_4p",
    "BestTrack_4p_ChiSquare_xy",
    "BestTrack_4p_ChiSquare_xz",
    "timestamp",
    "Phi_3p",
    "Theta_3p",
]

with uproot.open(path) as f:
    tree = f["AnalyzedData"]

    for branch in branches:
        print("Testing:", branch)
        try:
            arr = tree[branch].array(library="np")
            print("  OK:", len(arr))
        except Exception as e:
            print("  FAILED:", repr(e))