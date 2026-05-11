import os
import glob
import openmdao.api as om
import numpy as np

folders = sorted(glob.glob("main_dymos_mission_example*_out"))

for folder in folders:
    db = os.path.join(folder, "dymos_simulation.db")
    if not os.path.exists(db):
        continue

    cr = om.CaseReader(db)
    case = cr.get_case("final")

    t = case.get_val("traj.phase0.timeseries.time").ravel()
    theta = case.get_val("traj.phase0.timeseries.theta").ravel()
    phi = case.get_val("traj.phase0.timeseries.phi").ravel()

    print(f"{folder}")
    print(f"  final time   = {t[-1]:.3f} s")
    print(f"  final theta  = {np.degrees(theta[-1]):.4f} deg")
    print(f"  final phi    = {np.degrees(phi[-1]):.4f} deg")
    print()
