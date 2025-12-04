
import numpy as np
from pymol import cmd, cgo
cmd.load("/workspaces/Graphormer/TCR_Metrics/TCR_TOOLS/geometry/test/1ao7/tmpermgiwle/tmpermgiwle_oriented.pdb","aligned_tmpermgiwle_oriented")
cmd.load("/workspaces/Graphormer/TCR_Metrics/TCR_TOOLS/geometry/test/1ao7/tmpermgiwle/consensus_oriented.pdb","consensus_geom")
cmd.bg_color("white")
cmd.hide("everything","all")

cmd.show("cartoon","aligned_tmpermgiwle_oriented")
# --- FIX: Using standard PyMOL color names ---
cmd.color("marine","aligned_tmpermgiwle_oriented and chain A")
cmd.color("teal","aligned_tmpermgiwle_oriented and chain B")

cmd.show("cartoon","consensus_geom and polymer")
# --- FIX: Using a more standard gray color ---
cmd.color("gray","consensus_geom")
cmd.set("cartoon_transparency", 0.5, "consensus_geom and polymer")

# --- FIX: Explicitly create and show spheres for the centroids ---
cmd.pseudoatom("centroid_A", pos=[np.float64(0.0), np.float64(0.0), np.float64(0.0)], color="red")
cmd.pseudoatom("centroid_B", pos=[np.float64(26.809309875084196), np.float64(0.0), np.float64(0.0)], color="orange")
cmd.pseudoatom("PCA_A1", pos=[np.float64(-15.449157799039323), np.float64(47.55337551951893), np.float64(0.0)], color="white")
cmd.pseudoatom("PCA_A2", pos=[np.float64(-42.44908235257279), np.float64(0.0), np.float64(26.42111669525899)], color="white")
cmd.pseudoatom("PCA_B1", pos=[np.float64(41.34085716172647), np.float64(21.95884635095062), np.float64(-42.50462563523436)], color="white")
cmd.pseudoatom("PCA_B2", pos=[np.float64(73.77705951022294), np.float64(17.14731740567378), np.float64(0.0)], color="white")
cmd.show("spheres", "centroid_A or centroid_B or PCA_A1 or PCA_A2 or PCA_B1 or PCA_B2")
cmd.set("sphere_scale", 0.5, "centroid_A or centroid_B or PCA_A1 or PCA_A2 or PCA_B1 or PCA_B2")

# Load CGO arrows representing the scaled PCA axes
cmd.load_cgo([
        cgo.CYLINDER,0.000,0.000,0.000,
                     -15.449,47.553,0.000,
                     0.3,
                     0.2,0.5,1.0,
                     0.2,0.5,1.0,
        cgo.CONE,-15.449,47.553,0.000,
                 0.000,0.000,0.000,
                 0.44999999999999996,0.0,
                 0.2,0.5,1.0,
                 0.2,0.5,1.0,1.0
    ], "PC1_A")
cmd.load_cgo([
        cgo.CYLINDER,0.000,0.000,0.000,
                     -42.449,0.000,26.421,
                     0.3,
                     0.1,0.8,0.1,
                     0.1,0.8,0.1,
        cgo.CONE,-42.449,0.000,26.421,
                 0.000,0.000,0.000,
                 0.44999999999999996,0.0,
                 0.1,0.8,0.1,
                 0.1,0.8,0.1,1.0
    ], "PC2_A")
cmd.load_cgo([
        cgo.CYLINDER,26.809,0.000,0.000,
                     41.341,21.959,-42.505,
                     0.3,
                     1.0,0.5,0.2,
                     1.0,0.5,0.2,
        cgo.CONE,41.341,21.959,-42.505,
                 26.809,0.000,0.000,
                 0.44999999999999996,0.0,
                 1.0,0.5,0.2,
                 1.0,0.5,0.2,1.0
    ], "PC1_B")
cmd.load_cgo([
        cgo.CYLINDER,26.809,0.000,0.000,
                     73.777,17.147,0.000,
                     0.3,
                     0.8,0.8,0.1,
                     0.8,0.8,0.1,
        cgo.CONE,73.777,17.147,0.000,
                 26.809,0.000,0.000,
                 0.44999999999999996,0.0,
                 0.8,0.8,0.1,
                 0.8,0.8,0.1,1.0
    ], "PC2_B")
cmd.load_cgo([
        cgo.CYLINDER,0.000,0.000,0.000,
                     26.809,0.000,0.000,
                     0.3,
                     0.5,0.0,0.5,
                     0.5,0.0,0.5,
        cgo.CONE,26.809,0.000,0.000,
                 0.000,0.000,0.000,
                 0.44999999999999996,0.0,
                 0.5,0.0,0.5,
                 0.5,0.0,0.5,1.0
    ],"dc")


# --- measurements (wizard-equivalent) ---
cmd.distance("dc_len", "centroid_B", "centroid_A")                 # distance dc
cmd.angle("BC1_ang", "PCA_B1", "centroid_B", "centroid_A")         # angle BC1
cmd.angle("BC2_ang", "PCA_B2", "centroid_B", "centroid_A")         # angle BC2
cmd.angle("AC1_ang", "PCA_A1", "centroid_A", "centroid_B")         # angle AC1
cmd.angle("AC2_ang", "PCA_A2", "centroid_A", "centroid_B")         # angle AC2
cmd.dihedral("BA_dih", "PCA_B1", "centroid_B", "centroid_A", "PCA_A1")  # dihedral BA

# Ensure measurement objects are visible
cmd.enable("dc_len")
cmd.enable("BC1_ang")
cmd.enable("BC2_ang")
cmd.enable("AC1_ang")
cmd.enable("AC2_ang")
cmd.enable("BA_dih")

# Global styling for measurement dashes & labels (applies to all three)
cmd.set("dash_width", 3.0)
cmd.set("dash_gap", 0.0)
cmd.set("label_size", 18)
cmd.set("label_color", "black")
cmd.set("label_distance_digits", 2)  # for distances
cmd.set("label_angle_digits", 1)     # for angles/dihedrals

cmd.orient()
cmd.zoom("all", 1.2)
cmd.png("/workspaces/Graphormer/TCR_Metrics/TCR_TOOLS/geometry/test/1ao7/tmpermgiwle/vis/tmpermgiwle_final_vis.png", dpi=300, ray=1)
cmd.save("/workspaces/Graphormer/TCR_Metrics/TCR_TOOLS/geometry/test/1ao7/tmpermgiwle/vis/tmpermgiwle_final_vis.pse")
cmd.quit()
