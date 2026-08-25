import numpy as np
import matplotlib.pyplot as plt
import mulder
import argparse
import numpy.ma as ma
import os
from matplotlib.ticker import FuncFormatter

parser = argparse.ArgumentParser()

parser.add_argument("--phi-min", type=float, default=0.)
parser.add_argument("--phi-max", type=float, default=360.)
#parser.add_argument("--tag", type=str, required=True)

parser.add_argument("--rho", type=float, default=2.65E+03)

parser.add_argument("--el-min", type=float, default=0.)
parser.add_argument("--el-max", type=float, default=40.)

parser.add_argument("--d-phi", type=float, default=1.)
parser.add_argument("--d-el", type=float, default=1.)

parser.add_argument("--output-path",
                    type=str,
                    default="./images/rock_thickness")
parser.add_argument("--input-path", type=str, default="./")

parser.add_argument("--shift", action="store_true", help="Shift azimuth labels by +136 degrees")

args = parser.parse_args()

# ==============================================================
# Geometry (fixed CRS from before)
# ==============================================================
geometry = mulder.EarthGeometry(
    mulder.Layer(
        mulder.Grid(args.input_path + "vesuvio_5m_cut.asc", crs=32633),
        density=args.rho,
        material="Rock",
    ),
)

'''geometry = mulder.EarthGeometry(
    mulder.Layer(
        args.input_path + "GMRT.asc",
        density=args.rho,
        material="Rock",
    ),)'''

# ==============================================================
# Detector viewpoint
# ==============================================================
latitude = 40.810251
longitude = 14.411708
rock = geometry.layers[0]
altitude = rock.altitude(latitude, longitude)

az = np.arange(args.phi_min, args.phi_max, args.d_phi)   
el = np.arange(args.el_min, args.el_max, args.d_el)                    
AZ, EL = np.meshgrid(az, el)   

thickness = geometry.scan(
    latitude=latitude,
    longitude=longitude,
    altitude=altitude,
    azimuth=AZ,
    elevation=EL,
    output="thickness",
)[..., 0]

os.makedirs(
    args.output_path,
    exist_ok=True
)
outfile = os.path.join(
    args.output_path,
    f"rock_thickness_{args.d_phi}bin_{args.phi_min}-{args.phi_max}phi_{args.el_min}-{args.el_max}el.png"
)

# Mask air as white
masked_thickness = ma.masked_where(thickness < 1e-6, thickness)

cmap = plt.cm.inferno.copy()
cmap.set_bad(color="white")   # color for masked (air) pixels

fig, ax = plt.subplots(figsize=(10, 6))

im = ax.imshow(
    masked_thickness,
    origin="lower",
    extent=[az.min(), az.max(), el.min(), el.max()],
    aspect="auto",
    cmap=cmap,
)

if args.shift:
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{(x + 136) % 360:.0f}")
    )

fig.colorbar(im, ax=ax, label="Rock thickness [m]")
ax.set_xlabel("Azimuth [deg]", loc="right")
ax.set_ylabel("Elevation [deg]", loc="top")
ax.set_title("Vesuvius DEM rock thickness map")
ax.minorticks_on()

fig.savefig(outfile, dpi=150)
plt.show()
