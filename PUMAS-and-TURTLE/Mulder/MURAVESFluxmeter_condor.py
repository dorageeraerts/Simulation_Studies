import os
import argparse
import numpy as np
import matplotlib.pyplot as plot
from mulder import Reference, Fluxmeter, Position, Direction, Geometry, State, Layer


parser = argparse.ArgumentParser()
parser.add_argument("--phi-min", type=float, required=True)
parser.add_argument("--phi-max", type=float, required=True)
parser.add_argument("--tag", type=str, required=True)
parser.add_argument("--rho", type=float, default=2.65E+03)
args = parser.parse_args()


# define geometry
geometry = Geometry(
    Layer(
        material="Rock",
        density=args.rho,
        model="../../mulder/data/5mNew_projection.png"
    ),
)

# create a fluxmeter
fluxmeter = Fluxmeter(geometry)

# by default, the fluxmeter is configured in "continuous" mode. "discrete" and "mixed"
fluxmeter.mode = "discrete"

# altitude correction, tabulation to MCEq results
fluxmeter.reference = "../../mulder/data/flux-mceq-yfm-gsf-usstd.table"


# observation point
rock = geometry.layers[0]
xMURAVES = 40.810251
yMURAVES = 14.411708

proj = rock.project(latitude=xMURAVES, longitude=yMURAVES, height=0.0)
topo = rock.height(x=proj.x, y=proj.y)


# energy spectrum
E_min = 0.9
E_max = 3000.0
n_E   = 10000

energy = np.logspace(np.log10(E_min), np.log10(E_max), n_E)


# Angular grid
phi_min = args.phi_min
phi_max = args.phi_max
dphi    = 0.20

el_min  = 0.0
el_max  = 40.0
del_el  = 0.20

az_vals = np.arange(phi_min, phi_max, dphi)
el_vals = np.arange(el_min, el_max, del_el)

n_az = len(az_vals)
n_el = len(el_vals)

print(f"Running azimuth range: {phi_min} to {phi_max}", flush=True)
print(f"Number of azimuth bins: {n_az}", flush=True)
print(f"Number of elevation bins: {n_el}", flush=True)
print(f"Total angular bins in this job: {n_az * n_el}", flush=True)


rho = args.rho
geometry.layers[0].density = rho

outdir = "relativeApproach/condor_parts"
os.makedirs(outdir, exist_ok=True)

outfile = os.path.join(
    outdir,
    f"flux_5m_02bin_rho{rho}_discrete_200T_{args.tag}.txt"
)

N_EVENTS = 100

with open(outfile, "w") as ffile:
    for i, az in enumerate(az_vals):
        for j, el in enumerate(el_vals):

            s_obs = State(
                latitude  = xMURAVES,
                longitude = yMURAVES,
                height    = topo,
                azimuth   = az,
                elevation = el,
                energy    = energy
            )

            s_ref = fluxmeter.transport(s_obs, events=N_EVENTS)
            flux_mc = s_ref.flux(fluxmeter.reference)
            values = np.asarray(flux_mc.value)

            values_shape = values.reshape(n_E, N_EVENTS)
            # Integrated flux for each Monte Carlo replica
            flux_int = np.trapz(values_shape, energy, axis=0)
            # Monte Carlo uncertainty on the integrated flux
            flux_err = np.std(flux_int, ddof=1) / np.sqrt(flux_int.size)

            # Final flux estimate
            flux_ave = np.mean(flux_int)

            ffile.write(
                f"{az + 136} {el} {flux_ave:.6e} {flux_err:.6e} {flux_int.size}\n"
            )

            print(
                f"[{args.tag}] rho={rho} az={az:.2f} el={el:.2f} "
                f"bin={i * n_el + j + 1}/{n_az * n_el}",
                flush=True
            )

print(f"Finished. Output written to: {outfile}", flush=True)