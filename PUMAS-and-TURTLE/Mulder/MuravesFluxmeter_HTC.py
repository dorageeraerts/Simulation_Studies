'''updated for Mulder version 0.3.7'''
import os
import argparse
import numpy as np
import matplotlib.pyplot as plot
import mulder


parser = argparse.ArgumentParser()
parser.add_argument("--phi-min", type=float, required=True)
parser.add_argument("--phi-max", type=float, required=True)
parser.add_argument("--tag", type=str, required=True)
parser.add_argument("--rho", type=float, default=2.65E+03)
parser.add_argument("--el-min", type=float, required=True)
parser.add_argument("--el-max", type=float, required=True)
parser.add_argument("--n-E", type=int, default=10000)
parser.add_argument("--nEvents", type=int, default=1)
parser.add_argument("--d-phi", type=float, default=1.)
parser.add_argument("--d-el", type=float, default=1.)
parser.add_argument("--input-path", type=str, default="./")
parser.add_argument("--output-path", type=str, default="./mulder_output")
args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path

# define geometry
# NOTE: Layer's `data` argument is now positional (no `model=` keyword).
geometry = mulder.EarthGeometry(
    mulder.Layer(
        "GMRT.asc",
        density=args.rho,
        material="Rock",
    ),
)

# create a fluxmeter
fluxmeter = mulder.Fluxmeter(geometry=geometry)

# by default, the fluxmeter is configured in "continuous" mode. "discrete" and "mixed"
fluxmeter.mode = "discrete"

# altitude correction, tabulation to MCEq results
fluxmeter.reference = "flux-mceq-yfm-gsf-usstd.table"#"Gaisser90" #"../../mulder/data/flux-mceq-yfm-gsf-usstd.table"


# observation point
rock = geometry.layers[0]
xMURAVES = 40.810251
yMURAVES = 14.411708

# NOTE: rock.project() + rock.height() collapse into a single altitude() call
# in 0.3.7, working directly in geographic (lat/lon) coordinates.
topo = rock.altitude(xMURAVES, yMURAVES)


# energy spectrum
E_min = 0.9
E_max = 3000.0
n_E   = args.n_E

energy = np.logspace(np.log10(E_min), np.log10(E_max), n_E)


# Angular grid
phi_min = args.phi_min
phi_max = args.phi_max
dphi    = args.d_phi

el_min  = args.el_min
el_max  = args.el_max
del_el  = args.d_el

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

outdir = output_path #"relativeApproach/condor_parts"
os.makedirs(outdir, exist_ok=True)

outfile = os.path.join(
    outdir,
    f"flux_5m_02bin_rho{rho}_discrete_200T_{args.tag}.txt"
)

N_EVENTS = args.nEvents

with open(outfile, "w") as ffile:
    for i, az in enumerate(az_vals):
        for j, el in enumerate(el_vals):

            # NOTE: State -> GeographicStates, height -> altitude
            s_obs = mulder.GeographicStates(
                latitude  = xMURAVES,
                longitude = yMURAVES,
                altitude  = topo,
                azimuth   = az,
                elevation = el,
                energy    = energy,
            )

            #s_ref = fluxmeter.transport(s_obs, events=N_EVENTS)

            # NOTE: flux() now lives on Reference, not on the transported
            # states. It also returns a plain array, not an object with
            # a `.value` attribute.
            #flux_mc = fluxmeter.reference.flux(s_ref)
            #values = np.asarray(flux_mc)

            s_ref = fluxmeter.transport(s_obs, events=N_EVENTS)
            flux_ref = np.asarray(fluxmeter.reference.flux(s_ref))
            weight   = np.asarray(s_ref.weight)
            values   = flux_ref * weight

            #print("flux_ref (no weight):", flux_ref[:3])
            #print("weight:              ", weight[:3])
            #print("flux_ref * weight:    ", (flux_ref * weight)[:3])
            #print("fluxmeter.reference.elevation", fluxmeter.reference.elevation)
            #print("transported energies:", np.asarray(s_ref.energy)[:3])

            values_shape = values.reshape(n_E, N_EVENTS)

            # VERIFY BEFORE TRUSTING: confirm this reshape matches the
            # actual shape returned by transport()/flux() for a vector
            # `energy` input combined with `events=N_EVENTS`. Uncomment
            # the print below once, check the shape, then adjust the
            # reshape (or remove it) accordingly.
            #print(values.shape); import sys; sys.exit(0)

            values_shape = values.reshape(n_E, N_EVENTS)
            # Integrated flux for each Monte Carlo replica
            flux_int = np.trapezoid(values_shape, energy, axis=0)
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