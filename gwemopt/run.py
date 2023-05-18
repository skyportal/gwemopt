import os
from pathlib import Path

import numpy as np

import gwemopt.coverage
import gwemopt.efficiency
import gwemopt.lightcurve
import gwemopt.plotting
import gwemopt.segments
from gwemopt.args import parse_args
from gwemopt.catalogs import get_catalog
from gwemopt.io import get_skymap, read_skymap, summary
from gwemopt.params import params_struct
from gwemopt.paths import DEFAULT_BASE_OUTPUT_DIR
from gwemopt.plotting import (
    make_coverage_plots,
    make_efficiency_plots,
    make_tile_plots,
    plot_observability,
    plot_skymap,
)
from gwemopt.utils import calculate_observability


def run(args):
    args = parse_args(args)

    params = params_struct(args)

    if len(params["filters"]) != len(params["exposuretimes"]):
        raise ValueError(
            "The number of filters specified must match the number of exposure times."
        )

    params["skymap"] = get_skymap(event_name=args.event)

    # Function to read maps
    params, map_struct = read_skymap(params)

    # Set output directory
    if args.outputDir is not None:
        output_dir = Path(args.outputDir)
    else:
        output_dir = DEFAULT_BASE_OUTPUT_DIR.joinpath(
            f"{params['name']}/{'+'.join(params['telescopes'])}/"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    params["outputDir"] = output_dir
    print(f"Output directory: {output_dir}")

    params = gwemopt.segments.get_telescope_segments(params)

    print("Loading skymap...")

    if params["catalog"] is not None:
        print("Generating catalog...")
        map_struct, catalog_struct = get_catalog(params, map_struct)
    else:
        catalog_struct = None

    if args.doPlots:
        print("Plotting skymap...")
        plot_skymap(params, map_struct)

    if args.doObservability:
        print("Calculating observability")
        observability_struct = calculate_observability(params, map_struct)
        map_struct["observability"] = observability_struct
        if args.doPlots:
            print("Plotting observability...")
            plot_observability(params, map_struct)

    if params["splitType"] is not None:
        print("Splitting skymap...")
        map_struct["groups"] = gwemopt.mapsplit.similar_range(params, map_struct)

    if args.doTiles:
        if params["tilesType"] == "moc":
            print("Generating MOC struct...")
            moc_structs = gwemopt.moc.create_moc(params, map_struct=map_struct)
            print("Generating tile struct for MOC...")
            tile_structs = gwemopt.tiles.moc(params, map_struct, moc_structs)

        elif params["tilesType"] == "galaxy":
            print("Generating galaxy struct...")
            tile_structs = gwemopt.tiles.galaxy(params, map_struct, catalog_struct)
            for telescope in params["telescopes"]:
                params["config"][telescope]["tesselation"] = np.empty((0, 3))
                tiles_struct = tile_structs[telescope]
                for index in tiles_struct.keys():
                    ra, dec = tiles_struct[index]["ra"], tiles_struct[index]["dec"]
                    params["config"][telescope]["tesselation"] = np.append(
                        params["config"][telescope]["tesselation"],
                        [[index, ra, dec]],
                        axis=0,
                    )
        else:
            raise ValueError(f"Unknown tilesType: {params['tilesType']}")

        if args.doPlots:
            print("Plotting tiles struct...")
            make_tile_plots(params, map_struct, tile_structs)

    if args.doSchedule:
        if args.doTiles:
            print("Generating coverage...")
            tile_structs, coverage_struct = gwemopt.coverage.timeallocation(
                params, map_struct, tile_structs
            )
        else:
            print("Need to enable --doTiles to use --doSchedule")
            exit(0)
    elif args.doCoverage:
        print("Reading coverage from file...")
        coverage_struct = gwemopt.coverage.read_coverage_files(
            params, moc_structs=moc_structs
        )

    if args.doSchedule or args.doCoverage:
        print("Summary of coverage...")
        summary(params, map_struct, coverage_struct, catalog_struct=catalog_struct)

        print("Plotting coverage...")
        make_coverage_plots(
            params, map_struct, coverage_struct, catalog_struct=catalog_struct
        )

    if args.doEfficiency:
        if args.doSchedule or args.doCoverage:
            print("Computing efficiency...")
            if args.modelType == "file":
                lightcurve_structs = gwemopt.lightcurve.read_files(
                    params["lightcurveFiles"]
                )
            elif args.modelType == "Tophat":
                lightcurve_structs = gwemopt.lightcurve.tophat(
                    mag0=args.mag, dmag=args.dmag
                )
            efficiency_structs = {}
            for key in lightcurve_structs.keys():
                lightcurve_struct = lightcurve_structs[key]
                efficiency_struct = gwemopt.efficiency.compute_efficiency(
                    params,
                    map_struct,
                    lightcurve_struct,
                    coverage_struct,
                )
                efficiency_structs[key] = efficiency_struct
                efficiency_structs[key]["legend_label"] = lightcurve_struct[
                    "legend_label"
                ]
                if params["do_3d"]:
                    print(
                        f'Percent detections out of {params["Ninj"]} injected KNe: '
                        f'{efficiency_structs[key]["3D"]*100:.2f}% '
                    )

            if args.doPlots:
                print("Plotting efficiency...")
                make_efficiency_plots(params, map_struct, efficiency_structs)
        else:
            print("Need to enable --doSchedule or --doCoverage for --doEfficiency")
            exit(0)
