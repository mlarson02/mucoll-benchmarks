#!/usr/bin/env python3
"""Stage 1 of the condor stau analysis: one reco file in, one result file out.

    python3 stauAccumulator.py --input <sample>_reco<N>.edm4hep.root \\
                               --output <sample>_acc<N>.pkl [--coned 0|1] [--index N]

Runs edm4hepStauTrackAnalyzer.process_file() on the file, i.e. the same event
loop as a direct run of the analyzer, and saves its per-file result. For stage
2, list the directory of result files in the analyzer's file_names and run it
as usual: it loads them instead of processing reco files, then merges and
plots. condor/stauacc/ runs one job per reco file.
"""
import argparse
import os
import time

import edm4hepStauTrackAnalyzer as ana


def main():
    p = argparse.ArgumentParser(description="Save edm4hepStauTrackAnalyzer's per-file result of one reco file.")
    p.add_argument("--input", required=True, help="reco .edm4hep.root file")
    p.add_argument("--output", required=True, help="result file to write (<sample>_acc<N>.pkl)")
    p.add_argument("--coned", type=int, choices=(0, 1), default=None,
                   help="1 if the reco used the coned hit collections (default: 'coned' in the input path)")
    p.add_argument("--index", type=int, default=0,
                   help="file index in its sample; only index 0 records the [DIAG] hit-collection printout")
    args = p.parse_args()

    use_coned = bool(args.coned) if args.coned is not None else "coned" in args.input.lower()
    t0 = time.time()
    acc = ana.process_file((args.input, args.index, use_coned))
    ana.save_acc(acc, args.output)
    print(f">>> {args.input}: {acc['n_events']} events -> {args.output} "
          f"({os.path.getsize(args.output) / 1e6:.1f} MB) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
