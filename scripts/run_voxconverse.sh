#!/bin/bash
# Cross-corpus check on VoxConverse (no training): data, our three models, pyannote, DiariZen, figures.
#   systemd-inhibit --what=sleep:idle:handle-lid-switch bash scripts/run_voxconverse.sh
# The output is also written to voxconverse.log. Systems that already have a row in the
# results table are skipped, so the script can be started again after an interruption.

DATA=data/voxconverse
RESULTS=results/results_voxconverse.csv
done_already() { grep -q "^$1," $RESULTS 2>/dev/null; }
{
    .venv/bin/python scripts/prepare_data.py --dataset diarizers-community/voxconverse --config default \
        --out $DATA --splits test --max-meetings 30 || exit 1
    rm -f $DATA/test/test_014.*   # this recording is not used: 29 recordings are evaluated

    done_already B1 || .venv/bin/python scripts/baseline_pyannote.py --name B1 --data $DATA --results $RESULTS --out results/rttm/B1_vox_test
    if ! done_already B2; then
        .venv-diarizen/bin/python scripts/baseline_diarizen.py --data $DATA --name B2_vox   # skips finished recordings
        .venv/bin/python scripts/score_rttm.py --rttm results/rttm/B2_vox_test --name B2 --data $DATA --results $RESULTS
    fi
    for run in E1=e1_scratch E2=e2_frozen E3=e3_finetune; do
        done_already ${run%%=*} || .venv/bin/python scripts/evaluate.py --checkpoint runs/${run#*=}/best.pt \
            --name ${run%%=*} --data $DATA --results $RESULTS --out results/rttm/${run%%=*}_vox_test
    done

    .venv/bin/python scripts/make_plots.py --results $RESULTS --out results/figures_voxconverse --corpus VoxConverse
    echo; echo "=== VoxConverse results ==="; cat $RESULTS
} 2>&1 | tee voxconverse.log
