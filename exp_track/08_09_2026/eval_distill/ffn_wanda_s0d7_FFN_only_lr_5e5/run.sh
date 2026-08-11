# sleep 3 hours
for step in 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000 1050 1100 1150 1200 1250 1300 1320; do
    script="eval_global_step_${step}.sh"
    echo "Submitting $script"
    sbatch "$script"
done
