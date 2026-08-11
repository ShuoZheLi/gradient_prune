# sleep 4 hours before submitting all eval jobs
sleep 1h
for script in eval_global_step_*.sh; do
    echo "Submitting $script"
    sbatch "$script"
done
