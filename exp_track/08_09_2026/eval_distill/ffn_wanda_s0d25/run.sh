# sleep 1.5 hours
sleep 5.5h
for script in eval_global_step_*.sh; do
    echo "Submitting $script"
    sbatch "$script"
done
