sleep 9h

for script in eval_global_step_*.sh; do
    echo "Submitting $script"
    sbatch "$script"
done