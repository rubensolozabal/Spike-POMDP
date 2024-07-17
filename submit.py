import datetime
from itertools import product
import sys, time
print(sys.executable)

from simple_slurm import Slurm


# Create slurm object
slurm = Slurm(
        cpus_per_task=16,
        mem='20G',
        # array=range(0, 90),
        gres='gpu:1',
        qos='ml-faculty', #'it-pool',
        partition='ml-dept-g5', #'ml-dept-p4de', #'it-hpc',
        job_name='POMDP',
        output=f'slurm_logs/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
        time=datetime.timedelta(days=0, hours=12, minutes=0, seconds=0),
        # exclude=['g512-1', 'g512-1']
    )

slurm.sbatch(f"python main.py --cfg configs/pomdp/pendulum/v/snn.yml --cuda 0")