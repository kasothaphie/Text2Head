import os
import tyro


def main(start_id : int, end_id : int):
    for pid in range(start_id, end_id):
        cmd_prep = f'cd scripts/preprocessing ; ./run.sh {pid} --no-intrinsics_provided ; cd ../..'
        cmd_track = f'python scripts/inference/track.py --exp_name PretrainedMonoNPHM --ckpt 2500 --seq_tag {pid} --run_tag first_try --lambda_normals 5.2'
        os.system(cmd_prep)
        os.system(cmd_track)

if __name__ == '__main__':
    tyro.cli(main)