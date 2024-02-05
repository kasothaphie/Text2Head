import os
import tyro
import mediapy
from PIL import Image


def main(video_folder : str):
   valid_vids = [
    ('3y6Vjr45I34_0004_S287_E1254_L568_T0_R1464_B896.mp4', 67),
    ('5crEV5DbRyc_0009_S208_E1152_L1058_T102_R1712_B756.mp4', 0),
    ('A2800grpOzU_0002_S812_E1407_L227_T7_R1139_B919.mp4', 57),
    ('EGGsK7po68c_0007_S0_E1024_L786_T50_R1598_B862.mp4', 19),
    ('jpCrKYWjYD8_0002_S0_E768_L527_T68_R1215_B756.mp4', 31),
    ('SU8NSkuBkb0_0015_S826_E1397_L347_T69_R1099_B821.mp4', 46),
   ]

   output_folder = '/mnt/hdd/debug_prep/'

   #all_videos = [s for s in os.listdir(video_folder) if s in [v[0] for v in valid_vids]]
   #print(all_videos)
   for v, video_pair in enumerate(valid_vids):
       video = video_pair[0]
       vpath = f'{video_folder}/{video}'
       out = f'{output_folder}/th_{v:03d}/source/'
       os.makedirs(out, exist_ok=True)
       images = mediapy.read_video(vpath)
       start_frame = video_pair[1]
       for i, image in enumerate(images):
           if i < start_frame:
               continue
           I = Image.fromarray(image)
           I.save(f'{out}/{i-start_frame:05d}.png')
           if i > start_frame + 150:
               break



if __name__ == '__main__':
    tyro.cli(main)