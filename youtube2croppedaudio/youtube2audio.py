from __future__ import unicode_literals
import argparse
from subprocess import call
import pandas as pd
import os
from tqdm import tqdm
from joblib import Parallel, delayed


def downloader(row):
    link = row['video_link']
    if 'youtube' in link:
        try:
            output_dir = os.path.join(BASE_PATH, "raw_full", row["speaker"])
            output_path = os.path.join(BASE_PATH, "raw_full", row["speaker"], row["video_link"][-11:] + ".mp3")
            if not(os.path.exists(os.path.dirname(output_dir))):
                os.makedirs(os.path.dirname(output_dir))
            #download audio
            command = 'youtube-dl {link} -o {output_path} --extract-audio --audio-format mp3'.format(link=link, output_path=output_path)
            res1 = call(command, shell=True)
        except Exception as e:
            print(e)
    else: #using youtube-dl to download jon
        output_path = os.path.join(BASE_PATH, "raw_full", row["speaker"],str(row["video_fn"])[:-4] + ".mp3")
        print(link)
        command = 'youtube-dl {link} --extract-audio --audio-format mp3 -o "{output_path}"'.format(link=link, output_path=output_path)
        res1 = call(command, shell=True)

def save_interval(input_fn, start, end, output_fn):
    cmd = 'ffmpeg -i "%s"  -ss %s -to %s -strict -2 "%s" -y' % (
    input_fn, start, end, output_fn)
    res3 = call(cmd, shell=True)

def crop_tool(interval):
    try:
        start_time = str(pd.to_datetime(interval['start_time']).time())
        end_time = str(pd.to_datetime(interval['end_time']).time())
        video_id = (interval["video_link"][-11:])
        OUTPUT_DIR  = os.path.join(args.base_path, "raw" , interval['speaker'] + "_cropped")
        if not(os.path.isdir(OUTPUT_DIR)): os.makedirs(OUTPUT_DIR)
        if (interval["speaker"] == 'jon') and ('youtube' not in interval["video_link"]):
            video_id = (interval["video_fn"])[:-4]
        input_fn = os.path.join(args.base_path,"raw_full", interval['speaker'], video_id + ".mp3")
        output_fn = os.path.join(args.base_path, "raw" , interval['speaker'] + "_cropped", "%s.mp3"%(interval['interval_id'])) 
        if not(os.path.exists(output_fn)):
            save_interval(input_fn, str(start_time), str(end_time), output_fn)
    except Exception as e:
        print(e)
        print("couldn't crop interval: %s"%interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-base_path', '--base_path', help='base folder path of dataset')
    parser.add_argument('-speaker', '--speaker', help='download videos of a specific speaker')
    parser.add_argument('-interval_path', '--interval_path', help='path to the intervals_df file')
    args = parser.parse_args()
    BASE_PATH = args.base_path #Path which to create raw folder and has intervals_df and video_links.csv

    df = pd.read_csv(os.path.join(BASE_PATH, args.interval_path))
    if args.speaker:
        df = df[df['speaker'] == args.speaker]
    df_download = df.drop_duplicates(subset=['video_link'])
    Parallel(n_jobs=-1)(delayed(downloader)(row) for _, row in tqdm(df_download.iterrows(), total=df_download.shape[0]))
    Parallel(n_jobs=-1)(delayed(crop_tool)(interval) for _, interval in tqdm(df.iterrows(), total=df.shape[0]))
    command = "rm -r {}".format(os.path.join(BASE_PATH, "raw_full"))
    res1 = call(command, shell=True)