import json
import pandas as pd

import tensorflow as tf


NO_TOKEN = "no"
YES_TOKEN = "yes"


def condense_files_to_csv(json_dir, output_csv):
    files = tf.io.gfile.glob(f"{json_dir}/*.json")
    final_dict = {}
    for f in files:
        update_dict(f, final_dict)

    df = []
    for (article_id, rx_code, mesh_id), related in final_dict.items():
        df.append({
            'article_id': article_id,
            'rx_code': rx_code,
            'mesh_id': mesh_id,
            'related': related
        })

    if df:
        pd.DataFrame(df).sort_values(['article_id', 'rx_code', 'mesh_id']) \
                        .reindex(columns=['article_id', 'rx_code', 'mesh_id', 'related']) \
                        .to_csv(output_csv, index=False)


def update_dict(json_file: str, final_dict):
    with open(json_file, 'r') as jhandle:
        d = json.load(jhandle)

    for key, val in d.items():
        article_id, rx_code, mesh_id = key.split('_')[:3]
        truth_val = True if val[0] == YES_TOKEN else False
        key = (article_id, rx_code, mesh_id)
  
        if truth_val:
            final_dict[key] = truth_val
        elif key not in final_dict:
            final_dict[key] = truth_val


if __name__ == '__main__':
    condense_files_to_csv('outputs/predictions', 'sample.csv')
