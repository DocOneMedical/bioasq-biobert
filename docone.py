import json
import pandas as pd


NO_TOKEN = "no"
YES_TOKEN = "yes"


def json_to_csv(json_file: str, output_csv: str):
    with open(json_file, 'r') as jhandle:
        d = json.load(jhandle)

    final_dict = {}
    for key, val in d.items():
        print(key)
        article_id, rx_code, mesh_id = key.split('_')[:3]
        truth_val = True if val[0] == YES_TOKEN else False
        key = (article_id, rx_code, mesh_id)
      
        if truth_val:
            final_dict[key] = truth_val
        elif key not in final_dict:
            final_dict[key] = truth_val

    df = []
    for (article_id, rx_code, mesh_id), related in final_dict.items():
        df.append({
            'article_id': article_id, 
            'rx_code': rx_code, 
            'mesh_id': mesh_id, 
            'related': related
        })

    pd.DataFrame(df).to_csv(output_csv, index=False) 

if __name__ == '__main__':
    json_to_csv('outputs/testing/predictions/predictions_0.json', 'temp.csv')
