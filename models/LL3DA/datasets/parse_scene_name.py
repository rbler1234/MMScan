import json
from datasets.mmscan_config import embodied_scan_info_base
import os.path as osp

mp3d_mapping = json.load(open(osp.join(embodied_scan_info_base,"mp3d_mapping.json")))
trscan_mapping =  json.load(open(osp.join(embodied_scan_info_base,"3rscan_mapping.json")))

def map3rscan2origin(scan_id):
    if "3rscan" in scan_id:
        for key in trscan_mapping:
            if trscan_mapping[key]==scan_id:
                return key
    else:
        return scan_id
    
def parse_embodied_scan(scan_id):
    if "matterport3d" in scan_id:
        # return f"{mp3d_mapping[scan_id.splot('/')[-2]]}_{scan_id.splot('/')[-1]}"
        return scan_id.split('/')[-1]
    elif "3rscan" in scan_id:
        return scan_id.split('/')[-1]
        # return trscan_mapping(scan_id.split('/')[-1])
    elif "scannet" in scan_id:
        return scan_id.split('/')[-1]
    else:
        raise NotImplementedError

