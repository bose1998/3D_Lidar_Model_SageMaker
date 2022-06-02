import json
import logging
import os
import random
import urllib

import boto3
import numpy as np
import torch
from torchvision import transforms

from models.mmdet3d.apis import inference_detector, init_model, show_result_meshlab

import cv2 as cv
#from models.C3D_altered import C3D_altered
#from models.C3D_model import C3D
#from models.my_fc6 import my_fc6
#from models.score_regressor import score_regressor
#from opts import *
import flask

app = flask.Flask(__name__)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

logger.info("Loading ProcessPredict function...")

#torch.manual_seed(randomseed)
#torch.cuda.manual_seed_all(randomseed)
#random.seed(randomseed)
#np.random.seed(randomseed)
#torch.backends.cudnn.deterministic = True

current_path = os.path.abspath(os.getcwd())

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

s3 = boto3.client("s3")


# PING check used by creation of sagemaker endpoint
@app.route('/ping', methods=['GET'])
def ping_check():
    logger.info("PING!")
    return flask.Response(response=json.dumps({"ping_status": "ok"}), status=200)


# Lambda handler executed by lambda function
@app.route('/invocations', methods=['POST', 'PUT'])
def handler():
    logger.info("Received event.")

    data = json.loads(flask.request.data.decode('utf-8'))
    # Get the object from the event and show its content type
    aqa_data = data["aqa_data"]
    bucket = aqa_data["bucket_name"]
    key = aqa_data["object_key"]
    try:
        class_names = {
        0:'car', 1:'truck', 2:'construction_vehicle', 3:'bus', 4:'trailer',5: 'barrier',
        6:'motorcycle', 7:'bicycle', 8:'pedestrian', 9:'traffic_cone'
        }
        model = init_model('models/configs/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py', 'models/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20201001_135205-5db91e00.pth', 'cuda:0')
        temp_pcd_path = f"/tmp/{key}"
        s3.download_file(bucket, key, temp_pcd_path + '/' + '0.pcd.bin')
        result, dat = inference_detector(model, temp_pcd_path)

        scores=result[0]['pts_bbox']['scores_3d']
        li_bbox=[i for i in range(len(scores)) if scores[i]>0.3]

        with open('models/frames.json') as file1:
            data=json.load(file1)

        li=[]
        coun=0
        for i in range(len(li_bbox)):
            li.append({'id': '6214a7eb6937efeeffc34162', 'datasetId': '61df2c877e42622a71758961', 'type': 'cube_3d', 'label': 'Vehicle.Truck', 'attributes': [], 'metadata': {'system': {'status': None, 'startTime': 0, 'endTime': 200, 'frame': 0, 'endFrame': 200, 'snapshots_': [], 'parentId': None, 'clientId': 'ffbf0b57-bd11-4b95-b370-124bf893a44f', 'automated': False, 'objectId': '7', 'isOpen': False, 'isOnlyLocal': False, 'frameNumberBased': True, 'attributes': {}, 'clientParentId': None, 'system': False, 'description': None, 'itemLinks': [], 'openAnnotationVersion': '1.32.3-prod.8', 'recipeId': '61d6a9d1090fd05f635b17d0'}, 'user': {}}, 'creator': 'avi@dataloop.ai', 'createdAt': '2022-02-22T09:07:55.919Z', 'updatedBy': 'diwakar.negi@tcs.com', 'updatedAt': '2022-02-24T05:02:30.337Z', 'itemId': '61e06e0467146c2a1bfd3b9b', 'url': 'https://gate.dataloop.ai/api/v1/annotations/6214a7eb6937efeeffc34162', 'item': 'https://gate.dataloop.ai/api/v1/items/61e06e0467146c2a1bfd3b9b', 'dataset': 'https://gate.dataloop.ai/api/v1/datasets/61df2c877e42622a71758961', 'hash': '317a9d67fc54c955a079ce2e7c931ff456793d21', 'source': 'ui', 'coordinates': {'position': {'x': -10.264884948730469, 'y': -13.781776428222656, 'z': -0.2731616497039795}, 'scale': {'x': 2.8543715476989746, 'y': 12.071895599365234, 'z': 3.6031875610351562}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 1.6209666942046397e-09}}})
            save=result[0]['pts_bbox']['boxes_3d'].tensor[li_bbox[i]]
            li[coun]['label']=class_names[int(result[0]['pts_bbox']['labels_3d'][li_bbox[i]])]
            li[coun]['coordinates']['position']['x']=float(save[0])
            li[coun]['coordinates']['position']['y']=float(save[1])
            li[coun]['coordinates']['position']['z']=float(save[2])
            li[coun]['coordinates']['scale']['x']=float(save[3])
            li[coun]['coordinates']['scale']['y']=float(save[4])
            li[coun]['coordinates']['scale']['z']=float(save[5])
            li[coun]['coordinates']['rotation']['x']=float(0.0)
            li[coun]['coordinates']['rotation']['y']=float(0.0)
            li[coun]['coordinates']['rotation']['z']=float(save[7])
            #li.append(data['annotations'][i])
            coun=coun+1

        data['annotations']=li
        with open('models/samp.json','w') as outfile:
            json.dump(data,outfile)

        show_result_meshlab(
            dat,
            result,
            'models/demo',
            0.3,
            show=False,
            snapshot=False,
            task='det')
        
        f=open('models/samp.json')
        response=json.dumps(json.load(f))
        return flask.Response(response=response, status=200, mimetype='application/json')

    except Exception as e:
        print(e)
        raise e
    
    

    '''try:
        # Clean up old videos if there is any
        tmp_dir = "/tmp"
        tmp_items = os.listdir(tmp_dir)
        for item in tmp_items:
            if item.endswith(".mov") or item.endswith(".avi") or item.endswith(".mp4"):
                os.remove(os.path.join(tmp_dir, item))

        temp_video_path = f"/tmp/{key}"
        s3.download_file(bucket, key, temp_video_path)
        pred_score = make_prediction(temp_video_path)
        logger.info(f"=== Prediction score: {pred_score} ===")
        response = {"prediction": pred_score}
        response = json.dumps(response)
        return flask.Response(response=response, status=200, mimetype='application/json')
    except Exception as e:
        print(e)
        raise e'''


