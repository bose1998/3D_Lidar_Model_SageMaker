# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import json
from mmdet3d.apis import inference_detector, init_model, show_result_meshlab


def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()
    class_names = {
    0:'car', 1:'truck', 2:'construction_vehicle', 3:'bus', 4:'trailer',5: 'barrier',
    6:'motorcycle', 7:'bicycle', 8:'pedestrian', 9:'traffic_cone'
    }
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    result, dat = inference_detector(model, args.pcd)
    # show the results
    #print('initial_results')
    #print(result)
    #print(result[0]['pts_bbox']['boxes_3d'].tensor[0])
    scores=result[0]['pts_bbox']['scores_3d']
    li_bbox=[i for i in range(len(scores)) if scores[i]>args.score_thr]
    """for i in li_bbox:
      print(result[0]['pts_bbox']['boxes_3d'].tensor[i])"""
    with open('frames.json') as file1:
      data=json.load(file1)
    #print(result[0]['pts_bbox']['boxes_3d'].tensor[0])
    #print(len(data['annotations']))
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
      #print(data['annotations'][i])
    data['annotations']=li
    with open('samp.json','w') as outfile:
      json.dump(data,outfile)

    show_result_meshlab(
        dat,
        result,
        args.out_dir,
        args.score_thr,
        show=args.show,
        snapshot=args.snapshot,
        task='det')


if __name__ == '__main__':
    main()
