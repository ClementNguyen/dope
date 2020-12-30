import os
import sys
import argparse
import os.path as osp
import cv2
import numpy as np
import torch
import gzip
import pickle
import copy
from torchvision.transforms import ToTensor

from lcr_net_ppi_improved import LCRNet_PPI_improved
from model import dope_resnet50
import postprocess



if __name__=="__main__":

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    modelname = 'DOPE_v1_0_0'

    # load model
    ckpt_fname = osp.join('models', modelname + '.pth.tgz')
    print('Loading model', modelname)
    ckpt = torch.load(ckpt_fname, map_location=device)
    # ckpt['half'] = False # uncomment this line in case your device cannot handle half computation
    ckpt['dope_kwargs']['rpn_post_nms_top_n_test'] = 1000
    model = dope_resnet50(**ckpt['dope_kwargs'])
    if ckpt['half']: model = model.half()
    model = model.eval()
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)



    # Load videos
    vid_dir = '/home/clement_ngn/slt/data/videos'
    set_names = ['dev', 'test', 'train']

    for set_name in set_names:

        video_list = os.listdir(vid_dir + '/' + set_name)

        #
        n_keypoints = {'body': 13, 'face': 84, 'hand': 21}

        # load features
        filename = '/home/clement_ngn/slt/data/PHOENIX2014T/phoenix14t.pami0.' + set_name
        with gzip.open(filename, "rb") as f:
            features = pickle.load(f)
        name_list = [features[i]['name'] for i in range(len(features))]

    #    outputs_keypoints = []
    #    outputs_feat = []

        for video_idx, video in enumerate(video_list):

            print('Computing output for video {} / {}'.format(video_idx + 1, len(video_list)))

            # copy features
            output_keypoints = dict()
            output_keypoints['name'] = set_name + '/' + video[:-4]
            idx = name_list.index(output_keypoints['name'])
            output_keypoints['gloss'] = features[idx]['gloss']
            output_keypoints['text'] = features[idx]['text']
            output_keypoints['signer'] = features[idx]['signer']

            output_feat = copy.deepcopy(output_keypoints)

            # Extract frames from video
            vidcap = cv2.VideoCapture(vid_dir + '/' + set_name + '/' + video)
            frames = []
            success, frame = vidcap.read()
            frames.append(frame)
            while success:
                success, frame = vidcap.read()
                if success:
                    frames.append(frame)

            # DOPE output
            imlist = [ToTensor()(frame).to(device) for frame in frames]
            resolution = imlist[0].size()[-2:]

            processed_results = {'body_2d': [], 'body_3d': [], 'face_2d': [], 'face_3d': [],
                                 'left_hand_2d': [], 'left_hand_3d': [], 'right_hand_2d': [], 'right_hand_3d': []}
            tmp_feat = {
                'body_feat': [],
                'body_scores': [],
                'body_2d': [],
                'body_3d': [],
                'face_feat': [],
                'face_scores': [],
                'face_2d': [],
                'face_3d': [],
                'hand_feat_1': [],
                'hand_scores_1': [],
                'hand_2d_1': [],
                'hand_3d_1': [],
                'hand_feat_2': [],
                'hand_scores_2': [],
                'hand_2d_2': [],
                'hand_3d_2': []
            }

            for image in imlist:

                with torch.no_grad():
                    result = model([image], None)[0]

                # Get final keypoints

                # postprocess results (pose proposals integration, wrists/head assignment)

                parts = ['body', 'hand', 'face']

                res = {k: v.float().data.cpu().numpy() for k, v in result.items()}
                detections = {}
                for part in parts:
                    detections[part] = LCRNet_PPI_improved(res[part + '_scores'], res['boxes'], res[part + '_pose2d'],
                                                           res[part + '_pose3d'], resolution, **ckpt[part + '_ppi_kwargs'])

                # assignment of hands and head to body
                detections, body_with_wrists, body_with_head = postprocess.assign_hands_and_head_to_body(detections)

                for part in ['body', 'face']:
                    if len(detections[part]) == 0:
                        processed_results[part + '_2d'].append(torch.zeros((n_keypoints[part] * 2)))
                        processed_results[part + '_3d'].append(torch.zeros((n_keypoints[part] * 3)))
                    else:
                        processed_results[part + '_2d'].append(torch.tensor(detections[part][0]['pose2d']).flatten())
                        processed_results[part + '_3d'].append(torch.tensor(detections[part][0]['pose3d']).flatten())

                # hand
                left_hand_2d = torch.zeros((n_keypoints['hand'] * 2))
                right_hand_2d = torch.zeros((n_keypoints['hand'] * 2))
                left_hand_3d = torch.zeros((n_keypoints['hand'] * 3))
                right_hand_3d = torch.zeros((n_keypoints['hand'] * 3))

                right_hand_done, left_hand_done = False, False
                for i in range(len(detections['hand'])):
                    hand_i = detections['hand'][i]
                    if hand_i['hand_isright'] and not right_hand_done:
                        processed_results['right_hand_2d'].append(torch.tensor(hand_i['pose2d']).flatten())
                        processed_results['right_hand_3d'].append(torch.tensor(hand_i['pose3d']).flatten())
                        right_hand_done = True
                    elif not hand_i['hand_isright'] and not left_hand_done:
                        processed_results['left_hand_2d'].append(torch.tensor(hand_i['pose2d']).flatten())
                        processed_results['left_hand_3d'].append(torch.tensor(hand_i['pose3d']).flatten())
                        left_hand_done = True
                if not right_hand_done:
                    processed_results['right_hand_2d'].append(right_hand_2d)
                    processed_results['right_hand_3d'].append(right_hand_3d)
                if not left_hand_done:
                    processed_results['left_hand_2d'].append(left_hand_2d)
                    processed_results['left_hand_3d'].append(left_hand_3d)

                # Get features and model predictions

                pooled_features = model.roi_heads.pooled_features

                # body features
                body_scores = 1 - result['body_scores'][:, 0]
                best_body_roi_idx = torch.argmax(body_scores)
                body_features = pooled_features[best_body_roi_idx]
                body_2d_pred_from_feat = result['body_pose2d'][best_body_roi_idx]
                body_3d_pred_from_feat = result['body_pose3d'][best_body_roi_idx]
                tmp_feat['body_feat'].append(body_features)
                tmp_feat['body_scores'].append(result['body_scores'][best_body_roi_idx])
                tmp_feat['body_2d'].append(body_2d_pred_from_feat.flatten())
                tmp_feat['body_3d'].append(body_3d_pred_from_feat.flatten())

                # face features
                face_scores = 1 - result['face_scores'][:, 0]
                best_face_roi_idx = torch.argmax(face_scores)
                face_features = pooled_features[best_face_roi_idx]
                face_2d_pred_from_feat = result['face_pose2d'][best_face_roi_idx]
                face_3d_pred_from_feat = result['face_pose3d'][best_face_roi_idx]
                tmp_feat['face_feat'].append(face_features)
                tmp_feat['face_scores'].append(result['face_scores'][best_face_roi_idx])
                tmp_feat['face_2d'].append(face_2d_pred_from_feat.flatten())
                tmp_feat['face_3d'].append(face_3d_pred_from_feat.flatten())

                # hand features
                dets, indices, bestcls = postprocess.DOPE_NMS(result['hand_scores'], result['boxes'], result['hand_pose2d'],
                                                              result['hand_pose3d'], min_score=0.)
                hand_features = pooled_features[indices, :]
                hand_scores = 1 - result['hand_scores'][indices, 0]
                sorted, indices_sort = torch.sort(hand_scores, descending=True)
                best_hand_roi_idx_1 = indices_sort[0]
                best_hand_roi_idx_2 = indices_sort[1]
                hand_features_1 = hand_features[best_hand_roi_idx_1]
                hand_features_2 = hand_features[best_hand_roi_idx_2]
                hand_2d_pred_from_feat_1 = (result['hand_pose2d'][indices])[best_hand_roi_idx_1]
                hand_3d_pred_from_feat_1 = (result['hand_pose3d'][indices])[best_hand_roi_idx_1]
                hand_2d_pred_from_feat_2 = (result['hand_pose2d'][indices])[best_hand_roi_idx_2]
                hand_3d_pred_from_feat_2 = (result['hand_pose3d'][indices])[best_hand_roi_idx_2]
                tmp_feat['hand_feat_1'].append(hand_features_1)
                tmp_feat['hand_scores_1'].append((result['hand_scores'][indices, :])[best_hand_roi_idx_1])
                tmp_feat['hand_2d_1'].append(hand_2d_pred_from_feat_1.flatten())
                tmp_feat['hand_3d_1'].append(hand_3d_pred_from_feat_1.flatten())
                tmp_feat['hand_feat_2'].append(hand_features_2)
                tmp_feat['hand_scores_2'].append((result['hand_scores'][indices, :])[best_hand_roi_idx_2])
                tmp_feat['hand_2d_2'].append(hand_2d_pred_from_feat_2.flatten())
                tmp_feat['hand_3d_2'].append(hand_3d_pred_from_feat_2.flatten())

                # for key in list(tmp_feat.keys())[4:]:
                #   tmp_feat[key] = tmp_feat[key].cpu().detach()

            for key in tmp_feat.keys():
                output_feat[key] = torch.stack(tmp_feat[key])
                output_feat[key] = output_feat[key].cpu().detach()
#           outputs_feat.append(output_feat)

            for key in processed_results.keys():
                output_keypoints[key] = torch.stack(processed_results[key])
                output_keypoints[key] = output_keypoints[key].cpu().detach()
#           outputs_keypoints.append(output_keypoints)

            with open('/home/clement_ngn/slt/data/dope_outputs/pose_estimation/' + set_name + '/' + video[:-4]+'.p', 'wb') as handle:
                pickle.dump(output_keypoints, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('/home/clement_ngn/slt/data/dope_outputs/features/' + set_name + '/' + video[:-4]+'.p', 'wb') as handle:
                pickle.dump(output_feat, handle, protocol=pickle.HIGHEST_PROTOCOL)