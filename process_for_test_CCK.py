import logging
import pickle
import random
import sys
import time

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from reid_model.make_model import make_model
from utils.metrics import R1_mAP_eval
from gan.model import Model

def frozen_feature_layers(model):
    for name, module in model.named_children():
        if 'base' in name:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def open_all_layers(model):
    for name, module in model.named_children():
        module.train()
        for p in module.parameters():
            p.requires_grad = True


def load_all_landmark(gallery_pose_list):
    landmark_dict = dict()
    for type in gallery_pose_list:
        for gallery_pose in type:
            for file in gallery_pose:
                landmark = []
                with open(file, 'r') as f:
                    landmark_file = f.readlines()
                size = Image.open(file[:-4] + '.jpg').size
                for i, line in enumerate(landmark_file):
                    if i % 2 == 0:
                        h0 = int(float(line) * 224 / size[0])
                        if h0 < 0:
                            h0 = -1
                    else:
                        w0 = int(float(line) * 224 / size[1])
                        if w0 < 0:
                            w0 = -1
                        landmark.append(torch.Tensor([[w0, h0]]))
                landmark = torch.cat(landmark).long()
                # avoid to over fit
                ram = random.randint(0, 19)
                landmark[ram][0] = random.randint(0, 224 - 1)
                landmark[ram][1] = random.randint(0, 224 - 1)
                landmark_dict[file] = landmark
    return landmark_dict


def _load_landmark(path_list):
    landmark_list = []
    for file in path_list:
        landmark = []
        with open(file, 'r') as f:
            landmark_file = f.readlines()
        size = Image.open(file[:-4] + '.jpg').size
        for i, line in enumerate(landmark_file):
            if i % 2 == 0:
                h0 = int(float(line) * 224 / size[0])
                if h0 < 0:
                    h0 = -1
            else:
                w0 = int(float(line) * 224 / size[1])
                if w0 < 0:
                    w0 = -1
                landmark.append(torch.Tensor([[w0, h0]]))
        landmark = torch.cat(landmark).long()
        # avoid to over fit
        ram = random.randint(0, 19)
        landmark[ram][0] = random.randint(0, 224 - 1)
        landmark[ram][1] = random.randint(0, 224 - 1)
        landmark_list.append(landmark)
    return landmark_list


def file2pose_map(landmark_dict, gauss_sigma=5):
    pose_map_dict = dict()
    for landmark_key in tqdm(landmark_dict):
        landmark = landmark_dict[landmark_key]
        maps = []
        randnum = landmark.size(0) + 1
        gauss_sigma = random.randint(gauss_sigma - 1, gauss_sigma + 1)
        for i in range(landmark.size(0)):
            map = np.zeros([224, 224])
            if landmark[i, 0] != -1 and landmark[i, 1] != -1 and i != randnum:
                map[landmark[i, 0], landmark[i, 1]] = 1
                map = ndimage.filters.gaussian_filter(map, sigma=gauss_sigma)
                map = map / map.max()
            maps.append(map)
        maps = np.stack(maps, axis=0)
        pose_map_dict[landmark_key] = maps
        print(sys.getsizeof(maps) / 1024)
    return pose_map_dict


def _generate_pose_map(landmark_list, gauss_sigma=5):
    map_list = []
    for landmark in landmark_list:
        maps = []
        randnum = landmark.size(0) + 1
        gauss_sigma = random.randint(gauss_sigma - 1, gauss_sigma + 1)
        for i in range(landmark.size(0)):
            map = np.zeros([224, 224])
            if landmark[i, 0] != -1 and landmark[i, 1] != -1 and i != randnum:
                map[landmark[i, 0], landmark[i, 1]] = 1
                map = ndimage.filters.gaussian_filter(map, sigma=gauss_sigma)
                map = map / map.max()
            maps.append(map)
        maps = np.stack(maps, axis=0)
        map_list.append(maps)
    return map_list


def compute_distmat(query_feat, query_data, gen_gallery, ori_gallery, evaluator, cfg, gen_P, gen_neg_vec, P, neg_vec, device):
    num_gallery = len(ori_gallery['pid'])
    i = 1
    scale = 50
    current = 0
    gen_gallery['feature'] = torch.cat(gen_gallery['feature'], dim=0).to(device)
    ori_gallery['feature'] = torch.cat(ori_gallery['feature'], dim=0).to(device)
    while True:
        if current >= num_gallery:
            return
        end = current + i * scale
        evaluator.reset()
        pid = [query_data['pid']] + gen_gallery['pid'][current:end]
        feature = torch.cat((query_feat, gen_gallery['feature'][current:end]), dim=0)
        camid = [query_data['camid']] + gen_gallery['camera_id'][current:end]
        trackid = [query_data['trackid']] + gen_gallery['tid'][current:end]
        evaluator.update((feature, pid, camid, trackid))
        gen_distmat, _, _ = evaluator.compute(fic=cfg.TEST.FIC, fac=cfg.TEST.FAC, rm_camera=cfg.TEST.RM_CAMERA,
                                              save_dir=cfg.OUTPUT_DIR, crop_test=cfg.TEST.CROP_TEST,
                                              la=cfg.TEST.LA, P=gen_P, neg_vec=gen_neg_vec)
        evaluator.reset()
        feature = torch.cat((query_feat, ori_gallery['feature'][current:end]), dim=0)
        pid = [query_data['pid']] + ori_gallery['pid'][current:end]
        camid = [query_data['camid']] + ori_gallery['camera_id'][current:end]
        trackid = [query_data['trackid']] + ori_gallery['tid'][current:end]
        evaluator.update((feature, pid, camid, trackid))
        distmat, _, _ = evaluator.compute(fic=cfg.TEST.FIC, fac=cfg.TEST.FAC, rm_camera=cfg.TEST.RM_CAMERA,
                                          save_dir=cfg.OUTPUT_DIR, crop_test=cfg.TEST.CROP_TEST, la=cfg.TEST.LA,
                                          P=P, neg_vec=neg_vec)
        current = end
        i += 1
        distmat += gen_distmat * 0.5
        yield distmat, ori_gallery['file_name'],


def get_pose(query_data, device="cuda"):
    model = Model(device)
    model.reset_model_status()
    model.eval()
    with torch.no_grad():
        img = query_data['origin'].to(device)
        img = img.unsqueeze(0)
        # img = query_data['origin'].to(device)
        # img = img.unsqueeze(0)
        pose, type = model.get_pose_type(img)
        query_poseid = torch.argmax(pose, dim=1)
    return query_poseid


def do_inference_reid(cfg, query_data, device="cuda"):

    reid_model = make_model(cfg, num_class=1678)
    reid_model.load_param(cfg.TEST.WEIGHT)
    reid_model.to(device)
    reid_model.eval()
    with torch.no_grad():
        img = query_data['origin'].to(device)
        img = img.unsqueeze(0)
        # img = query_data['origin'].to(device)
        # img = img.unsqueeze(0)
        if cfg.TEST.FLIP_FEATS == 'on':
            for i in range(2):
                if i == 1:
                    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                    img = img.index_select(3, inv_idx)
                    f1 = reid_model(img)
                else:
                    f2 = reid_model(img)
            feat = f2 + f1
        else:
            feat = reid_model(img)
    return feat


def do_inference(cfg, query_data, query_feats, query_poseid, resultWindow=None):
    logger = logging.getLogger("reid_baseline.test")
    logger.info("Enter inferencing")
    device = "cuda"

    # model.eval()
    # model = model.to(device)
    # reid_model = make_model(cfg, num_class=1678)
    # reid_model.load_param(cfg.TEST.WEIGHT)
    # reid_model.to("cuda")
    # reid_model.eval()

    # Compute Original_query to Original_gallery distance matrix
    logger = logging.getLogger("reid_baseline.test")
    logger.info("Enter inferencing")
    if device and torch.cuda.device_count() > 1:
        print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
    # num_query = int(len(query_data) / 5)
    num_query = 1
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING,
                            dataset=cfg.DATASETS.NAMES, reranking_track=cfg.TEST.RE_RANKING_TRACK)
    evaluator.reset()

    gallery_path = f'gallery_features/{query_poseid.item()}'
    with open(os.path.join(gallery_path, 'gen_gallery_feature'), 'rb') as f:
        gen_gallery = pickle.load(f)
    with open(os.path.join(gallery_path, 'gen_gallery_P'), 'rb') as f:
        gen_P = pickle.load(f)
    with open(os.path.join(gallery_path, 'gen_gallery_vec'), 'rb') as f:
        gen_neg_vec = pickle.load(f)
    with open('gallery_features/orig_gallery_feature_test', 'rb') as f:
        ori_gallery = pickle.load(f)
    with open("gallery_features/orig_gallery_P_test", 'rb') as f:
        P = pickle.load(f)
    with open("gallery_features/orig_gallery_vec_test", 'rb') as f:
        neg_vec = pickle.load(f)

    start = time.time()
    # # compute gen_gallery and query image
    # # with torch.no_grad():
    # #     img = query_data['origin'].to(device)
    # #     img = img.unsqueeze(0)
    # #     query_poseid = get_pose(img, device="cuda")
    # #     feat = do_inference_reid(cfg, img, device=device)
    #     img = query_data['origin'].to(device)
    #     img = img.unsqueeze(0)
    #     pose, type = model.get_pose_type(img)
    #     query_poseid = torch.argmax(pose, dim=1)
    #     if cfg.TEST.FLIP_FEATS == 'on':
    #         for i in range(2):
    #             if i == 1:
    #                 inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
    #                 img = img.index_select(3, inv_idx)
    #                 f1 = reid_model(img)
    #             else:
    #                 f2 = reid_model(img)
    #         feat = f2 + f1
    #     else:
    #         feat = reid_model(img)
    # feat = query_feats
    # gen_gallery = gen_gallery[query_poseid]
    # gen_P = gen_P[query_poseid]
    # gen_neg_vec = gen_neg_vec[query_poseid]
    combine_distmat = np.zeros((1, len(ori_gallery['pid'])))
    distmats = compute_distmat(query_feats, query_data, gen_gallery, ori_gallery, evaluator, cfg, gen_P, gen_neg_vec, P,
                               neg_vec, device)
    current_distmat = np.empty(shape=0)
    current = 0
    query_paths = query_data['file_name']
    gallary_paths = gen_gallery['file_name']

    for i, (distmat, gallery_name) in enumerate(distmats):
        end = distmat.shape[1] + current
        # sorted current result: use sorted_current can choose to return show image
        current_distmat = np.append(current_distmat, distmat)
        current_name = gallery_name[:end]
        zipped = list(zip(current_name, current_distmat))
        sorted_current = sorted(zipped, key=lambda x: x[1])
        # return current result to UI
        if resultWindow != None:
            resultWindow.printSignal.emit(sorted_current)
        # combine final result
        combine_distmat[..., current:end] = distmat
        current = end
        print(f"{'-'*10}{end}/{len(gallary_paths)}{'-'*10}")

    print("query stage -- time: ")

    df = pd.DataFrame(combine_distmat, index=[query_paths], columns=gallary_paths)
    df.to_csv("similiar_img_distmat.csv")
    # df.sort_values(by=query_paths[0], axis=1, ascending=True) # this place can sort value, and you can pick rank-k
    print('Using totally {:.2f}S to compute'.format(time.time() - start))

    print("Finish")
