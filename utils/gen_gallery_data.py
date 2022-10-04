import argparse
import pickle

from tqdm import tqdm
import os
from config import cfg
from data_process import *
from gan.model import Model
from reid_model import make_model
from utils.ficfac_torch import compute_P2


def generate_feature(cfg, model, gallery_loader, gallery_pose_list):
    device = "cuda:0"

    Smoothing = GaussianSmoothing(20, 21, 5, device)
    Smoothing = Smoothing.to(device)

    model = model.to(device)
    model.eval()

    reid_model = make_model(cfg, num_class=1678)
    reid_model.load_param(cfg.TEST.WEIGHT)
    reid_model.to(device)
    reid_model.eval()

    numPose = 8
    gen_gallery = [{'feature': [], 'camera_id': [], 'pid': [], 'tid': [], 'vtype_id': [], 'file_name': []} for _ in
                   range(numPose)]

    # origin gallery feature
    # '''
    ori_gallery = {'feature': [], 'camera_id': [], 'pid': [], 'tid': [], 'vtype_id': [], 'file_name': []}
    landmark_dict = load_all_landmark(gallery_pose_list)
    for input in tqdm(gallery_loader):
        with torch.no_grad():
            img = input['origin'].to(device)
            _, type = model.get_pose_type(img)
            type = torch.argmax(type, dim=1)

            for i in range(2):
                if i == 1:
                    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().to(device)
                    img = img.index_select(3, inv_idx)
                    f1 = reid_model(img)
                else:
                    f2 = reid_model(img)
            feat = f2 + f1

            ori_gallery['pid'] += input['pid']
            ori_gallery['tid'] += input['trackid']
            ori_gallery['camera_id'] += input['camid']
            rename = [file_name[3:] for file_name in input['file_name']]
            ori_gallery['file_name'] += rename
            ori_gallery['vtype_id'] += type.tolist()
            ori_gallery['feature'] += feat.view(-1, 1, feat.size(1))

    feats = torch.cat(ori_gallery['feature'], dim=0).to(device)
    g_camids = np.asarray(ori_gallery['camera_id'])
    P, neg_vec = compute_P2(None, feats, g_camids, la=0.02)
    with open('../gallery_features/orig_gallery_feature_test', 'wb') as f:
        pickle.dump(ori_gallery, f)
    with open('../gallery_features/orig_gallery_P_test', 'wb') as f:
        pickle.dump(P, f)
    with open('../gallery_features/orig_gallery_vec_test', 'wb') as f:
        pickle.dump(neg_vec, f)

    '''
    landmark_dict = load_all_landmark(gallery_pose_list)
    gen_gallery_type = []

    # generate gallery feature
    with torch.no_grad():
        for gallery_data in tqdm(gallery_loader):
            gallery_img = gallery_data['origin'].to(device)
            _, type = model.get_pose_type(gallery_img)
            query_type = torch.argmax(type, dim=1)
            for b, type in enumerate(query_type):
                pose_file = []
                gen_gallery_type.append(type.item())
                # pose_file[n_img][8 pose]
                for pose_id in range(8):
                    if len(gallery_pose_list[type][pose_id]) != 0:
                        pose_file.append(gallery_pose_list[type][pose_id][0])
                    else:
                        pose_file.append(gallery_pose_list[type][0][0])
                landmark_tensor = torch.FloatTensor([])
                for c, file in enumerate(pose_file):
                    landmark_tensor = torch.cat((landmark_tensor, landmark_dict[file].unsqueeze(0)), dim=0)
                landmark_tensor = landmark_tensor.to(device)
                pose_maps = Smoothing(landmark_tensor)
                gen_query2gal = model.generate(gallery_img[b].unsqueeze(0), pose_maps)

                img = gen_query2gal.to(device)
                img = F.interpolate(img, size=cfg.INPUT.SIZE_TEST[0])

                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().to(device)
                        img = img.index_select(3, inv_idx)
                        f1 = reid_model(img)
                    else:
                        f2 = reid_model(img)
                feat = f1 + f2
                for pose in range(8):
                    gen_gallery[pose]['pid'].append(gallery_data['pid'][b].item())
                    gen_gallery[pose]['tid'].append(gallery_data['trackid'][b].item())
                    gen_gallery[pose]['camera_id'].append(gallery_data['camid'][b].item())
                    gen_gallery[pose]['file_name'].append(gallery_data['file_name'][b][3:])
                    gen_gallery[pose]['vtype_id'].append(type.item())
                    gen_gallery[pose]['feature'].append(feat[pose].view(1, -1))

    for pose in range(8):
        savePath = f'../gallery_features/{pose}'
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        feats = torch.cat(gen_gallery[pose]['feature'], dim=0)
        g_camids = np.asarray(gen_gallery[pose]['camera_id'])
        P, neg_vec = compute_P2(None, feats, g_camids, la=0.02)

        with open(os.path.join(savePath, 'gen_gallery_feature'), 'wb') as f:
            pickle.dump(gen_gallery[pose], f)
        with open(os.path.join(savePath, 'gen_gallery_P'), 'wb') as f:
            pickle.dump(P, f)
        with open(os.path.join(savePath, 'gen_gallery_vec'), 'wb') as f:
            pickle.dump(neg_vec, f)
        '''

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = 'cuda:0'
    query_loader, gallery_loader, gallery_pose_list = get_data('../../AIC21/veri_pose')
    model = Model(device)
    model.reset_model_status()
    generate_feature(cfg, model, gallery_loader, gallery_pose_list)


if __name__ == '__main__':

    main()
