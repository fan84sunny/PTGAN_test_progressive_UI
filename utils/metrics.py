import numpy as np
import torch

from utils.ficfac_torch import run_fic
from utils.reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()


def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        # tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False, dataset='aic', reranking_track=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.dataset = dataset
        self.reranking_track = reranking_track

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.tids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid, trackid = output
        self.feats = feat
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.tids.extend(np.asarray(trackid))
        self.unique_tids = list(set(self.tids))

    def track_ranking(self, qf, gf, gallery_tids, unique_tids, fic=False):
        origin_dist = euclidean_distance(qf, gf)
        gallery_tids = np.asarray(gallery_tids)
        m, n = qf.shape[0], gf.shape[0]
        dist = np.zeros((m, n))
        gf_tids = sorted(list(set(gallery_tids)))
        track_gf = []
        for i, tid in enumerate(gf_tids):
            temp_dist = origin_dist[:, gallery_tids == tid]
            temp_min = np.min(temp_dist, axis=1)
            index = np.where(temp_min < 0.6)[0]
            if len(index) < 1:
                index = np.where(temp_min == np.min(temp_min))[0]
            weight = temp_dist[index, :].mean(axis=0)
            weight = 1.0 / (weight + 0.01)
            weight = weight / np.sum(weight)
            weight = torch.tensor(weight).cuda().unsqueeze(0)
            temp_gf = torch.mm(weight, gf[gallery_tids == tid, :])
            track_gf.append(torch.mm(weight, gf[gallery_tids == tid, :]))
        track_gf = torch.cat(track_gf)
        origin_track_dist = re_ranking(qf, track_gf, k1=1, k2=2, lambda_value=0.6)

        # cam_dist = np.load('./track_cam_rk.npy')
        # view_dist = np.load('./track_view_rk.npy')

        # track_dist = origin_track_dist - 0.1* cam_dist - 0.05*view_dist
        track_dist = origin_track_dist

        for i, tid in enumerate(gf_tids):
            dist[:, gallery_tids == tid] = track_dist[:, i:(i + 1)]
        return dist, origin_track_dist

    def compute(self, fic=False, fac=False, crop_test=False, rm_camera=False, la=0.18, save_dir='./', P=None, neg_vec=None):
        if crop_test:
            # feats = feats[::2] + feats[1::2]
            # self.pids = self.pids[::2]
            # self.camids = self.camids[::2]
            # self.tids = self.tids[::2]
            # self.num_query = int(self.num_query/2)
            feats = self.feats*2
        if self.feat_norm:
            # print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # f = open(os.path.join(save_dir, 'fic.txt'), 'w')

        # gallery
        self.gf = feats[self.num_query:]
        self.g_pids = np.asarray(self.pids[self.num_query:])
        self.g_camids = np.asarray(self.camids[self.num_query:])
        self.gallery_tids = np.asarray(self.tids[self.num_query:])
        # query
        self.qf = feats[:self.num_query]
        self.q_pids = np.asarray(self.pids[:self.num_query])
        self.q_camids = np.asarray(self.camids[:self.num_query])
        if fic:
            qf1, gf1, P, neg_vec = run_fic(self.qf, self.gf, self.q_camids, self.g_camids, P, neg_vec, la=la)

        if self.reranking_track:
            # print('=> Enter track reranking')
            distmat, origin_track_dist = self.track_ranking(self.qf, self.gf, self.gallery_tids, self.unique_tids)
            if fic:
                distmat1, origin_track_dist1 = self.track_ranking(qf1, gf1, self.gallery_tids, self.unique_tids,
                                                                  fic=True)
                distmat = (distmat + distmat1) / 2.0
                origin_track_dist = 0.5 * origin_track_dist + 0.5 * origin_track_dist1
        elif self.reranking:
            # print('=> Enter reranking')
            distmat = re_ranking(self.qf, self.gf, k1=50, k2=15, lambda_value=0.3)
            if fic:
                distmat += re_ranking(qf1, gf1, k1=50, k2=15, lambda_value=0.3)
        else:
            # print('=> Computing DistMat with euclidean distance')
            distmat = euclidean_distance(self.qf, self.gf)
            if fic:
                distmat += euclidean_distance(qf1, gf1)
        if rm_camera:
            cam_matches = (self.g_camids == self.q_camids[:, np.newaxis]).astype(np.int32)
            distmat = distmat + 10.0 * cam_matches
            cam_matches = ((self.g_camids >= 40).astype(np.int32) != (self.q_camids[:, np.newaxis] >= 40).astype(
                np.int32)).astype(
                np.int32)
            distmat = distmat + 10.0 * cam_matches

        return distmat, P, neg_vec

    def calcu_R1_mAP(self, distmat):
        cmc, mAP = eval_func(distmat, self.q_pids, self.g_pids, self.q_camids, self.g_camids)
        return cmc, mAP, self.pids, self.camids, self.qf, self.gf


class R1_mAP_eval_gen:
    def __init__(self, P, neg_vec, max_rank=50, feat_norm=True, reranking=False, dataset='aic', reranking_track=False):
        super(R1_mAP_eval_gen, self).__init__()
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.dataset = dataset
        self.reranking_track = reranking_track
        self.P = P
        self.neg_vec = neg_vec

    def reset(self, gallery_features):
        self.gallery_features = gallery_features
        self.query_feature = []
        self.pids = []
        self.camids = []
        self.tids = []

    def update(self, output):  # called once for each batch
        feat, camid, trackid = output
        self.query_feature.append(feat)
        self.camids.extend(np.asarray(camid))
        self.tids.extend(np.asarray(trackid))
        self.unique_tids = list(set(self.tids))

    def track_ranking(self, qf, gf, gallery_tids, unique_tids, fic=False):
        origin_dist = euclidean_distance(qf, gf)
        gallery_tids = np.asarray(gallery_tids)
        m, n = qf.shape[0], gf.shape[0]
        dist = np.zeros((m, n))
        gf_tids = sorted(list(set(gallery_tids)))
        track_gf = []
        for i, tid in enumerate(gf_tids):
            temp_dist = origin_dist[:, gallery_tids == tid]
            temp_min = np.min(temp_dist, axis=1)
            index = np.where(temp_min < 0.6)[0]
            if len(index) < 1:
                index = np.where(temp_min == np.min(temp_min))[0]
            weight = temp_dist[index, :].mean(axis=0)
            weight = 1.0 / (weight + 0.01)
            weight = weight / np.sum(weight)
            weight = torch.tensor(weight).cuda().unsqueeze(0)
            track_gf.append(torch.mm(weight, gf[gallery_tids == tid, :]))
        track_gf = torch.cat(track_gf)
        origin_track_dist = re_ranking(qf, track_gf, k1=7, k2=2, lambda_value=0.6)

        track_dist = origin_track_dist
        for i, tid in enumerate(gf_tids):
            dist[:, gallery_tids == tid] = track_dist[:, i:(i + 1)]
        return dist, origin_track_dist

    def compute(self, la=0.18):
        feats = torch.cat(self.query_feature, dim=0)
        feats = feats * 2
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel

        # gallery
        gf = self.gallery_features['gf']
        g_camids = np.asarray(self.gallery_features['g_camids'])
        gallery_tids = np.asarray(self.gallery_features['gallery_tids'])
        # query
        qf = feats
        q_camids = np.asarray(self.camids)
        qf1, gf1, _, _ = run_fic(qf, gf, q_camids, g_camids, self.P, self.neg_vec, la=la)
        print('=> Enter track reranking')
        distmat, origin_track_dist = self.track_ranking(qf, gf, gallery_tids, self.unique_tids)
        distmat1, origin_track_dist1 = self.track_ranking(qf1, gf1, gallery_tids, self.unique_tids, fic=True)
        distmat = (distmat + distmat1) / 2.0

        cam_matches = (g_camids == q_camids[:, np.newaxis]).astype(np.int32)
        distmat = distmat + 10.0 * cam_matches
        cam_matches = ((g_camids >= 40).astype(np.int32) != (q_camids[:, np.newaxis] >= 40).astype(
            np.int32)).astype(
            np.int32)
        distmat = distmat + 10.0 * cam_matches
        return distmat


class R1_mAP():
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP
