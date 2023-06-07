#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import copy
import datetime
import json
import logging
import math
import time

import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
from shapely.geometry import Polygon
from tqdm import tqdm


def compute_area(x0, x1, x2, x3, y0, y1, y2, y3):
    return 0.5 * abs(x0 * y1 - x1 * y0 + x1 * y2 - x2 * y1 + x2 * y3 - x3 * y2 + x3 * y0 - x0 * y3)


def compute_angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle


class CarSpace(COCO):

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO()
        res.dataset['images'] = [img for img in self.dataset['images']]
        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
            'Results do not correspond to current coco set'
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id + 1
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                if 'segmentation' not in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2] * bb[3]
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = maskUtils.area(ann['segmentation'])
                if 'bbox' not in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        elif 'keypoints' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                s = ann['keypoints']
                x = s[0::3]
                y = s[1::3]
                x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann['id'] = id + 1
                ann['bbox'] = [x0, y0, x1 - x0, y1 - y0]
                # The only difference with original version
                ann['area'] = compute_area(*(s[::3] + s[1::3]))
        print('DONE (t={:0.2f}s)'.format(time.time() - tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res


class CarSpaceParams(Params):

    def __init__(self, iou_type='keypoints', iou_thr=0.9, mthr=0.005, lthr=0.01, minl=50, cls_type='pld'):
        self.mthr = mthr
        self.lthr = lthr
        self.iou_thr = iou_thr
        self.minl = minl
        self.cls_type = cls_type
        super(CarSpaceParams, self).__init__(iou_type)

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        self.maxDets = [20]
        w, h = 1000, 1000
        self.areaRng = [[0, w * h]]
        self.areaRngLbl = ['all']
        self.areaRngMapping = dict([(x, y) for x, y in zip(self.areaRngLbl, self.areaRng)])
        self.subCats = ['all', 'vertical', 'parallel', 'cross']
        self.subCatIds = [0, 1, 2, 3]
        self.subCatMapping = dict(zip(self.subCats, self.subCatIds))
        self.useCats = 1
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / 0.01)) + 1, endpoint=True)
        self.iouThrs = np.linspace(0.8, 0.98, int(np.round((0.98 - 0.8) / 0.02)) + 1, endpoint=True)


class CarSpaceEval(COCOeval):
    """ keypoint divided by line, in coco eval format.
    """

    def __init__(self, cocoGt, cocoDt, cfg):
        super(CarSpaceEval, self).__init__(cocoGt, cocoDt, cfg.iou_type)
        self.logger = logging.getLogger()
        self.params = CarSpaceParams(cfg.iou_type, cfg.iou_thr, cls_type=cfg.cls_type)
        if cocoGt is not None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())
        self.computeOks = self.computeAvg

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        self.logger.info('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if p.useSegm is not None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            self.logger.info('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        self.logger.info('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId)
                     for imgId in tqdm(p.imgIds)
                     for catId in catIds}

        maxDet = p.maxDets[-1]
        self.evalImgs = [self.evaluateImg(imgId, catId, areaRng, subCatId, maxDet)
                         for catId in catIds
                         for subCatId in p.subCatIds
                         for areaRng in p.areaRng
                         for imgId in tqdm(p.imgIds)]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        self.logger.info('DONE (t={:0.2f}s).'.format(toc - tic))

    def evaluateImg(self, imgId, catId, aRng, subCatId, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            kpt = np.array(g['keypoints']).reshape(-1, 3)[:, :2]
            l1 = ((kpt[0] - kpt[1]) ** 2).sum()
            l2 = ((kpt[0] - kpt[3]) ** 2).sum()
            angle = compute_angle(kpt[:2].reshape(-1), np.roll(kpt, 1, 0)[:2].reshape(-1))
            if 75 < angle < 105:
                if (l1 >= l2 * 1.21):
                    g['parkingType'] = 2
                else:
                    g['parkingType'] = 1
            else:
                g['parkingType'] = 3

            vg = kpt[2::3]
            if self.params.cls_type == 'pld' and vg[:2].sum() == 0:
                g['ignore'] = 1
            elif vg.sum() == 0:
                g['ignore'] = 1

            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]) or \
                    (subCatId > 0 and g.get('parkingType', 0) != subCatId):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0
        for d in dt:
            kpt = np.array(d['keypoints']).reshape(-1, 3)[:, :2]
            l1 = ((kpt[0] - kpt[1]) ** 2).sum()
            l2 = ((kpt[0] - kpt[3]) ** 2).sum()
            angle = compute_angle(kpt[:2].reshape(-1), np.roll(kpt, 1, 0)[:2].reshape(-1))
            if 75 < angle < 105:
                if (l1 >= l2 * 1.21):
                    d['parkingType'] = 2
                else:
                    d['parkingType'] = 1
            else:
                d['parkingType'] = 3

            if (d['area'] < aRng[0] or d['area'] > aRng[1]) or \
                    (subCatId > 0 and d.get('parkingType', 0) != subCatId):
                d['_ignore'] = 1
            else:
                d['_ignore'] = 0
        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))

        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']

        # set unmatched detections outside of area range to ignore
        a = np.array([d['_ignore'] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            'image_id': imgId,
            'category_id': catId,
            'aRng': aRng,
            'subCatId': subCatId,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
        }

    def get_badcase(self, iouThr=None, subCat='all', areaRng='all', maxDets=20):
        if iouThr is None:
            iouThr = self.params.iou_thr
        subCatId = self.params.subCatMapping[subCat]
        aRng = self.params.areaRngMapping[areaRng]
        evalimgs = filter(lambda x: x and x['aRng'] == aRng and x['maxDet'] == maxDets
                          and x['subCatId'] == subCatId, self.evalImgs)

        pr = collections.defaultdict(lambda: collections.defaultdict(int))
        badcase = collections.defaultdict(lambda: collections.defaultdict(list))
        index = np.where(self.params.iouThrs == iouThr)[0][0]
        for item in evalimgs:
            fn, tp, fp = [], [], []
            imgId = item['image_id']
            catId = item['category_id']
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
            # sort dt highest score first, sort gt ignore last
            gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
            gt = [gt[i] for i in gtind]
            dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
            dt = [dt[i] for i in dtind[0:maxDets]]
            dtm = set(item['dtMatches'][index])
            gtm = set(item['gtMatches'][index])
            gtIg = item['gtIgnore']
            dtIg = item['dtIgnore'][index]
            for i, _gt in enumerate(gt):
                if gtIg[i] > 0:
                    continue
                if _gt['id'] not in dtm:
                    # this is a miss detection
                    fn.append([_gt['keypoints'], catId, 0])
            for i, _dt in enumerate(dt):
                if dtIg[i] > 0:
                    continue
                if _dt['id'] in gtm:
                    tp.append([_dt['keypoints'], catId, _dt['score']])
                else:
                    fp.append([_dt['keypoints'], catId, _dt['score']])

            badcase[imgId]['fn'].extend(fn)
            badcase[imgId]['fp'].extend(fp)
            pr[catId]['tp'] += len(tp)
            pr[catId]['fp'] += len(fp)
            pr[catId]['fn'] += len(fn)

        for catId, ret in pr.items():
            TP, FP, FN = ret['tp'], ret['fp'], ret['fn']
            self.logger.info(f'{catId} false detection rate: {FP} / {FP + TP} = {FP / (FP + TP)}')
            self.logger.info(f'{catId} miss detection rate: {FN} / {TP + FN} = {FN / (TP + FN)}')

        return badcase

    def computeAvg(self, imgId, catId):
        """ Calculate point distance to entrance and backline of carspace.

        Change from both COCO and MPII's mpckh.
        equation: iou score = (delta(d0 or d1)/l01 + delta(d2 or d3)/l23) / 4
        """
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        # compute pckh between each detection and ground truth object
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        for j, gt in enumerate(gts):
            g = np.array(gt['keypoints'])
            xg = g[0::3]
            yg = g[1::3]
            vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            lx = xg[1::2] - xg[0::2]
            ly = yg[1::2] - yg[0::2]
            le = (lx**2 + ly**2)
            if self.params.cls_type == 'iou':
                gt_poly = Polygon(g.reshape(4, 3)[:, :2]).convex_hull
            elif self.params.cls_type == 'center':
                # 1pixel = 15mm, 10pixel = 15cm
                kpt = g.reshape(4, 3)[:, :2]
                # length = sum(sorted(np.sqrt(((kpt - np.roll(kpt, 1, 0)) ** 2).sum(1)))[:2]) / 2
                length = 100
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                if self.params.cls_type == 'iou':
                    dt_poly = Polygon(d.reshape(4, 3)[:, :2]).convex_hull
                    inter_area = gt_poly.intersection(dt_poly).area
                    union_area = max(min(gt_poly.area, dt_poly.area), 1)
                    ious[i, j] = inter_area / union_area
                    continue
                elif self.params.cls_type == 'center':
                    dt_kpt = d.reshape(4, 3)[:, :2]
                    dis = np.sqrt(((kpt.mean(0) - dt_kpt.mean(0))**2).sum())
                    ious[i, j] = max(0, 1 - dis / length)
                    continue

                d_s = np.copy(d)
                min_dis = 0xffffff
                for sign in [1, - 1]:
                    if sign < 0:
                        d_s = d_s.reshape(-1, 3)[::-1, :].reshape(-1)
                    for _ in range(4):
                        xd, yd = d_s[0::3], d_s[1::3]
                        dis = (xd - xg) ** 2 + (yd - yg) ** 2
                        dis = np.sum(dis[vg > 0])
                        if dis < min_dis:
                            min_dis = dis
                            d = d_s
                        d_s = np.roll(d_s, 3)
                xd = d[0::3]
                yd = d[1::3]
                if k1 > 0:
                    # measure the per-keypoint distance if gt keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                    e = (dx**2 + dy**2)
                    assert len(dx) % 2 == 0, "Error: keypoints should be even."
                    if self.params.cls_type == "pld":
                        e[0: len(dx) // 2] /= (le[0] + np.spacing(1))  # divided by entrance line length
                        e[len(dx) // 2:] /= (le[1] + np.spacing(1))  # divided by backline
                        e = e**0.5
                        vg[2:] = 0
                    else:
                        e = e**0.5
                        e[0: len(dx) // 2] /= 100
                        e[len(dx) // 2:] /= 100

                    # visible: 2, invisible and in-image-plane: 1, out-of-image: 0
                    # e = e[vg > 0] # both consider the visible and invisible points
                    e = e[vg > 0]
                    # if len(vg) == 4:
                    #     mask = e > 0.1
                    #     if len(np.nonzero(mask)) == 1:
                    #         e[np.nonzero(mask)] = 0
                    #         error_point[np.nonzero(mask)] += 1
                    if e.shape[0] == 0:
                        e = np.ones((len(xd), 1))
                else:
                    e = np.ones((len(xd), 1))
                ious[i, j] = max(0, 1 - (np.sum(e) / e.shape[0]))
        return ious

    def accumulate(self, p=None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        self.logger.info('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            self.logger.info('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        C = len(p.subCats)
        M = len(p.maxDets)
        precision = -np.ones((T, R, K, C, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, C, A, M))
        scores = -np.ones((T, R, K, C, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setC = set(_pe.subCats)
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        c_list = [n for n, c in enumerate(p.subCats) if c in setC]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        C0 = len(_pe.subCats)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * C0 * A0 * I0
            for c, c0 in enumerate(c_list):
                Nc = c0 * A0 * I0
                for a, a0 in enumerate(a_list):
                    Na = a0 * I0
                    for m, maxDet in enumerate(m_list):
                        E = [self.evalImgs[Nk + Nc + Na + i] for i in i_list]
                        E = [e for e in E if e is not None]
                        if len(E) == 0:
                            continue
                        dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                        # different sorting method generates slightly different results.
                        # mergesort is used to be consistent as Matlab implementation.
                        inds = np.argsort(-dtScores, kind='mergesort')
                        dtScoresSorted = dtScores[inds]

                        dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                        dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                        gtIg = np.concatenate([e['gtIgnore'] for e in E])
                        npig = np.count_nonzero(gtIg == 0)
                        if npig == 0:
                            continue
                        tps = np.logical_and(dtm, np.logical_not(dtIg))
                        fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                        tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                        fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                        for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                            tp = np.array(tp)
                            fp = np.array(fp)
                            nd = len(tp)
                            rc = tp / npig
                            pr = tp / (fp + tp + np.spacing(1))
                            q = np.zeros((R,))
                            ss = np.zeros((R,))

                            if nd:
                                recall[t, k, c, a, m] = rc[-1]
                            else:
                                recall[t, k, c, a, m] = 0

                            # numpy is slow without cython optimization for accessing elements
                            # use python array gets significant speed improvement
                            pr = pr.tolist()
                            q = q.tolist()

                            for i in range(nd - 1, 0, -1):
                                if pr[i] > pr[i - 1]:
                                    pr[i - 1] = pr[i]

                            inds = np.searchsorted(rc, p.recThrs, side='left')
                            try:
                                for ri, pi in enumerate(inds):
                                    q[ri] = pr[pi]
                                    ss[ri] = dtScoresSorted[pi]
                            except:
                                pass
                            precision[t, :, k, c, a, m] = np.array(q)
                            scores[t, :, k, c, a, m] = np.array(ss)

        self.eval = {
            'params': p,
            'counts': [T, R, K, C, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall': recall,
            'scores': scores,
        }
        toc = time.time()
        self.logger.info('DONE (t={:0.2f}s).'.format(toc - tic))

    def summarize(self, logger):
        '''
        Compute and display summary metrics for evaluation results.
        '''
        def _summarize(ap=1, iouThr=None, areaRng='all', subCat='all', maxDets=20):
            p = self.params
            iStr = '{:<18} {} @[ IoU={:<9} | parking={:>6s} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            cind = [i for i, sCat in enumerate(p.subCats) if subCat == sCat]
            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxCxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, cind, aind, mind]
                if len(p.catIds) > 1:
                    for i in range(len(p.catIds)):
                        logger.info('Category {0} Precision: {1}'.format(i, np.mean(s[:, :, i])))
            else:
                # dimension of recall: [TxKxCxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, cind, aind, mind]
                if len(p.catIds) > 1:
                    for i in range(len(p.catIds)):
                        logger.info('Category {0} Recall: {1}'.format(i, np.mean(s[:, i])))

            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])

            logger.info(iStr.format(titleStr, typeStr, iouStr, subCat, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeKpts():
            stats = [
                _summarize(1, maxDets=self.params.maxDets[-1]),
                _summarize(1, iouThr=self.params.iou_thr, subCat='vertical', maxDets=self.params.maxDets[-1]),
                _summarize(1, iouThr=self.params.iou_thr, subCat='parallel', maxDets=self.params.maxDets[-1]),
                _summarize(1, iouThr=self.params.iou_thr, subCat='cross', maxDets=self.params.maxDets[-1]),
                _summarize(1, iouThr=self.params.iou_thr, maxDets=self.params.maxDets[-1]),
                _summarize(1, iouThr=float(self.params.iouThrs[0]), maxDets=self.params.maxDets[-1]),

                _summarize(0, maxDets=self.params.maxDets[-1]),
                _summarize(0, iouThr=self.params.iou_thr, subCat='vertical', maxDets=self.params.maxDets[-1]),
                _summarize(0, iouThr=self.params.iou_thr, subCat='parallel', maxDets=self.params.maxDets[-1]),
                _summarize(0, iouThr=self.params.iou_thr, subCat='cross', maxDets=self.params.maxDets[-1]),
                _summarize(0, iouThr=self.params.iou_thr, maxDets=self.params.maxDets[-1]),
                _summarize(0, iouThr=float(self.params.iouThrs[0]), maxDets=self.params.maxDets[-1]),
            ]
            return np.array(stats)
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'keypoints':
            summarize = _summarizeKpts
        else:
            raise Exception('iouType error: only support keypoints.')
        self.stats = summarize()
        # print('error_point:', error_point)
