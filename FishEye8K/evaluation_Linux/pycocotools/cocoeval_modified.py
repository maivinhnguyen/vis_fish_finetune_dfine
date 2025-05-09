# __author__ = 'tsungyi' # Thường không cần thiết khi chia sẻ code, nhưng giữ lại nếu bạn muốn

import numpy as np
import datetime
import time
from collections import defaultdict
# Đảm bảo bạn có thư viện pycocotools đúng cách, ví dụ:
# from pycocotools import mask as maskUtils
# Hoặc nếu mask.py nằm cùng thư mục:
from . import mask as maskUtils # Dành cho trường hợp import tương đối trong một package
import copy

class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if cocoGt is not None: # Sửa lỗi "not cocoGt is None" thành "cocoGt is not None" cho dễ đọc
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())


    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            # Sửa lỗi logic: iscrowd nên là một điều kiện OR với ignore hiện tại, không phải gán đè.
            # Tuy nhiên, COCO gốc dường như ưu tiên iscrowd.
            # gt['ignore'] = gt['ignore'] or ('iscrowd' in gt and gt['iscrowd'])
            # Theo logic COCO gốc, iscrowd được ưu tiên để set ignore
            if 'iscrowd' in gt and gt['iscrowd']:
                gt['ignore'] = 1 # Luôn ignore nếu iscrowd
            # else: # giữ nguyên giá trị ignore ban đầu nếu không phải iscrowd
            #    gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0

            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if p.useSegm is not None: # Sửa lỗi "not p.useSegm is None"
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        else: # Thêm xử lý cho trường hợp iouType không hợp lệ
            raise ValueError(f"Unsupported iouType: {p.iouType}")

        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0: # Điều kiện này không đúng, chỉ cần 1 trong 2 rỗng là đủ
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = np.array(p.kpt_oks_sigmas) # Đảm bảo sigmas là np.array
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            # Dòng print này có thể tạo ra quá nhiều output, nên comment lại trừ khi debug
            # if g['ignore']:
            #     print(g['ignore'])
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        # Sửa lỗi tiềm ẩn: ious có thể là mảng rỗng, cần kiểm tra len(dt) trước khi truy cập
        if len(dt) == 0: # Nếu không có detection nào sau khi sort và cắt theo maxDet
            ious_sorted_dt = np.array([]) # Hoặc xử lý phù hợp
        elif len(self.ious[imgId, catId]) > 0 : # Nếu có ious đã tính
            # Đảm bảo ious[imgId, catId] có đủ hàng để lấy dtind
            # ious có shape (D_raw, G), D_raw là số dt ban đầu trước khi cắt maxDet
            # dtind là chỉ số của dt đã sort và cắt, nên cần lấy dtind_original
            # Tuy nhiên, logic gốc là lấy theo dt đã được sort và cắt rồi,
            # nên ious cần được sort theo score của dt trước, sau đó mới lấy theo gtind
            # Đoạn này hơi phức tạp, cần xem lại logic gốc của COCO
            # Giả sử self.ious[imgId, catId] được tính với dt đã sort theo score và cắt maxDet
            # Nếu self.ious được tính với dt gốc, thì phải lấy theo dtind trước khi cắt.
            # Logic hiện tại của computeIoU là sort dt và cắt maxDet BÊN TRONG computeIoU.
            # Nên self.ious[imgId, catId] đã tương ứng với dt đã sort và cắt.
            current_ious = self.ious[imgId, catId]
            # if current_ious.shape[0] != len(dt): # This should not happen if computeIoU is correct
            #    print("Warning: Mismatch in ious dimension and dt length in evaluateImg")

            ious_sorted_dt = current_ious[:, gtind] if len(current_ious) > 0 else current_ious

        else: # Nếu self.ious[imgId,catId] rỗng (không có gt hoặc dt)
             ious_sorted_dt = self.ious[imgId,catId]


        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))

        if not len(ious_sorted_dt)==0: # Sử dụng ious đã được sắp xếp theo gt
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious_sorted_dt[dind,gind] < iou: # Sử dụng ious_sorted_dt
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious_sorted_dt[dind,gind] # Sử dụng ious_sorted_dt
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
            return # Thêm return để tránh lỗi nếu evalImgs rỗng

        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        precision_new = -np.ones((T,K,A,M)) # Precision at the operating point
        recall      = -np.ones((T,K,A,M)) # Recall at the operating point
        f1_score    = -np.ones((T,K,A,M)) # F1-score at the operating point
        scores      = -np.ones((T,R,K,A,M))


        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k_val in enumerate(p.catIds)  if k_val in setK] # Sửa tên biến k thành k_val
        m_list = [m_val for n, m_val in enumerate(p.maxDets) if m_val in setM] # Sửa tên biến m thành m_val
        a_list = [n for n, a_val in enumerate(map(lambda x: tuple(x), p.areaRng)) if a_val in setA] # Sửa tên biến a thành a_val
        i_list = [n for n, i_val in enumerate(p.imgIds)  if i_val in setI] # Sửa tên biến i thành i_val
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        
        for k_idx, k0 in enumerate(k_list): # catIDs loop index k_idx, actual category index in p.catIds is k0
            Nk = k0*A0*I0 

            for a_idx, a0 in enumerate(a_list):  #areaRanges loop index a_idx, actual area index in p.areaRng is a0
                Na = a0*I0 
                for m_idx, maxDet_val in enumerate(m_list):  #maxDets loop index m_idx, actual maxDet value
                    E = [self.evalImgs[Nk + Na + i_val_idx] for i_val_idx in i_list] # Nk, Na, i_val_idx là chỉ số trong self.evalImgs
                    E = [e for e in E if e is not None] # Sửa "not e is None" thành "e is not None"
                    if len(E) == 0:
                        continue
                    # Lấy dtScores, đảm bảo chỉ lấy maxDet_val phần tử
                    dtScores = np.concatenate([e['dtScores'][0:maxDet_val] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds] # Scores đã được sắp xếp
                    
                    # Lấy dtm và dtIg, đảm bảo chỉ lấy theo cột tương ứng với maxDet_val
                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet_val] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet_val]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 ) # Number of positive ground truths (not ignored)

                    if npig == 0: # Nếu không có ground truth nào để đánh giá
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) ) # True positives
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) ) # False positives
                    
                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)

                    for t_idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)): # t_idx là chỉ số ngưỡng IoU
                        tp_arr = np.array(tp) # Chuyển sang np.array
                        fp_arr = np.array(fp) # Chuyển sang np.array
                        nd = len(tp_arr) # Number of detections considered
                        
                        rc_curve = tp_arr / npig # Recall curve
                        pr_curve = tp_arr / (fp_arr + tp_arr + np.spacing(1)) # Precision curve
                        
                        q_pr_at_recThrs  = np.zeros((R,)) # Precision at standard recall thresholds
                        s_scores_at_recThrs = np.zeros((R,)) # Scores at standard recall thresholds
                        
                        if nd: # Nếu có detections
                            # Recall, Precision, F1 tại điểm hoạt động (sau khi duyệt hết detections)
                            recall_op = rc_curve[-1]
                            precision_op = pr_curve[-1]
                            f1_op = 0.0
                            if (precision_op + recall_op) > 0: # Tránh chia cho 0
                                f1_op = 2 * precision_op * recall_op / (precision_op + recall_op + np.spacing(1)) # Sửa lỗi 1e-9

                            recall[t_idx, k_idx, a_idx, m_idx] = recall_op
                            precision_new[t_idx, k_idx, a_idx, m_idx] = precision_op
                            f1_score[t_idx, k_idx, a_idx, m_idx] = f1_op
                        else: # Nếu không có detections nào
                            recall[t_idx, k_idx, a_idx, m_idx] = 0
                            precision_new[t_idx, k_idx, a_idx, m_idx] = 0
                            f1_score[t_idx, k_idx, a_idx, m_idx] = 0
                        
                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr_list = pr_curve.tolist(); q_pr_list = q_pr_at_recThrs.tolist()

                        # Interpolate precision for PR curve (standard COCO mAP calculation)
                        for i_interp in range(nd-1, 0, -1):
                            if pr_list[i_interp] > pr_list[i_interp-1]:
                                pr_list[i_interp-1] = pr_list[i_interp]
                        
                        # Find precision values at standard recall thresholds
                        inds_recall_match = np.searchsorted(rc_curve, p.recThrs, side='left')
                        try:
                            for ri, pi_idx in enumerate(inds_recall_match): # ri là index của recThrs, pi_idx là index trong pr_list
                                if pi_idx < len(pr_list): # Đảm bảo pi_idx không vượt quá độ dài pr_list
                                    q_pr_list[ri] = pr_list[pi_idx]
                                    if pi_idx < len(dtScoresSorted): # Đảm bảo pi_idx không vượt quá dtScoresSorted
                                        s_scores_at_recThrs[ri] = dtScoresSorted[pi_idx]
                                else: # Nếu pi_idx vượt quá, nghĩa là không đạt được recall threshold đó
                                     q_pr_list[ri] = 0 # Hoặc giá trị phù hợp
                                     s_scores_at_recThrs[ri] = 0 # Hoặc giá trị phù hợp

                        except IndexError: # Bắt lỗi nếu có
                            pass # Hoặc xử lý lỗi cụ thể
                        
                        # Lưu precision và scores đã nội suy
                        precision[t_idx, :, k_idx, a_idx, m_idx] = np.array(q_pr_list)
                        scores[t_idx, :, k_idx, a_idx, m_idx] = np.array(s_scores_at_recThrs)

        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,       # PR curve (interpolated precision at recall thresholds)
            'recall':   recall,           # Recall at operating point
            'scores': scores,             # Scores corresponding to PR curve points
            'precision_point': precision_new, # Precision at operating point (thay "precision_new")
            'f1_score': f1_score          # F1-score at operating point
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize(ap_type=1, iouThr=None, areaRng='all', maxDets=100): # Đổi tên ap thành ap_type
            p = self.params
            # Sửa lỗi format string, đảm bảo số lượng placeholder khớp với số lượng biến
            iStr = ' {:<30} {:<4} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}' # Thêm 1 placeholder cho typeStr
            
            titleStr, typeStr = "", "" # Khởi tạo
            if ap_type == 1 : # mAP (trung bình precision trên đường cong PR)
                titleStr = 'Average Precision'
                typeStr = '(AP)'
            elif ap_type == 0 : # Average Recall (tại điểm hoạt động)
                titleStr = 'Average Recall'
                typeStr = '(AR)'
            elif ap_type == 2: # Average F1-score (tại điểm hoạt động)
                titleStr = 'Average F1-score'
                typeStr = '(F1)'
            elif ap_type == 3: # Average Precision (tại điểm hoạt động, không phải mAP)
                titleStr = 'Average Precision'
                typeStr = '(P)' # Ký hiệu P cho precision tại điểm hoạt động
            else: # Trường hợp không xác định
                 raise ValueError(f"Unsupported ap_type: {ap_type}")

            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng_label in enumerate(p.areaRngLbl) if aRng_label == areaRng] # Sửa aRng thành aRng_label
            mind = [i for i, mDet_val in enumerate(p.maxDets) if mDet_val == maxDets] # Sửa mDet thành mDet_val

            s_values = None # Khởi tạo
            if ap_type == 1: # mAP
                # dimension of precision: [TxRxKxAxM]
                s_values = self.eval['precision']
                if iouThr is not None:
                    t = np.where(np.isclose(p.iouThrs, iouThr))[0] # Dùng isclose cho so sánh float
                    if len(t) == 0: raise ValueError(f"iouThr={iouThr} not found in p.iouThrs")
                    s_values = s_values[t]
                s_values = s_values[:,:,:,aind,mind] # Lấy tất cả recall thresholds (:)
            elif ap_type == 0: # AR
                # dimension of recall: [TxKxAxM]
                s_values = self.eval['recall']
                if iouThr is not None:
                    t = np.where(np.isclose(p.iouThrs, iouThr))[0]
                    if len(t) == 0: raise ValueError(f"iouThr={iouThr} not found in p.iouThrs")
                    s_values = s_values[t]
                s_values = s_values[:,:,aind,mind] # Recall là 1 giá trị cho mỗi (T,K,A,M)
            elif ap_type == 2: # F1
                s_values = self.eval['f1_score']
                if iouThr is not None:
                    t = np.where(np.isclose(p.iouThrs, iouThr))[0]
                    if len(t) == 0: raise ValueError(f"iouThr={iouThr} not found in p.iouThrs")
                    s_values = s_values[t]
                s_values = s_values[:,:,aind,mind]
            elif ap_type == 3: # Precision tại điểm hoạt động
                s_values = self.eval['precision_point'] # Sử dụng 'precision_point'
                if iouThr is not None:
                    t = np.where(np.isclose(p.iouThrs, iouThr))[0]
                    if len(t) == 0: raise ValueError(f"iouThr={iouThr} not found in p.iouThrs")
                    s_values = s_values[t]
                s_values = s_values[:,:,aind,mind]

            if s_values is None or len(s_values[s_values > -1]) == 0: # Kiểm tra s_values có giá trị hợp lệ không
                mean_s = -1.0 # Hoặc np.nan
            else:
                mean_s = np.mean(s_values[s_values > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((25,)) # Tăng kích thước stats lên 25 để chứa F1@0.5
            # AP (mAP - Average Precision over PR curve)
            stats[0] = _summarize(ap_type=1) # AP @ IoU=0.50:0.95 | area=all | maxDets=100
            stats[1] = _summarize(ap_type=1, iouThr=.5, maxDets=self.params.maxDets[2]) # AP @ IoU=0.50
            stats[2] = _summarize(ap_type=1, iouThr=.75, maxDets=self.params.maxDets[2]) # AP @ IoU=0.75
            stats[3] = _summarize(ap_type=1, areaRng='small', maxDets=self.params.maxDets[2]) # AP_small
            stats[4] = _summarize(ap_type=1, areaRng='medium', maxDets=self.params.maxDets[2]) # AP_medium
            stats[5] = _summarize(ap_type=1, areaRng='large', maxDets=self.params.maxDets[2]) # AP_large
            # AR (Average Recall at operating point)
            stats[6] = _summarize(ap_type=0, maxDets=self.params.maxDets[0]) # AR @ maxDets=1
            stats[7] = _summarize(ap_type=0, maxDets=self.params.maxDets[1]) # AR @ maxDets=10
            stats[8] = _summarize(ap_type=0, maxDets=self.params.maxDets[2]) # AR @ maxDets=100
            stats[9] = _summarize(ap_type=0, areaRng='small', maxDets=self.params.maxDets[2]) # AR_small
            stats[10] = _summarize(ap_type=0, areaRng='medium', maxDets=self.params.maxDets[2]) # AR_medium
            stats[11] = _summarize(ap_type=0, areaRng='large', maxDets=self.params.maxDets[2]) # AR_large
            
            # P (Average Precision at operating point) - Các chỉ số này thường ít dùng hơn AP/AR/F1
            stats[12] = _summarize(ap_type=3, maxDets=self.params.maxDets[0]) # P @ maxDets=1
            stats[13] = _summarize(ap_type=3, maxDets=self.params.maxDets[1]) # P @ maxDets=10
            stats[14] = _summarize(ap_type=3, maxDets=self.params.maxDets[2]) # P @ maxDets=100
            stats[15] = _summarize(ap_type=3, areaRng='small', maxDets=self.params.maxDets[2]) # P_small
            stats[16] = _summarize(ap_type=3, areaRng='medium', maxDets=self.params.maxDets[2]) # P_medium
            stats[17] = _summarize(ap_type=3, areaRng='large', maxDets=self.params.maxDets[2]) # P_large

            # F1 (Average F1-score at operating point)
            stats[18] = _summarize(ap_type=2, maxDets=self.params.maxDets[0]) # F1 @ IoU=0.50:0.95 | maxDets=1
            stats[19] = _summarize(ap_type=2, maxDets=self.params.maxDets[1]) # F1 @ IoU=0.50:0.95 | maxDets=10
            stats[20] = _summarize(ap_type=2, maxDets=self.params.maxDets[2]) # F1 @ IoU=0.50:0.95 | maxDets=100
            stats[21] = _summarize(ap_type=2, areaRng='small', maxDets=self.params.maxDets[2]) # F1_small
            stats[22] = _summarize(ap_type=2, areaRng='medium', maxDets=self.params.maxDets[2]) # F1_medium
            stats[23] = _summarize(ap_type=2, areaRng='large', maxDets=self.params.maxDets[2]) # F1_large
            
            # Thêm F1 @ IoU=0.5
            stats[24] = _summarize(ap_type=2, iouThr=.5, maxDets=self.params.maxDets[2]) # F1 @ IoU=0.50 | maxDets=100
            return stats

        def _summarizeKps(): # Giữ nguyên cho keypoints nếu cần
            stats = np.zeros((10,))
            stats[0] = _summarize(ap_type=1, maxDets=20)
            stats[1] = _summarize(ap_type=1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(ap_type=1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(ap_type=1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(ap_type=1, maxDets=20, areaRng='large')
            stats[5] = _summarize(ap_type=0, maxDets=20)
            stats[6] = _summarize(ap_type=0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(ap_type=0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(ap_type=0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(ap_type=0, maxDets=20, areaRng='large')
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize_func = _summarizeDets # Sửa tên biến
        elif iouType == 'keypoints':
            summarize_func = _summarizeKps # Sửa tên biến
        else: # Thêm xử lý cho iouType không hợp lệ
            raise ValueError(f"Unsupported iouType for summarization: {iouType}")
        self.stats = summarize_func()

    def __str__(self):
        self.summarize()
        return "" # __str__ nên trả về một string

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        #self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        #replaced (đây là thay đổi của bạn, giữ nguyên)
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 64 ** 2],[64 ** 2, 192 ** 2], [192 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        # Đảm bảo kpt_oks_sigmas là một list hoặc tuple trước khi gán cho np.array trong computeOks
        self.kpt_oks_sigmas = [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89] # Chia cho 10.0 sau
        self.kpt_oks_sigmas = np.array(self.kpt_oks_sigmas)/10.0


    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None

# Ví dụ cách sử dụng (nếu bạn chạy file này trực tiếp)
if __name__ == '__main__':
    # Giả sử bạn có cocoGt và cocoDt đã được tải
    # from pycocotools.coco import COCO
    # gt_file = 'path/to/your/ground_truth.json'
    # dt_file = 'path/to/your/detections.json'
    # cocoGt = COCO(gt_file)
    # cocoDt = cocoGt.loadRes(dt_file)

    # evaluator = COCOeval(cocoGt, cocoDt, iouType='bbox')
    # evaluator.evaluate()
    # evaluator.accumulate()
    # evaluator.summarize()

    # In ra các chỉ số stats
    # print("\nAll stats:")
    # for i, stat_val in enumerate(evaluator.stats):
    #     print(f"stats[{i}] = {stat_val:.3f}")

    # Cụ thể F1@0.5
    # print(f"\nF1-score @ IoU=0.50 | area=all | maxDets=100: {evaluator.stats[24]:.3f}")
    print("COCOeval_modified class defined. Ready for use with COCO ground truth and detection objects.")
    print("To use, instantiate COCOeval(cocoGt, cocoDt, iouType), then call evaluate(), accumulate(), summarize().")
    print("F1-score @ IoU=0.5 will be available in evaluator.stats[24].")