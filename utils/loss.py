# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true, Activation = "sigmoid"):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        if Activation == "sigmoid":
            pred_prob = torch.sigmoid(pred)
        else:
            pred_prob = torch.softmax(pred)  # prob from logits

        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets
        # cp正样例、cn负样例

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g, "softmax"), FocalLoss(BCEobj, g, "softmax")

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc
        self.nc_total = sum(self.nc)
        # self.nc1 = m.nc[0]  # number of classes
        # self.nc2 = m.nc[1]  # classes2 --->m这里修改了
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        self.label_mapping = {
            0: [1, 0],
            1: [1, 0],
            2: [1, 1],
            3: [1, 1],
            4: [1, 2],
            5: [1, 2],
            6: [1, 3],
            7: [1, 3],
            8: [0, 0],
            9: [0, 0],
            10: [0, 1],
            11: [0, 1],
            12: [0, 2],
            13: [0, 2],
            14: [0, 3],
            15: [0, 3],
            16: [2, 0],
            17: [2, 0],
            18: [2, 1],
            19: [2, 1],
            20: [2, 2],
            21: [2, 2],
            22: [2, 3],
            23: [2, 3],
            24: [3, 9],
            25: [0, 9],
            26: [1, 9],
            27: [2, 9],
            28: [4, 9],
            29: [0, 5],
            30: [1, 5],
            31: [2, 5],
            32: [0, 9],
            33: [1, 9],
            34: [2, 9],
            35: [1, 8],
            36: [0, 8],
            37: [1, 8],
            38: [1, 8],
            39: [1, 4],
            40: [0, 4],
            41: [2, 4],
            42: [0, 6],
            43: [1, 6],
            44: [0, 7],
            45: [1, 7],
            46: [1, 9],
            47: [0, 9],
            48: [1, 9],
            49: [0, 9]
        }
        # self.label_mapping = torch.tensor(list(self.label_mapping.values()))
        # self.label_mapping.to(self.device)


    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(len(self.nc), device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj
            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc_total), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                tcls_res = {}
                for k in range(len(self.nc)):
                    tcls_res[k] = []
                    # tcls_res[i] = []
                # tcls1 = []
                # tcls2 = []
                # print("tcls type {}, tcls{}".format(type(tcls), tcls))
                for lab in tcls:
                    values_tensor = torch.tensor(list(self.label_mapping.values()), device=self.device)
                    lab = lab.to(self.device)
                    # print("lab device {}".format(lab.device))
                    # lab.to(self.device)
                    temp_res = {}
                    for k in range(len(self.nc)):
                        temp_res = values_tensor[lab, k]
                        tcls_res[k].append(temp_res)
                    # temp_res = values_tensor[lab, 0]
                    # temp_res[] = values_tensor[lab, 1]

                    # tcls_res.append(temp_tcls1)
                    # tcls2.append(temp_tcls2)

                shift_tcls = 0
                for k in range(len(self.nc)):
                    # Classification
                    if self.nc[k] > 1:  # cls loss (only if multiple classes)

                        t = torch.full_like(pcls[:,shift_tcls:shift_tcls+self.nc[k],...], self.cn, device=self.device)  # targets
                        t[range(n), tcls_res[k][i]] = self.cp
                        lcls[k] += self.BCEcls(pcls[:,shift_tcls:shift_tcls+self.nc[k],...], t)  # BCE
                        shift_tcls += self.nc[k]


                    # # Classification
                    # if self.nc2 > 1:  # cls loss (only if multiple classes)
                    #     t2 = torch.full_like(pcls2, self.cn, device=self.device)  # targets
                    #     t2[range(n), tcls2[i]] = self.cp
                    #     lcls2 += self.BCEcls(pcls2, t2)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        for i in range(len(self.nc)):
            lcls[i] *= self.hyp[f'cls{i+1}'] # 类别权重参数
            if i == 1:
                lcls[i] *= 4 # 针对交通灯任务加个权重
        # lcls1 *= self.hyp['cls1']
        # lcls2 *= self.hyp['cls2']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls.sum()) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        nt1 = 5
        nt2 = 10

        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
