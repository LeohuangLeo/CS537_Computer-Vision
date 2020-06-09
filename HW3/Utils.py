import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
try:
    from PIL import Image
except ImportError:
    import Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import torch
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt


cv2_scale = lambda x: cv2.resize(x, dsize=(32, 32),
                                 interpolation=cv2.INTER_LINEAR)

np_reshape = lambda x: np.reshape(x, (32, 32, 1))


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x


def visualize_kps(img, xys, show_patch=True, r=4):
    img = img.copy()
    image1 = Image.fromarray(np.array(img), "RGB")
    draw = ImageDraw.Draw(image1)
    for x, y in xys:
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(255,0,0,255))
        draw.text((x+r, y+r), "({:},{:})".format(x, y), font = ImageFont.truetype("arial.ttf", 25), fill=(255,0,0,255))
    num_images = len(xys) + 1
    plt.figure(figsize = (15, 15))
    plt.imshow(np.array(image1))
    plt.show()

    # get gray patches, need to convert original image to gray first
    img = np.array(img, dtype=np.uint8)
    grayed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    patches = getPatches(xys, grayed_img, 64)
    r = 10
    # draw patches
    if show_patch:
        for i in range(len(xys)):
            one_patch = patches[i].view(64, 64).numpy()#.expand(64, 64, 3)
            ct_patch = len(one_patch)//2
            image_patch = Image.fromarray(one_patch)#, "RGB")
            draw = ImageDraw.Draw(image_patch)
            draw.ellipse((ct_patch-r, ct_patch-r, ct_patch+r, ct_patch+r))
            draw.line([(ct_patch-r, ct_patch), (ct_patch+r, ct_patch)])
            draw.line([(ct_patch, ct_patch-r), (ct_patch, ct_patch+r)])
            draw.text((ct_patch-3*r, ct_patch+r), "({:},{:})".format(xys[i][0], xys[i][1]), font = ImageFont.truetype("arial.ttf", 12))
            plt.figure(figsize = (8, 8))
            plt.imshow(np.array(image_patch))
            plt.show()

def getPatches(kps, img, size=32, num=500):
    res = torch.zeros(num, 1, size, size)
    if type(img) is np.ndarray:
        img = torch.from_numpy(img)
    h, w = img.shape      # note: for image, the x direction is the verticle, y-direction is the horizontal...
    for i in range(num):
        cx, cy = kps[i]
        cx, cy = int(cx), int(cy)
        dd = int(size/2)
        xmin, xmax = max(0, cx - dd), min(w, cx + dd )
        ymin, ymax = max(0, cy - dd), min(h, cy + dd )

        xmin_res, xmax_res = dd - min(dd,cx), dd + min(dd, w - cx)
        ymin_res, ymax_res = dd - min(dd,cy), dd + min(dd, h - cy)
        res[i, 0, ymin_res: ymax_res, xmin_res: xmax_res] = img[ymin: ymax, xmin: xmax]
    return res


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c= img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,4)
        img1 = cv2.circle(img1,tuple(pt1[:2]),10,color,-1)
        img2 = cv2.circle(img2,tuple(pt2[:2]),10,color,-1)
    return img1,img2


def drawlines1(img1,img2,line1_F0, line1_F1, line2_F0, line2_F1,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines
    '''
    '''
    FOR HW3
    '''
    w,h, channel= img1.shape
#     img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
#     img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    color1 = (255, 0, 255)
    color2 = (0, 255, 255)
#     color2 = tuple(np.random.randint(0,255,3).tolist())
    F0x0,F0y0 = map(int, [0, -line1_F0[2]/line1_F0[1] ])
    F0x1,F0y1 = map(int, [h, -(line1_F0[2]+line1_F0[0]*h)/line1_F0[1]])
    img1 = cv2.line(img1, (F0x0, F0y0), (F0x1, F0y1), color1, 4)

    if line1_F1 is not None:
        F1x0,F1y0 = map(int, [0, -line1_F1[2]/line1_F1[1] ])
        F1x1,F1y1 = map(int, [h, -(line1_F1[2]+line1_F1[0]*h)/line1_F1[1]])
        img1 = cv2.line(img1, (F1x0, F1y0), (F1x1, F1y1), color1, 4)

    F0x0,F0y0 = map(int, [0, -line2_F0[2]/line2_F0[1] ])
    F0x1,F0y1 = map(int, [h, -(line2_F0[2]+line2_F0[0]*h)/line2_F0[1]])
    img2 = cv2.line(img2, (F0x0, F0y0), (F0x1, F0y1), color2, 4)

    if line2_F1 is not None:
        F1x0,F1y0 = map(int, [0, -line2_F1[2]/line2_F1[1] ])
        F1x1,F1y1 = map(int, [h, -(line2_F1[2]+line2_F1[0]*h)/line2_F1[1]])
        img2 = cv2.line(img2, (F1x0, F1y0), (F1x1, F1y1), color2, 4)

    # draw points
    img1 = cv2.circle(img1,tuple(pts1),20,color2,-1)
    img2 = cv2.circle(img2,tuple(pts2),20,color1,-1)

    return img1,img2

def drawKptsLines(img1, img2, kpt, line2_F0, line2_F1, line2_F2):
    '''
    img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines

    USED FOR HW3
    '''
    w,h, channel= img1.shape
    color1 = (255, 0, 255)
    color2 = (0, 255, 255)
    color3 = (255, 255, 0)
    # draw kpt on img 1
    img1 = cv2.circle(img1,tuple(kpt),20,color1,-1)

    # draw line2s on img2
    if line2_F0 is not None:
        F0x0,F0y0 = map(int, [0, -line2_F0[2]/line2_F0[1] ])
        F0x1,F0y1 = map(int, [h, -(line2_F0[2]+line2_F0[0]*h)/line2_F0[1]])
        img2 = cv2.line(img2, (F0x0, F0y0), (F0x1, F0y1), color1, 4)

    if line2_F1 is not None:
        F1x0,F1y0 = map(int, [0, -line2_F1[2]/line2_F1[1] ])
        F1x1,F1y1 = map(int, [h, -(line2_F1[2]+line2_F1[0]*h)/line2_F1[1]])
        img2 = cv2.line(img2, (F1x0, F1y0), (F1x1, F1y1), color2, 4)

    if line2_F1 is not None:
        F1x0,F1y0 = map(int, [0, -line2_F2[2]/line2_F2[1] ])
        F1x1,F1y1 = map(int, [h, -(line2_F2[2]+line2_F2[0]*h)/line2_F2[1]])
        img2 = cv2.line(img2, (F1x0, F1y0), (F1x1, F1y1), color3, 4)

    return img1,img2
def drawEpipoles(img1, kpt, color = (255, 0, 255)):
    '''
    draw kpt on img1
    '''
    img1 = cv2.circle(img1,tuple(kpt),20,color,-1)
    return img1


def get_kps_pair_appearance_cost(des1, des2):
    # des1 = N x 128
    # des2 = N x 128
    N = des1.shape[0]
    des11 = des1.view(N, 1, 128).expand(N, N, 128).contiguous().view(-1, 128).double()
    des22 = des2.view(1, N, 128).expand(N, N, 128).contiguous().view(-1, 128).double()
    simi = (1 + F.cosine_similarity(des11, des22))/2
    simi = simi.view(N, N)
    cost = 1 - simi

    # take abs, and normalize
    cost = cost.abs()
    cost = cost / cost.max()
    return cost

def get_kps_pair_geometric_cost(kps1, kps2, fund):
    # kps1 = 30 x 2
    # kps2 = 30 x 2
    # fund = 3 x 3, tensor
    N = kps1.shape[0]
    pts1 = torch.cat((kps1, torch.ones(N, 1)), dim=1).double()
    pts2 = torch.cat((kps2, torch.ones(N, 1)), dim=1).double()
    pts2_fund = torch.mm(pts2, fund.double())
    cost = torch.mm(pts2_fund, pts1.t())

    # take abs, and normalize
    cost = cost.abs()
    cost = cost / cost.max()
    return cost


def getCost_one2one(des1, des2, kps_num=20, topk=10):
    # des1 = 20 x 128
    # des2 = 20 x 128
    cost = get_kps_pair_appearance_cost(des1, des2)
    return getCost_one2one_with_cost(cost, kps_num, topk)


def getCost_one2one_with_cost(cost, kps_num=20, topk=10):
    # cost: 100 x 100
    row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())

    # calculate l2 distance
    cost_list = cost.view(-1)

    # select matched pair
    ind = row_ind * kps_num + col_ind
    cost_list = cost_list[ind]

    # take top k
    _, idx = torch.topk(cost_list, topk, dim=0, largest=False )
    ind = ind[idx.cpu().numpy()]
    row = np.floor(ind/kps_num).astype(int)
    col = np.floor(ind%kps_num).astype(int)
    return row, col


def epipoleSVD(F):
    V = cv2.SVDecomp(F)[2]
    return V[-1]/V[-1,-1]


def epipoleSVD1(f):
    _, _, V = linalg.svd(f)
    e = V[-1]
    return e/e[-1]
