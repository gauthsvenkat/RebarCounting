import torch
import torch.nn.functional as F
import numpy as np 
from skimage.morphology import watershed
from skimage.segmentation import find_boundaries
from scipy import ndimage
import utils as ut

def lc_loss(model, batch):
    
    model.train() #set model to training mode (batchnorm and dropout layers work accordingly)
    N =  batch["images"].size(0) #get the number of images in batch
    assert N == 1 #make sure batch size is 1

    blob_dict = get_blob_dict(model, batch) #get all blob information
    # put variables in gpu memory
    images = batch["images"].cuda()
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()

    O = model(images) #output of model (1, k, h, w)
    S = F.softmax(O, 1) #softmax along the k dimension
    S_log = F.log_softmax(O, 1) #log softmax along k dimension

    # IMAGE LOSS
    loss = compute_image_loss(S, counts)

    # POINT LOSS
    loss += F.nll_loss(S_log, points, 
                        ignore_index=0,
                        reduction='sum')
    # FP loss
    if blob_dict["n_fp"] > 0:
        loss += compute_fp_loss(S_log, blob_dict)

    # split_mode loss
    if blob_dict["n_multi"] > 0:
        loss += compute_split_loss(S_log, S, points, blob_dict)

    
    # Global loss 
    S_npy = ut.t2n(S.squeeze()) #convert to numpy
    points_npy = ut.t2n(points).squeeze() #convert to numpy and squeeze
    for l in range(1, S.shape[1]): #iterate over all classes
        points_class = (points_npy==l).astype(int) #locations of that particular class

        if points_class.sum() == 0: #if no points in the class skip
            continue

        T = watersplit(S_npy[l], points_class) #watershed segmentation on the probabilities
        T = 1 - T #convert the borders to 0 (background)
        scale = float(counts.sum())
        loss += float(scale) * F.nll_loss(S_log, torch.LongTensor(T).cuda()[None],
                        ignore_index=1, reduction='mean')


    return loss


# Loss Utils
def compute_image_loss(S, Counts):
    n,k,h,w = S.size()

    # GET TARGET
    
    ones = torch.ones(Counts.size(0), 1).long().cuda() #initialize ones tensor with shape (K, 1)
    BgFgCounts = torch.cat([ones, Counts], 1) #background forground counts
    Target = (BgFgCounts.view(n*k).view(-1) > 0).view(-1).float() #flatten and get target

    # GET INPUT
    Smax = S.view(n, k, h*w).max(2)[0].view(-1) #get the model predictions
    
    loss = F.binary_cross_entropy(Smax, Target, reduction='sum') #BCE of predictions and target
    
    return loss

def compute_fp_loss(S_log, blob_dict):

    blobs = blob_dict["blobs"] #contains the actual blobs
    
    scale = 1.
    loss = 0.

    for b in blob_dict["blobList"]:
        if b["n_points"] != 0: #ignore if not false positives
            continue

        T = np.ones(blobs.shape[-2:]) #get (H,W)
        T[blobs[b["class"]] == b["label"]] = 0 #get the prediction of a particular blob from a particular class and set it to 0 (Background)

        loss += scale * F.nll_loss(S_log, torch.LongTensor(T).cuda()[None],
                        ignore_index=1, reduction='mean')
    return loss 

def compute_split_loss(S_log, S, points, blob_dict):
    blobs = blob_dict["blobs"]
    S_numpy = ut.t2n(S[0]) #convert to numpy array
    points_numpy = ut.t2n(points).squeeze() #convert to numpy array and squeeze it
    
    loss = 0.

    for b in blob_dict["blobList"]:
        if b["n_points"] < 2: #ignore if single point in a blob
            continue

        l = b["class"] + 1 #get the class (+1 cause screw background)
        probs = S_numpy[b["class"] + 1] #get corresponding predictions

        points_class = (points_numpy==l).astype("int") #get the particular class' annotation
        blob_ind = blobs[b["class"] ] == b["label"] #get the particlar blob of a particlar class

        T = watersplit(probs, points_class*blob_ind)*blob_ind #watershed segmentation on the probabilities
        T = 1 - T #convert the borders to 0 (background)

        scale = b["n_points"] + 1 #loss scale
        loss += float(scale) * F.nll_loss(S_log, torch.LongTensor(T).cuda()[None],
                        ignore_index=1, reduction='mean')

    return loss 


def watersplit(_probs, _points):
   points = _points.copy()

   points[points!=0] = np.arange(1, points.sum()+1)
   points = points.astype(float)

   probs = ndimage.black_tophat(_probs.copy(), 7)   
   seg =  watershed(probs, points)

   return find_boundaries(seg)


@torch.no_grad() #temporarily set requires_grad to false 
def get_blob_dict(model, batch, training=False): 
    blobs = model.predict(batch, method="blobs").squeeze() #predicted blobs of shape (k, h, w) with labeled blobs
    points = ut.t2n(batch["points"]).squeeze() #actual location of objects

    if blobs.ndim == 2:      #If k == 1 squeeze would also remove that 
        blobs = blobs[None]  #so to add an extra dimension if k==1

    blobList = []

    n_multi = 0
    n_single = 0
    n_fp = 0

    for l in range(blobs.shape[0]): #iterating through k classes
        class_blobs = blobs[l] #class_blobs now contains labelled blobs of that specific class
        points_mask = points == (l+1) #points_mask now contains actual location of object of that particular class
        # Intersecting
        blob_uniques, blob_counts = np.unique(class_blobs * (points_mask), return_counts=True) #blob_uniques now only contains labelled points that are inside a predicted blob
        uniques = np.delete(np.unique(class_blobs), blob_uniques)  #Delete correctly predicted blobs

        for u in uniques: #iterate over falsely predicted blobs
            blobList += [{"class":l, "label":u, "n_points":0}]
            n_fp += 1 #counter to increment false positives

        for i, u in enumerate(blob_uniques): #iterate over the labelled blobs
            if u == 0: #ignore if background
                continue

            blob_ind = class_blobs==u #work with one specific labelled blob

            locs = np.where(blob_ind * (points_mask)) #get location of all the points in the labelled blob

            if blob_counts[i] == 1: #If only one point in blob
                n_single += 1

            else: #if multiple points in blob
                n_multi += 1
            blobList += [{"class":l, "label":u, "n_points":blob_counts[i]}]

    blob_dict = {"blobs":blobs, #labelled blobs from morph.label
                 "blobList":blobList, #information on each blob
                 "n_fp":n_fp, #number of false positives
                 "n_single":n_single, #number of blobs with only one point
                 "n_multi":n_multi,} #number of blobs with multiple points

    return blob_dict
    #################### FLAG ENDS ####################