import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

from pointnet2_utils import (
    gather_operation,
    furthest_point_sample,
)

DEBUG_FLAG = False

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed



def sample_pts_feats(pts, feats, npoint=2048, return_index=False):
    '''
        pts: B*N*3
        feats: B*N*C
    '''
    # sample_idx = furthest_point_sample(pts, npoint)
    sample_idx = furthest_point_sample(pts, torch.tensor(npoint))
    if DEBUG_FLAG:
        # print(f"[Torch] FurthestPointSampling output: {sample_idx}")
        flat_sample_idx = sample_idx.cpu().detach().numpy().reshape(-1)
        with open('output/torch_furthest_point_sampling.txt', 'a') as f:
            f.write('--- furthest_point_sample ---\n')
            f.write(' '.join(f'{x:.6f}' for x in flat_sample_idx) + '\n')
    
    pts = gather_operation(pts.transpose(1,2).contiguous(), sample_idx)
    if DEBUG_FLAG:
        flat_pts = pts.cpu().detach().numpy().reshape(-1)
        # print(f"[Torch] gather_operation(pts) output: {flat_pts}")
        with open('output/torch_gather_operation.txt', 'a') as f:
            f.write('--- gather_operation(pts) ---\n')
            f.write(' '.join(f'{x:.6f}' for x in flat_pts) + '\n')
    pts = pts.transpose(1,2).contiguous()
    feats = gather_operation(feats.transpose(1,2).contiguous(), sample_idx)
    if DEBUG_FLAG:
        flat_feats = feats.cpu().detach().numpy().reshape(-1)
        # print(f"[Torch] gather_operation(feats) output: {flat_feats}")
        with open('output/torch_gather_operation.txt', 'a') as f:
            f.write('--- gather_operation(feats) ---\n')
            f.write(' '.join(f'{x:.6f}' for x in flat_feats) + '\n')
    feats = feats.transpose(1,2).contiguous()
    if return_index:
        return pts, feats, sample_idx
    else:
        return pts, feats


def get_chosen_pixel_feats(img, choose):
    shape = img.size()
    if len(shape) == 3:
        pass
    elif len(shape) == 4:
        B, C, H, W = shape
        img = img.reshape(B, C, H*W)
    else:
        assert False

    choose = choose.unsqueeze(1).repeat(1, C, 1)
    x = torch.gather(img, 2, choose).contiguous()
    return x.transpose(1,2).contiguous()


def pairwise_distance(
    x: torch.Tensor, y: torch.Tensor, normalized: bool = False, channel_first: bool = False
) -> torch.Tensor:
    r"""Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2), y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    sq_distances = sq_distances.clamp(min=0.0)
    return sq_distances


def compute_feature_similarity(feat1, feat2, type='cosine', temp=1.0, normalize_feat=True):
    r'''
    Args:
        feat1 (Tensor): (B, N, C)
        feat2 (Tensor): (B, M, C)

    Returns:
        atten_mat (Tensor): (B, N, M)
    '''
    if normalize_feat:
        feat1 = F.normalize(feat1, p=2, dim=2)
        feat2 = F.normalize(feat2, p=2, dim=2)

    if type == 'cosine':
        atten_mat = feat1 @ feat2.transpose(1,2)
    elif type == 'L2':
        atten_mat = torch.sqrt(pairwise_distance(feat1, feat2))
    else:
        assert False

    atten_mat = atten_mat / temp

    return atten_mat



def aug_pose_noise(gt_r, gt_t,
                std_rots=[15, 10, 5, 1.25, 1],
                max_rot=45,
                sel_std_trans=[0.2, 0.2, 0.2],
                max_trans=0.8):

    B = gt_r.size(0)
    device = gt_r.device

    std_rot = np.random.choice(std_rots)
    angles = torch.normal(mean=0, std=std_rot, size=(B, 3)).to(device=device)
    angles = angles.clamp(min=-max_rot, max=max_rot)
    ones = gt_r.new(B, 1, 1).zero_() + 1
    zeros = gt_r.new(B, 1, 1).zero_()
    a1 = angles[:,0].reshape(B, 1, 1) * np.pi / 180.0
    a1 = torch.cat(
        [torch.cat([torch.cos(a1), -torch.sin(a1), zeros], dim=2),
        torch.cat([torch.sin(a1), torch.cos(a1), zeros], dim=2),
        torch.cat([zeros, zeros, ones], dim=2)], dim=1
    )
    a2 = angles[:,1].reshape(B, 1, 1) * np.pi / 180.0
    a2 = torch.cat(
        [torch.cat([ones, zeros, zeros], dim=2),
        torch.cat([zeros, torch.cos(a2), -torch.sin(a2)], dim=2),
        torch.cat([zeros, torch.sin(a2), torch.cos(a2)], dim=2)], dim=1
    )
    a3 = angles[:,2].reshape(B, 1, 1) * np.pi / 180.0
    a3 = torch.cat(
        [torch.cat([torch.cos(a3), zeros, torch.sin(a3)], dim=2),
        torch.cat([zeros, ones, zeros], dim=2),
        torch.cat([-torch.sin(a3), zeros, torch.cos(a3)], dim=2)], dim=1
    )
    rand_rot = a1 @ a2 @ a3

    rand_trans = torch.normal(
        mean=torch.zeros([B, 3]).to(device),
        std=torch.tensor(sel_std_trans, device=device).view(1, 3),
    )
    rand_trans = torch.clamp(rand_trans, min=-max_trans, max=max_trans)

    rand_rot = gt_r @ rand_rot
    rand_trans = gt_t + rand_trans
    rand_trans[:,2] = torch.clamp(rand_trans[:,2], min=1e-6)

    return rand_rot.detach(), rand_trans.detach()


def compute_coarse_Rt(
    atten,
    pts1,
    pts2,
    model_pts=None,
    n_proposal1=6000,
    n_proposal2=300,
):
    WSVD = WeightedProcrustes()

    B, N1, _ = pts1.size()
    N2 = pts2.size(1)
    device = pts1.device

    if model_pts is None:
        model_pts = pts2
    expand_model_pts = model_pts.unsqueeze(1).repeat(1,n_proposal2,1,1).reshape(B*n_proposal2, -1, 3)

    # compute soft assignment matrix
    pred_score = torch.softmax(atten, dim=2) * torch.softmax(atten, dim=1)
    pred_label1 = torch.max(pred_score[:,1:,:], dim=2)[1]
    pred_label2 = torch.max(pred_score[:,:,1:], dim=1)[1]
    weights1 = (pred_label1>0).float()
    weights2 = (pred_label2>0).float()

    pred_score = pred_score[:, 1:, 1:].contiguous()
    pred_score = pred_score * weights1.unsqueeze(2) * weights2.unsqueeze(1)
    pred_score = pred_score.reshape(B, N1*N2) ** 1.5

    # sample pose hypothese
    cumsum_weights = torch.cumsum(pred_score, dim=1)
    cumsum_weights /= (cumsum_weights[:, -1].unsqueeze(1).contiguous()+1e-8)
    
    # idx = torch.searchsorted(cumsum_weights, torch.rand(B, n_proposal1*3, device=device))  #Exporting the operator 'aten::searchsorted' to ONNX opset version 20 is not supported
    # Replace torch.searchsorted with ONNX compatible implementation

    idx = CustomSearchSorted.apply(cumsum_weights, torch.rand(B, n_proposal1*3, device=device))
    if DEBUG_FLAG:
        flat_idx = idx.cpu().detach().numpy().reshape(-1)
        with open('output/torch_search_sorted.txt', 'a') as f:
            f.write('--- torch CustomSearchSorted ---\n')
            f.write(' '.join(f'{x:.2f}' for x in flat_idx) + '\n')
    
    # idx = weighted_sampling_onnx_compatible(cumsum_weights, B, n_proposal1*3, device)
    # if DEBUG_FLAG:
    #     flat_idx = idx.cpu().detach().numpy().reshape(-1)
    #     with open('output/torch_search_sorted.txt', 'a') as f:
    #         f.write('--- torch CustomSearchSorted onnx compatible ---\n')
    #         f.write(' '.join(f'{x:.6f}' for x in flat_idx) + '\n')

    idx1, idx2 = idx.div(N2, rounding_mode='floor'), idx % N2
    idx1 = torch.clamp(idx1, max=N1-1).unsqueeze(2).repeat(1,1,3)
    idx2 = torch.clamp(idx2, max=N2-1).unsqueeze(2).repeat(1,1,3)

    p1 = torch.gather(pts1, 1, idx1).reshape(B,n_proposal1,3,3).reshape(B*n_proposal1,3,3)
    p2 = torch.gather(pts2, 1, idx2).reshape(B,n_proposal1,3,3).reshape(B*n_proposal1,3,3)
    pred_rs, pred_ts = WSVD(p2, p1, None)
    pred_rs = pred_rs.reshape(B, n_proposal1, 3, 3)
    pred_ts = pred_ts.reshape(B, n_proposal1, 1, 3)

    p1 = p1.reshape(B, n_proposal1, 3, 3)
    p2 = p2.reshape(B, n_proposal1, 3, 3)
    dis = torch.norm((p1 - pred_ts) @ pred_rs - p2, dim=3).mean(2)
    idx = torch.topk(dis, n_proposal2, dim=1, largest=False)[1]
    pred_rs = torch.gather(pred_rs, 1, idx.reshape(B,n_proposal2,1,1).repeat(1,1,3,3))
    pred_ts = torch.gather(pred_ts, 1, idx.reshape(B,n_proposal2,1,1).repeat(1,1,1,3))

    # pose selection
    transformed_pts = (pts1.unsqueeze(1) - pred_ts) @ pred_rs
    transformed_pts = transformed_pts.reshape(B*n_proposal2, -1, 3)
    dis = torch.sqrt(pairwise_distance(transformed_pts, expand_model_pts))
    dis = dis.min(2)[0].reshape(B, n_proposal2, -1)
    scores = weights1.unsqueeze(1).sum(2) / ((dis * weights1.unsqueeze(1)).sum(2) + + 1e-8)
    idx = scores.max(1)[1]
    pred_R = torch.gather(pred_rs, 1, idx.reshape(B,1,1,1).repeat(1,1,3,3)).squeeze(1)
    pred_t = torch.gather(pred_ts, 1, idx.reshape(B,1,1,1).repeat(1,1,1,3)).squeeze(2).squeeze(1)
    return pred_R, pred_t


def weighted_sampling_onnx_compatible(cumsum_weights, batch_size, num_samples, device):
    """
    [Not used] This will cause huge memory overhead, is not a good implementation method, and is not conducive to cuda infer
    ONNX compatible weighted sampling implementation, replacing torch.searchsorted
    
    Args:
        cumsum_weights: (B, N) cumulative weights
        batch_size: batch size
        num_samples: number of samples
        device: device
    
    Returns:
        idx: (B, num_samples) sample index
    """
    # Generate random numbers
    random_values = torch.rand(batch_size, num_samples, device=device)

    # cumsum_weights: (B, N) -> (B, 1, N)
    # random_values: (B, num_samples) -> (B, num_samples, 1)
    cumsum_expanded = cumsum_weights.unsqueeze(1)  # (B, 1, N)
    random_expanded = random_values.unsqueeze(2)   # (B, num_samples, 1)
    
    # Compare: cumsum_weights >= random_values
    # Result: (B, num_samples, N)
    comparison = cumsum_expanded >= random_expanded
    idx = torch.argmax(comparison.float(), dim=2)  # (B, num_samples)
    
    return idx


def compute_fine_Rt(
    atten,
    pts1,
    pts2,
    model_pts=None,
    dis_thres=0.15
):
    if model_pts is None:
        model_pts = pts2

    # compute pose
    WSVD = WeightedProcrustes(weight_thresh=0.0)
    assginment_mat = torch.softmax(atten, dim=2) * torch.softmax(atten, dim=1)
    label1 = torch.max(assginment_mat[:,1:,:], dim=2)[1]
    label2 = torch.max(assginment_mat[:,:,1:], dim=1)[1]

    assginment_mat = assginment_mat[:, 1:, 1:] * (label1>0).float().unsqueeze(2) * (label2>0).float().unsqueeze(1)
    # max_idx = torch.max(assginment_mat, dim=2, keepdim=True)[1]
    # pred_pts = torch.gather(pts2, 1, max_idx.expand_as(pts1))
    normalized_assginment_mat = assginment_mat / (assginment_mat.sum(2, keepdim=True) + 1e-6)
    pred_pts = normalized_assginment_mat @ pts2

    assginment_score = assginment_mat.sum(2)
    pred_R, pred_t = WSVD(pred_pts, pts1, assginment_score)

    # compute score
    pred_pts = (pts1 - pred_t.unsqueeze(1)) @ pred_R
    dis = torch.sqrt(pairwise_distance(pred_pts, model_pts)).min(2)[0]
    mask = (label1>0).float()
    pose_score = (dis < dis_thres).float()
    pose_score = (pose_score * mask).sum(1) / (mask.sum(1) + 1e-8)
    pose_score = pose_score * mask.mean(1)

    return pred_R, pred_t, pose_score


def weighted_procrustes(
    src_points,
    ref_points,
    weights=None,
    weight_thresh=0.0,
    eps=1e-5,
    return_transform=False,
    src_centroid = None,
    ref_centroid = None,
):
    r"""Compute rigid transformation from `src_points` to `ref_points` using weighted SVD.

    Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).

    Args:
        src_points: torch.Tensor (B, N, 3) or (N, 3)
        ref_points: torch.Tensor (B, N, 3) or (N, 3)
        weights: torch.Tensor (B, N) or (N,) (default: None)
        weight_thresh: float (default: 0.)
        eps: float (default: 1e-5)
        return_transform: bool (default: False)

    Returns:
        R: torch.Tensor (B, 3, 3) or (3, 3)
        t: torch.Tensor (B, 3) or (3,)
        transform: torch.Tensor (B, 4, 4) or (4, 4)
    """
    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        ref_points = ref_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0])
    weights = torch.where(torch.lt(weights, weight_thresh), torch.zeros_like(weights), weights)
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)
    weights = weights.unsqueeze(2)  # (B, N, 1)

    if src_centroid is None:
        src_centroid = torch.sum(src_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    elif len(src_centroid.size()) == 2:
        src_centroid = src_centroid.unsqueeze(1)
    src_points_centered = src_points - src_centroid  # (B, N, 3)

    if ref_centroid is None:
        ref_centroid = torch.sum(ref_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    elif len(ref_centroid.size()) == 2:
        ref_centroid = ref_centroid.unsqueeze(1)
    ref_points_centered = ref_points - ref_centroid  # (B, N, 3)

    H = src_points_centered.permute(0, 2, 1) @ (weights * ref_points_centered)
    
    # U, _, V = torch.svd(H) #Exporting the operator 'aten::svd' to ONNX opset version 20 is not supported
    # Replace torch.svd with ONNX compatible implementation
    print(f"[weighted_procrustes] torch.svd input shape: {H.shape}")
    U, _, V = CustomSVD.apply(H)
    if DEBUG_FLAG:
        flat_H = H.cpu().detach().numpy().reshape(-1)
        flat_U = U.cpu().detach().numpy().reshape(-1)
        flat_V = V.cpu().detach().numpy().reshape(-1)
        with open('output/torch_svd.txt', 'a') as f:
            f.write('--- torch custom_svd (input) ---\n')
            f.write(' '.join(f'{x:.2f}' for x in flat_H) + '\n')
            f.write('--- torch custom_svd (U) ---\n')
            f.write(' '.join(f'{x:.2f}' for x in flat_U) + '\n')
            f.write('--- torch custom_svd (V) ---\n')
            f.write(' '.join(f'{x:.2f}' for x in flat_V) + '\n')

    # U, _, V = svd_eigh_replace(H)
    # U, _, V = svd_onnx_compatible(H)
    
    Ut, V = U.transpose(1, 2), V
    eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(src_points.device)
    # eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    det_V_Ut = CustomDet.apply(V @ Ut)
    if DEBUG_FLAG:
        flat_det_V_Ut = det_V_Ut.cpu().detach().numpy().reshape(-1)
        with open('output/torch_det.txt', 'a') as f:
            f.write('--- torch_det(V @ Ut) ---\n')
            f.write(' '.join(f'{x:.6f}' for x in flat_det_V_Ut) + '\n')
    eye[:, -1, -1] = torch.sign(det_V_Ut)
    R = V @ eye @ Ut

    t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    t = t.squeeze(2)

    if return_transform:
        transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(src_points.device)
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        if squeeze_first:
            transform = transform.squeeze(0)
        return transform
    else:
        if squeeze_first:
            R = R.squeeze(0)
            t = t.squeeze(0)
        return R, t



class WeightedProcrustes(nn.Module):
    def __init__(self, weight_thresh=0.5, eps=1e-5, return_transform=False):
        super(WeightedProcrustes, self).__init__()
        self.weight_thresh = weight_thresh
        self.eps = eps
        self.return_transform = return_transform

    def forward(self, src_points, tgt_points, weights=None,src_centroid = None,ref_centroid = None):
        return weighted_procrustes(
            src_points,
            tgt_points,
            weights=weights,
            weight_thresh=self.weight_thresh,
            eps=self.eps,
            return_transform=self.return_transform,
            src_centroid=src_centroid,
            ref_centroid=ref_centroid
        )


class CustomSearchSorted(torch.autograd.Function):
    def __init__(self):
        super(CustomSearchSorted, self).__init__()

    @staticmethod
    def forward(ctx, cumsum_weights, random_values):
        return torch.searchsorted(cumsum_weights, random_values)
    
    @staticmethod
    def symbolic(g, cumsum_weights, random_values):
        return g.op("CustomSearchSorted", cumsum_weights, random_values, outputs=1)

class CustomSVD(torch.autograd.Function):
    def __init__(self):
        super(CustomSVD, self).__init__()
    
    @staticmethod
    def forward(ctx, H):
        U, S, V = torch.svd(H)
        ctx.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def symbolic(g: torch.Graph, H: torch.Tensor) :
        return g.op("CustomSVD", H, outputs=3)

class CustomDet(torch.autograd.Function):
    def __init__(self):
        super(CustomDet, self).__init__()
    
    @staticmethod
    def forward(ctx, H):
        det = torch.det(H)
        ctx.save_for_backward(det)
        return det

    @staticmethod
    def symbolic(g: torch.Graph, H: torch.Tensor) :
        return g.op("CustomDet", H, outputs=1)


class CustomDebugNode(torch.autograd.Function):
    def __init__(self):
        super(CustomDebugNode, self).__init__()

    @staticmethod
    def forward(ctx, input):
        flat_input = input.cpu().detach().numpy().reshape(-1)
        print(f"[Shape]: {input.shape} , type: {type(input)}")
        with open('output/torch_debug_node.txt', 'a') as f:
            f.write('--- torch custom_debug_node (input) ---\n')
            f.write(f"[Shape]: {input.shape} , dtype: {type(input)}\n")
            # Only keep the first 100 data
            max_elements = min(100, len(flat_input))
            f.write(' '.join(f'{x:.2f}' for x in flat_input[:max_elements]) + '\n')
            if len(flat_input) > 100:
                f.write(f'... (truncated, total {len(flat_input)} elements)\n')
            f.write('\n')
        return input
    
    @staticmethod
    def symbolic(g, input):
        return g.op("CustomDebugNode", input, outputs=1)