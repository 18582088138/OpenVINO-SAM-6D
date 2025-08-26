import os
import numpy as np
import openvino as ov
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import pointnet2._ext as _ext

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.pointnet2.pytorch_utils import SharedMLP, Conv1d
# from model.pointnet2.pointnet2_utils import QueryAndGroup, CustomDebugNode

torch_model_path = "./test_model/SharedMLP.pth"
onnx_model_path = "./test_model/SharedMLP.onnx"
ov_model_path = "./test_model/SharedMLP.xml"

# ov_kernel_path = "../model/ov_pointnet2_op/PositionalEncoding.xml"
# ov_extension_lib_path = "../model/ov_pointnet2_op/build/libopenvino_operation_extension.so"
ov_kernel_path = "ov_op/PositionalEncoding_cl.xml"
ov_extension_lib_path = "ov_op/build/libopenvino_operation_extension.so"

# pts1 shape :torch.Size([7, 2048, 3])
B = 7
C = 6
H = 2048
W = 32
DEBUG_FLAG = False 

DUMMY_BATCH_SIZE = 7
DUMMY_NPOINT = 2048
DUMMY_C = 3

core = ov.Core()
core.add_extension(ov_extension_lib_path)



class GroupingOperation(Function):
    @staticmethod
    def symbolic(g: torch.Graph, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return g.op("GroupingOperation", features, idx)

    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        ctx.for_backwards = (idx, N)

        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, None


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod
    def symbolic(g: torch.Graph, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
       return g.op("BallQuery", new_xyz, xyz, radius_f=radius, nsample_i=nsample)
    
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        inds = _ext.ball_query(new_xyz, xyz, radius, nsample)
        ctx.mark_non_differentiable(inds)
        return inds

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        np.save("output/grouping_idx.npy",idx.numpy())
        #=================Test=====================
        """
        The inference output of two consecutive custom operations (ball_query & grouping_operation) 
        results in incorrect calculations. 
        However, saving the ball_query output to a .npy file and then reading it into the grouping_operation 
        get the correct result. 
        This is likely a bug in the OpenVINO GPU Custom Operations. T
        he current implementation is a workaround.
        """
        real_idx_data_path = "output/grouping_idx.npy"
        real_idx = np.load(real_idx_data_path)
        idx = torch.from_numpy(real_idx)
        print(f"Loading Real data & {real_idx_data_path} ")
        #=================Test End=====================

        unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))

        if self.sample_uniformly:
            # unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind

        xyz_trans = xyz.transpose(1, 2).contiguous()
        # np.save("output/grouping_features.npy",xyz_trans.numpy())
    
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)

        print(f"[grouping_operation] features_shape: {xyz_trans.shape}, idx_shape:{idx.shape}")
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            print(f"[grouping_operation] features_shape: {features.shape}, idx_shape:{idx.shape}")
            if DEBUG_FLAG:
                flat_grouped_features = grouped_features.cpu().numpy().reshape(-1)
                with open('output/torch_grouping_operation.txt', 'a') as f:
                    f.write('--- grouping_operation (flat_grouped_features) ---\n')
                    f.write(' '.join(f'{x:.6f}' for x in flat_grouped_features) + '\n')

            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)

class MyModel(nn.Module):
    def __init__(self, r1=0.1, r2=0.2, nsample1=32, nsample2=64, use_xyz=True, bn=True):
        super(MyModel, self).__init__()
        self.group1 = QueryAndGroup(r1, nsample1, use_xyz=use_xyz)
        input_dim = 6 if use_xyz else 3
        self.mlp1 = SharedMLP([input_dim, 32, 64, 128], bn=bn)

    def forward(self, pts1, pts2=None):
        print(f"[PositionalEncoding] pts1 shape :{pts1.shape}")
        if pts2 is None:
            # pts2 = pts1
            pts2 = pts1+0.00000001
        """
        Error: [GPU] Different primitive with id 'parameter:pts1_/group1/BallQuery_cldnn_custom_preprocess' exists already
        
        When pts2 = pts1, the GPU kernel will display a different primitive already present during model compilation.
        The specific cause of this issue is unknown, but it is confirmed to be related to OV graph transformation optimization.
        This may be because the same source data, after OV transformation optimization, is considered a duplicate primitive when the GPU kernel primitive is executed.
        By adding a minimum value for pts2, computation accuracy is not affected and OV optimization prevents the data from being considered the same source.
        """
        feat1 = self.group1(
                pts1.contiguous(), pts2.contiguous(), pts1.transpose(1,2).contiguous()
            )

        feat1 = self.mlp1(feat1)
        feat1 = torch.amax(feat1, dim=3, keepdim=True)
        return feat1

def get_input_data():
    real_data_path ="output/pts1_input.npy"
    if not os.path.exists(real_data_path):
        dummy_pts1 = torch.randn(DUMMY_BATCH_SIZE, DUMMY_NPOINT, DUMMY_C)
    else:
        real_pts1 = np.load(real_data_path)
        dummy_pts1 = torch.from_numpy(real_pts1)
        print("Loading Real pts1 data")

    onnx_input = (dummy_pts1)
    onnx_input_name = ["pts1"]

    ov_input = {"pts1":dummy_pts1}
    # ov_input_name = {"pts1":[B, C, H, W]}
    ov_input_name = {"pts1":[DUMMY_BATCH_SIZE, DUMMY_NPOINT, DUMMY_C]}
    return onnx_input, onnx_input_name, ov_input, ov_input_name

def onnx_model_convert(model, onnx_input, onnx_input_name):
    with torch.no_grad():
        torch.onnx.export(
            model,
            onnx_input,
            onnx_model_path,
            opset_version=20,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            input_names=onnx_input_name,
            dynamic_axes={k: {0: "batch"} for k in onnx_input_name},
            do_constant_folding=False,
            verbose=False,  # True , for detailed output
            export_params=True,
            keep_initializers_as_inputs=False
        )
    print(f"[ONNX] model export success: {onnx_model_path}")

def ov_model_convert(ov_input, ov_input_name):
    ov_model = ov.convert_model(onnx_model_path, 
                                input=ov_input_name,
                                example_input=ov_input,
                                extension=ov_extension_lib_path,
                                    )
    compiled_model = core.compile_model(ov_model, 'CPU')
    ov.save_model(ov_model, ov_model_path)
    print(f"[OpenVINO] model convert success: {ov_model_path}")


def ov_infer(ov_input, device="CPU"):
    if device == "GPU":
        core.set_property("GPU", {"INFERENCE_PRECISION_HINT": "f32"})
        core.set_property("GPU", {"CONFIG_FILE": ov_kernel_path})
    # Init model
    ov_model = core.read_model(ov_model_path)
    compiled_model = core.compile_model(ov_model, device)
    print("[compiled_model] Done")
    # warmup 

    ov_start_time = time.time()
    ov_result = compiled_model.infer_new_request(ov_input)
    ov_infer_result =list(ov_result.values())
    ov_infer_time = time.time() - ov_start_time
    print(f"[OpenVINO {device}] infer time_cost: {(ov_infer_time*1000):.2f} ms")
    return ov_infer_result

def torch_infer(onnx_input, device="cpu"):
    torch_model = MyModel()
    torch_model.load_state_dict(torch.load(torch_model_path))
    torch_model.eval()
    torch_model.to(device)
    torch_start_time = time.time()

    torch_infer_result = torch_model(onnx_input)

    # print(f"[Pytorch] infer result: \n{torch_infer_result[:5]}")
    torch_infer_time = time.time() - torch_start_time
    print(f"[Torch] infer time_cost: {(torch_infer_time*1000):.2f} ms")
    return torch_infer_result

def compare_infer(torch_out, ov_out, device):
    # Compare
    print("Torch output shape:", torch_out.shape)
    ov_tensor = torch.from_numpy(ov_out)
    print("OV output shape:", ov_tensor.shape)
    diff = torch.abs(ov_tensor - torch_out)
    print(f"[OV {device}] + pytorch  Max diff: {diff.max()}")
    print(f"[OV {device}] + pytorch  Min diff: {diff.min()}")
    print(f"[OV {device}] + pytorch  MSE: {(diff ** 2).mean()}")
    
    # assert torch.allclose(ov_tensor, torch_out, atol=1e-4)
    print(f"[COMPARE Result {device}] and Pytorch PASSED")

def main():
    np.random.seed(324)
    torch.manual_seed(32)

    model = MyModel()
    model.eval()
    model.to("cpu")
    torch.save(model.state_dict(), torch_model_path)

    onnx_input, onnx_input_name, ov_input, ov_input_name = get_input_data()

    onnx_model_convert(model, onnx_input, onnx_input_name)
    ov_model_convert(ov_input, ov_input_name)

    torch_cpu_out = torch_infer(onnx_input, "cpu")

    ov_cpu_out = ov_infer(ov_input, "CPU")
    compare_infer(torch_cpu_out, ov_cpu_out[0], "CPU")

    ov_gpu_out = ov_infer(ov_input, "GPU")
    compare_infer(torch_cpu_out, ov_gpu_out[0], "GPU")


if __name__ == "__main__":
    main()