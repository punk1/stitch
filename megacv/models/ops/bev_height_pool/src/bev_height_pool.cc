/*
 * @Description  :
 * @Version      : 1.0
 * @Author       : chengzuo.qi
 * @Date         : 2022-07-25 17:58:27
 * @LastEditTime : 2022-08-09 11:36:11
 * Copyright (C) 2022 mega. All rights reserved.
 */

#include <torch/torch.h>
#include <stdio.h>

void bev_height_pool_forward_function(int B, int N, int D, int C, int H, int W, int h, int w, int downsample,
                                      const float *_input_feats_ptr, const float *_points_ptr, const bool *_maskes_ptr,
                                      float *_out_ptr, unsigned char *_out_index_ptr, unsigned char *_out_num_ptr, int pool_method, bool overlap, int interpolate);
void bev_height_pool_backward_function(int B, int N, int D, int C, int H, int W, int h, int w, int downsample,
                                       const float *_out_grad_ptr, const float *_points_ptr, const bool *_maskes_ptr, const unsigned char *_out_index_ptr,
                                       const unsigned char *_out_num_ptr, float *_in_grad_ptr, int pool_method, bool overlap, int interpolate);

/*
    Function: bev pillar height pooling
*/
std::vector<at::Tensor> bev_height_pool_forward(
    const at::Tensor _input_feats,
    const at::Tensor _points,
    const at::Tensor _maskes,
    int downsample = 4,
    int pool_method = 0,
    bool overlap = false,
    int interpolate = 0)
{
    int B = _input_feats.size(0), N = _input_feats.size(1), C = _input_feats.size(2), H = _input_feats.size(3), W = _input_feats.size(4);
    int D = _points.size(2), h = _points.size(3), w = _points.size(4);

    const float *_input_feats_ptr = _input_feats.data_ptr<float>();
    const float *_points_ptr = _points.data_ptr<float>();
    const bool *_maskes_ptr = _maskes.data_ptr<bool>();

    auto options = torch::TensorOptions().dtype(_input_feats.dtype()).device(_input_feats.device());
    at::Tensor _out = torch::zeros({B, D, C, h / downsample, w / downsample}, options);
    auto options_index = torch::TensorOptions().dtype(torch::kUInt8).device(_input_feats.device());

    at::Tensor _out_index = torch::zeros({1, 1}, options_index);
    if (pool_method == 0)
    {
        _out_index = torch::zeros({B, N, D, C, h / downsample, w / downsample}, options_index);
    }
    options_index = torch::TensorOptions().dtype(torch::kUInt8).device(_input_feats.device());
    at::Tensor _out_num = torch::zeros({B, D, h / downsample, w / downsample}, options_index);
    float *_out_ptr = _out.data_ptr<float>();
    unsigned char *_out_index_ptr = NULL;
    if (pool_method == 0)
    {
        _out_index_ptr = _out_index.data_ptr<unsigned char>();
    }

    unsigned char *_out_num_ptr = _out_num.data_ptr<unsigned char>();
    bev_height_pool_forward_function(B, N, D, C, H, W, h, w, downsample, _input_feats_ptr, _points_ptr, _maskes_ptr, _out_ptr, _out_index_ptr, _out_num_ptr, pool_method, overlap, interpolate);

    std::vector<at::Tensor> tmp;
    tmp.push_back(_out);
    tmp.push_back(_out_index);
    tmp.push_back(_out_num);
    return tmp;
}

at::Tensor bev_height_pool_backward(
    const at::Tensor _out_grad,
    const at::Tensor _points,
    const at::Tensor _maskes,
    const at::Tensor _out_index,
    const at::Tensor _out_num,
    int H, int W, int downsample, int pool_method, bool overlap, int interpolate)
{
    int B = _points.size(0), N = _points.size(1), D = _points.size(2), h = _points.size(3), w = _points.size(4);
    int C = _out_grad.size(2);
    const float *_out_grad_ptr = _out_grad.data_ptr<float>();
    const float *_points_ptr = _points.data_ptr<float>();
    const bool *_maskes_ptr = _maskes.data_ptr<bool>();
    const unsigned char *_out_index_ptr = NULL;
    if (pool_method == 0)
    {
        _out_index_ptr = _out_index.data_ptr<unsigned char>();
    }
    // const unsigned char *_out_index_ptr = _out_index.data_ptr<unsigned char>();
    const unsigned char *_out_num_ptr = _out_num.data_ptr<unsigned char>();
    auto options = torch::TensorOptions().dtype(_out_grad.dtype()).device(_out_grad.device());
    at::Tensor _in_grad = torch::zeros({B, N, C, H, W}, options);
    float *_in_grad_ptr = _in_grad.data_ptr<float>();
    bev_height_pool_backward_function(B, N, D, C, H, W, h, w, downsample, _out_grad_ptr, _points_ptr, _maskes_ptr, _out_index_ptr, _out_num_ptr, _in_grad_ptr, pool_method, overlap, interpolate);
    return _in_grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("bev_height_pool_forward", &bev_height_pool_forward,
          "bev_height_pool_forward");
    m.def("bev_height_pool_backward", &bev_height_pool_backward,
          "bev_height_pool_backward");
}
