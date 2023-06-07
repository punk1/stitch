#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <time.h>

enum POOL
{
    MAX = 0,
    SUM = 1,
};
enum INTERPOLATE
{
    NEAREST = 0,
    BILINEAR = 1,
};

__device__ bool within_bounds_2d(int x, int y, int w, int h)
{
    return x >= 0 && x < w && y >= 0 && y < h;
}
__device__ bool within_bounds_3d(int b_idx, int x, int y, int b, int w, int h)
{
    return b_idx >= 0 && b_idx < b && within_bounds_2d(x, y, w, h);
}

__global__ void bev_height_pool_forward_kernel(int B, int N, int D, int C, int H, int W, int h, int w, int out_h, int out_w, int downsample, int grid_step,
                                               const float *__restrict__ _input_feats_ptr, const float *__restrict__ _points_ptr, const bool *__restrict__ _maskes_ptr,
                                               float *__restrict__ _out_ptr, unsigned char *__restrict__ _out_index_ptr, unsigned char *__restrict__ _out_num_ptr, int pool_method, bool overlap, int interpolate)
{
    /*
    pool_method: 0->max 1->sum
    */
    // sum of bin * bin feature
    int blockx = blockIdx.x % grid_step;
    int blocky = blockIdx.x / grid_step;

    int out_index_x = blockx * blockDim.x + threadIdx.x;
    int out_index_y = blocky * blockDim.y + threadIdx.y;

    if (within_bounds_2d(out_index_x, out_index_y, out_w, out_h))
    {
        int start_index_x = out_index_x * downsample;
        int start_index_y = out_index_y * downsample;

        // get b_idx, n_idx, d_idx
        int b_idx = blockIdx.z / D;
        int d_idx = blockIdx.z % D;
        if (!within_bounds_2d(b_idx, d_idx, B, D))
        {
            return;
        }

        bool debug = false;
        if (out_index_x == 0 && out_index_y == 0 && b_idx == 0 && d_idx == 0 && blockIdx.y == 0)
        {
            debug = false;
        }
        long long out_index_head = (long long)blockIdx.z * C * out_h * out_w + (long long)blockIdx.y * out_h * out_w;
        long long out_index = out_index_head + out_index_y * out_w + out_index_x;
        long long out_top_index = (long long)B * D * C * out_h * out_w;
        int out_num_index = b_idx * (D * out_h * out_w) + d_idx * (out_h * out_w) + out_index_y * out_w + out_index_x;
        long long out_max_index_head = b_idx * (N * D * C * out_h * out_w) + d_idx * (C * out_h * out_w) + blockIdx.y * out_h * out_w + out_index_y * out_w + out_index_x;

        // printf("out_index_head: %d, out_index: %d, out_num_index: %d, out_max_index_head: %d\n", out_index_head, out_index, out_num_index, out_max_index_head);
        float out_value = 0.0;
        int out_num = 0;

        for (int n_idx = 0; n_idx < N; n_idx++)
        {
            long long feature_index_head = (long long)b_idx * N * C * H * W + (long long)n_idx * C * H * W + (long long)blockIdx.y * H * W;
            long long points_index_head = (long long)b_idx * (N * D * h * w * 2) + (long long)n_idx * (D * h * w * 2) + (long long)d_idx * (h * w * 2);
            long long out_max_index = out_max_index_head + (long long)n_idx * (D * C * out_h * out_w);

            float update_value = -10000.0;
            bool update_flag = false;

            int max_index = 0;
            if (pool_method == POOL::SUM)
            {
                update_value = 0.0;
            }
            for (int index_y = start_index_y; index_y < start_index_y + downsample; index_y++)
            {
                for (int index_x = start_index_x; index_x < start_index_x + downsample; index_x++)
                {
                    if (!within_bounds_3d(blockIdx.z, index_x, index_y, B * D, w, h))
                    {
                        continue;
                    }
                    long long points_index = points_index_head + index_y * (w * 2) + index_x * 2;
                    long long maskes_index = points_index / 2;
                    if ((points_index + 1) >= B * N * D * h * w * 2)
                    {
                        printf("points_index exceed: %d\n", points_index);
                        continue;
                    }
                    float f_feat_x_index = _points_ptr[points_index];
                    float f_feat_y_index = _points_ptr[points_index + 1];

                    if (maskes_index >= B * N * D * h * w)
                    {
                        printf("maskes_index exceed: %d\n", maskes_index);
                        continue;
                    }
                    bool feat_mask = _maskes_ptr[maskes_index];

                    if (debug)
                    {
                        printf("b: %d, n: %d, d: %d, index_x: %d, index_y: %d, mask: %d, points_index: %d, vaule: %f\n", b_idx, n_idx, d_idx, index_x, index_y, feat_mask, points_index, update_value);
                    }
                    if (!feat_mask)
                    {
                        continue;
                    }
                    // TODOS add bilinear interpolate
                    float tmp_value = -10000;
                    if (pool_method == POOL::SUM)
                    {
                        tmp_value = 0.0;
                    }

                    if (interpolate == INTERPOLATE::NEAREST)
                    {
                        int feat_x_index = (int)f_feat_x_index;
                        int feat_y_index = (int)f_feat_y_index;
                        if (!within_bounds_2d(feat_x_index, feat_y_index, W, H))
                        {
                            printf("feat index exceed! %d, %d\n", feat_x_index, feat_y_index);
                            continue;
                        }

                        long long feature_index = feature_index_head + feat_y_index * W + feat_x_index;
                        tmp_value = _input_feats_ptr[feature_index];
                    }
                    else if (interpolate == INTERPOLATE::BILINEAR)
                    {
                        // get NE, NW, SE, SW pixel values from (x, y)
                        float tmp = 0.0;
                        int ix_nw = floor(f_feat_x_index);
                        int iy_nw = floor(f_feat_y_index);
                        int ix_ne = ix_nw + 1;
                        int iy_ne = iy_nw;
                        int ix_sw = ix_nw;
                        int iy_sw = iy_nw + 1;
                        int ix_se = ix_nw + 1;
                        int iy_se = iy_nw + 1;
                        float nw = (ix_se - f_feat_x_index) * (iy_se - f_feat_y_index);
                        float ne = (f_feat_x_index - ix_sw) * (iy_sw - f_feat_y_index);
                        float sw = (ix_ne - f_feat_x_index) * (f_feat_y_index - iy_ne);
                        float se = (f_feat_x_index - ix_nw) * (f_feat_y_index - iy_nw);

                        if (within_bounds_2d(ix_nw, iy_nw, W, H))
                        {
                            long long nw_index = feature_index_head + iy_nw * W + ix_nw;
                            if (nw_index >= B * N * C * H * W)
                            {
                                printf("nw index exced, %d\n", nw_index);
                                continue;
                            }
                            tmp += nw * _input_feats_ptr[nw_index];
                        }
                        if (within_bounds_2d(ix_ne, iy_ne, W, H))
                        {

                            long long ne_index = feature_index_head + iy_ne * W + ix_ne;
                            if (ne_index >= B * N * C * H * W)
                            {
                                printf("ne_index exced, %d\n", ne_index);
                                continue;
                            }
                            tmp += ne * _input_feats_ptr[ne_index];
                        }
                        if (within_bounds_2d(ix_sw, iy_sw, W, H))
                        {
                            long long sw_index = feature_index_head + iy_sw * W + ix_sw;
                            if (sw_index >= B * N * C * H * W)
                            {
                                printf("sw_index exced, %d\n", sw_index);
                                continue;
                            }
                            tmp += sw * _input_feats_ptr[sw_index];
                        }
                        if (within_bounds_2d(ix_se, iy_se, W, H))
                        {
                            long long se_index = feature_index_head + iy_se * W + ix_se;
                            if (se_index >= B * N * C * H * W)
                            {
                                printf("se_index exced, %d\n", se_index);
                                continue;
                            }
                            tmp += se * _input_feats_ptr[se_index];
                        }
                        tmp_value = tmp;
                    }
                    if (pool_method == POOL::MAX)
                    {
                        if (tmp_value > update_value)
                        {
                            update_value = tmp_value;
                            max_index = (index_y - start_index_y) * downsample + index_x - start_index_x;
                            update_flag = feat_mask;
                        }
                    }
                    else if (pool_method == POOL::SUM)
                    {
                        update_value = update_value + tmp_value;
                        update_flag = feat_mask;
                        if (debug)
                        {
                            printf("tmp_value: %f, update_flag: %d, update_value: %f\n", tmp_value, update_flag, update_value);
                        }
                    }
                }
            }
            if (pool_method == POOL::MAX && !update_flag)
            {
                update_value = 0.0;
            }
            // TODOS out_max_index B * N * D * C * out_h * out_w exceed int max value
            long long top_index = (long long)B * N * D * C * out_h * out_w;
            if (out_max_index >= top_index)
            {
                printf("out max index excedd, out_max_index: %d, dimsize: %#llx\n", out_max_index, top_index);
                printf("B, N, D, C, out_h, out_w, %d, %d, %d, %d, %d, %d\n", B, N, D, C, out_h, out_w);
                continue;
            }
            if (pool_method == POOL::MAX)
            {
                _out_index_ptr[out_max_index] = (unsigned char)max_index;
            }
            out_num += int(update_flag);
            out_value += update_value;
            if (debug)
            {
                printf("update_value: %f, max_index: %d, out_num: %d, out_value: %f\n", update_value, max_index, out_num, out_value);
            }
        }
        if (blockIdx.y == 0)
        {
            if (out_num_index >= B * D * out_h * out_w)
            {
                printf("out num index excedd, %d\n", out_num_index);
                return;
            }
            _out_num_ptr[out_num_index] = (unsigned char)out_num;
        }
        if (overlap && out_num > 0)
        {
            out_value /= out_num;
        }
        if (out_index >= out_top_index)
        {
            printf("out index excedd, out_index: %#llx, out_top_index: %#llx\n", out_index, out_top_index);
            return;
        }
        _out_ptr[out_index] = out_value;
    }
}

__global__ void bev_height_pool_backward_kernel(int B, int N, int D, int C, int H, int W, int h, int w, int out_h, int out_w, int downsample, int grid_step,
                                                const float *__restrict__ _out_grad_ptr, const float *__restrict__ _points_ptr, const bool *__restrict__ _maskes_ptr, const unsigned char *__restrict__ _out_index_ptr,
                                                const unsigned char *__restrict__ _out_num_ptr, float *__restrict__ _in_grad_ptr, int pool_method, bool overlap, int interpolate)
{

    // sum of bin * bin feature
    int blockx = blockIdx.x % grid_step;
    int blocky = blockIdx.x / grid_step;

    int out_index_x = blockx * blockDim.x + threadIdx.x;
    int out_index_y = blocky * blockDim.y + threadIdx.y;

    if (out_index_x >= 0 && out_index_x < out_w && out_index_y >= 0 && out_index_y < out_h)
    {
        int start_index_x = out_index_x * downsample;
        int start_index_y = out_index_y * downsample;

        // get b_idx, n_idx, d_idx
        int b_idx = blockIdx.z / D;
        int d_idx = blockIdx.z % D;

        // int points_index_head = blockIdx.z * h * w * 2;
        long long out_index_head = (long long)blockIdx.z * C * out_h * out_w + (long long)blockIdx.y * out_h * out_w;
        long long out_max_index_head = (long long)b_idx * (N * D * C * out_h * out_w) + (long long)d_idx * (C * out_h * out_w) + (long long)blockIdx.y * out_h * out_w + (long long)out_index_y * out_w + (long long)out_index_x;

        long long out_index = out_index_head + out_index_y * out_w + out_index_x;
        int out_num_index = b_idx * (D * out_h * out_w) + d_idx * (out_h * out_w) + out_index_y * out_w + out_index_x;

        float out_grad = _out_grad_ptr[out_index];
        int out_num = (int)_out_num_ptr[out_num_index];

        for (int n_idx = 0; n_idx < N; n_idx++)
        {
            if (b_idx < B && n_idx < N && blockIdx.y < C)
            {
                long long out_max_index = out_max_index_head + n_idx * (D * C * out_h * out_w);
                int max_index = 0;
                if (pool_method == POOL::MAX)
                {
                    max_index = _out_index_ptr[out_max_index];
                }

                int index_x_ = max_index % downsample;
                int index_y_ = max_index / downsample;
                long long feature_index_head = b_idx * N * C * H * W + n_idx * C * H * W + blockIdx.y * H * W;
                long long points_index_head = b_idx * (N * D * h * w * 2) + n_idx * (D * h * w * 2) + d_idx * (h * w * 2);

                for (int index_y = start_index_y; index_y < start_index_y + downsample; index_y++)
                {
                    for (int index_x = start_index_x; index_x < start_index_x + downsample; index_x++)
                    {
                        if (pool_method == POOL::MAX && !((index_x - start_index_x) == index_x_ && (index_y - start_index_y) == index_y_))
                        {
                            continue;
                        }
                        if (!within_bounds_3d(blockIdx.z, index_x, index_y, B * D, w, h))
                        {
                            continue;
                        }
                        long long points_index = points_index_head + index_y * (w * 2) + index_x * 2;
                        long long maskes_index = points_index / 2;
                        float f_feat_x_index = _points_ptr[points_index];
                        float f_feat_y_index = _points_ptr[points_index + 1];

                        // TODOS add bilinear grad back
                        bool feat_mask = _maskes_ptr[maskes_index];

                        if (!feat_mask)
                        {
                            continue;
                        }
                        if (interpolate == INTERPOLATE::NEAREST)
                        {

                            int feat_x_index = (int)f_feat_x_index;
                            int feat_y_index = (int)f_feat_y_index;
                            if (!within_bounds_2d(feat_x_index, feat_y_index, W, H))
                            {
                                printf("feat index exceed! %d, %d\n", feat_x_index, feat_y_index);
                                continue;
                            }

                            long long feature_index = feature_index_head + feat_y_index * W + feat_x_index;

                            if (overlap && out_num > 0)
                            {
                                atomicAdd(&_in_grad_ptr[feature_index], out_grad / out_num);
                            }
                            else if (!overlap)
                            {
                                atomicAdd(&_in_grad_ptr[feature_index], out_grad);
                            }
                        }
                        else if (interpolate == INTERPOLATE::BILINEAR)
                        {
                            int ix_nw = floor(f_feat_x_index);
                            int iy_nw = floor(f_feat_y_index);
                            int ix_ne = ix_nw + 1;
                            int iy_ne = iy_nw;
                            int ix_sw = ix_nw;
                            int iy_sw = iy_nw + 1;
                            int ix_se = ix_nw + 1;
                            int iy_se = iy_nw + 1;
                            float nw = (ix_se - f_feat_x_index) * (iy_se - f_feat_y_index);
                            float ne = (f_feat_x_index - ix_sw) * (iy_sw - f_feat_y_index);
                            float sw = (ix_ne - f_feat_x_index) * (f_feat_y_index - iy_ne);
                            float se = (f_feat_x_index - ix_nw) * (f_feat_y_index - iy_nw);
                            float tmp = out_grad;
                            if (overlap && out_num > 0)
                            {
                                tmp = tmp / out_num;
                            }
                            else if (overlap)
                            {
                                continue;
                            }
                            if (within_bounds_2d(ix_nw, iy_nw, W, H))
                            {
                                long long nw_index = feature_index_head + iy_nw * W + ix_nw;
                                atomicAdd(&_in_grad_ptr[nw_index], tmp * nw);
                            }
                            if (within_bounds_2d(ix_ne, iy_ne, W, H))
                            {
                                long long ne_index = feature_index_head + iy_ne * W + ix_ne;
                                atomicAdd(&_in_grad_ptr[ne_index], tmp * ne);
                            }
                            if (within_bounds_2d(ix_sw, iy_sw, W, H))
                            {
                                long long sw_index = feature_index_head + iy_sw * W + ix_sw;
                                atomicAdd(&_in_grad_ptr[sw_index], tmp * sw);
                            }
                            if (within_bounds_2d(ix_se, iy_se, W, H))
                            {
                                long long se_index = feature_index_head + iy_se * W + ix_se;
                                atomicAdd(&_in_grad_ptr[se_index], tmp * se);
                            }
                        }
                    }
                }
            }
        }
    }
}

void bev_height_pool_forward_function(int B, int N, int D, int C, int H, int W, int h, int w, int downsample,
                                      const float *_input_feats_ptr, const float *_points_ptr, const bool *_maskes_ptr,
                                      float *_out_ptr, unsigned char *_out_index_ptr, unsigned char *_out_num_ptr, int pool_method, bool overlap, int interpolate)
{
    int block_x = 32, block_y = 16;
    int grid_step = (int)ceil((float)w / downsample / block_x);
    int grid_x = (int)ceil((float)w / downsample / block_x) * (int)ceil((float)h / downsample / block_y);
    int grid_y = C;
    int grid_z = B * D;
    int out_h = (int)(h / downsample);
    int out_w = (int)(w / downsample);

    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y);
    // printf("block, x,y: %d, %d, grid x,y,z: %d, %d, %d, grid_step: %d\n", block_x, block_y, grid_x, grid_y, grid_z, grid_step);
    bev_height_pool_forward_kernel<<<grid, block>>>(B, N, D, C, H, W, h, w, out_h, out_w, downsample, grid_step, _input_feats_ptr, _points_ptr, _maskes_ptr, _out_ptr, _out_index_ptr, _out_num_ptr, pool_method, overlap, interpolate);
    cudaDeviceSynchronize();
}

void bev_height_pool_backward_function(int B, int N, int D, int C, int H, int W, int h, int w, int downsample,
                                       const float *_out_grad_ptr, const float *_points_ptr, const bool *_maskes_ptr, const unsigned char *_out_index_ptr,
                                       const unsigned char *_out_num_ptr, float *_in_grad_ptr, int pool_method, bool overlap, int interpolate)
{
    int block_x = 32, block_y = 16;
    int grid_step = (int)ceil((float)w / downsample / block_x);
    int grid_x = (int)ceil((float)w / downsample / block_x) * (int)ceil((float)h / downsample / block_y);
    int grid_y = C;
    int grid_z = B * D;
    int out_h = (int)(h / downsample);
    int out_w = (int)(w / downsample);

    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y);

    bev_height_pool_backward_kernel<<<grid, block>>>(B, N, D, C, H, W, h, w, out_h, out_w, downsample, grid_step, _out_grad_ptr, _points_ptr, _maskes_ptr, _out_index_ptr, _out_num_ptr, _in_grad_ptr, pool_method, overlap, interpolate);
    cudaDeviceSynchronize();
}

// int main()
// {
//     std::srand((unsigned)time(NULL));

//     int B = 8, N = 6, D = 8, C = 768, H = 16, W = 44;
//     int size = 64;
//     int out_size = 64;
//     int downsample = size / out_size;
//     int pool_method = 1;
//     bool overlap = true;
//     int interpolate = 1;

//     float *c_input_feat;
//     c_input_feat = (float *)malloc(sizeof(float) * B * N * C * H * W);
//     for (int i = 0; i < B * N * C * H * W; i++)
//     {
//         c_input_feat[i] = rand() / double(RAND_MAX);
//     }
//     float *c_points;
//     c_points = (float *)malloc(sizeof(float) * B * N * D * size * size * 2);
//     bool *c_maskes;
//     c_maskes = (bool *)malloc(sizeof(bool) * B * N * D * size * size);
//     for (int i = 0; i < B * N * D * size * size; i++)
//     {
//         float x = (W - 1) * (float)rand() / RAND_MAX;
//         float y = (H - 1) * (float)rand() / RAND_MAX;
//         c_points[i * 2] = x;
//         c_points[i * 2 + 1] = y;
//         bool mask = rand() % 2;
//         c_maskes[i] = mask;
//     }

//     float *g_input_feat;
//     cudaMalloc((void **)&g_input_feat, sizeof(float) * B * N * C * H * W);
//     cudaMemcpy(g_input_feat, c_input_feat, sizeof(float) * B * N * C * H * W, cudaMemcpyHostToDevice);
//     float *g_points;
//     cudaMalloc((void **)&g_points, sizeof(float) * B * N * D * size * size * 2);
//     cudaMemcpy(g_points, c_points, sizeof(float) * B * N * D * size * size * 2, cudaMemcpyHostToDevice);
//     bool *g_maskes;
//     cudaMalloc((void **)&g_maskes, sizeof(bool) * B * N * D * size * size);
//     cudaMemcpy(g_maskes, c_maskes, sizeof(bool) * B * N * D * size * size, cudaMemcpyHostToDevice);
//     std::cout << "cuda malloc successed!\n";
//     // out
//     float *g_out;
//     cudaMalloc((void **)&g_out, sizeof(float) * B * D * C * out_size * out_size);
//     cudaMemset(g_out, 0.0, sizeof(float) * B * D * C * out_size * out_size);
//     unsigned char *g_out_index;
//     cudaMalloc((void **)&g_out_index, sizeof(unsigned char) * B * N * D * C * out_size * out_size);
//     cudaMemset(g_out_index, 0, sizeof(unsigned char) * B * N * D * C * out_size * out_size);
//     unsigned char *g_out_num;
//     cudaMalloc((void **)&g_out_num, sizeof(unsigned char) * B * D * out_size * out_size);
//     cudaMemset(g_out_num, 0, sizeof(unsigned char) * B * D * out_size * out_size);

//     bev_height_pool_forward_function(B, N, D, C, H, W, out_size, out_size, downsample, g_input_feat, g_points, g_maskes, g_out, g_out_index, g_out_num, pool_method, overlap, interpolate);

//     float *c_out;
//     c_out = (float *)malloc(sizeof(float) * B * D * C * out_size * out_size);
//     unsigned char *c_out_index;
//     c_out_index = (unsigned char *)malloc(sizeof(unsigned char) * B * N * D * C * out_size * out_size);
//     unsigned char *c_out_num;
//     c_out_num = (unsigned char *)malloc(sizeof(unsigned char) * B * D * out_size * out_size);

//     std::cout << "start memory device to host\n";
//     cudaMemcpy(c_out, g_out, sizeof(float) * B * D * C * out_size * out_size, cudaMemcpyDeviceToHost);
//     cudaMemcpy(c_out_index, g_out_index, sizeof(unsigned char) * B * N * D * C * out_size * out_size, cudaMemcpyDeviceToHost);
//     cudaMemcpy(c_out_num, g_out_num, sizeof(unsigned char) * B * D * out_size * out_size, cudaMemcpyDeviceToHost);

//     cudaFree(g_out);
//     cudaFree(g_out_index);
//     cudaFree(g_out_num);
//     cudaFree(g_input_feat);
//     cudaFree(g_points);
//     cudaFree(g_maskes);

//     for (int i = 0; i < 100; i++)
//     {
//         std::cout << "in_feature " << i << "th " << c_input_feat[i] << std::endl;
//         std::cout << "points " << i << "th " << c_points[i] << std::endl;
//         std::cout << "maskes " << i << "th " << c_maskes[i] << std::endl;
//         std::cout << "out_feature " << i << "th " << c_out[i] << std::endl;
//     }
//     std::cout << "in_feature: 1048576th " << c_input_feat[1048576] << std::endl;
//     return 0;
// }
