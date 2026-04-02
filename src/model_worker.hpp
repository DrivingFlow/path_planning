#pragma once

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/core.hpp>

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <atomic>
#include <functional>

class ModelWorker {
public:
    static constexpr int GRID_SIZE = 201;
    static constexpr int N_FRAMES = 5;
    static constexpr int N_CHANNELS = 2;  // occ + delta

    struct Result {
        std::vector<cv::Mat> pred_grids;  // N_FRAMES x (GRID_SIZE x GRID_SIZE) CV_8UC1
        std::chrono::steady_clock::time_point input_time;
        float inference_ms = 0.0f;
        bool valid = false;
    };

    using LogFn = std::function<void(const std::string&)>;

    ModelWorker(const std::string& model_path, bool use_cuda, LogFn log_fn)
        : use_cuda_(use_cuda), log_(std::move(log_fn)), stop_(false) {

        auto device = use_cuda_ ? torch::kCUDA : torch::kCPU;
        log_("Loading TorchScript model from: " + model_path);
        model_ = torch::jit::load(model_path, device);
        model_.eval();
        if (use_cuda_) {
            model_.to(torch::kHalf);
            log_("Model loaded on CUDA with fp16");
        } else {
            log_("Model loaded on CPU with fp32");
        }

        // Warmup inference to trigger JIT compilation
        {
            auto dtype = use_cuda_ ? torch::kHalf : torch::kFloat32;
            auto dummy_grids = torch::zeros({1, N_FRAMES, N_CHANNELS, GRID_SIZE, GRID_SIZE},
                                             torch::TensorOptions().dtype(dtype).device(device));
            auto dummy_motion = torch::zeros({1, N_FRAMES, 2},
                                              torch::TensorOptions().dtype(dtype).device(device));
            torch::NoGradGuard no_grad;
            auto out = model_.forward({dummy_grids, dummy_motion}).toTensor();
            if (use_cuda_) torch::cuda::synchronize();
            log_("Warmup inference complete, output shape: [" +
                 std::to_string(out.size(0)) + "," + std::to_string(out.size(1)) + "," +
                 std::to_string(out.size(2)) + "," + std::to_string(out.size(3)) + "," +
                 std::to_string(out.size(4)) + "]");
        }

        worker_ = std::thread(&ModelWorker::workerLoop, this);
    }

    ~ModelWorker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_one();
        if (worker_.joinable()) worker_.join();
    }

    ModelWorker(const ModelWorker&) = delete;
    ModelWorker& operator=(const ModelWorker&) = delete;

    // Build tensors from EgoFrame-style data and submit for inference.
    // grids: N_FRAMES cv::Mat (GRID_SIZE x GRID_SIZE, CV_8UC1, 0 or 255)
    // deltas: N_FRAMES cv::Mat (same dims, delta encoding)
    // motion_fwd: N_FRAMES forward speeds
    // motion_yaw: N_FRAMES yaw rates
    void submit(const std::vector<cv::Mat>& grids,
                const std::vector<cv::Mat>& deltas,
                const std::vector<float>& motion_fwd,
                const std::vector<float>& motion_yaw,
                std::chrono::steady_clock::time_point input_time) {

        auto device = use_cuda_ ? torch::kCUDA : torch::kCPU;
        auto dtype = use_cuda_ ? torch::kHalf : torch::kFloat32;

        // Build (1, 5, 2, 201, 201) tensor
        auto x_grids = torch::zeros({1, N_FRAMES, N_CHANNELS, GRID_SIZE, GRID_SIZE},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        auto grids_acc = x_grids.accessor<float, 5>();

        for (int t = 0; t < N_FRAMES; ++t) {
            const cv::Mat& g = grids[t];
            const cv::Mat& d = deltas[t];
            for (int r = 0; r < GRID_SIZE; ++r) {
                const uchar* g_row = g.ptr<uchar>(r);
                const uchar* d_row = d.ptr<uchar>(r);
                for (int c = 0; c < GRID_SIZE; ++c) {
                    grids_acc[0][t][0][r][c] = g_row[c] ? 1.0f : 0.0f;
                    // Delta: -1 (newly free), 0 (unchanged), 1 (newly occupied)
                    // Stored in CV_8UC1: 0=free(-1 or 0), 255=occupied(1)
                    // For the delta channel, match what toAgentCenteredInput produces
                    grids_acc[0][t][1][r][c] = d_row[c] ? 1.0f : 0.0f;
                }
            }
        }

        // Build (1, 5, 2) motion tensor
        auto x_motion = torch::zeros({1, N_FRAMES, 2},
                                      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        auto motion_acc = x_motion.accessor<float, 3>();
        for (int t = 0; t < N_FRAMES; ++t) {
            motion_acc[0][t][0] = motion_fwd[t];
            motion_acc[0][t][1] = motion_yaw[t];
        }

        // Move to device and convert dtype
        x_grids = x_grids.to(device).to(dtype);
        x_motion = x_motion.to(device).to(dtype);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending_grids_ = std::move(x_grids);
            pending_motion_ = std::move(x_motion);
            pending_time_ = input_time;
            has_pending_ = true;
        }
        cv_.notify_one();
    }

    Result getLatestResult() {
        std::lock_guard<std::mutex> lock(result_mutex_);
        Result r = std::move(latest_result_);
        latest_result_.valid = false;
        return r;
    }

    // Non-destructive peek: check if a result is available and how old it is
    bool hasResult() const {
        std::lock_guard<std::mutex> lock(result_mutex_);
        return latest_result_.valid;
    }

private:
    void workerLoop() {
        while (true) {
            torch::Tensor grids, motion;
            std::chrono::steady_clock::time_point input_time;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return has_pending_ || stop_; });
                if (stop_ && !has_pending_) return;
                grids = std::move(pending_grids_);
                motion = std::move(pending_motion_);
                input_time = pending_time_;
                has_pending_ = false;
            }

            auto t0 = std::chrono::steady_clock::now();

            torch::Tensor pred;
            {
                torch::NoGradGuard no_grad;
                pred = model_.forward({grids, motion}).toTensor();  // (1,5,1,201,201)
            }
            if (use_cuda_) torch::cuda::synchronize();

            auto t1 = std::chrono::steady_clock::now();
            float inference_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

            // Convert output to cv::Mat vector
            pred = pred.to(torch::kFloat32).to(torch::kCPU);  // (1,5,1,201,201)
            pred = pred.squeeze(0).squeeze(1);                  // (5,201,201)

            Result result;
            result.input_time = input_time;
            result.inference_ms = inference_ms;
            result.valid = true;
            result.pred_grids.reserve(N_FRAMES);

            auto pred_acc = pred.accessor<float, 3>();
            for (int t = 0; t < N_FRAMES; ++t) {
                cv::Mat m(GRID_SIZE, GRID_SIZE, CV_8UC1);
                for (int r = 0; r < GRID_SIZE; ++r) {
                    uchar* row = m.ptr<uchar>(r);
                    for (int c = 0; c < GRID_SIZE; ++c) {
                        row[c] = pred_acc[t][r][c] > 0.5f ? 255 : 0;
                    }
                }
                result.pred_grids.push_back(std::move(m));
            }

            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                latest_result_ = std::move(result);
            }
        }
    }

    torch::jit::script::Module model_;
    bool use_cuda_;
    LogFn log_;

    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool has_pending_ = false;
    std::atomic<bool> stop_;
    torch::Tensor pending_grids_;
    torch::Tensor pending_motion_;
    std::chrono::steady_clock::time_point pending_time_;

    mutable std::mutex result_mutex_;
    Result latest_result_;
};
