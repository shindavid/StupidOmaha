#pragma once

#include <torch/script.h>

#include <Params.hpp>
#include <mutex>
#include <pokerstove/penum/SimpleDeck.hpp>
#include <thread>
#include <vector>

class SharedData {
 public:
  using input_vec_t = std::vector<torch::jit::IValue>;

  SharedData(const Params& params) : params_(params) {
    if (params.empty_net()) return;

    std::cout << "Loading network from " << params.network_file << std::endl;
    module_ = torch::jit::load(params.network_file.c_str());
    module_.to(at::Device(params.cuda_device));

    for (int i = 0; i < batch_size(); ++i) {
      inputs_.push_back(torch::empty({input_size()}, torch::kFloat32));
    }
    input_gpu_ = torch::empty({batch_size(), input_size()}, torch::kFloat32)
                     .to(at::Device(params.cuda_device));
    output_ = torch::empty({batch_size(), 1}, torch::kFloat32);
    input_vec_.push_back(input_gpu_);

    thread_ = new std::thread([&] { loop(); });
  }

  ~SharedData() {
    closed_ = true;
    eval_cv_.notify_all();
    loop_cv_.notify_all();
    if (thread_) {
      thread_->join();
      delete thread_;
    }
  }

  void join() {
    closed_ = true;
    loop_cv_.notify_one();
    if (thread_) {
      thread_->join();
      delete thread_;
      thread_ = nullptr;
    }
  }

  void encode(torch::Tensor& tensor, int seat, pokerstove::CardSet cards,
              const std::vector<bool>& callers) const {
    tensor.zero_();

    uint64_t mask = cards.mask();
    while (mask) {
      int i = std::countr_zero(mask);
      mask &= ~(1ULL << i);
      tensor[i] = 1;
    }

    tensor[52 + seat] = 1;
    for (int i = 0; i < n_acting_players(); ++i) {
      tensor[52 + n_acting_players() + i] = callers[i];
    }
  }

  double eval(int thread_id, int seat, pokerstove::CardSet cards,
              const std::vector<bool>& callers) {
    if (params_.empty_net()) {
      return 0.1;  // arbitrary positive number = always call
    }
    std::unique_lock lock(mutex_);
    eval_cv_.wait(lock, [&] { return write_ok_; });
    lock.unlock();

    torch::Tensor& tensor = inputs_[thread_id];
    encode(tensor, seat, cards, callers);

    lock.lock();
    write_count_++;
    loop_cv_.notify_one();

    eval_cv_.wait(lock, [&] { return read_ok_; });
    double x = output_[thread_id][0].item<double>();

    read_count_++;
    loop_cv_.notify_one();

    return x;
  }

  const Params& params() const { return params_; }
  int n_hands() const { return params_.n_hands; }
  int n_runs() const { return params_.n_runs; }
  int n_players() const { return params_.n_players; }
  int batch_size() const { return params_.batch_size; }
  std::string output_dir() const { return params_.output_dir; }

  int n_acting_players() const { return n_players() - 1; }  // no BB action
  int input_size() const { return 52 + 2 * n_acting_players(); }

 private:
  void loop() {
    while (!closed_) {
      std::unique_lock lock(mutex_);
      prepare_for_writes();
      loop_cv_.wait(lock,
                    [&] { return write_count_ == batch_size() || closed_; });
      if (closed_) return;

      copy_input_to_gpu();
      auto out = module_.forward(input_vec_).toTuple()->elements()[0].toTensor();
      output_.copy_(out.detach());

      prepare_for_reads();
      loop_cv_.wait(lock,
                    [&] { return read_count_ == batch_size() || closed_; });
    }
  }

  void copy_input_to_gpu() {
    torch::Tensor tmp = torch::empty({batch_size(), input_size()}, torch::kUInt8);
    for (int i = 0; i < batch_size(); ++i) {
      tmp.index_put_({i}, inputs_[i]);
      // tmp[i] = inputs_[i];
    }
    input_gpu_.copy_(tmp);
  }

  void prepare_for_writes() {
    write_count_ = 0;
    read_count_ = 0;
    write_ok_ = true;
    read_ok_ = false;
    eval_cv_.notify_all();
  }

  void prepare_for_reads() {
    write_ok_ = false;
    read_ok_ = true;
    eval_cv_.notify_all();
  }

  const Params params_;
  torch::jit::script::Module module_;
  input_vec_t input_vec_;
  torch::Tensor input_gpu_;
  std::vector<torch::Tensor> inputs_;
  torch::Tensor output_;

  std::thread* thread_ = nullptr;
  std::mutex mutex_;
  std::condition_variable loop_cv_;
  std::condition_variable eval_cv_;
  int write_count_ = 0;
  int read_count_ = 0;
  bool write_ok_ = false;
  bool read_ok_ = false;
  bool closed_ = false;
};
