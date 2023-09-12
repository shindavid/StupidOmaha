#pragma once

#include <GameThread.hpp>
#include <Params.hpp>
#include <SharedData.hpp>
#include <boost/filesystem.hpp>

class Manager {
 public:
  Manager(const Params& params) : shared_data_(params) {
    if (!params.output_dir.empty()) {
      boost::filesystem::create_directories(params.output_dir);
    }
    for (int b = 0; b < batch_size(); ++b) {
      game_threads_.push_back(new GameThread(&shared_data_, b));
    }
  }

  int n_hands() const { return shared_data_.n_hands(); }
  int n_runs() const { return shared_data_.n_runs(); }
  int n_players() const { return shared_data_.n_players(); }
  int batch_size() const { return shared_data_.batch_size(); }

  void run() {
    bool check_kill_file = !shared_data_.params().kill_file.empty();
    boost::filesystem::path kill_file(shared_data_.params().kill_file);

    int remaining_hands = n_hands();
    bool infinite = remaining_hands == 0;
    while (infinite || remaining_hands > 0) {
      remaining_hands -= batch_size();
      for (GameThread* thread : game_threads_) {
        thread->launch(shared_data_.params().verbose);
      }
      for (GameThread* thread : game_threads_) {
        thread->join();
      }

      if (check_kill_file) {
        // break if kill_file exists:
        if (boost::filesystem::exists(kill_file)) {
          break;
        }
      }
    }

    int n_training_rows = 0;

    shared_data_.join();
    for (GameThread* thread : game_threads_) {
      thread->flush();
      n_training_rows += thread->n_training_rows();
    }

    std::cout << "RESULT n_training_rows: " << n_training_rows << std::endl;

    GameThread::count_map_t call_counts;
    GameThread::count_map_t fold_counts;
    for (GameThread* thread : game_threads_) {
      for (const auto& call_count : thread->call_counts()) {
        int seat = call_count.first;
        for (const auto& submap : call_count.second) {
          int hand = submap.first;
          int count = submap.second;
          call_counts[seat][hand] += count;
        }
      }
      for (const auto& fold_count : thread->fold_counts()) {
        int seat = fold_count.first;
        for (const auto& submap : fold_count.second) {
          int hand = submap.first;
          int count = submap.second;
          fold_counts[seat][hand] += count;
        }
      }
    }

    // dump call_counts:
    for (const auto& call_count : call_counts) {
      int seat = call_count.first;
      for (const auto& submap : call_count.second) {
        int n_prev_callers = submap.first;
        int count = submap.second;
        std::cout << "RESULT call-" << seat << "-" << n_prev_callers << ": "
                  << count << std::endl;
      }
    }

    for (const auto& fold_count : fold_counts) {
      int seat = fold_count.first;
      for (const auto& submap : fold_count.second) {
        int n_prev_callers = submap.first;
        int count = submap.second;
        std::cout << "RESULT fold-" << seat << "-" << n_prev_callers << ": "
                  << count << std::endl;
      }
    }
  }

 private:
  SharedData shared_data_;
  std::vector<GameThread*> game_threads_;
};
