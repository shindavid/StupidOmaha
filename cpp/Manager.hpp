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
        thread->launch();
      }

      if (check_kill_file) {
        // break if kill_file exists:
        if (boost::filesystem::exists(kill_file)) {
          break;
        }
      }
    }

    for (GameThread* thread : game_threads_) {
      thread->flush();
    }
  }

 private:
  SharedData shared_data_;
  std::vector<GameThread*> game_threads_;
};
