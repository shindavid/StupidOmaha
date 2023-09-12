#pragma once

#include <pokerstove/penum/ShowdownEnumerator.h>
#include <pokerstove/peval/OmahaHighHandEvaluator.h>
#include <torch/serialize/archive.h>

#include <SharedData.hpp>
#include <list>
#include <map>
#include <pokerstove/penum/SimpleDeck.hpp>
#include <vector>

class GameThread {
 public:
  using count_submap_t = std::map<int, int>;
  using count_map_t = std::map<int, count_submap_t>;

  GameThread(SharedData* shared_data, int thread_id)
      : shared_data_(shared_data), thread_id_(thread_id) {
    std::random_device rd;
    std::mt19937 g(rd());
    prng_ = g;

    hands_.resize(n_players());
    callers_.resize(n_players());
    callers_[n_players() - 1] = true;  // BB forced to call
    evaluator_ = pokerstove::PokerHandEvaluator::alloc("o");

    int n_remaining_cards = 52 - 4 * n_players();
    for (int i = 0; i < n_remaining_cards; ++i) {
      indices_.push_back(i);
    }
  }

  ~GameThread() {
    if (thread_) {
      thread_->join();
      delete thread_;
    }
  }

  int n_players() const { return shared_data_->n_players(); }
  int n_runs() const { return shared_data_->n_runs(); }
  int batch_size() const { return shared_data_->batch_size(); }
  int n_acting_players() const { return shared_data_->n_acting_players(); }
  int input_size() const { return shared_data_->input_size(); }

  int n_training_rows() const { return n_rows_; }
  const count_map_t& call_counts() const { return call_counts_; }
  const count_map_t& fold_counts() const { return fold_counts_; }

  void launch(bool verbose = false) {
    thread_ = new std::thread([&] { run(verbose); });
  }

  void join() {
    if (thread_) {
      thread_->join();
      delete thread_;
      thread_ = nullptr;
    }
  }

  void flush() {
    // combine all training_data inputs into one tensor
    torch::Tensor input = torch::empty({n_rows_, input_size()}, torch::kUInt8);
    torch::Tensor output = torch::empty({n_rows_, 1}, torch::kFloat32);
    int row = 0;
    for (const training_row_t& training_row : training_data_) {
      input.index_put_({row}, training_row.input);
      output[row][0] = training_row.output;
      ++row;
    }

    // output filename is output_dir/<thread_id>.pt
    std::stringstream ss;
    ss << shared_data_->output_dir() << "/" << thread_id_ << "-" << n_rows_
       << ".pt";
    std::string output_filename = ss.str();

    using tensor_map_t = std::map<std::string, torch::Tensor>;
    tensor_map_t tensor_map;
    tensor_map["input"] = input;
    tensor_map["output"] = output;

    torch::serialize::OutputArchive archive(
        std::make_shared<torch::jit::CompilationUnit>());
    for (auto it : tensor_map) {
      archive.write(it.first, it.second);
    }

    archive.save_to(output_filename);
    std::cout << "Wrote to: " << output_filename << std::endl;
  }

 private:
  struct training_row_t {
    torch::Tensor input;
    double output;
  };
  using training_row_list_t = std::list<training_row_t>;

  int num_callers() const {
    int n = 0;
    for (int p = 0; p < n_players(); ++p) {
      if (callers_[p]) {
        n++;
      }
    }
    return n;
  }

  void run(bool verbose) {
    init_hand();

    std::vector<std::string> log;
    char buf[1024];

    for (int seat = 0; seat < n_acting_players(); ++seat) {
      double predicted_ev =
          shared_data_->eval(thread_id_, seat, hands_[seat], callers_);
      bool call = predicted_ev >= 0;

      // counterfactual sim
      callers_[seat] = true;
      for (int seat2 = seat + 1; seat2 < n_acting_players(); ++seat2) {
        double predicted_ev2 =
            shared_data_->eval(thread_id_, seat2, hands_[seat2], callers_);
        callers_[seat2] = predicted_ev2 >= 0;
      }

      estimate_evs();
      double counterfactual_ev = evs_[seat];

      if (verbose) {
        std::string action_str = get_action_str(seat);
        sprintf(buf, "%d %s %+5.3f %+5.3f %s %c", seat,
                hands_[seat].str().c_str(), counterfactual_ev, predicted_ev,
                action_str.c_str(), call ? 'C' : 'F');
        log.push_back(buf);
      }

      // undo counterfactual sim
      callers_[seat] = false;
      for (int seat2 = seat + 1; seat2 < n_acting_players(); ++seat2) {
        callers_[seat2] = false;
      }

      record_training_data(hands_[seat], seat, callers_, counterfactual_ev);
      int num_previous_callers = num_callers() - 1;  // omit BB call
      if (call) {
        call_counts_[seat][num_previous_callers]++;
      } else {
        fold_counts_[seat][num_previous_callers]++;
      }
      callers_[seat] = call;
    }

    if (verbose) {
      int seat = n_players() - 1;
      std::string action_str = get_action_str(seat);
      sprintf(buf, "%d %s               %s C", seat, hands_[seat].str().c_str(),
              action_str.c_str());
      log.push_back(buf);

      estimate_evs();
      printf("\n");
      for (int p = 0; p < n_players(); ++p) {
        if (callers_[p]) {
          printf("%s %+5.3f\n", log[p].c_str(), evs_[p]);
        } else {
          printf("%s\n", log[p].c_str());
        }
      }
    }
  }

  std::string get_action_str(int seat) const {
    char buf[16];
    for (int p = 0; p < n_players(); ++p) {
      if (p == seat) {
        buf[p] = callers_[p] ? 'C' : 'F';
      } else {
        buf[p] = callers_[p] ? 'c' : 'f';
      }
    }
    buf[n_players()] = '\0';
    return std::string(buf);
  }

  void record_training_data(pokerstove::CardSet hand, int seat,
                            const std::vector<bool>& callers, double ev) {
    torch::Tensor tensor = torch::empty({input_size()}, torch::kUInt8);
    shared_data_->encode(tensor, seat, hand, callers);

    training_data_.push_back(training_row_t{tensor, ev});
    n_rows_++;
  }

  void init_hand() {
    deck_.shuffle();
    for (int p = 0; p < n_players(); ++p) {
      hands_[p] = deck_.deal(4);
      callers_[p] = false;
    }
    callers_[n_players() - 1] = true;  // BB forced to call
  }

  pokerstove::CardSet deal() {
    std::shuffle(indices_.begin(), indices_.end(), prng_);
    pokerstove::CardSet board;
    for (int i = 0; i < 5; ++i) {
      board |= deck_[indices_[i]];
    }
    return board;
  }

  int get_caller_index(int seat) const {
    int call_index = 0;
    for (int p = 0; p < n_players(); ++p) {
      if (!callers_[p]) continue;
      if (p == seat) return call_index;
      ++call_index;
    }
    throw std::exception();
  }

  void init_hand_distrs() {
    hand_dists_.clear();
    evs_.clear();
    for (int p = 0; p < n_players(); ++p) {
      if (callers_[p]) {
        hand_dists_.emplace_back(hands_[p]);
      }
      evs_.push_back(0);
    }
  }

  void estimate_evs() {
    init_hand_distrs();
    int n_callers = num_callers();

    if (n_callers < 2) {
      return;  // no need to run out boards
    }

    // int caller_index = get_caller_index(seat);

    // double ev = 0;
    for (int r = 0; r < n_runs(); ++r) {
      pokerstove::CardSet board = deal();

      pokerstove::ShowdownEnumerator showdown;
      auto results = showdown.calculateEquity(hand_dists_, board, evaluator_);
      for (int c = 0; c < n_callers; ++c) {
        evs_[c] += results[c].winShares + results[c].tieShares;
      }
    }

    double pot_size = n_callers;
    double mult = pot_size / n_runs();

    // shift evs_ indexing
    int c = n_callers - 1;
    for (int p = n_players() - 1; p >= 0; --p) {
      if (callers_[p]) {
        evs_[p] = evs_[c] * mult - 1;  // - 1 for call
        --c;
      } else {
        evs_[p] = 0;
      }
    }
  }

  SharedData* shared_data_;
  const int thread_id_;
  std::mt19937 prng_;
  pokerstove::SimpleDeck deck_;
  std::vector<pokerstove::CardSet> hands_;
  boost::shared_ptr<pokerstove::PokerHandEvaluator> evaluator_;
  std::vector<int> indices_;

  std::vector<bool> callers_;                             // reset on each hand
  std::vector<pokerstove::CardDistribution> hand_dists_;  // reset on each hand
  std::vector<double> evs_;

  /*
   * call_counts_[x][y] is the number of times seat x voluntarily called after y
   * players voluntarily called in front of him.
   *
   * fold_counts_[x][y] is the number of times seat x folded after y players
   * voluntarily called in front of him.
   */
  count_map_t call_counts_;
  count_map_t fold_counts_;

  std::thread* thread_ = nullptr;
  training_row_list_t training_data_;
  int n_rows_ = 0;
};
