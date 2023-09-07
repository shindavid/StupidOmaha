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
  GameThread(SharedData* shared_data, int thread_id)
      : shared_data_(shared_data), thread_id_(thread_id) {
    hands_.resize(n_players());
    callers_.resize(n_players());
    evaluator_ = pokerstove::PokerHandEvaluator::alloc("o");

    int n_remaining_cards = 52 - 4 * n_players();
    for (int i = 0; i < n_remaining_cards; ++i) {
      indices_.push_back(i);
    }
  }

  int n_players() const { return shared_data_->n_players(); }
  int n_runs() const { return shared_data_->n_runs(); }
  int batch_size() const { return shared_data_->batch_size(); }
  int n_acting_players() const { return shared_data_->n_acting_players(); }
  int input_size() const { return shared_data_->input_size(); }

  void launch() {
    init_hand();

    for (int seat = 0; seat < n_acting_players(); ++seat) {
      double predicted_ev =
          shared_data_->eval(thread_id_, seat, hands_[seat], callers_);

      // counterfactual sim
      callers_[seat] = true;
      for (int seat2 = seat + 1; seat2 < n_acting_players(); ++seat2) {
        double predicted_ev2 =
            shared_data_->eval(thread_id_, seat2, hands_[seat2], callers_);
        callers_[seat2] = predicted_ev2 >= 0;
      }

      double counterfactual_ev = estimate_ev(seat);

      // undo counterfactual sim
      callers_[seat] = false;
      for (int seat2 = seat + 1; seat2 < n_acting_players(); ++seat2) {
        callers_[seat2] = false;
      }

      record_training_data(hands_[seat], seat, callers_, counterfactual_ev);
      callers_[seat] = predicted_ev >= 0;
    }
  }

  void flush() {
    // combine all training_data inputs into one tensor
    int n_rows = training_data_.size();
    torch::Tensor input =
        torch::empty({n_rows, input_size()}, torch::kUInt8);
    torch::Tensor output = torch::empty({n_rows, 1}, torch::kFloat32);
    int row = 0;
    for (const training_row_t& training_row : training_data_) {
      input.index_put_({row}, training_row.input);
      // input[row] = training_row.input;
      // input.index_put_({row}, training_row.input);
      output[row][0] = training_row.output;
      ++row;
    }

    // output filename is output_dir/<thread_id>.pt
    std::stringstream ss;
    ss << shared_data_->output_dir() << "/" << thread_id_ << ".pt";
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

  void record_training_data(pokerstove::CardSet hand, int seat,
                            const std::vector<bool>& callers, double ev) {
    torch::Tensor tensor = torch::empty({input_size()}, torch::kUInt8);
    shared_data_->encode(tensor, seat, hand, callers);

    training_data_.push_back(training_row_t{tensor, ev});
  }

  void init_hand() {
    deck_.shuffle();
    for (int p = 0; p < n_players(); ++p) {
      hands_[p] = deck_.deal(4);
      callers_[p] = false;
    }
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

  int init_hand_distrs() {
    int num_callers = 0;
    hand_dists_.clear();
    for (int p = 0; p < n_players(); ++p) {
      if (callers_[p]) {
        num_callers++;
        hand_dists_.emplace_back(hands_[p]);
      }
    }
    return num_callers;
  }

  double estimate_ev(int seat) {
    int num_callers = init_hand_distrs();

    if (num_callers < 2) {
      return 0;  // no need to run out boards
    }

    int caller_index = get_caller_index(seat);

    double ev = 0;
    for (int r = 0; r < n_runs(); ++r) {
      pokerstove::CardSet board = deal();

      pokerstove::ShowdownEnumerator showdown;
      auto results = showdown.calculateEquity(hand_dists_, board, evaluator_);
      ev += results[caller_index].winShares + results[caller_index].tieShares;
    }

    double pot_size = num_callers;
    return ev * pot_size / n_runs() - 1;  // -1 for call
  }

  SharedData* shared_data_;
  const int thread_id_;
  std::mt19937 prng_;
  pokerstove::SimpleDeck deck_;
  std::vector<pokerstove::CardSet> hands_;
  boost::shared_ptr<pokerstove::PokerHandEvaluator> evaluator_;

  std::vector<int> indices_;
  std::vector<bool> callers_;
  std::vector<pokerstove::CardDistribution> hand_dists_;

  training_row_list_t training_data_;
};
