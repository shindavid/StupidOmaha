#include <pokerstove/penum/ShowdownEnumerator.h>
#include <pokerstove/penum/SimpleDeck.hpp>
#include <pokerstove/peval/OmahaHighHandEvaluator.h>

#include <boost/program_options.hpp>
#include <iostream>
#include <vector>

using namespace pokerstove;
namespace po = boost::program_options;
using namespace std;

int main(int argc, char** argv) {
  po::options_description desc("evaluates equity of an Omaha hand against a random range\n");

  desc.add_options()("help,?", "produce help message")(
      "hand,h", po::value<vector<string>>(), "hand(s) for evaluation")(
      "num-iters,n", po::value<int>()->default_value(10000), "Monte carlo sim count"
      );

  // make hand a positional argument
  po::positional_options_description p;
  p.add("hand", -1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .style(po::command_line_style::unix_style)
                .options(desc)
                .positional(p)
                .run(),
            vm);
  po::notify(vm);

  // check for help
  if (vm.count("help") || argc == 1) {
    cout << desc << endl;
    return 1;
  }

  // extract the options
  vector<string> hands = vm["hand"].as<vector<string>>();
  int n = vm["num-iters"].as<int>();

  std::cout << "n=" << n << std::endl;

  SimpleDeck deck;
  CardSet board;
  boost::shared_ptr<PokerHandEvaluator> evaluator =
      PokerHandEvaluator::alloc("o");

  vector<CardDistribution> handDists;
  for (const string& hand_str : hands) {
    CardSet hand1(hand_str);
    if (hand1.size() != 4) {
      cerr << "Error: hand must be 4 cards: " << hand_str << endl;
      return 1;
    }

    std::cout << "Computing equity for " << hand_str << "..." << std::endl;
    EquityResult total1;
    EquityResult total2;
    for (int i = 0; i < n; ++i) {
      deck.shuffle();
      deck.remove(hand1);

      CardSet board = deck.deal(5);
      CardSet hand2 = deck.deal(4);

      handDists.clear();
      handDists.emplace_back(hand1);
      handDists.emplace_back(hand2);

      ShowdownEnumerator showdown;
      auto results = showdown.calculateEquity(handDists, board, evaluator);
      auto result1 = results[0];
      auto result2 = results[1];
      total1 += result1;
      total2 += result2;
    }

    double shares1 = total1.winShares + total1.tieShares;
    double shares2 = total2.winShares + total2.tieShares;
    double equity = shares1 / (shares1 + shares2);
    std::cout << "...computed: " << equity << std::endl;
  }
}
