#include <ProgressBar.hpp>

#include <pokerstove/penum/ShowdownEnumerator.h>
#include <pokerstove/peval/OmahaHighHandEvaluator.h>

#include <boost/program_options.hpp>
#include <functional>
#include <iostream>
#include <pokerstove/penum/SimpleDeck.hpp>
#include <unordered_set>
#include <vector>

using namespace pokerstove;
namespace po = boost::program_options;

template <>
struct std::hash<CardSet> {
  std::size_t operator()(const CardSet& set) const {
    return std::hash<uint64_t>()(set.mask());
  }
};

int main(int argc, char** argv) {
  po::options_description desc("evaluates equity of an Omaha hand against a random range\n");

  desc.add_options()("help,?", "produce help message")(
    "num-iters,n", po::value<int>()->default_value(10000), "Monte carlo sim count")(
    "output-file,o", po::value<std::string>()->default_value("equity-vs-random.txt"), "output file"
    );

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .style(po::command_line_style::unix_style)
                .options(desc)
                .run(),
            vm);
  po::notify(vm);

  // check for help
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  int n = vm["num-iters"].as<int>();
  std::string output_file = vm["output-file"].as<std::string>();

  SimpleDeck deck;
  CardSet board;
  boost::shared_ptr<PokerHandEvaluator> evaluator =
      PokerHandEvaluator::alloc("o");

  std::unordered_set<CardSet> hand_set;
  for (int a = 0; a < 52; ++a) {
    for (int b = a + 1; b < 52; ++b) {
      for (int c = b + 1; c < 52; ++c) {
        for (int d = c + 1; d < 52; ++d) {
          uint64_t mask = 1UL << a | 1UL << b | 1UL << c | 1UL << d;
          CardSet hand(mask);
          hand_set.insert(hand.canonize());
        }
      }
    }
  }

  std::cout << "writing to " << output_file << std::endl;
  FILE *fp = fopen(output_file.c_str(), "w");

  progressbar bar;
  bar.set_niter(hand_set.size());

  std::vector<CardDistribution> handDists;
  for (CardSet hand1 : hand_set) {
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
    fprintf(fp, "%s %.5f\n", hand1.str().c_str(), equity);
    bar.update();
  }

  fclose(fp);
}
