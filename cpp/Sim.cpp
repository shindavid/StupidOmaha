#include <pokerstove/penum/ShowdownEnumerator.h>
#include <pokerstove/peval/OmahaHighHandEvaluator.h>
#include <torch/script.h>

#include <GameThread.hpp>
#include <Manager.hpp>
#include <ProgressBar.hpp>
#include <SharedData.hpp>
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
  po::options_description desc(
      "evaluates equity of an Omaha hand against a random range\n");

  desc.add_options()("help,?", "produce help message")(
      "hands,h", po::value<int>()->default_value(10000), "number of hands")(
      "runs,r", po::value<int>()->default_value(100), "number of boards to run")(
      "players,p", po::value<int>()->default_value(6), "number of players")(
      "batch-size,b", po::value<int>()->default_value(128), "batch size")(
      "network-file,f", po::value<std::string>()->default_value(""), "path of neural network")(
      "cuda-device,c", po::value<std::string>()->default_value("cuda:0"), "cuda device")(
      "kill-file,k", po::value<std::string>()->default_value(""), "kill file")(
      "output-dir,o", po::value<std::string>(), "output directory");

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

  Params params(vm);
  Manager manager(params);
  manager.run();
}
