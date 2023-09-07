#pragma once

#include <boost/program_options.hpp>
#include <string>

struct Params {
  Params(const boost::program_options::variables_map& vm) {
    n_hands = vm["hands"].as<int>();
    n_runs = vm["runs"].as<int>();
    n_players = vm["players"].as<int>();
    batch_size = vm["batch-size"].as<int>();
    network_file = vm["network-file"].as<std::string>();
    cuda_device = vm["cuda-device"].as<std::string>();
    output_dir = vm["output-dir"].as<std::string>();
    kill_file = vm["kill-file"].as<std::string>();
  }

  bool empty_net() const { return network_file.empty(); }

  int n_hands;
  int n_runs;
  int n_players;
  int batch_size;
  std::string network_file;
  std::string cuda_device;
  std::string output_dir;
  std::string kill_file;
};
