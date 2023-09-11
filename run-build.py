#!/usr/bin/env python3

"""
Script to automate the build process of the project using CMake and Ninja.
"""
import os
import shutil
import subprocess
from pathlib import Path
import argparse


def build_project(build_dir: Path, build_type: str):
  """
  Build the project using CMake and Ninja. After building, it lists all the binaries created by Ninja.

  Args:
      build_dir (Path): The directory where the build files will be generated.
      build_type (str): The type of build, either 'Debug' or 'Release'.
  """
  cmake_cmd = [
      'cmake',
      '-GNinja',
      '-DCMAKE_BUILD_TYPE=' + build_type,
      '../..'
  ]
  build_cmd = ['ninja']

  os.makedirs(build_dir, exist_ok=True)
  os.chdir(build_dir)

  subprocess.run(cmake_cmd, check=True)
  subprocess.run(build_cmd, check=True)

  # Query Ninja for all targets it is responsible for building, so we can print them out.
  targets_cmd = build_cmd + ['-t', 'targets']
  result = subprocess.run(targets_cmd, capture_output=True, text=True, check=True)
  candidates = set([line.split(":")[0].strip() for line in result.stdout.splitlines()])

  # Search the 'bin' directory for all executable files
  bin_dir = build_dir / 'bin'
  binaries = sorted([b for b in bin_dir.rglob('*') if b.is_file() and os.access(b, os.X_OK)])

  print('')
  for binary in binaries:
    if binary.name in candidates:
      print(binary.relative_to(build_dir.parent.parent))


def main():
  parser = argparse.ArgumentParser(description='Build script')
  parser.add_argument('-c', '--rm-cores', action='store_true',
                    help='Remove core.* from script directory')
  parser.add_argument('-C', '--clean-build', action='store_true',
                    help='Blow away build/ first')
  parser.add_argument('-d', '--debug', action='store_true',
                    help='Build in debug mode')
  args = parser.parse_args()

  script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
  build_base_dir = script_dir / 'build'

  # Determine build type and respective build directory
  build_type = 'Debug' if args.debug else 'Release'
  build_dir = build_base_dir / build_type

  # Handle core dump cleanup if requested
  if args.rm_cores:
    for core_file in script_dir.glob('core.*'):
      print('Removing', core_file)
      core_file.unlink()

  # Clean the entire build directory if requested
  if args.clean_build:
    shutil.rmtree(build_base_dir, ignore_errors=True)

  build_project(build_dir, build_type)


if __name__ == '__main__':
  main()
