#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path
import argparse

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
  if args.rm_cores:
    for core_file in script_dir.glob('core.*'):
      core_file.unlink()

  build_dir = script_dir / 'build'
  if args.clean_build:
    subprocess.run(['rm', '-rf', build_dir])

  build_project(build_dir, args.debug)


def build_project(build_dir: Path, debug: bool):
  cmake_build_type = 'Debug' if debug else 'Release'
  cmake_cmd = [
      'cmake',
      '-DCMAKE_BUILD_TYPE=' + cmake_build_type,
      '..'
  ]
  build_cmd = ['ninja']

  os.makedirs(build_dir, exist_ok=True)
  os.chdir(build_dir)

  subprocess.check_call(cmake_cmd)
  subprocess.check_call(build_cmd)

  bin_dir = build_dir / 'bin'
  binaries = sorted([b for b in bin_dir.rglob('*') if b.is_file() and os.access(b, os.X_OK)])
  for binary in binaries:
    print(binary.relative_to(build_dir.parent))

if __name__ == '__main__':
  main()
