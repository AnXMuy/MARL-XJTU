# Findings
- Existing origin is https://github.com/AnXMuy/MARL-XJTU.git; main contains an unrelated previous project with many files already deleted in the working tree.
- Supplied SOP exists at repository root and is untracked.
- Default Python: 3.13.13. No sumo executable found in PATH.
- NVIDIA driver 595.71.05; reported CUDA compatibility 13.2; RTX 4090, 49140 MiB.
- PyTorch 2.14.0+cu130; torch reports CUDA runtime 13.0 and CUDA available on RTX 4090 (driver 595.71.05).
- SUMO 1.25 wheel metadata reachable, but repeated 105.8 MB download stalled at ~30-40 KB/s; stopped after two attempts. No system SUMO executable.
- `traffic-marl/tests`: 7 passed. Python compileall succeeds. PPO actor/value shapes checked.
- No traffic simulation, learned checkpoint, or empirical performance data exists as of this update.
