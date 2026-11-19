SUMO/Python installation notes (2026-11-19)

- Tsinghua PyPI returned HTTP 403 for package files.
- USTC mirror's package index resolved metadata but its redirected wheel URL returned HTTP 403.
- Tencent PyPI mirror served eclipse-sumo 1.25.0 wheel (105.8 MB) at about 2.8 MB/s; traci and sumolib also installed successfully.
- Verified wheel SHA256: 57bee7cd36f9859e20e2143ca67163f7aa4e8fd36f73b078af8a3b443d4845b3.
- The wheel installs `sumo`, `netconvert`, `sumo-gui` and related commands into the active environment's `bin` directory. The project venv's bin must be added to PATH.
- Python used: 3.13.13. `torch` reports 2.14.0+cu130 / CUDA runtime 13.0; the system CUDA 13.2 driver compatibility does not mean Torch uses a cu132 runtime.
