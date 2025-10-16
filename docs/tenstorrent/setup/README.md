# Setup & Configuration

This directory contains setup and configuration guides for the Tenstorrent backend.

## Documents

- **METALIUM_SETUP_GUIDE.md**: Complete guide for installing and configuring the TT-Metalium SDK
- **CI.md**: Continuous integration setup and local CI parity instructions
- **local_build_guide.md**: Detailed walkthrough of the local build process with troubleshooting

## Quick Start

### Mock Mode (No Hardware)
```bash
bash maint/scripts/local_build_and_test_tt.sh --skip-deps --jobs 4
```

### Real Mode (With SDK)
```bash
export TT_METAL_HOME=/path/to/tt-metal
bash maint/scripts/local_build_and_test_tt.sh --with-metalium --skip-deps --jobs 4
```

## Build Modes

- **Mock Mode**: Development without hardware, uses mock Metalium APIs
- **Real Mode**: Hardware execution with actual Metalium SDK

See individual guides for detailed instructions.