# Contributing to Legend of Elya N64

Thank you for your interest in contributing to Legend of Elya — the world's first LLM running on Nintendo 64 hardware!

## Project Overview

This is an N64 homebrew ROM that runs a nano-GPT transformer for live AI inference on the MIPS R4300i CPU. The project includes:

- `legend_of_elya.c` - Main ROM code
- `nano_gpt.c` / `nano_gpt.h` - GPT inference engine
- `train_sophia*.py` - Training scripts
- `gen_sophia_host.c` - Host-side weight conversion

## How to Contribute

### Reporting Issues

- Check existing issues before creating new ones
- Use clear, descriptive titles
- Include steps to reproduce bugs
- Include details about your setup (emulator or real hardware)

### Suggesting Enhancements

Open an issue with the "enhancement" label describing:

- The feature and its benefits
- How it fits with the project's goals
- Reference similar implementations if applicable

### Pull Requests

1. Fork the repository
2. Create a descriptive branch:
   - `fix/your-fix-name`
   - `feat/your-feature-name`
3. Make your changes
4. Test thoroughly
5. Submit a pull request against `main`

### Code Style

- Follow existing code conventions in the project
- Keep functions focused and modular
- Add comments for N64-specific code (MIPS assembly, memory constraints)
- Document any hardware-specific behavior

## Development Setup

### Prerequisites

- **N64 Development**: modern gcc with MIPS N64 toolchain
- **Emulator**: [ares](https://ares-emu.net) for testing
- **Real Hardware**: EverDrive 64 for N64 cartridge

### Building

```bash
make
```

### Testing

Test on both emulator and real hardware if possible, as performance characteristics differ significantly.

## What We Need Help With

- **Performance optimization**: Improving tok/s on real hardware
- **Memory optimization**: Reducing RDRAM usage
- **New features**: Additional prompts, character interactions
- **Documentation**: Improving README and code comments
- **Testing**: Testing on different emulators and hardware setups

## Questions?

Feel free to open an issue for any questions about contributing.

Thank you for helping bring AI to retro gaming!
