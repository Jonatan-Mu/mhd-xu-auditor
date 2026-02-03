# Contributing to MHD Xu-Auditor

Thank you for your interest in improving the Xu-Auditor protocol!

## How to Contribute
1. **Reporting Bugs:** Please open an Issue on GitHub describing the error, providing a sample snapshot if possible, and the expected vs. actual output.
2. **Feature Requests:** Open an Issue to discuss new metrics or support for different MHD formulations.
3. **Pull Requests:** - Fork the repository.
   - Create a new branch (`feat/your-feature`).
   - Ensure all tests pass by running `pytest`.
   - Submit the PR with a clear description of the changes.

## Development Setup
- Install development dependencies: `pip install -r requirements-dev.txt`
- Run the test suite: `pytest tests/test_algebraic_consistency.py`

## Code Style
Please ensure any new code follows the existing structure in `src/mhd_xu_auditor/` and includes type hints where applicable.
