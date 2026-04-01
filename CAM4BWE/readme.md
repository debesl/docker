# Environment Requirements
The setup has been verified with:

- **Operating System:** Windows
- **IDE:** PyCharm 2021.3
- **Python:** 3.9
- **TensorFlow:** 2.10
- **NumPy:** 1.24

## Important Notes

- TensorFlow **2.10 is the last version that supports GPU acceleration on Windows** (native, non-WSL).
- Python versions **higher than 3.9 are not compatible** with TensorFlow GPU support on Windows.
- Using a different Python or TensorFlow version will result in **CPU-only execution**.

## Required Package Versions

```txt
Needs to be adjusted manuly:
python==3.9
tensorflow==2.10
numpy==1.24
