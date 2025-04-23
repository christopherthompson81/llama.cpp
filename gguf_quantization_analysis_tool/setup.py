from setuptools import setup, find_packages

setup(
    name="gguf_quantization_analysis_tool",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pyside6",
    ],
    entry_points={
        "console_scripts": [
            "quantize-stats=quantize_stats:main",
        ],
    },
    author="GGUF Quantization Analysis Tool Contributors",
    author_email="example@example.com",
    description="A tool for analyzing quantization effects on GGUF model tensors",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gguf_quantization_analysis_tool",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
