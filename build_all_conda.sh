PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ "${CONDA_DEFAULT_ENV:-}" != "foundationpose" ]; then
  echo "Please activate the foundationpose conda environment first:"
  echo "  conda activate foundationpose"
  exit 1
fi

# Install mycpp
cd ${PROJ_ROOT}/mycpp/ && \
rm -rf build && mkdir -p build && cd build && \
cmake \
  -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") \
  -DPYTHON_EXECUTABLE=$(which python) \
  .. && \
make -j$(nproc)

# Install mycuda
cd ${PROJ_ROOT}/bundlesdf/mycuda && \
rm -rf build *egg* *.so && \
python -m pip install --no-build-isolation -e .

cd ${PROJ_ROOT}
