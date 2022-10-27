# Remove `operon` directory, if it exists.
if [ -d operon ]; then
    rm -rf operon
fi

# Clone Operon.
git clone https://github.com/heal-research/operon operon
cd operon

# Check out an appropriate commit.
git checkout 9e7ee4e90734951ab4ce27f9c84bb4a8b78a22f9

# Extract a custom `CMakeLists.txt` file.
\cp ../custom/CMakeLists.txt ./

# Create temporary directory for building Operon.
mkdir build
cd build

# Set some appropriate environment variables.
export CC=gcc
export CXX=g++

# Build Operon.
SOURCE_DATE_EPOCH=`date +%s` cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_OPENLIBM=ON -DCERES_TINY_SOLVER=ON -DUSE_SINGLE_PRECISION=ON -DFASTFLOAT_TEST=ON -DBUILD_TESTS=ON -DUSE_JEMALLOC=ON

# Utilize some custom files for the purposes of profiling.
\cp ../../custom/node.hpp ../include/operon/core
\cp ../../custom/metrics.hpp ../include/operon/core
\cp ../../custom/functions.hpp ../include/operon/interpreter
\cp ../../custom/infix.hpp ../include/operon/parser
\cp ../../custom/lexer.hpp ./_deps/infix_parser-src/include
\cp ../../custom/parser.hpp ./_deps/infix_parser-src/include
\cp ../../custom/sexpr.hpp ./_deps/infix_parser-src/src
\cp ../../custom/evaluation.cpp ../test/performance

# Build custom Operon target.
make -j operon-test
