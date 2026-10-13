FROM --platform=linux/arm64 python:3.11-slim-bookworm@sha256:dabc8c628ee25179a9fcb8ab81e592938ee2bad3ffb15e64157f8fe428d5279f AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends g++ cmake ninja-build pkg-config nlohmann-json3-dev && rm -rf /var/lib/apt/lists/*
COPY cppwrapper/pytorch/requirements-torch.lock /tmp/requirements-torch.lock
RUN python -m pip install --no-cache-dir --only-binary=:all: --require-hashes -r /tmp/requirements-torch.lock --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
WORKDIR /repos
COPY lmctl-ai/lmcpp /src/lmcpp
RUN cmake -S /src/lmcpp -B /build-lmcpp -G Ninja -DCMAKE_BUILD_TYPE=Release -DLMCPP_BUILD_TESTS=OFF -DLMCPP_STATIC_STDLIB=OFF -DCMAKE_INSTALL_PREFIX=/opt/lmcpp \
 && cmake --build /build-lmcpp --parallel 2 \
 && cmake --install /build-lmcpp
COPY cppwrapper/pytorch /src/pytorch
RUN cmake -S /src/pytorch -B /build-torchcli -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
    -Dlmcpp_DIR=/opt/lmcpp/lib/cmake/lmcpp \
    && cmake --build /build-torchcli --parallel 2

FROM --platform=linux/arm64 python:3.11-slim-bookworm@sha256:dabc8c628ee25179a9fcb8ab81e592938ee2bad3ffb15e64157f8fe428d5279f AS test
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /opt/lmcpp /opt/lmcpp
COPY --from=builder /build-torchcli /build
COPY --from=builder /src/pytorch/tests /src/pytorch/tests
COPY --from=builder /src/pytorch/bin /src/pytorch/bin
COPY cppwrapper/pytorch/requirements-test.lock /tmp/requirements-test.lock
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 cmake ninja-build && rm -rf /var/lib/apt/lists/* \
 && python -m pip install --no-cache-dir --only-binary=:all: --require-hashes --no-deps -r /tmp/requirements-test.lock
ENV OMP_NUM_THREADS=2
CMD ["ctest", "--test-dir", "/build", "--output-on-failure"]

FROM --platform=linux/arm64 python:3.11-slim-bookworm@sha256:dabc8c628ee25179a9fcb8ab81e592938ee2bad3ffb15e64157f8fe428d5279f AS runtime
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /build-torchcli/bin/torchcli /usr/local/bin/torchcli
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
ENV OMP_NUM_THREADS=2
ENTRYPOINT ["/usr/local/bin/torchcli"]
