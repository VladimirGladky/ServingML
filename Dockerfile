FROM golang:1.24 AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

RUN wget https://github.com/daulet/tokenizers/releases/download/v1.20.2/libtokenizers.linux-amd64.tar.gz \
    && tar -xzf libtokenizers.linux-amd64.tar.gz -C /usr/lib \
    && rm libtokenizers.linux-amd64.tar.gz

COPY . .
COPY docker_libs/libonnxruntime.so /usr/lib/libonnxruntime.so

ENV CGO_LDFLAGS="-L/usr/lib -ltokenizers -lonnxruntime"

RUN cd cmd/server && \
    CGO_ENABLED=1 GOOS=linux go build -o /server

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y libstdc++6 libgcc-s1

COPY --from=builder /server /app/server
COPY --from=builder /app/internal/modelWrapper/data /app/internal/modelWrapper/data
COPY --from=builder /usr/lib/libonnxruntime.so /usr/lib/libonnxruntime.so
COPY --from=builder /usr/lib/libtokenizers.a /usr/lib/libtokenizers.a
COPY --from=builder /app/config/config.yaml /app/config/config.yaml

ENV LD_LIBRARY_PATH=/usr/lib
ENV ONNXRUNTIME_SHARED_LIB_PATH=/usr/lib/libonnxruntime.so

RUN chmod +x /app/server

WORKDIR /app
EXPOSE 6060

CMD ["/app/server"]