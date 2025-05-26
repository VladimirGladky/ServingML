FROM golang:1.21-alpine AS builder

RUN apk add --no-cache git gcc musl-dev cmake make

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
COPY /usr/lib/libonnxruntime.so /usr/lib/libonnxruntime.so

RUN cd cmd/server && \
    CGO_ENABLED=1 GOOS=linux go build -ldflags="-s -w" -o /server

FROM alpine:3.18

RUN apk add --no-cache libstdc++

COPY --from=builder /server /app/server
COPY --from=builder /app/internal/modelWrapper/data /app/internal/modelWrapper/data
COPY --from=builder /usr/lib/libonnxruntime.so /usr/lib/libonnxruntime.so

ENV LD_LIBRARY_PATH=/usr/lib
ENV ONNXRUNTIME_SHARED_LIB_PATH=/usr/lib/libonnxruntime.so

WORKDIR /app
EXPOSE 6060

CMD ["/app/server"]