package grpc

import (
	"ServingML/gen/proto/model"
	"ServingML/internal/batcher"
	"context"

	"google.golang.org/grpc"
)

type MLServer struct {
	model.UnimplementedBertServiceServer
	mlService *batcher.ServiceBatcher
}

func NewMLServer(mlService *batcher.ServiceBatcher) *MLServer {
	return &MLServer{mlService: mlService}
}

func Register(s *grpc.Server, mlService *batcher.ServiceBatcher) {
	model.RegisterBertServiceServer(s, NewMLServer(mlService))
}

func (s *MLServer) Predict(ctx context.Context, req *model.BertRequest) (*model.BertResponse, error) {
	if req.Text == "" {
		return nil, nil
	}
	id, err := s.mlService.Predict(ctx, req.Text, req.ModelName)
	if err != nil {
		return nil, err
	}
	return &model.BertResponse{Result: id}, nil
}
