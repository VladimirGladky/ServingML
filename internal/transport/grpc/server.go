package grpc

import (
	"ServingML/gen/proto/model"
	"ServingML/internal/service"
	"context"

	"google.golang.org/grpc"
)

type MLServer struct {
	model.UnimplementedBertServiceServer
	mlService *service.MLService
}

func NewMLServer(mlService *service.MLService) *MLServer {
	return &MLServer{mlService: mlService}
}

func Register(s *grpc.Server, mlService *service.MLService) {
	model.RegisterBertServiceServer(s, NewMLServer(mlService))
}

func (s *MLServer) PredictSentiment(ctx context.Context, req *model.BertRequest) (*model.BertResponse, error) {
	id, err := s.mlService.PredictSentiment(ctx, req.Text)
	if err != nil {
		return nil, err
	}
	return &model.BertResponse{Result: id}, nil
}

func (s *MLServer) PredictEmotion(ctx context.Context, req *model.BertRequest) (*model.BertResponse, error) {
	id, err := s.mlService.PredictEmotion(ctx, req.Text)
	if err != nil {
		return nil, err
	}
	return &model.BertResponse{Result: id}, nil
}
