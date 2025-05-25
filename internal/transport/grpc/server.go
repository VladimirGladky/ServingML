package grpc

import (
	"ServingML/gen/proto/model"
	"ServingML/internal/service"
	"context"
	"google.golang.org/grpc"
)

type MLServer struct {
	model.UnimplementedBertServiceServer
	mlService service.MLServiceInterface
}

func NewMLServer(mlService service.MLServiceInterface) *MLServer {
	return &MLServer{mlService: mlService}
}

func Register(s *grpc.Server, mlService service.MLServiceInterface) {
	model.RegisterBertServiceServer(s, NewMLServer(mlService))
}

func (s *MLServer) Predict(ctx context.Context, req *model.BertRequest) (*model.BertResponse, error) {
	id, err := s.mlService.Predict(ctx, req.Text)
	if err != nil {
		return nil, err
	}
	return &model.BertResponse{Result: id}, nil
}

//func (s *MLServer) GetResult(ctx context.Context, req *model.GetResultRequest) (*model.GetResultResponse, error) {
//	result, err := s.mlService.GetResult(ctx, req.Id)
//	if err != nil {
//		return nil, err
//	}
//	return &model.GetResultResponse{Result: result}, nil
//}
