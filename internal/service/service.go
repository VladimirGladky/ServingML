package service

import (
	"ServingML/internal/repository"
	"context"
	"github.com/google/uuid"
)

type MLServiceInterface interface {
	Predict(ctx context.Context, text string) (string, error)
}

type MLService struct {
	repository repository.MLRepositoryInterface
}

func (s *MLService) Predict(ctx context.Context, text string) (string, error) {
	id := uuid.New().String()
	err := s.repository.SaveExpr(ctx,id, text)
	if err != nil {
		return "", err
	}
	return id, nil
}
