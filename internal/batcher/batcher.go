package batcher

import (
	"ServingML/internal/domain/models"
	"ServingML/internal/inference"
	"ServingML/internal/modelWrapper"
	"ServingML/pkg/converter"
	"ServingML/pkg/logger"
	"context"
	"sync"
	"sync/atomic"
	"time"
)

type ServiceBatcherInterface interface {
	PredictFirstModel(ctx context.Context, text string) (string, error)
	PredictSecondModel(ctx context.Context, text string) (string, error)
	StartBatchProcessor()
}

type ServiceBatcher struct {
	ctx         context.Context
	queue       chan *models.PredictionRequest
	firstModel  *modelWrapper.WrapperModel
	secondModel *modelWrapper.WrapperModel
	initialized uint32
	initOnce    sync.Once
	inf         inference.ServiceInferenceInterface
}

func New(ctx context.Context, sentimentModel *modelWrapper.WrapperModel, emotionModel *modelWrapper.WrapperModel, inf inference.ServiceInferenceInterface) *ServiceBatcher {
	return &ServiceBatcher{
		ctx:         ctx,
		queue:       make(chan *models.PredictionRequest, 2000),
		firstModel:  sentimentModel,
		secondModel: emotionModel,
		inf:         inf,
	}
}

func (s *ServiceBatcher) PredictFirstModel(ctx context.Context, text string) (string, error) {
	s.initOnce.Do(func() {
		go s.StartBatchProcessor()
		logger.GetLoggerFromCtx(s.ctx).Info("BatchProcessor запущен по требованию")
	})
	respCh := make(chan *models.PredictionResponse, 1)
	select {
	case s.queue <- &models.PredictionRequest{
		Text:           text,
		ResponseCh:     respCh,
		PredictionType: models.FirstPrediction,
	}:
	case <-ctx.Done():
		return "", ctx.Err()
	}

	select {
	case resp := <-respCh:
		if resp.Error != nil {
			return "", resp.Error
		}
		return converter.ConvertFirstModel(resp.Probabilities), nil
	}
}

func (s *ServiceBatcher) PredictSecondModel(ctx context.Context, text string) (string, error) {
	s.initOnce.Do(func() {
		go s.StartBatchProcessor()
		logger.GetLoggerFromCtx(s.ctx).Info("BatchProcessor запущен по требованию")
	})
	respCh := make(chan *models.PredictionResponse, 1)
	select {
	case s.queue <- &models.PredictionRequest{
		Text:           text,
		ResponseCh:     respCh,
		PredictionType: models.SecondPrediction,
	}:
	case <-ctx.Done():
		return "", ctx.Err()
	}

	select {
	case resp := <-respCh:
		if resp.Error != nil {
			return "", resp.Error
		}
		return converter.ConvertSecondModel(resp.Probabilities), nil
	}
}

func (s *ServiceBatcher) StartBatchProcessor() {
	if !atomic.CompareAndSwapUint32(&s.initialized, 0, 1) {
		return
	}
	modelFirstBatch := make([]*models.PredictionRequest, 0)
	modelSecondBatch := make([]*models.PredictionRequest, 0)
	timer := time.NewTimer(50 * time.Millisecond)

	for {
		select {
		case req := <-s.queue:
			switch req.PredictionType {
			case models.FirstPrediction:
				modelFirstBatch = append(modelFirstBatch, req)
				if len(modelFirstBatch) >= s.firstModel.BatchSize {
					s.inf.ProcessBatch(modelFirstBatch, s.firstModel)
					modelFirstBatch = nil
				}
			case models.SecondPrediction:
				modelSecondBatch = append(modelSecondBatch, req)
				if len(modelSecondBatch) >= s.secondModel.BatchSize {
					s.inf.ProcessBatch(modelSecondBatch, s.secondModel)
					modelSecondBatch = nil
				}
			}
			timer.Reset(50 * time.Millisecond)

		case <-timer.C:
			if len(modelFirstBatch) > 0 {
				s.inf.ProcessBatch(modelFirstBatch, s.firstModel)
				modelFirstBatch = nil
			}
			if len(modelSecondBatch) > 0 {
				s.inf.ProcessBatch(modelSecondBatch, s.secondModel)
				modelSecondBatch = nil
			}
			timer.Reset(50 * time.Millisecond)
		}
	}
}
