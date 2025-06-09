package batcher

import (
	"ServingML/internal/config"
	"ServingML/internal/domain/models"
	"ServingML/internal/inference"
	"ServingML/pkg/converter"
	"ServingML/pkg/logger"
	"context"
	"fmt"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

type ServiceBatcherInterface interface {
	Predict(ctx context.Context, modelName string, text string) (string, error)
	StartBatchProcessor()
}

type ServiceBatcher struct {
	ctx         context.Context
	queues      map[string]chan *models.PredictionRequest
	initialized uint32
	inf         *inference.ServiceInference
	mu          sync.Mutex
	batchSizes  map[string]int
	cfg         *config.Config
	idleTimer   *time.Timer
	timeout     int
}

func New(ctx context.Context, inf *inference.ServiceInference, batchSizes map[string]int, cfg *config.Config) *ServiceBatcher {
	queues := make(map[string]chan *models.PredictionRequest)
	for modelName := range batchSizes {
		queues[modelName] = make(chan *models.PredictionRequest, 1000)
	}
	timeout, err := strconv.Atoi(cfg.Timeout)
	if err != nil {
		timeout = 5
	}
	return &ServiceBatcher{
		ctx:        ctx,
		queues:     queues,
		inf:        inf,
		batchSizes: batchSizes,
		cfg:        cfg,
		timeout:    timeout,
		idleTimer:  time.NewTimer(time.Duration(timeout) * time.Millisecond),
	}
}

func (s *ServiceBatcher) Predict(ctx context.Context, modelName string, text string) (string, error) {
	if atomic.LoadUint32(&s.initialized) == 0 {
		if atomic.CompareAndSwapUint32(&s.initialized, 0, 1) {
			go s.StartBatchProcessor()
		}
	} else {
		s.resetIdleTimer()
	}
	queue, exists := s.queues[modelName]
	if !exists {
		return "", fmt.Errorf("model %s not found", modelName)
	}
	respCh := make(chan *models.PredictionResponse, 1)
	select {
	case queue <- &models.PredictionRequest{
		Text:       text,
		ResponseCh: respCh,
		ModelName:  modelName,
	}:
	case <-ctx.Done():
		return "", ctx.Err()
	}

	select {
	case resp := <-respCh:
		if resp.Error != nil {
			return "", resp.Error
		}
		return converter.Convert(modelName, resp.Probabilities), nil
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

func (s *ServiceBatcher) StartBatchProcessor() {
	defer atomic.StoreUint32(&s.initialized, 0)
	batches := make(map[string][]*models.PredictionRequest)
	timers := make(map[string]*time.Timer)
	batchTimeout, err := strconv.Atoi(s.cfg.BatchTimeout)
	if err != nil {
		batchTimeout = 50
	}

	for modelName := range s.queues {
		batches[modelName] = make([]*models.PredictionRequest, 0)
		timers[modelName] = time.NewTimer(time.Duration(batchTimeout) * time.Millisecond)
	}

	for {
		select {
		case <-s.idleTimer.C:
			logger.GetLoggerFromCtx(s.ctx).Info("StartBatchProcessor stopped")
			return
		default:
			processed := false
			for modelName, queue := range s.queues {
				select {
				case req := <-queue:
					processed = true
					batches[modelName] = append(batches[modelName], req)
					if len(batches[modelName]) >= s.batchSizes[modelName] {
						s.inf.ProcessBatch(batches[modelName], modelName)
						batches[modelName] = nil
						timers[modelName].Reset(time.Duration(batchTimeout) * time.Millisecond)
					}
				case <-timers[modelName].C:
					processed = true
					if len(batches[modelName]) > 0 {
						s.inf.ProcessBatch(batches[modelName], modelName)
						batches[modelName] = nil
					}
					timers[modelName].Reset(time.Duration(batchTimeout) * time.Millisecond)
				}
			}
			if processed {
				s.resetIdleTimer()
			}
		}
	}
}

func (s *ServiceBatcher) resetIdleTimer() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.idleTimer.Stop() {
		select {
		case <-s.idleTimer.C:
		default:
		}
	}
	s.idleTimer.Reset(time.Duration(s.timeout) * time.Second)
}
