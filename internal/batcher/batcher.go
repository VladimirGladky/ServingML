package batcher

import (
	"ServingML/internal/config"
	"ServingML/internal/domain/models"
	"ServingML/internal/inference"
	"ServingML/pkg/converter"
	"ServingML/pkg/logger"
	"context"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

type ServiceBatcherInterface interface {
	Predict(ctx context.Context, text string, modelName string) (string, error)
	StartBatchProcessor()
}

type ServiceBatcher struct {
	ctx          context.Context
	queue        map[string]chan *models.PredictionRequest
	initialized  uint32
	inf          *inference.ServiceInference
	mu           sync.Mutex
	batchSizes   map[string]int
	cfg          *config.Config
	idleTimer    *time.Timer
	batchTimeout int
	timeout      int
	initOnce     sync.Once
}

func New(ctx context.Context, inf *inference.ServiceInference, batchSizes map[string]int, cfg *config.Config) *ServiceBatcher {
	queues := make(map[string]chan *models.PredictionRequest)
	for modelName := range batchSizes {
		queues[modelName] = make(chan *models.PredictionRequest, 1000)
	}
	batchTimeout, err := strconv.Atoi(cfg.BatchTimeout)
	if err != nil {
		batchTimeout = 50
	}
	timeout, err := strconv.Atoi(cfg.Timeout)
	if err != nil {
		timeout = 5
	}
	return &ServiceBatcher{
		ctx:          ctx,
		queue:        queues,
		inf:          inf,
		batchSizes:   batchSizes,
		cfg:          cfg,
		batchTimeout: batchTimeout,
		timeout:      timeout,
		idleTimer:    time.NewTimer(time.Duration(timeout) * time.Second),
	}
}

func (s *ServiceBatcher) Predict(ctx context.Context, text string, modelName string) (string, error) {
	if atomic.LoadUint32(&s.initialized) == 0 {
		if atomic.CompareAndSwapUint32(&s.initialized, 0, 1) {
			go s.StartBatchProcessor()
		}
	} else {
		s.resetIdleTimer()
	}
	respCh := make(chan *models.PredictionResponse, 1)
	select {
	case s.queue[modelName] <- &models.PredictionRequest{
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
	logger.GetLoggerFromCtx(s.ctx).Info("StartBatchProcessor")
	defer atomic.StoreUint32(&s.initialized, 0)

	var wg sync.WaitGroup
	stopChan := make(chan struct{})
	idleTimer := time.NewTimer(time.Duration(s.timeout) * time.Second)

	for modelName, queue := range s.queue {
		wg.Add(1)
		go func(name string, q chan *models.PredictionRequest) {
			defer wg.Done()

			batch := make([]*models.PredictionRequest, 0, s.batchSizes[name])
			timer := time.NewTimer(time.Duration(s.batchTimeout) * time.Millisecond)
			defer timer.Stop()

			for {
				select {
				case req := <-q:
					if !idleTimer.Stop() {
						select {
						case <-idleTimer.C:
						default:
						}
					}
					idleTimer.Reset(time.Duration(s.timeout) * time.Second)
					batch = append(batch, req)
					if len(batch) >= cap(batch) {
						s.inf.ProcessBatch(batch, name)
						batch = batch[:0]
						timer.Reset(time.Duration(s.batchTimeout) * time.Millisecond)
					}
				case <-timer.C:
					if len(batch) > 0 {
						s.inf.ProcessBatch(batch, name)
						batch = batch[:0]
					}
					timer.Reset(time.Duration(s.batchTimeout) * time.Millisecond)
				case <-idleTimer.C:
					return

				case <-stopChan:
					logger.GetLoggerFromCtx(s.ctx).Info("Stopped StartBatchProcessor")
					return
				}
			}
		}(modelName, queue)
	}

	select {
	case <-idleTimer.C:
		close(stopChan)
	case <-s.ctx.Done():
		close(stopChan)
	}

	wg.Wait()
}

func (s *ServiceBatcher) resetIdleTimer() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.idleTimer != nil {
		if !s.idleTimer.Stop() {
			select {
			case <-s.idleTimer.C:
			default:
			}
		}
		s.idleTimer.Reset(time.Duration(s.timeout) * time.Second)
	}
}
