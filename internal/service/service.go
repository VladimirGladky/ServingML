package service

import (
	"ServingML/internal/domain/models"
	"ServingML/internal/modelWrapper"
	"ServingML/internal/repository"
	"ServingML/pkg/modelUtils"
	"context"
	"fmt"
	"github.com/daulet/tokenizers"
	ort "github.com/yalue/onnxruntime_go"
	"time"
)

type MLServiceInterface interface {
	Predict(ctx context.Context, text string) (string, error)
	startBatchProcessor()
	processBatch(batch []*models.PredictionRequest)
}

type MLService struct {
	repository repository.MLRepositoryInterface
	ctx        context.Context
	queue      chan *models.PredictionRequest
	model      *modelWrapper.WrapperModel
}

func New(ctx context.Context) *MLService {
	return &MLService{ctx: ctx}
}

func (s *MLService) Predict(ctx context.Context, text string) (string, error) {
	respCh := make(chan *models.PredictionResponse, 1)
	select {
	case s.queue <- &models.PredictionRequest{
		Text:       text,
		ResponseCh: respCh,
	}:
	case <-ctx.Done():
		return "", ctx.Err()
	}

	select {
	case resp := <-respCh:
		if resp.Error != nil {
			return "", resp.Error
		}
		if resp.Probabilities[0] > resp.Probabilities[1] && resp.Probabilities[0] > resp.Probabilities[2] {
			return "neutral", nil
		} else if resp.Probabilities[1] > resp.Probabilities[0] && resp.Probabilities[1] > resp.Probabilities[2] {
			return "positive", nil
		}
		return "negative", nil
	}
}

func (s *MLService) startBatchProcessor() {
	var batch []*models.PredictionRequest
	timer := time.NewTimer(100 * time.Millisecond)

	for {
		select {
		case req := <-s.queue:
			batch = append(batch, req)
			if len(batch) >= 32 {
				s.processBatch(batch)
				batch = nil
				timer.Reset(100 * time.Millisecond)
			}
		case <-timer.C:
			if len(batch) > 0 {
				s.processBatch(batch)
				batch = nil
			}
			timer.Reset(100 * time.Millisecond)
		}
	}
}

func (s *MLService) processBatch(batch []*models.PredictionRequest) {
	var allIDs, allTypeIDs, allAttentionMasks [][]uint32

	for _, req := range batch {
		encoding := s.model.Tokenizer.EncodeWithOptions(
			req.Text,
			true,
			tokenizers.WithReturnTypeIDs(),
			tokenizers.WithReturnAttentionMask(),
		)
		allIDs = append(allIDs, encoding.IDs)
		allTypeIDs = append(allTypeIDs, encoding.TypeIDs)
		allAttentionMasks = append(allAttentionMasks, encoding.AttentionMask)
	}

	inputTensors := make([]ort.Value, 3)
	for i, data := range [][][]uint32{allIDs, allTypeIDs, allAttentionMasks} {
		tensor, err := modelUtils.CreateInputTensor(data)
		if err != nil {
			sendErrorToBatch(batch, err)
			return
		}
		defer tensor.Destroy()
		inputTensors[i] = tensor
	}

	outputShape := ort.NewShape(int64(len(batch)), 3)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		sendErrorToBatch(batch, err)
		return
	}
	defer outputTensor.Destroy()

	s.model.ModelMutex.Lock()
	err = s.model.Session.Run(inputTensors, []ort.Value{outputTensor})
	s.model.ModelMutex.Unlock()

	if err != nil {
		sendErrorToBatch(batch, err)
		return
	}

	outputData := outputTensor.GetData()
	results := modelUtils.BatchResults(outputData, len(batch))

	for i, req := range batch {
		if i < len(results) {
			req.ResponseCh <- &models.PredictionResponse{Probabilities: results[i]}
		} else {
			req.ResponseCh <- &models.PredictionResponse{Error: fmt.Errorf("result index out of range")}
		}
		close(req.ResponseCh)
	}
}

func sendErrorToBatch(batch []*models.PredictionRequest, err error) {
	for _, req := range batch {
		req.ResponseCh <- &models.PredictionResponse{Error: err}
		close(req.ResponseCh)
	}
}
