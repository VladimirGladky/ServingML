package inference

import (
	"ServingML/internal/domain/models"
	"ServingML/internal/modelWrapper"
	"ServingML/pkg/modelUtils"
	"fmt"

	"github.com/daulet/tokenizers"
	ort "github.com/yalue/onnxruntime_go"
)

type ServiceInferenceInterface interface {
	ProcessBatch(batch []*models.PredictionRequest, model *modelWrapper.WrapperModel)
}

type ServiceInference struct{}

func New() *ServiceInference {
	return &ServiceInference{}
}

func (s *ServiceInference) ProcessBatch(batch []*models.PredictionRequest, model *modelWrapper.WrapperModel) {
	var allIDs, allTypeIDs, allAttentionMasks [][]uint32
	for _, req := range batch {
		encoding := model.Tokenizer.EncodeWithOptions(
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
	outputSize := model.OutputSize
	outputShape := ort.NewShape(int64(len(batch)), int64(outputSize))
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		sendErrorToBatch(batch, err)
		return
	}
	defer outputTensor.Destroy()

	model.ModelMutex.Lock()
	err = model.Session.Run(inputTensors, []ort.Value{outputTensor})
	model.ModelMutex.Unlock()

	if err != nil {
		sendErrorToBatch(batch, err)
		return
	}

	outputData := outputTensor.GetData()
	results := modelUtils.BatchResults(outputData, len(batch), model.OutputSize)

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
