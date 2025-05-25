package service

import (
	"ServingML/internal/domain/models"
	"ServingML/internal/modelWrapper"
	"ServingML/internal/repository"
	"ServingML/pkg/modelUtils"
	"context"
	"fmt"
	"time"

	"github.com/daulet/tokenizers"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	SentimentOutputSize = 3
	EmotionOutputSize   = 28
)

type MLServiceInterface interface {
	Predict(ctx context.Context, text string) (string, error)
	StartBatchProcessor()
	processBatch(batch []*models.PredictionRequest)
}

type MLService struct {
	repository     repository.MLRepositoryInterface
	ctx            context.Context
	queue          chan *models.PredictionRequest
	sentimentModel *modelWrapper.WrapperModel
	emotionModel   *modelWrapper.WrapperModel
}

func New(ctx context.Context, sentimentModel *modelWrapper.WrapperModel, emotionModel *modelWrapper.WrapperModel) *MLService {
	return &MLService{
		ctx:            ctx,
		queue:          make(chan *models.PredictionRequest, 2000),
		sentimentModel: sentimentModel,
		emotionModel:   emotionModel,
	}
}

func (s *MLService) PredictSentiment(ctx context.Context, text string) (string, error) {
	respCh := make(chan *models.PredictionResponse, 1)
	select {
	case s.queue <- &models.PredictionRequest{
		Text:           text,
		ResponseCh:     respCh,
		PredictionType: models.SentimentPrediction,
	}:
	case <-ctx.Done():
		return "", ctx.Err()
	}

	select {
	case resp := <-respCh:
		if resp.Error != nil {
			return "", resp.Error
		}
		return convertSentiment(resp.Probabilities), nil
	}
}

func (s *MLService) PredictEmotion(ctx context.Context, text string) (string, error) {
	respCh := make(chan *models.PredictionResponse, 1)
	select {
	case s.queue <- &models.PredictionRequest{
		Text:           text,
		ResponseCh:     respCh,
		PredictionType: models.EmotionPrediction,
	}:
	case <-ctx.Done():
		return "", ctx.Err()
	}

	select {
	case resp := <-respCh:
		if resp.Error != nil {
			return "", resp.Error
		}
		return convertEmotion(resp.Probabilities), nil
	}
}

func (s *MLService) StartBatchProcessor() {
	sentimentBatch := make([]*models.PredictionRequest, 0)
	emotionBatch := make([]*models.PredictionRequest, 0)
	timer := time.NewTimer(50 * time.Millisecond)

	for {
		select {
		case req := <-s.queue:
			switch req.PredictionType {
			case models.SentimentPrediction:
				sentimentBatch = append(sentimentBatch, req)
				if len(sentimentBatch) >= 512 {
					s.processBatch(sentimentBatch, s.sentimentModel, "sentiment")
					sentimentBatch = nil
				}
			case models.EmotionPrediction:
				emotionBatch = append(emotionBatch, req)
				if len(emotionBatch) >= 512 {
					s.processBatch(emotionBatch, s.emotionModel, "emotion")
					emotionBatch = nil
				}
			}
			timer.Reset(50 * time.Millisecond)

		case <-timer.C:
			if len(sentimentBatch) > 0 {
				s.processBatch(sentimentBatch, s.sentimentModel, "sentiment")
				sentimentBatch = nil
			}
			if len(emotionBatch) > 0 {
				s.processBatch(emotionBatch, s.emotionModel, "emotion")
				emotionBatch = nil
			}
			timer.Reset(50 * time.Millisecond)
		}
	}
}

func (s *MLService) processBatch(batch []*models.PredictionRequest, model *modelWrapper.WrapperModel, typeModel string) {
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
	if typeModel != "emotion" && typeModel != "sentiment" {
		panic("wrong type")
	}
	outputSize := SentimentOutputSize
	if typeModel == "emotion" {
		outputSize = EmotionOutputSize
	}
	outputShape := ort.NewShape(int64(len(batch)), int64(outputSize))
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		sendErrorToBatch(batch, err)
		return
	}
	defer outputTensor.Destroy()

	s.sentimentModel.ModelMutex.Lock()
	err = model.Session.Run(inputTensors, []ort.Value{outputTensor})
	s.sentimentModel.ModelMutex.Unlock()

	if err != nil {
		sendErrorToBatch(batch, err)
		return
	}

	outputData := outputTensor.GetData()
	var results [][]float64
	if typeModel == "emotion" {
		results = modelUtils.BatchResults(outputData, len(batch), "emotion")
	} else {
		results = modelUtils.BatchResults(outputData, len(batch), "sentiment")
	}

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

func convertSentiment(probabilities []float64) string {
	if probabilities[0] > probabilities[1] && probabilities[0] > probabilities[2] {
		return "neutral"
	} else if probabilities[1] > probabilities[0] && probabilities[1] > probabilities[2] {
		return "positive"
	}
	return "negative"
}

func convertEmotion(probabilities []float64) string {
	emotions := []string{
		"восхищение",      // admiration
		"веселье",         // amusement
		"злость",          // anger
		"раздражение",     // annoyance
		"одобрение",       // approval
		"забота",          // caring
		"непонимание",     // confusion
		"любопытство",     // curiosity
		"желание",         // desire
		"разочарование",   // disappointment
		"неодобрение",     // disapproval
		"отвращение",      // disgust
		"смущение",        // embarrassment
		"возбуждение",     // excitement
		"страх",           // fear
		"признательность", // gratitude
		"горе",            // grief
		"радость",         // joy
		"любовь",          // love
		"нервозность",     // nervousness
		"оптимизм",        // optimism
		"гордость",        // pride
		"осознание",       // realization
		"облегчение",      // relief
		"раскаяние",       // remorse
		"грусть",          // sadness
		"удивление",       // surprise
		"нейтральность",   // neutral
	}

	if len(probabilities) != len(emotions) {
		return "неизвестно"
	}

	maxIndex := 0
	for i, p := range probabilities {
		if p > probabilities[maxIndex] {
			maxIndex = i
		}
	}

	return emotions[maxIndex]
}
