package models

type PredictionType int

const (
	SentimentPrediction PredictionType = iota
	EmotionPrediction
)

type PredictionRequest struct {
	Text string
	PredictionType
	ResponseCh chan *PredictionResponse
}
