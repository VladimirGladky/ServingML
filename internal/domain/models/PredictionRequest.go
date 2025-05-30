package models

type PredictionType int

const (
	FirstPrediction PredictionType = iota
	SecondPrediction
)

type PredictionRequest struct {
	Text string
	PredictionType
	ResponseCh chan *PredictionResponse
}
