package models

type PredictionRequest struct {
	Text       string
	ResponseCh chan *PredictionResponse
}
