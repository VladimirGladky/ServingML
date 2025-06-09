package models

type PredictionRequest struct {
	Text       string
	ModelName  string
	ResponseCh chan *PredictionResponse
}
