package models

type PredictionResponse struct {
	Probabilities []float64
	Error         error
}
