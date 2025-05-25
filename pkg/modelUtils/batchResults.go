package modelUtils

const (
	SentimentOutputSize = 3
	EmotionOutputSize   = 28
)

func BatchResults(outputData []float32, batchSize int, modelType string) [][]float64 {
	var outputSize int
	if modelType != "emotion" && modelType != "sentiment" {
		panic("wrong type")
	}
	switch modelType {
	case "emotion":
		outputSize = EmotionOutputSize
	case "sentiment":
		outputSize = SentimentOutputSize
	default:
		outputSize = SentimentOutputSize
	}

	results := make([][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		start := i * outputSize
		end := start + outputSize
		if end > len(outputData) {
			break
		}
		sm, _ := SoftMax(outputData[start:end])
		results[i] = sm
	}
	return results
}
