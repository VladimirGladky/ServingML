package modelUtils

const (
	SentimentOutputSize = 3
	EmotionOutputSize   = 28
)

func BatchResults(outputData []float32, batchSize int, outputSize int) [][]float64 {
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
