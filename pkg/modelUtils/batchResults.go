package modelUtils

func BatchResults(outputData []float32, batchSize int) [][]float64 {
	results := make([][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		start := i * 3
		end := start + 3
		if end > len(outputData) {
			break
		}
		sm, _ := SoftMax(outputData[start:end])
		results[i] = sm
	}
	return results
}
