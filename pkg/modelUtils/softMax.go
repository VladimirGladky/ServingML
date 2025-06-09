package modelUtils

import "math"

func SoftMax(array []float32) ([]float64, error) {
	input := make([]float64, len(array))
	for i, x := range array {
		input[i] = float64(x)
	}
	s := 0.0
	c := math.Inf(-1)
	for _, e := range input {
		c = math.Max(e, c)
	}
	for _, e := range input {
		s += math.Exp(e - c)
	}
	sm := make([]float64, len(input))
	for i, v := range input {
		sm[i] = math.Exp(v-c) / s
	}
	return sm, nil
}
