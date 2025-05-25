package modelUtils

import ort "github.com/yalue/onnxruntime_go"

func CreateInputTensor(data [][]uint32) (*ort.Tensor[int64], error) {
	maxLen := 0
	for _, seq := range data {
		if len(seq) > maxLen {
			maxLen = len(seq)
		}
	}

	flatData := make([]int64, 0, len(data)*maxLen)
	for _, seq := range data {
		padded := make([]int64, maxLen)
		for i := 0; i < maxLen; i++ {
			if i < len(seq) {
				padded[i] = int64(seq[i])
			} else {
				padded[i] = 0
			}
		}
		flatData = append(flatData, padded...)
	}

	shape := ort.NewShape(int64(len(data)), int64(maxLen))
	return ort.NewTensor(shape, flatData)
}
