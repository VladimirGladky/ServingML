package converter

func Convert(modelName string, probabilities []float64) string {
	switch modelName {
	case "sentiment-analysis":
		return ConvertFirstModel(probabilities)
	case "emotion-detection":
		return ConvertSecondModel(probabilities)
	default:
		return "unknown model"
	}
}

func ConvertFirstModel(probabilities []float64) string {
	if probabilities[0] > probabilities[1] && probabilities[0] > probabilities[2] {
		return "neutral"
	} else if probabilities[1] > probabilities[0] && probabilities[1] > probabilities[2] {
		return "positive"
	}
	return "negative"
}

func ConvertSecondModel(probabilities []float64) string {
	emotions := []string{
		"восхищение",      // admiration
		"веселье",         // amusement
		"злость",          // anger
		"раздражение",     // annoyance
		"одобрение",       // approval
		"забота",          // caring
		"непонимание",     // confusion
		"любопытство",     // curiosity
		"желание",         // desire
		"разочарование",   // disappointment
		"неодобрение",     // disapproval
		"отвращение",      // disgust
		"смущение",        // embarrassment
		"возбуждение",     // excitement
		"страх",           // fear
		"признательность", // gratitude
		"горе",            // grief
		"радость",         // joy
		"любовь",          // love
		"нервозность",     // nervousness
		"оптимизм",        // optimism
		"гордость",        // pride
		"осознание",       // realization
		"облегчение",      // relief
		"раскаяние",       // remorse
		"грусть",          // sadness
		"удивление",       // surprise
		"нейтральность",   // neutral
	}

	if len(probabilities) != len(emotions) {
		return "неизвестно"
	}

	maxIndex := 0
	for i, p := range probabilities {
		if p > probabilities[maxIndex] {
			maxIndex = i
		}
	}

	return emotions[maxIndex]
}
