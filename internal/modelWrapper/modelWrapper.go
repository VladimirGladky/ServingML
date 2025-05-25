package modelWrapper

import (
	"fmt"
	"sync"

	"github.com/daulet/tokenizers"
	ort "github.com/yalue/onnxruntime_go"
)

type WrapperModel struct {
	Session    *ort.DynamicAdvancedSession
	Tokenizer  *tokenizers.Tokenizer
	InputNames []string
	OutputName []string
	ModelMutex sync.Mutex
}

func NewWrapperModel(tokenizerPath string, modelPath string) (*WrapperModel, error) {
	tk, err := tokenizers.FromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %v", err)
	}

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input_ids", "token_type_ids", "attention_mask"},
		[]string{"logits"},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %v", err)
	}

	return &WrapperModel{
		Session:    session,
		Tokenizer:  tk,
		InputNames: []string{"input_ids", "token_type_ids", "attention_mask"},
		OutputName: []string{"logits"},
	}, nil
}

func (m *WrapperModel) Close() {
	m.ModelMutex.Lock()
	defer m.ModelMutex.Unlock()

	if m.Session != nil {
		m.Session.Destroy()
	}
	if m.Tokenizer != nil {
		m.Tokenizer.Close()
	}
	ort.DestroyEnvironment()
}
