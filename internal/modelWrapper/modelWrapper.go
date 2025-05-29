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
	BatchSize  int
	OutputSize int
	ModelMutex sync.Mutex
}

func NewWrapperModel(tokenizerPath string, modelPath string, batchSize, outputSize int) (*WrapperModel, error) {
	tk, err := tokenizers.FromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %v", err)
	}
	input, output, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load info: %v", err)
	}
	var inputs []string
	for _, x := range input {
		inputs = append(inputs, x.Name)
	}
	var outputs []string
	for _, x := range output {
		outputs = append(outputs, x.Name)
	}
	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input_ids", "token_type_ids", "attention_mask"},
		outputs,
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %v", err)
	}

	return &WrapperModel{
		Session:    session,
		Tokenizer:  tk,
		InputNames: inputs,
		OutputName: outputs,
		BatchSize:  batchSize,
		OutputSize: outputSize,
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
