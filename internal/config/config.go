package config

import (
	"ServingML/pkg/postgres"
	"github.com/ilyakaznacheev/cleanenv"
)

type Config struct {
	Postgres            postgres.Config `yaml:"Postgres" env:"POSTGRES"`
	GrpcPort            string          `yaml:"grpc_port" env:"GRPC_PORT" env-default:"6060"`
	GrpcHost            string          `yaml:"grpc_host" env:"GRPC_HOST" env-default:"localhost"`
	Model1TokenizerPath string          `yaml:"model1_tokenizer_path" env:"MODEL1_TOKENIZER_PATH" env-default:"/home/smooth/Рабочий стол/ServingML/internal/modelWrapper/data/rubert-tiny2-russian-sentiment-onnx/tokenizer.json"`
	Model1Path          string          `yaml:"model1_model_path" env:"MODEL1_MODEL_PATH" env-default:"/home/smooth/Рабочий стол/ServingML/internal/modelWrapper/data/rubert-tiny2-russian-sentiment-onnx/model.onnx"`
	Model1OutputSize    int             `yaml:"model1_output_size" env:"MODEL1_OUTPUT_SIZE" env-default:"3"`
	Model1BatchSize     int             `yaml:"model1_batch_size" env:"MODEL1_BATCH_SIZE" env-default:"4"`
	Model2TokenizerPath string          `yaml:"model2_tokenizer_path" env:"MODEL2_TOKENIZER_PATH" env-default:"/home/smooth/Рабочий стол/ServingML/internal/modelWrapper/data/rubert-tiny2-russian-emotion-detection-ru-go-emotions/tokenizer.json"`
	Model2Path          string          `yaml:"model2_model_path" env:"MODEL2_MODEL_PATH" env-default:"/home/smooth/Рабочий стол/ServingML/internal/modelWrapper/data/rubert-tiny2-russian-emotion-detection-ru-go-emotions/model.onnx"`
	Model2OutputSize    int             `yaml:"model2_output_size" env:"MODEL2_OUTPUT_SIZE" env-default:"28"`
	Model2BatchSize     int             `yaml:"model2_batch_size" env:"MODEL2_BATCH_SIZE" env-default:"4"`
}

func New() (*Config, error) {
	var cfg Config
	if err := cleanenv.ReadConfig("./config/config.yaml", &cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}
