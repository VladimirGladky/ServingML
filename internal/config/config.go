package config

import (
	"github.com/ilyakaznacheev/cleanenv"
)

type ModelConfig struct {
	Name          string `yaml:"name"`
	TokenizerPath string `yaml:"tokenizer_path"`
	ModelPath     string `yaml:"model_path"`
	InputSize     int    `yaml:"input_size"`
	OutputSize    int    `yaml:"output_size"`
	BatchSize     int    `yaml:"batch_size"`
}

type Config struct {
	GrpcPort     string        `yaml:"grpc_port" env:"GRPC_PORT" env-default:"6060"`
	GrpcHost     string        `yaml:"grpc_host" env:"GRPC_HOST" env-default:"localhost"`
	BatchTimeout string        `yaml:"batch_timeout" env:"BATCH_TIMEOUT" env-default:"50"`
	Timeout      string        `yaml:"timeout" env:"TIMEOUT" env-default:"50"`
	Models       []ModelConfig `yaml:"models"`
}

func New() (*Config, error) {
	var cfg Config
	if err := cleanenv.ReadConfig("./config/config.yaml", &cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}
